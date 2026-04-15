"""
Uni-Mol LoRA Trainer — Shared Utilities

Provides the UniMolLoRAClassifier and UniMolLoRATrainer used by all
finetune_unimol_*.py scripts, mirroring the HuggingFace Trainer + PEFT
pattern used for ChemBERTa and MolFormer in the main EffiChem-2.0 pipeline.

Multimodal Input Design
-----------------------
ChemBERTa / MolFormer:  SMILES token sequence  →  1D language model
Uni-Mol (this module):  Two additional modalities derived from the same SMILES

  Modality 1 — 3D conformer (primary Uni-Mol input)
      SMILES → RDKit ETKDGv3 + MMFF94 → atom types + 3D coordinates
      Processed by Uni-Mol's SE(3)-equivariant Transformer (hidden=512, 15 layers)

  Modality 2 — Morgan fingerprint (ECFP4, 2048-bit)
      SMILES → RDKit Morgan fingerprint (radius=2, 2048 bits)
      Concatenated with Uni-Mol CLS embedding for a hybrid 2D+3D representation

Final feature vector per molecule: 512 (Uni-Mol CLS) + 2048 (Morgan FP) = 2560-dim
Downstream classifier projects 2560 → num_classes.

LoRA target modules: in_proj (combined QKV) and out_proj in every Uni-Mol
attention layer.  The encoder is finetuned via direct UniMolModel.forward()
calls that maintain the autograd graph — NOT via UniMolRepr.get_repr() which
runs under torch.no_grad() and returns numpy, breaking gradient flow.

Gradient flow fix
-----------------
UniMolRepr is a plain Python class (not nn.Module).  This has two consequences:

  1. state_dict() only captures head weights; the encoder is invisible to PyTorch.
     Fix: save encoder weights separately as unimol_encoder.pt alongside
     pytorch_model.pt (head) and load them back in load_finetuned_unimol().

  2. get_repr() runs the encoder through an internal Trainer.inference() pipeline
     that wraps everything in torch.no_grad() and returns numpy arrays.
     Wrapping the output in torch.tensor() creates a detached leaf, so LoRA
     adapters receive zero gradients and are never updated.
     Fix: pre-process SMILES once with DataHub (conformer gen + tokenisation,
     non-differentiable) and call UniMolModel.forward() directly during training
     to maintain the computation graph.
"""

import gc
import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from unimol_tools import UniMolRepr
    from unimol_tools.data import DataHub
    from unimol_tools.utils import pad_1d_tokens, pad_2d, pad_coords
    UNIMOL_AVAILABLE = True
except ImportError:
    UNIMOL_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNIMOL_HIDDEN  = 512   # Uni-Mol CLS embedding dimension
MORGAN_BITS    = 2048  # ECFP4 fingerprint length
COMBINED_DIM   = UNIMOL_HIDDEN + MORGAN_BITS  # 2560

_PLACEHOLDER_SMILES = "CCO"   # ethanol — always valid for Uni-Mol


# ── Modality 2: Morgan fingerprint ────────────────────────────────────────────

def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = MORGAN_BITS) -> np.ndarray:
    """ECFP4 Morgan fingerprint (2048-bit) from SMILES. Returns zeros on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def batch_morgan_fps(smiles_list: List[str]) -> np.ndarray:
    """Returns (N, 2048) float32 array of Morgan fingerprints."""
    return np.vstack([smiles_to_morgan_fp(s) for s in smiles_list])


# ── UniMol conformer preprocessing ────────────────────────────────────────────

def _canonicalize(smi: str) -> str:
    """Return RDKit canonical SMILES, or placeholder on failure."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return _PLACEHOLDER_SMILES
        return Chem.MolToSmiles(mol)
    except Exception:
        return _PLACEHOLDER_SMILES


def preprocess_smiles_for_unimol(smiles_list: List[str], repr_model) -> List[Dict]:
    """
    Convert a list of SMILES to per-molecule UniMol input dicts.

    Uses DataHub + ConformerGen for conformer generation and tokenisation —
    non-differentiable preprocessing done ONCE at dataset creation time.
    The resulting dicts are stored and fed directly to UniMolModel.forward()
    during training so the autograd graph is maintained and LoRA adapters
    receive gradients.

    Returns a list of dicts, each with keys:
        src_tokens   (N_atoms,)        int64 — atom token ids
        src_distance (N_atoms, N_atoms) float32 — pairwise distances
        src_coord    (N_atoms, 3)       float32 — 3D coordinates
        src_edge_type(N_atoms, N_atoms) int64 — atom-pair edge types
    """
    sanitised = [_canonicalize(s) for s in smiles_list]
    hub = DataHub(
        data=sanitised,
        task="repr",
        is_train=False,
        data_type=repr_model.params.get("data_type", "molecule"),
        remove_hs=repr_model.params.get("remove_hs", False),
        model_name=repr_model.params.get("model_name", "unimolv1"),
    )
    return hub.data["unimol_input"]


def _collate_unimol_inputs(samples: List[Dict], padding_idx: int) -> Dict[str, torch.Tensor]:
    """
    Pad a list of per-molecule UniMol input dicts into batch tensors.
    Mirrors UniMolModel.batch_collate_fn but returns a plain dict.
    """
    batch: Dict[str, torch.Tensor] = {}
    for k in samples[0].keys():
        if k == "src_coord":
            batch[k] = pad_coords(
                [torch.tensor(s[k]).float() for s in samples], pad_idx=0.0
            )
        elif k == "src_edge_type":
            batch[k] = pad_2d(
                [torch.tensor(s[k]).long() for s in samples], pad_idx=padding_idx
            )
        elif k == "src_distance":
            batch[k] = pad_2d(
                [torch.tensor(s[k]).float() for s in samples], pad_idx=0.0
            )
        elif k == "src_tokens":
            batch[k] = pad_1d_tokens(
                [torch.tensor(s[k]).long() for s in samples], pad_idx=padding_idx
            )
    return batch


# ── Dataset ────────────────────────────────────────────────────────────────────

class MolDataset(Dataset):
    """
    Holds per-molecule data: labels, Morgan FPs, and pre-computed UniMol inputs.

    unimol_inputs is a list of dicts produced by preprocess_smiles_for_unimol().
    Storing them here avoids re-running the non-differentiable conformer pipeline
    every epoch.
    """

    def __init__(
        self,
        smiles:        List[str],
        labels:        List[int],
        unimol_inputs: List[Dict],
    ):
        self.smiles        = smiles
        self.labels        = labels
        self.unimol_inputs = unimol_inputs
        logging.info(f"  Precomputing Morgan fingerprints for {len(smiles)} molecules…")
        self.morgan_fps = batch_morgan_fps(smiles)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "smiles":       self.smiles[idx],
            "label":        self.labels[idx],
            "morgan_fp":    self.morgan_fps[idx],
            "unimol_input": self.unimol_inputs[idx],
        }


def collate_fn(batch: List[Dict], padding_idx: int) -> Dict:
    """
    Collate a list of MolDataset samples into a batch dict.

    Pass to DataLoader via functools.partial:
        loader = DataLoader(..., collate_fn=partial(collate_fn, padding_idx=pi))
    """
    return {
        "unimol_batch": _collate_unimol_inputs(
            [b["unimol_input"] for b in batch], padding_idx
        ),
        "labels":    torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "morgan_fps": torch.tensor(
            np.stack([b["morgan_fp"] for b in batch]), dtype=torch.float32
        ),
    }


# ── Model ──────────────────────────────────────────────────────────────────────

class UniMolLoRAClassifier(nn.Module):
    """
    Uni-Mol encoder (optionally LoRA-adapted) + Morgan FP + classification head.

    Forward input:  unimol_batch (Dict of padded tensors) + morgan_fps (Tensor N×2048)
    Forward output: (logits, combined_embedding)
        logits            : (N, num_classes)
        combined_embedding: (N, 512 + 2048) — used for downstream extraction

    The encoder is called via _get_cls_from_batch() which invokes
    UniMolModel.forward() directly, maintaining the autograd graph and allowing
    LoRA adapter gradients to flow.
    """

    def __init__(
        self,
        num_classes:  int,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        if not UNIMOL_AVAILABLE:
            raise ImportError(
                "unimol_tools is required. "
                "Install: pip install unimol_tools"
            )

        self.num_classes = num_classes
        # UniMolRepr is a plain Python wrapper (not nn.Module); the actual
        # nn.Module is at self._repr.model (a UniMolModel).
        self._repr = UniMolRepr(data_type="molecule", remove_hs=False)

        # Classification head: 2560 → num_classes
        self.head = nn.Sequential(
            nn.LayerNorm(COMBINED_DIM),
            nn.Dropout(head_dropout),
            nn.Linear(COMBINED_DIM, 512),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, num_classes),
        )

    # ------------------------------------------------------------------
    # Core forward helpers
    # ------------------------------------------------------------------

    def _get_cls_from_batch(self, unimol_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Call UniMolModel.forward() with the pre-batched conformer tensors.
        Returns CLS token embeddings, shape (N, 512), WITH gradient tracking.

        This is the gradient-enabled replacement for get_repr().  By calling the
        nn.Module directly (instead of going through UniMolRepr.get_repr() →
        Trainer.inference() → numpy), the LoRA adapter parameters participate
        in the computation graph and receive non-zero gradients on backward().
        """
        src_tokens    = unimol_batch["src_tokens"].to(DEVICE)
        src_distance  = unimol_batch["src_distance"].to(DEVICE)
        src_coord     = unimol_batch["src_coord"].to(DEVICE)
        src_edge_type = unimol_batch["src_edge_type"].to(DEVICE)
        return self._repr.model(
            src_tokens, src_distance, src_coord, src_edge_type,
            return_repr=True, return_atomic_reprs=False,
        )  # (N, 512)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        unimol_batch: Dict[str, torch.Tensor],
        morgan_fps:   torch.Tensor,            # (N, 2048)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_emb    = self._get_cls_from_batch(unimol_batch)   # (N, 512)
        morgan_fps = morgan_fps.to(DEVICE)
        combined   = torch.cat([cls_emb, morgan_fps], dim=-1)  # (N, 2560)
        logits     = self.head(combined)
        return logits, combined

    # ------------------------------------------------------------------
    # Embedding extraction (no grad, no classification head)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_embeddings(
        self,
        smiles_list: List[str],
        batch_size:  int = 512,
    ) -> np.ndarray:
        """
        Returns (N, 2560) float32 combined embeddings:
          - [:, :512]   — Uni-Mol CLS (3D-structure-aware, finetuned encoder)
          - [:, 512:]   — Morgan ECFP4 fingerprint (2D structural)

        Pre-processes all SMILES once (conformer gen + tokenisation) then calls
        the encoder via _get_cls_from_batch() in inference mode.
        """
        self.eval()
        self._repr.model.eval()

        unimol_inputs = preprocess_smiles_for_unimol(smiles_list, self._repr)
        morgan_all    = batch_morgan_fps(smiles_list)
        padding_idx   = self._repr.model.padding_idx
        all_embs      = []

        for i in range(0, len(smiles_list), batch_size):
            batch_inputs = unimol_inputs[i : i + batch_size]
            unimol_batch = _collate_unimol_inputs(batch_inputs, padding_idx)
            batch_fp     = torch.tensor(
                morgan_all[i : i + batch_size], dtype=torch.float32
            )
            _, emb = self.forward(unimol_batch, batch_fp)
            all_embs.append(emb.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)


# ── LoRA application ───────────────────────────────────────────────────────────

def apply_lora_to_unimol(
    model:       UniMolLoRAClassifier,
    r:           int,
    lora_alpha:  int,
    dropout:     float,
) -> UniMolLoRAClassifier:
    """
    Apply PEFT LoRA adapters to Uni-Mol's attention projection layers.

    UniMolModel uses in_proj (combined QKV) rather than separate q/k/v_proj.
    LoRA is applied to in_proj and out_proj when present.

    Falls back to full fine-tuning of the encoder if PEFT is unavailable
    or no matching layer names are found.
    """
    if not PEFT_AVAILABLE:
        logging.warning("peft not available — training full Uni-Mol encoder without LoRA")
        return model

    encoder = getattr(model._repr, "model", None)
    if encoder is None:
        logging.warning("Cannot locate Uni-Mol encoder (_repr.model) — skipping LoRA")
        return model

    all_module_names = {name.split(".")[-1] for name, _ in encoder.named_modules()}

    # UniMolModel uses in_proj (combined QKV); older or different versions may
    # use separate q_proj/k_proj/v_proj.  Detect which is present.
    if "q_proj" in all_module_names and "k_proj" in all_module_names:
        target_modules = ["q_proj", "k_proj", "v_proj"]
        logging.info("  LoRA target: q_proj, k_proj, v_proj (split-QKV attention)")
    elif "in_proj" in all_module_names:
        target_modules = ["in_proj"]
        logging.info("  LoRA target: in_proj (combined-QKV attention)")
    else:
        logging.warning(
            "No recognised attention projection layers found in Uni-Mol encoder. "
            "Training full encoder without LoRA."
        )
        return model

    if "out_proj" in all_module_names:
        target_modules.append("out_proj")

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )

    model._repr.model = get_peft_model(encoder, peft_config)
    model._repr.model.print_trainable_parameters()
    return model


# ── Loss functions ─────────────────────────────────────────────────────────────

def focal_loss(
    logits:  torch.Tensor,
    labels:  torch.Tensor,
    gamma:   float = 2.0,
    alpha:   Optional[torch.Tensor] = None,  # per-class weights, shape (num_classes,)
) -> torch.Tensor:
    """Focal loss with optional per-class alpha weighting.

    alpha should be a (num_classes,) tensor of per-class weights, e.g. computed
    as (1 - class_frequency) to up-weight minority classes.
    If None, all classes are weighted equally (alpha=1).
    """
    log_prob   = F.log_softmax(logits, dim=-1)
    prob       = torch.exp(log_prob)
    targets_oh = F.one_hot(labels, num_classes=logits.shape[-1]).float()
    pt         = (prob * targets_oh).sum(dim=-1)
    if alpha is not None:
        at = (alpha.to(logits.device) * targets_oh).sum(dim=-1)
    else:
        at = 1.0
    fl = -at * (1 - pt) ** gamma * (log_prob * targets_oh).sum(dim=-1)
    return fl.mean()


def weighted_cross_entropy(
    logits:       torch.Tensor,
    labels:       torch.Tensor,
    class_counts: torch.Tensor,   # number of samples per class
) -> torch.Tensor:
    """Inverse-frequency weighted cross-entropy."""
    total   = class_counts.float().sum()
    n_cls   = len(class_counts)
    weights = total / (n_cls * class_counts.float())
    weights = (weights / weights.sum() * n_cls).to(logits.device)
    return F.cross_entropy(logits, labels, weight=weights)


# ── Trainer ────────────────────────────────────────────────────────────────────

class UniMolLoRATrainer:
    """
    Custom trainer for UniMolLoRAClassifier.
    Mirrors the HuggingFace Trainer pattern (EarlyStopping, best-model restore,
    metric tracking) used in the ChemBERTa/MolFormer scripts.
    """

    def __init__(
        self,
        model:        UniMolLoRAClassifier,
        loss_type:    str,               # "focal" | "weighted"
        class_counts: Optional[torch.Tensor] = None,
        lr:           float = 1e-4,
        max_epochs:   int   = 30,
        batch_size:   int   = 32,
        patience:     int   = 5,
        weight_decay: float = 1e-2,
        seed:         int   = 42,
        wandb_run     = None,
    ):
        self.model        = model.to(DEVICE)
        self.loss_type    = loss_type
        self.class_counts = class_counts
        self.lr           = lr
        self.max_epochs   = max_epochs
        self.batch_size   = batch_size
        self.patience     = patience
        self.weight_decay = weight_decay
        self.wandb_run    = wandb_run

        torch.manual_seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "focal":
            alpha = None
            if self.class_counts is not None:
                total = self.class_counts.float().sum()
                alpha = 1.0 - self.class_counts.float() / total
            return focal_loss(logits, labels, alpha=alpha)
        if self.loss_type == "weighted" and self.class_counts is not None:
            return weighted_cross_entropy(logits, labels, self.class_counts)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------

    def train(
        self,
        train_smiles: List[str], train_labels: List[int],
        val_smiles:   List[str], val_labels:   List[int],
        train_unimol: List[Dict] = None,
        val_unimol:   List[Dict] = None,
    ) -> Dict:
        """Train the model. Returns best-validation metrics dict.

        Args:
            train_unimol: Pre-computed DataHub outputs for training set.
                If None, will be computed from train_smiles (expensive for
                large datasets with many sweep trials — pass pre-computed
                values to avoid redundant conformer generation).
            val_unimol:   Same for validation set.
        """

        padding_idx = self.model._repr.model.padding_idx

        # Pre-process conformers once (non-differentiable; cached in dataset)
        if train_unimol is None:
            logging.info("  Pre-processing conformers for train split…")
            train_unimol = preprocess_smiles_for_unimol(train_smiles, self.model._repr)
        else:
            logging.info("  Using pre-computed conformers for train split.")
        if val_unimol is None:
            logging.info("  Pre-processing conformers for val split…")
            val_unimol = preprocess_smiles_for_unimol(val_smiles, self.model._repr)
        else:
            logging.info("  Using pre-computed conformers for val split.")

        train_ds = MolDataset(train_smiles, train_labels, train_unimol)
        val_ds   = MolDataset(val_smiles,   val_labels,   val_unimol)

        _collate = partial(collate_fn, padding_idx=padding_idx)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=True, collate_fn=_collate, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size,
            shuffle=False, collate_fn=_collate, num_workers=0,
        )

        # Separate LRs: encoder (LoRA) gets base LR, head gets 5× LR
        optimizer = AdamW(
            [
                {"params": self.model._repr.model.parameters(), "lr": self.lr},
                {"params": self.model.head.parameters(), "lr": self.lr * 5},
            ],
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        best_mcc     = -1.0
        best_metrics = {}
        patience_cnt = 0
        best_state   = None

        for epoch in range(self.max_epochs):
            # ── Train ──
            self.model.train()
            self.model._repr.model.train()   # UniMolRepr sets eval(); override
            epoch_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                labels     = batch["labels"].to(DEVICE)
                morgan_fps = batch["morgan_fps"]

                logits, _ = self.model(batch["unimol_batch"], morgan_fps)
                loss      = self._loss(logits, labels)

                loss.backward()
                all_params = (
                    list(self.model._repr.model.parameters())
                    + list(self.model.head.parameters())
                )
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)

            # ── Validate ──
            val_metrics = self._evaluate(val_loader)
            val_mcc     = val_metrics["mcc"]

            logging.info(
                f"Epoch {epoch+1:3d}/{self.max_epochs} | "
                f"loss={avg_loss:.4f} | val_mcc={val_mcc:.4f} | "
                f"val_auc={val_metrics.get('auc', float('nan')):.4f}"
            )

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "epoch":           epoch + 1,
                    "train_loss":      avg_loss,
                    "eval/mcc_metric": val_mcc,
                    "eval/auc":        val_metrics.get("auc", 0.0),
                    "eval/accuracy":   val_metrics.get("accuracy", 0.0),
                })

            if val_mcc > best_mcc:
                best_mcc     = val_mcc
                best_metrics = val_metrics
                patience_cnt = 0
                # Save both head and encoder states (encoder is not in state_dict)
                best_state = {
                    "head":    {k: v.clone() for k, v in self.model.state_dict().items()},
                    "encoder": {k: v.clone() for k, v in self.model._repr.model.state_dict().items()},
                }
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    logging.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"(best MCC={best_mcc:.4f})"
                    )
                    break

        # Restore best weights for both head and encoder
        if best_state is not None:
            self.model.load_state_dict(best_state["head"])
            self.model._repr.model.load_state_dict(best_state["encoder"])

        return best_metrics

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict:
        self.model.eval()
        self.model._repr.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for batch in loader:
            labels     = batch["labels"].to(DEVICE)
            morgan_fps = batch["morgan_fps"]
            logits, _  = self.model(batch["unimol_batch"], morgan_fps)
            probs      = F.softmax(logits, dim=-1)
            preds      = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

        preds_arr  = np.array(all_preds)
        labels_arr = np.array(all_labels)
        probs_arr  = np.array(all_probs)

        metrics: Dict = {}
        metrics["mcc"]       = float(matthews_corrcoef(labels_arr, preds_arr))
        metrics["accuracy"]  = float(accuracy_score(labels_arr, preds_arr))
        metrics["f1_macro"]  = float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0))
        metrics["f1_micro"]  = float(f1_score(labels_arr, preds_arr, average="micro"))
        metrics["precision"] = float(precision_score(labels_arr, preds_arr, average="macro", zero_division=0))
        metrics["recall"]    = float(recall_score(labels_arr, preds_arr, average="macro", zero_division=0))

        n_cls = probs_arr.shape[1]
        if n_cls == 2:
            try:
                metrics["auc"] = float(roc_auc_score(labels_arr, probs_arr[:, 1]))
            except ValueError:
                metrics["auc"] = 0.0
        else:
            try:
                metrics["auc"] = float(
                    roc_auc_score(labels_arr, probs_arr, multi_class="ovr", average="macro")
                )
            except ValueError:
                metrics["auc"] = 0.0

        return metrics

    # ------------------------------------------------------------------

    def save(self, save_dir: Path, extra_info: Optional[Dict] = None) -> None:
        """
        Save the finetuned model + config.

        Files written:
            save_dir/
              pytorch_model.pt    — classification head weights
              unimol_encoder.pt   — Uni-Mol encoder weights (saved separately
                                    because UniMolRepr is not nn.Module and is
                                    therefore absent from state_dict())
              config.json          — num_classes, dims, training info
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        # Merge LoRA adapters into base encoder weights before saving
        encoder = getattr(self.model._repr, "model", None)
        if PEFT_AVAILABLE and encoder is not None and hasattr(encoder, "merge_and_unload"):
            self.model._repr.model = encoder.merge_and_unload()
            logging.info("  LoRA weights merged into base encoder.")

        # Save head weights
        torch.save(self.model.state_dict(), str(save_dir / "pytorch_model.pt"))

        # Save encoder weights separately (invisible to state_dict)
        enc = getattr(self.model._repr, "model", None)
        if enc is not None and hasattr(enc, "state_dict"):
            torch.save(enc.state_dict(), str(save_dir / "unimol_encoder.pt"))
            logging.info("  Uni-Mol encoder weights saved to unimol_encoder.pt")
        else:
            logging.warning("  Could not save encoder weights: _repr.model not found.")

        config = {
            "num_classes":  self.model.num_classes,
            "unimol_dim":   UNIMOL_HIDDEN,
            "morgan_bits":  MORGAN_BITS,
            "combined_dim": COMBINED_DIM,
            "model_type":   "UniMolLoRAClassifier",
        }
        if extra_info:
            config.update(extra_info)

        with open(str(save_dir / "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logging.info(f"  Model saved to: {save_dir}")


# ── Model loading ──────────────────────────────────────────────────────────────

def load_finetuned_unimol(
    save_dir:    Path,
    num_classes: Optional[int] = None,
    device:      torch.device  = DEVICE,
) -> UniMolLoRAClassifier:
    """
    Load a previously saved UniMolLoRAClassifier from save_dir.

    Restores both the classification head (pytorch_model.pt) and the finetuned
    Uni-Mol encoder (unimol_encoder.pt).  If unimol_encoder.pt is missing,
    the base pretrained encoder is used and a warning is logged.

    num_classes is read from config.json automatically if not supplied.
    """
    save_dir = Path(save_dir)

    if not save_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {save_dir}\n"
            "Run the corresponding finetune_unimol_*.py script first."
        )

    weights_path = save_dir / "pytorch_model.pt"
    config_path  = save_dir / "config.json"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "The directory exists but training may not have completed."
        )

    # Read num_classes from saved config if not provided
    if num_classes is None:
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found: {config_path}\n"
                "Pass num_classes explicitly or ensure config.json exists."
            )
        with open(str(config_path)) as f:
            cfg = json.load(f)
        num_classes = cfg["num_classes"]

    logging.info(f"Loading UniMolLoRAClassifier from {save_dir}  (num_classes={num_classes})")

    model = UniMolLoRAClassifier(num_classes=num_classes)

    # Restore head weights
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state, strict=False)

    # Restore finetuned encoder weights.
    # UniMolRepr is not nn.Module so the encoder is absent from state_dict().
    # Without this step both FL and WL models would use identical base pretrained
    # encoder weights, producing identical embeddings regardless of loss variant.
    encoder_path = save_dir / "unimol_encoder.pt"
    if encoder_path.exists():
        enc_state = torch.load(str(encoder_path), map_location=device)
        model._repr.model.load_state_dict(enc_state)
        logging.info("  Uni-Mol encoder weights restored from unimol_encoder.pt")
    else:
        logging.warning(
            "unimol_encoder.pt not found — using base pretrained encoder. "
            "Re-run finetuning with the fixed trainer to persist LoRA-adapted weights."
        )

    model.to(device)
    model.eval()
    model._repr.model.eval()

    logging.info("  Model loaded successfully.")
    return model