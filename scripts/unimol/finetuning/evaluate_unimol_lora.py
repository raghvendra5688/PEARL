"""
Evaluate all Uni-Mol LoRA finetuned models and produce per-dataset CSV tables
matching the format of EffiChem_Extras/{DATASET}/best_model_comparison/all_metrics.csv:

    Configuration, ML_Model, AUC, AUPR, MCC, F1_macro, Avg_Score,
    Accuracy, Precision, Recall

One CSV is written per dataset to:
    results/unimol_finetuning/{dataset}/unimol_lora_metrics.csv

Usage:
    cd "Finetuned Model Scripts/Uni-Mol/lora-finetuning-scripts"
    python evaluate_unimol_lora.py
    python evaluate_unimol_lora.py --datasets bace bbbp   # subset
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from functools import partial

from unimol_lora_trainer import (   # noqa: E402
    MolDataset,
    collate_fn,
    load_finetuned_unimol,
    preprocess_smiles_for_unimol,
)

REPO_ROOT   = _SCRIPT_DIR.parent.parent   # EffChem-2.0/
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
SMILES_COL  = "Standardized SMILES"

# ── Dataset registry ──────────────────────────────────────────────────────────
#   Each entry:
#     data_dir    : path to train/valid/test_clean.csv
#     label_col   : column name (string labels → LabelEncoder for multiclass)
#     multiclass  : True if label_col is a string class needing encoding
#     models      : list of (Configuration label, loss_type tag, model save dir)
DATASETS = {
    "bace": dict(
        data_dir   = REPO_ROOT / "data" / "clean" / "bace_datasets",
        label_col  = "Class",
        multiclass = False,
        models     = [
            ("Uni-Mol (LoRA)", "Focal Loss",    EXTRAS_ROOT / "focal_loss_BACE"    / "dptech__Uni__Mol_LoRA_Finetuned"),
            ("Uni-Mol (LoRA)", "Weighted Loss", EXTRAS_ROOT / "weighted_loss_BACE" / "dptech__Uni__Mol_LoRA_Finetuned"),
        ],
    ),
    "bbbp": dict(
        data_dir   = REPO_ROOT / "data" / "clean" / "bbbp_datasets",
        label_col  = "p_np",
        multiclass = False,
        models     = [
            ("Uni-Mol (LoRA)", "Focal Loss",    EXTRAS_ROOT / "focal_loss_BBBP"    / "dptech__Uni__Mol_LoRA_Finetuned"),
            ("Uni-Mol (LoRA)", "Weighted Loss", EXTRAS_ROOT / "weighted_loss_BBBP" / "dptech__Uni__Mol_LoRA_Finetuned"),
        ],
    ),
    "clintox": dict(
        data_dir   = REPO_ROOT / "data" / "clean" / "clintox_datasets",
        label_col  = "FDA_APPROVED",
        multiclass = False,
        models     = [
            ("Uni-Mol (LoRA)", "Focal Loss",    EXTRAS_ROOT / "focal_loss_clintox"    / "dptech__Uni__Mol_LoRA_Finetuned"),
            ("Uni-Mol (LoRA)", "Weighted Loss", EXTRAS_ROOT / "weighted_loss_clintox" / "dptech__Uni__Mol_LoRA_Finetuned"),
        ],
    ),
    "flavor": dict(
        data_dir   = REPO_ROOT / "data" / "clean" / "flavor_datasets",
        label_col  = "Canonicalized Taste",
        multiclass = True,
        models     = [
            ("Uni-Mol (LoRA)", "Focal Loss",    EXTRAS_ROOT / "focal_loss_flavor"    / "dptech__Uni__Mol_LoRA_Finetuned"),
            ("Uni-Mol (LoRA)", "Weighted Loss", EXTRAS_ROOT / "weighted_loss_flavor" / "dptech__Uni__Mol_LoRA_Finetuned"),
        ],
    ),
}


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Data loading ──────────────────────────────────────────────────────────────
def load_splits(
    data_dir: Path,
    label_col: str,
    multiclass: bool,
) -> Tuple[List[str], List[int], Optional[LabelEncoder]]:
    """
    Load test split SMILES and integer labels.
    For multiclass datasets the LabelEncoder is fit on the training set
    (mirrors finetune_unimol_flavor_*.py) and returned for reference.
    """
    train_df = pd.read_csv(data_dir / "train_clean.csv")
    test_df  = pd.read_csv(data_dir / "test_clean.csv")

    smiles = test_df[SMILES_COL].tolist()

    if multiclass:
        le = LabelEncoder()
        le.fit(train_df[label_col].astype(str))
        labels = le.transform(test_df[label_col].astype(str)).tolist()
        return smiles, labels, le

    labels = test_df[label_col].astype(int).tolist()
    return smiles, labels, None


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model_dir: Path,
    smiles: List[str],
    labels: List[int],
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load model from model_dir, run inference on (smiles, labels).
    Returns (labels_arr, preds_arr, probs_arr).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_finetuned_unimol(model_dir)
    model.eval()
    model.to(device)

    logging.info(f"    Pre-computing conformers for {len(smiles)} test molecules…")
    unimol_inputs = preprocess_smiles_for_unimol(smiles, model._repr)

    padding_idx = model._repr.model.padding_idx
    dataset     = MolDataset(smiles, labels, unimol_inputs)
    _collate    = partial(collate_fn, padding_idx=padding_idx)
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=_collate)

    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        unimol_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch["unimol_batch"].items()}
        morgan_fps   = batch["morgan_fps"].to(device)
        logits, _    = model(unimol_batch, morgan_fps)
        probs        = F.softmax(logits, dim=-1)
        preds        = logits.argmax(dim=-1)

        all_labels.extend(batch["labels"].cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())

    del model
    torch.cuda.empty_cache()

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


# ── Metrics computation ───────────────────────────────────────────────────────
def compute_metrics(
    labels_arr: np.ndarray,
    preds_arr:  np.ndarray,
    probs_arr:  np.ndarray,
    multiclass: bool,
) -> Dict[str, float]:
    n_classes = probs_arr.shape[1]

    mcc       = float(matthews_corrcoef(labels_arr, preds_arr))
    accuracy  = float(accuracy_score(labels_arr, preds_arr))
    f1_macro  = float(f1_score(labels_arr, preds_arr, average="macro",  zero_division=0))
    f1_micro  = float(f1_score(labels_arr, preds_arr, average="micro",  zero_division=0))
    precision = float(precision_score(labels_arr, preds_arr, average="macro", zero_division=0))
    recall    = float(recall_score(labels_arr, preds_arr, average="macro", zero_division=0))

    if not multiclass:
        try:
            auc  = float(roc_auc_score(labels_arr, probs_arr[:, 1]))
        except ValueError:
            auc  = float("nan")
        try:
            aupr = float(average_precision_score(labels_arr, probs_arr[:, 1]))
        except ValueError:
            aupr = float("nan")
        avg_score = float(np.nanmean([auc, aupr, mcc, f1_macro, accuracy, precision, recall]))
        return dict(AUC=auc, AUPR=aupr, MCC=mcc, F1_macro=f1_macro, F1_micro=f1_micro,
                    Avg_Score=avg_score, Accuracy=accuracy,
                    Precision=precision, Recall=recall)
    else:
        # Multiclass: macro-OvR AUC and macro-average AUPR
        try:
            auc = float(roc_auc_score(
                labels_arr, probs_arr,
                multi_class="ovr", average="macro",
            ))
        except ValueError:
            auc = float("nan")

        # Per-class AUPR averaged to macro
        aupr_vals = []
        for c in range(n_classes):
            y_bin = (labels_arr == c).astype(int)
            if y_bin.sum() > 0:
                aupr_vals.append(average_precision_score(y_bin, probs_arr[:, c]))
        aupr = float(np.mean(aupr_vals)) if aupr_vals else float("nan")

        avg_score = float(np.nanmean([auc, aupr, mcc, f1_macro, accuracy, precision, recall]))
        return dict(AUC=auc, AUPR=aupr, MCC=mcc, F1_macro=f1_macro, F1_micro=f1_micro,
                    Avg_Score=avg_score, Accuracy=accuracy,
                    Precision=precision, Recall=recall)


# ── Per-dataset evaluation ────────────────────────────────────────────────────
def evaluate_dataset(dataset_name: str, cfg: dict) -> pd.DataFrame:
    data_dir   = cfg["data_dir"]
    label_col  = cfg["label_col"]
    multiclass = cfg["multiclass"]
    models     = cfg["models"]

    logging.info(f"\n{'='*60}")
    logging.info(f"Dataset: {dataset_name.upper()}")
    logging.info(f"{'='*60}")

    smiles, labels, _le = load_splits(data_dir, label_col, multiclass)
    logging.info(f"  Test set: {len(smiles)} molecules")

    rows = []
    for configuration, ml_model_label, model_dir in models:
        if not model_dir.exists():
            logging.warning(f"  Model not found: {model_dir} — skipping.")
            continue

        logging.info(f"  [{ml_model_label}] Loading {model_dir.parent.name}…")
        try:
            labels_arr, preds_arr, probs_arr = run_inference(
                model_dir, smiles, labels
            )
            metrics = compute_metrics(labels_arr, preds_arr, probs_arr, multiclass)
        except Exception as e:
            logging.error(f"  Inference failed for {ml_model_label}: {e}", exc_info=True)
            continue

        row = {"Configuration": configuration, "ML_Model": ml_model_label}
        row.update({k: round(v, 4) for k, v in metrics.items()})
        rows.append(row)

        logging.info(
            f"    MCC={metrics['MCC']:.4f}  AUC={metrics['AUC']:.4f}"
            f"  AUPR={metrics['AUPR']:.4f}  F1={metrics['F1_macro']:.4f}"
        )

    if not rows:
        return pd.DataFrame()

    col_order = ["Configuration", "ML_Model", "AUC", "AUPR", "MCC",
                 "F1_macro", "F1_micro", "Avg_Score", "Accuracy", "Precision", "Recall"]
    df = pd.DataFrame(rows)
    df = df[[c for c in col_order if c in df.columns]]
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Uni-Mol LoRA models and produce all_metrics-style CSVs."
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Inference batch size (default: 16)",
    )
    args = parser.parse_args()

    setup_logging()

    for dataset_name in args.datasets:
        cfg = DATASETS[dataset_name]
        df  = evaluate_dataset(dataset_name, cfg)

        if df.empty:
            logging.warning(f"No results for {dataset_name}.")
            continue

        out_dir = REPO_ROOT / "results" / "unimol_finetuning" / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "unimol_lora_metrics.csv"
        df.to_csv(out_path, index=False)
        logging.info(f"\nSaved → {out_path}")
        print(f"\n{dataset_name.upper()} results:")
        print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
