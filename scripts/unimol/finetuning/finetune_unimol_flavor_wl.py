"""
Uni-Mol LoRA Finetuning — FLAVOR | Weighted Loss

Mirrors finetune_flavor_wl.py from lora-finetuning-scripts/ but uses
Uni-Mol as the backbone with two input modalities:
  1. 3D conformer  (ETKDGv3 + MMFF94 via RDKit)    — Uni-Mol native input
  2. Morgan ECFP4  (2048-bit, radius=2)             — additional 2D modality

WandB Bayesian sweep over:
  lr         : [1e-5, 5e-4]  (Uni-Mol is more LR-sensitive than BERT-family)
  r          : [4, 8, 16, 32]
  lora_alpha : [8, 16, 32, 64]
  dropout    : [0.0, 0.1, 0.2]

Best model saved to:
  EffiChem_Extras/weighted_flavor/dptech__Uni__Mol_LoRA_Finetuned/

Conformer pre-computation note:
  DataHub generates 3D conformers from SMILES (ETKDGv3 + MMFF94).  For
  ~14k training molecules this takes ~5 minutes per call.  With 30 sweep
  trials that would waste ~2.5 hours if done inside run_training().
  Instead, conformers are pre-computed ONCE in __main__ and passed via
  trainer.train(train_unimol=..., val_unimol=...) at every trial.
"""

import gc
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

# ── Project imports ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from unimol_lora_trainer import (
    UniMolLoRAClassifier,
    UniMolLoRATrainer,
    apply_lora_to_unimol,
    preprocess_smiles_for_unimol,
    MolDataset,
    collate_fn,
)
from functools import partial
from torch.utils.data import DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = REPO_ROOT / "data" / "clean" / "flavor_datasets"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
SAVE_DIR    = EXTRAS_ROOT / "weighted_loss_flavor" / "dptech__Uni__Mol_LoRA_Finetuned"
RESULTS_DIR = REPO_ROOT / "results" / "unimol_finetuning" / "flavor"
LOG_DIR     = REPO_ROOT / "logs"

os.makedirs(str(RESULTS_DIR), exist_ok=True)
os.makedirs(str(LOG_DIR), exist_ok=True)

SMILES_COL   = "Standardized SMILES"
LABEL_COL    = "Canonicalized Taste"
LOSS_TYPE    = "weighted"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / "unimol_flavor_wl.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())

# ── WandB setup ────────────────────────────────────────────────────────────────
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    logging.warning("WANDB_API_KEY not set — running without WandB logging")

# ── Module-level globals (populated in __main__ before sweep) ──────────────────
_label_encoder = LabelEncoder()

_TRAIN_SMILES  = None
_VAL_SMILES    = None
_TEST_SMILES   = None
_TRAIN_LABELS  = None
_VAL_LABELS    = None
_TEST_LABELS   = None
_NUM_CLASSES   = None
_CLASS_COUNTS  = None
_TRAIN_UNIMOL  = None   # pre-computed DataHub outputs — cached once
_VAL_UNIMOL    = None
_TEST_UNIMOL   = None


# ── WandB sweep config ─────────────────────────────────────────────────────────
sweep_config = {
    "name":   "FLAVOR_WL_UniMol_Tuning",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "eval/mcc_metric"},
    "parameters": {
        "lr":         {"distribution": "uniform", "min": 1e-5, "max": 5e-4},
        "r":          {"values": [4, 8, 16, 32]},
        "lora_alpha": {"values": [8, 16, 32, 64]},
        "dropout":    {"values": [0.0, 0.1, 0.2]},
    },
}

# ── Sweep run ──────────────────────────────────────────────────────────────────
def run_training():
    run    = wandb.init(project="FLAVOR_UniMol_WL")
    config = run.config

    # Build model
    model = UniMolLoRAClassifier(num_classes=_NUM_CLASSES, head_dropout=config.dropout)
    model = apply_lora_to_unimol(model, r=config.r, lora_alpha=config.lora_alpha, dropout=config.dropout)

    # Trainer
    trainer = UniMolLoRATrainer(
        model        = model,
        loss_type    = LOSS_TYPE,
        class_counts = _CLASS_COUNTS,
        lr           = config.lr,
        max_epochs   = 30,
        batch_size   = 128,
        patience     = 5,
        wandb_run    = run,
    )

    # Pass pre-computed conformers — no redundant DataHub calls per trial
    best_metrics = trainer.train(
        _TRAIN_SMILES, _TRAIN_LABELS,
        _VAL_SMILES,   _VAL_LABELS,
        train_unimol=_TRAIN_UNIMOL,
        val_unimol=_VAL_UNIMOL,
    )
    logging.info(f"Best val metrics: {best_metrics}")

    # Test evaluation using pre-computed test conformers
    test_ds     = MolDataset(_TEST_SMILES, _TEST_LABELS, _TEST_UNIMOL)
    _collate    = partial(collate_fn, padding_idx=trainer.model._repr.model.padding_idx)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=_collate)
    test_metrics = trainer._evaluate(test_loader)
    logging.info(f"Test metrics: {test_metrics}")

    if wandb.run is not None:
        wandb.run.summary.update({f"test_{k}": v for k, v in test_metrics.items()})

    # Save model
    trainer.save(SAVE_DIR, extra_info={
        "dataset":      "flavor",
        "loss_type":    LOSS_TYPE,
        "best_val_mcc": best_metrics.get("mcc"),
        "test_mcc":     test_metrics.get("mcc"),
        "r":            config.r,
        "lora_alpha":   config.lora_alpha,
        "lr":           config.lr,
        "dropout":      config.dropout,
    })

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Uni-Mol LoRA Finetuning | FLAVOR | Weighted Loss")
    logging.info("Additional modalities: 3D conformer (ETKDGv3) + Morgan ECFP4")
    logging.info("=" * 60)

    # ── Load data once ─────────────────────────────────────────────────────
    logging.info("Loading data…")
    train_df = pd.read_csv(str(DATA_DIR / "train_clean.csv"))
    val_df   = pd.read_csv(str(DATA_DIR / "valid_clean.csv"))
    test_df  = pd.read_csv(str(DATA_DIR / "test_clean.csv"))

    _TRAIN_SMILES = train_df[SMILES_COL].tolist()
    _VAL_SMILES   = val_df[SMILES_COL].tolist()
    _TEST_SMILES  = test_df[SMILES_COL].tolist()

    _TRAIN_LABELS = _label_encoder.fit_transform(
        train_df[LABEL_COL].astype(str)
    ).tolist()
    _VAL_LABELS  = _label_encoder.transform(val_df[LABEL_COL].astype(str)).tolist()
    _TEST_LABELS = _label_encoder.transform(test_df[LABEL_COL].astype(str)).tolist()
    _NUM_CLASSES = len(_label_encoder.classes_)

    unique, counts = np.unique(_TRAIN_LABELS, return_counts=True)
    _CLASS_COUNTS  = torch.zeros(_NUM_CLASSES)
    for u, c in zip(unique, counts):
        _CLASS_COUNTS[int(u)] = c

    # ── Pre-compute conformers once before sweep ────────────────────────────
    # preprocess_smiles_for_unimol uses only repr_model.params (config
    # strings), not weights — so a temporary model is sufficient.
    logging.info(
        f"Pre-computing conformers: train={len(_TRAIN_SMILES)}, "
        f"val={len(_VAL_SMILES)}, test={len(_TEST_SMILES)} molecules…"
    )
    _tmp_model = UniMolLoRAClassifier(num_classes=_NUM_CLASSES)
    logging.info("  Processing train split…")
    _TRAIN_UNIMOL = preprocess_smiles_for_unimol(_TRAIN_SMILES, _tmp_model._repr)
    logging.info("  Processing val split…")
    _VAL_UNIMOL   = preprocess_smiles_for_unimol(_VAL_SMILES,   _tmp_model._repr)
    logging.info("  Processing test split…")
    _TEST_UNIMOL  = preprocess_smiles_for_unimol(_TEST_SMILES,  _tmp_model._repr)
    del _tmp_model
    gc.collect()
    logging.info("Conformer pre-computation complete. Starting sweep…")

    sweep_id = wandb.sweep(sweep_config, project="FLAVOR_UniMol_WL")
    wandb.agent(sweep_id, function=run_training, count=30)
    logging.info("Sweep complete.")
