"""
Uni-Mol LoRA Finetuning — DILI | Focal Loss

Mirrors finetune_dili_fl.py from lora-finetuning-scripts/ but uses
Uni-Mol as the backbone with two input modalities:
  1. 3D conformer  (ETKDGv3 + MMFF94 via RDKit)    — Uni-Mol native input
  2. Morgan ECFP4  (2048-bit, radius=2)             — additional 2D modality

WandB Bayesian sweep over:
  lr         : [1e-5, 5e-4]  (Uni-Mol is more LR-sensitive than BERT-family)
  r          : [4, 8, 16, 32]
  lora_alpha : [8, 16, 32, 64]
  dropout    : [0.0, 0.1, 0.2]

Best model saved to:
  EffiChem_Extras/focal_DILI/dptech__Uni__Mol_LoRA_Finetuned/
"""

import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv


# ── Project imports ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from unimol_lora_trainer import (
    UniMolLoRAClassifier,
    UniMolLoRATrainer,
    apply_lora_to_unimol,
    is_new_best,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR    = REPO_ROOT / "data" / "clean" / "dili_datasets"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
SAVE_DIR    = EXTRAS_ROOT / "focal_loss_DILI" / "dptech__Uni__Mol_LoRA_Finetuned"
RESULTS_DIR = REPO_ROOT / "results" / "unimol_finetuning" / "dili"
LOG_DIR     = REPO_ROOT / "logs"

os.makedirs(str(RESULTS_DIR), exist_ok=True)
os.makedirs(str(LOG_DIR), exist_ok=True)

SMILES_COL   = "Standardized SMILES"
LABEL_COL    = "DILI_Label"
LOSS_TYPE    = "focal"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / "unimol_dili_fl.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())

# ── WandB setup ────────────────────────────────────────────────────────────────
# WandB's own local run cache defaults to CWD ("wandb/" under the repo root) --
# redirect it to EffiChem_Extras_v2 too, same rationale as SAVE_DIR/EXTRAS_ROOT above.
os.environ.setdefault("WANDB_DIR", str(EXTRAS_ROOT / "wandb_logs"))
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    logging.warning("WANDB_API_KEY not set — running without WandB logging")

# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    train_df = pd.read_csv(str(DATA_DIR / "train_clean.csv"))
    val_df   = pd.read_csv(str(DATA_DIR / "valid_clean.csv"))
    test_df  = pd.read_csv(str(DATA_DIR / "test_clean.csv"))
    return train_df, val_df, test_df



# ── WandB sweep config ─────────────────────────────────────────────────────────
sweep_config = {
    "name":   "DILI_FL_UniMol_Tuning",
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
    run    = wandb.init(project="DILI_UniMol_FL")
    config = run.config

    train_df, val_df, test_df = load_data()

    train_smiles = train_df[SMILES_COL].tolist()
    val_smiles   = val_df[SMILES_COL].tolist()
    test_smiles  = test_df[SMILES_COL].tolist()

    train_labels = train_df[LABEL_COL].astype(int).tolist()
    val_labels   = val_df[LABEL_COL].astype(int).tolist()
    test_labels  = test_df[LABEL_COL].astype(int).tolist()
    num_classes  = 2

    # Class counts for weighted loss
    unique, counts = np.unique(train_labels, return_counts=True)
    class_counts   = torch.zeros(num_classes)
    for u, c in zip(unique, counts):
        class_counts[int(u)] = c

    # Build model
    model = UniMolLoRAClassifier(num_classes=num_classes, head_dropout=config.dropout)
    model = apply_lora_to_unimol(model, r=config.r, lora_alpha=config.lora_alpha, dropout=config.dropout)

    # Trainer
    trainer = UniMolLoRATrainer(
        model        = model,
        loss_type    = LOSS_TYPE,
        class_counts = class_counts,
        lr           = config.lr,
        max_epochs   = 30,
        batch_size   = 128,
        patience     = 5,
        wandb_run    = run,
    )

    best_metrics = trainer.train(train_smiles, train_labels, val_smiles, val_labels)
    logging.info(f"Best val metrics: {best_metrics}")

    # Test evaluation
    from functools import partial
    from torch.utils.data import DataLoader
    from unimol_lora_trainer import MolDataset, collate_fn, preprocess_smiles_for_unimol
    import torch.nn.functional as F
    from sklearn.metrics import matthews_corrcoef

    test_unimol = preprocess_smiles_for_unimol(test_smiles, trainer.model._repr)
    test_ds     = MolDataset(test_smiles, test_labels, test_unimol)
    _collate    = partial(collate_fn, padding_idx=trainer.model._repr.model.padding_idx)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=_collate)
    test_metrics = trainer._evaluate(test_loader)
    logging.info(f"Test metrics: {test_metrics}")

    if wandb.run is not None:
        wandb.run.summary.update({f"test_{k}": v for k, v in test_metrics.items()})

    # Save model -- but only if this trial beats every prior trial in the
    # sweep (see is_new_best() docstring: every trial would otherwise
    # unconditionally overwrite SAVE_DIR regardless of hyperparameter quality)
    val_score = best_metrics.get('mcc')
    if val_score is not None and is_new_best(SAVE_DIR, val_score):
        trainer.save(SAVE_DIR, extra_info={
            "dataset":      "dili",
            "loss_type":    LOSS_TYPE,
            "best_val_mcc": best_metrics.get("mcc"),
            "test_mcc":     test_metrics.get("mcc"),
            "r":            config.r,
            "lora_alpha":   config.lora_alpha,
            "lr":           config.lr,
            "dropout":      config.dropout,
        })
        logging.info(f"New best (val_mcc={val_score:.4f}) -- saved to {SAVE_DIR}")
    else:
        logging.info(f"Trial val_mcc={val_score} did not beat the current best -- skipped save.")

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Uni-Mol LoRA Finetuning | DILI | Focal Loss")
    logging.info("Additional modalities: 3D conformer (ETKDGv3) + Morgan ECFP4")
    logging.info("=" * 60)

    sweep_id = wandb.sweep(sweep_config, project="DILI_UniMol_FL")
    wandb.agent(sweep_id, function=run_training, count=30)
    logging.info("Sweep complete.")
