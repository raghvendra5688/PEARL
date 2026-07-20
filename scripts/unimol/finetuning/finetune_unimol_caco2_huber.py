"""
Uni-Mol LoRA Finetuning -- CACO2 | HUBER Loss (regression)

Mirrors finetune_unimol_bace_fl.py's structure but for a regression target,
using the task_type="regression" / loss_type="huber" support added to
unimol_lora_trainer.py for Phase 5 (see editor_response_suggestions.md).

WandB Bayesian sweep over:
  lr         : [1e-5, 5e-4]
  r          : [4, 8, 16, 32]
  lora_alpha : [8, 16, 32, 64]
  dropout    : [0.0, 0.1, 0.2]

Best model saved to:
  $PEARL_EXTRAS_V2/huber_loss_CACO2/dptech__Uni__Mol_LoRA_Finetuned/
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


# -- Project imports ------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from unimol_lora_trainer import (
    UniMolLoRAClassifier,
    UniMolLoRATrainer,
    apply_lora_to_unimol,
    is_new_best,
)

# -- Paths ------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR    = REPO_ROOT / "data" / "clean" / "caco2_datasets"
# NEW artifacts go to PEARL_EXTRAS_V2 (the original PEARL_EXTRAS default was
# found read-only from this host during Phase 3 -- see editor_response_suggestions.md).
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
SAVE_DIR    = EXTRAS_ROOT / "huber_loss_CACO2" / "dptech__Uni__Mol_LoRA_Finetuned"
RESULTS_DIR = REPO_ROOT / "results" / "unimol_finetuning" / "caco2"
LOG_DIR     = REPO_ROOT / "logs"

os.makedirs(str(RESULTS_DIR), exist_ok=True)
os.makedirs(str(LOG_DIR), exist_ok=True)

SMILES_COL        = "Standardized SMILES"
LABEL_COL         = "Caco2_LogPapp"
TARGET_TRANSFORM  = None  # None or "log1p"
LOSS_TYPE         = "huber"

# -- Logging ------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / "unimol_caco2_huber.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())

# -- WandB setup ------------------------------------------------------------------------
# WandB's own local run cache defaults to CWD ("wandb/" under the repo root) --
# redirect it to EffiChem_Extras_v2 too, same rationale as SAVE_DIR/EXTRAS_ROOT above.
os.environ.setdefault("WANDB_DIR", str(EXTRAS_ROOT / "wandb_logs"))
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    logging.warning("WANDB_API_KEY not set -- running without WandB logging")


def apply_target_transform(y):
    return np.log1p(y) if TARGET_TRANSFORM == "log1p" else y


def invert_target_transform(y):
    return np.expm1(y) if TARGET_TRANSFORM == "log1p" else y


# -- Data loading ------------------------------------------------------------------
def load_data():
    train_df = pd.read_csv(str(DATA_DIR / "train_clean.csv"))
    val_df   = pd.read_csv(str(DATA_DIR / "valid_clean.csv"))
    test_df  = pd.read_csv(str(DATA_DIR / "test_clean.csv"))
    return train_df, val_df, test_df


# -- WandB sweep config ------------------------------------------------------------------
sweep_config = {
    "name":   "CACO2_HUBER_UniMol_Tuning",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "eval/r2"},
    "parameters": {
        "lr":         {"distribution": "uniform", "min": 1e-5, "max": 5e-4},
        "r":          {"values": [4, 8, 16, 32]},
        "lora_alpha": {"values": [8, 16, 32, 64]},
        "dropout":    {"values": [0.0, 0.1, 0.2]},
    },
}

# -- Sweep run ------------------------------------------------------------------------
def run_training():
    run    = wandb.init(project="CACO2_UniMol_HUBER")
    config = run.config

    train_df, val_df, test_df = load_data()

    train_smiles = train_df[SMILES_COL].tolist()
    val_smiles   = val_df[SMILES_COL].tolist()
    test_smiles  = test_df[SMILES_COL].tolist()

    train_labels = apply_target_transform(train_df[LABEL_COL].astype(float).values).tolist()
    val_labels   = apply_target_transform(val_df[LABEL_COL].astype(float).values).tolist()
    test_labels  = apply_target_transform(test_df[LABEL_COL].astype(float).values).tolist()

    # Build model (regression head: out_dim=1)
    model = UniMolLoRAClassifier(num_classes=1, head_dropout=config.dropout, task_type="regression")
    model = apply_lora_to_unimol(model, r=config.r, lora_alpha=config.lora_alpha, dropout=config.dropout)

    # Trainer
    trainer = UniMolLoRATrainer(
        model        = model,
        loss_type    = LOSS_TYPE,
        lr           = config.lr,
        max_epochs   = 30,
        batch_size   = 128,
        patience     = 5,
        wandb_run    = run,
        task_type    = "regression",
    )

    best_metrics = trainer.train(train_smiles, train_labels, val_smiles, val_labels)
    logging.info(f"Best val metrics: {best_metrics}")

    # Test evaluation (metrics computed on the ORIGINAL target scale via invert_fn,
    # matching pc_only_modelling.py / chemprop_baseline.py / gcn_baseline.py convention)
    from functools import partial
    from torch.utils.data import DataLoader
    from unimol_lora_trainer import MolDataset, collate_fn, preprocess_smiles_for_unimol

    test_unimol = preprocess_smiles_for_unimol(test_smiles, trainer.model._repr)
    test_ds     = MolDataset(test_smiles, test_labels, test_unimol)
    _collate    = partial(collate_fn, padding_idx=trainer.model._repr.model.padding_idx, regression=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=_collate)
    test_metrics = trainer._evaluate(test_loader, invert_fn=invert_target_transform)
    logging.info(f"Test metrics (original scale): {test_metrics}")

    if wandb.run is not None:
        wandb.run.summary.update({f"test_{k}": v for k, v in test_metrics.items()})

    # Save model -- but only if this trial beats every prior trial in the
    # sweep (see is_new_best() docstring: every trial would otherwise
    # unconditionally overwrite SAVE_DIR regardless of hyperparameter quality)
    val_score = best_metrics.get('r2')
    if val_score is not None and is_new_best(SAVE_DIR, val_score):
        trainer.save(SAVE_DIR, extra_info={
            "dataset":       "caco2",
            "loss_type":     LOSS_TYPE,
            "task_type":     "regression",
            "target_transform": TARGET_TRANSFORM,
            "best_val_r2":   best_metrics.get("r2"),
            "test_r2":       test_metrics.get("r2"),
            "test_spearman": test_metrics.get("spearman"),
            "r":             config.r,
            "lora_alpha":    config.lora_alpha,
            "lr":            config.lr,
            "dropout":       config.dropout,
        })
        logging.info(f"New best (val_r2={val_score:.4f}) -- saved to {SAVE_DIR}")
    else:
        logging.info(f"Trial val_r2={val_score} did not beat the current best -- skipped save.")

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()


# -- Main ------------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Uni-Mol LoRA Finetuning | CACO2 | HUBER Loss (regression)")
    logging.info("Additional modalities: 3D conformer (ETKDGv3) + Morgan ECFP4")
    logging.info("=" * 60)

    sweep_id = wandb.sweep(sweep_config, project="CACO2_UniMol_HUBER")
    wandb.agent(sweep_id, function=run_training, count=30)
    logging.info("Sweep complete.")
