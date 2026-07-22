"""
Uni-Mol Embedding Extraction — herg/dili/caco2/half_life

Mirrors unimol_embeddings.py (the original bace/bbbp/clintox/flavor script)
for the 4 new TDC datasets. Separate script because these live under
$PEARL_EXTRAS_V2 (not $PEARL_EXTRAS) and caco2/half_life use a single Huber
loss variant (regression) rather than FL+WL.

Output format matches finetuned_model_embeddings_new_datasets.py so the same
downstream ml-scripts can consume either modality's embeddings unchanged.

Embedding columns produced per split CSV:
  UniMol_FL_embeddings    — herg/dili only
  UniMol_WL_embeddings    — herg/dili only
  UniMol_Huber_embeddings — caco2/half_life only

Usage:
    python unimol_embeddings_new_datasets.py
    python unimol_embeddings_new_datasets.py --dataset herg
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from unimol_lora_trainer import COMBINED_DIM, load_finetuned_unimol

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
CLEAN_ROOT  = REPO_ROOT / "data" / "clean"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
# Mirrors the original unimol_embeddings.py's EXTRAS_ROOT/"unimol_embeddings"
# convention, routed through _v2. Kept in a sibling dir from the HF script's
# "finetuned_embeddings" output so the two never collide on the same filename.
OUTPUT_ROOT = EXTRAS_ROOT / "unimol_embeddings"
LOG_DIR     = REPO_ROOT / "logs"

SMILES_COL = "Standardized SMILES"

DATASETS = {
    "herg": {
        "clean_dir": "herg_datasets", "output_dir": "HERG_Embeddings", "file_prefix": "herg",
        "labels": ["hERG_Inhib"], "task_type": "classification", "num_classes": 2,
        "losses": {"focal_loss": ("HERG", "UniMol_FL_embeddings"), "weighted_loss": ("HERG", "UniMol_WL_embeddings")},
        "splits": {"train": "train", "eval": "valid", "test": "test"},
    },
    "dili": {
        "clean_dir": "dili_datasets", "output_dir": "DILI_Embeddings", "file_prefix": "dili",
        "labels": ["DILI_Label"], "task_type": "classification", "num_classes": 2,
        "losses": {"focal_loss": ("DILI", "UniMol_FL_embeddings"), "weighted_loss": ("DILI", "UniMol_WL_embeddings")},
        "splits": {"train": "train", "eval": "valid", "test": "test"},
    },
    "caco2": {
        "clean_dir": "caco2_datasets", "output_dir": "CACO2_Embeddings", "file_prefix": "caco2",
        "labels": ["Caco2_LogPapp"], "task_type": "regression", "num_classes": 1,
        "losses": {"huber_loss": ("CACO2", "UniMol_Huber_embeddings")},
        "splits": {"train": "train", "eval": "valid", "test": "test"},
    },
    "half_life": {
        "clean_dir": "half_life_datasets", "output_dir": "HALF_LIFE_Embeddings", "file_prefix": "half_life",
        "labels": ["Half_Life_Hours"], "task_type": "regression", "num_classes": 1,
        "losses": {"huber_loss": ("HALF_LIFE", "UniMol_Huber_embeddings")},
        "splits": {"train": "train", "eval": "valid", "test": "test"},
    },
}

MODEL_FOLDER = "dptech__Uni__Mol_LoRA_Finetuned"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / "unimol_embeddings_new_datasets.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def emb_to_str(emb_row: np.ndarray) -> str:
    """np.array2string truncates arrays >1000 elements ('...') -- explicit join
    guarantees all 2560 values survive round-tripping through CSV."""
    return "[" + ",".join(f"{x:.8f}" for x in emb_row) + "]"


def process_dataset(dataset_key: str, cfg: dict) -> None:
    out_dir = OUTPUT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_models = {}
    for loss_type, (extras_suffix, col_name) in cfg["losses"].items():
        model_path = EXTRAS_ROOT / f"{loss_type}_{extras_suffix}" / MODEL_FOLDER
        if not model_path.exists():
            logging.warning(f"Model not found, skipping: {model_path}")
            continue
        logging.info(f"Loading {col_name} from {model_path}")
        try:
            m = load_finetuned_unimol(model_path, num_classes=cfg["num_classes"], task_type=cfg["task_type"])
            loaded_models[col_name] = m
            logging.info(f"  Loaded: {col_name}")
        except Exception as e:
            logging.error(f"  Failed to load {model_path}: {e}")

    if not loaded_models:
        logging.warning(f"No models loaded for {dataset_key} — skipping")
        return

    for out_split, clean_split in cfg["splits"].items():
        input_csv = CLEAN_ROOT / cfg["clean_dir"] / f"{clean_split}_clean.csv"
        if not input_csv.exists():
            logging.warning(f"Input not found: {input_csv}")
            continue

        df = pd.read_csv(str(input_csv))
        smiles_list = df[SMILES_COL].astype(str).tolist()

        base_cols = [SMILES_COL] + [c for c in cfg["labels"] if c in df.columns]
        out_df = df[base_cols].copy()

        logging.info(f"  {dataset_key}/{out_split}: {len(smiles_list)} molecules")

        for col_name, model in loaded_models.items():
            logging.info(f"    Extracting: {col_name}")
            embs = model.get_embeddings(smiles_list, batch_size=16)
            out_df[col_name] = [emb_to_str(row) for row in embs]
            logging.info(f"    Shape: {embs.shape}")

        out_path = out_dir / f"{cfg['file_prefix']}_{out_split}_embed.csv"
        out_df.to_csv(str(out_path), index=False)
        logging.info(f"  Saved: {out_path}")

    del loaded_models
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Uni-Mol embeddings (new TDC datasets)")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None)
    args = parser.parse_args()

    setup_logging()
    logging.info("=" * 60)
    logging.info("Uni-Mol Embedding Extraction (herg/dili/caco2/half_life)")
    logging.info(f"Output dim: {COMBINED_DIM} (512 Uni-Mol CLS + 2048 Morgan ECFP4)")
    logging.info("=" * 60)

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    for ds_key, cfg in targets.items():
        logging.info(f"\n{'='*40}")
        logging.info(f"Dataset: {ds_key}")
        process_dataset(ds_key, cfg)

    logging.info("\nAll new-dataset Uni-Mol embeddings extracted.")


if __name__ == "__main__":
    main()
