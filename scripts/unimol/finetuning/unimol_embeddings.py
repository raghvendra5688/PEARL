"""
Uni-Mol Embedding Extraction

Loads the finetuned UniMolLoRAClassifier for each dataset × loss-type
combination and extracts multimodal embeddings for all splits (train/eval/test).

Output format matches finetuned_model_embeddings.py exactly so downstream
ml-scripts and RAG modelling scripts can consume them unchanged.

Embedding columns produced per split CSV:
  UniMol_FL_embeddings   — 2560-dim, comma-separated string (focal loss model)
  UniMol_WL_embeddings   — 2560-dim, comma-separated string (weighted loss model)

The 2560-dim vector is:
  [:512]   — Uni-Mol CLS token  (3D-structure-aware, ETKDGv3 conformer)
  [512:]   — Morgan ECFP4 2048-bit fingerprint (2D structural modality)

Usage:
    python unimol_embeddings.py
    python unimol_embeddings.py --dataset bace   # single dataset
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Project imports ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from unimol_lora_trainer import (
    COMBINED_DIM,
    UniMolLoRAClassifier,
    load_finetuned_unimol,
)

# ── Paths ───────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
CLEAN_ROOT  = REPO_ROOT / "data" / "clean"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
OUTPUT_ROOT = EXTRAS_ROOT / "unimol_embeddings"
LOG_DIR     = REPO_ROOT / "logs"

SMILES_COL = "Standardized SMILES"

# ── Dataset config ──────────────────────────────────────────────────────────────
DATASETS = {
    "bace": {
        "clean_dir":   "bace_datasets",
        "output_dir":  "BACE_Embeddings",
        "model_dir":   "BACE",
        "file_prefix": "bace",
        "labels":      ["Class"],
        "num_classes": 2,
        "splits":      {"train": "train", "eval": "valid", "test": "test"},
    },
    "bbbp": {
        "clean_dir":   "bbbp_datasets",
        "output_dir":  "BBBP_Embeddings",
        "model_dir":   "BBBP",
        "file_prefix": "bbbp",
        "labels":      ["p_np"],
        "num_classes": 2,
        "splits":      {"train": "train", "eval": "valid", "test": "test"},
    },
    "clintox": {
        "clean_dir":   "clintox_datasets",
        "output_dir":  "clintox_Embeddings",
        "model_dir":   "clintox",
        "file_prefix": "clintox",
        "labels":      ["FDA_APPROVED", "CT_TOX"],
        "num_classes": 2,
        "splits":      {"train": "train", "eval": "valid", "test": "test"},
    },
    "flavor": {
        "clean_dir":   "flavor_datasets",
        "output_dir":  "flavor_Embeddings",
        "model_dir":   "flavor",
        "file_prefix": "fart",
        "labels":      ["Canonicalized Taste"],
        "num_classes": None,   # determined at runtime
        "splits":      {"train": "train", "eval": "valid", "test": "test"},
    },
}

# Loss type → (col_name_suffix, folder_name_in_EffiChem_Extras)
LOSS_VARIANTS = {
    "focal_loss":    "UniMol_FL_embeddings",
    "weighted_loss": "UniMol_WL_embeddings",
}

MODEL_FOLDER = "dptech__Uni__Mol_LoRA_Finetuned"


# ── Logging ─────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / "unimol_embeddings.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


# ── Embedding extraction ─────────────────────────────────────────────────────────
def extract_embeddings(
    model:       UniMolLoRAClassifier,
    smiles_list: list,
    batch_size:  int = 16,
) -> np.ndarray:
    """Returns (N, 2560) float32 — Uni-Mol CLS + Morgan FP."""
    return model.get_embeddings(smiles_list, batch_size=batch_size)


def emb_to_str(emb_row: np.ndarray) -> str:
    """Format a 1D float32 array as a JSON-compatible bracketed string.

    np.array2string truncates arrays longer than threshold=1000, producing '...'
    in the output which makes the embedding unrecoverable. Using an explicit
    join guarantees all 2560 values are written.
    """
    return "[" + ",".join(f"{x:.8f}" for x in emb_row) + "]"


# ── Main ─────────────────────────────────────────────────────────────────────────
def process_dataset(dataset_key: str, cfg: dict) -> None:
    out_dir = OUTPUT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load both loss-type models once per dataset
    loaded_models = {}
    for loss_type, col_name in LOSS_VARIANTS.items():
        model_path = EXTRAS_ROOT / f"{loss_type}_{cfg['model_dir']}" / MODEL_FOLDER
        if not model_path.exists():
            logging.warning(f"Model not found, skipping: {model_path}")
            continue

        num_classes = cfg["num_classes"]   # None for flavor → load_finetuned_unimol reads from config.json
        logging.info(f"Loading {col_name} from {model_path}")
        try:
            m = load_finetuned_unimol(model_path, num_classes=num_classes)
            loaded_models[col_name] = m
            logging.info(f"  Loaded: {col_name}")
        except Exception as e:
            logging.error(f"  Failed to load {model_path}: {e}")

    if not loaded_models:
        logging.warning(f"No models loaded for {dataset_key} — skipping")
        return

    # Process each split
    for out_split, clean_split in cfg["splits"].items():
        input_csv = CLEAN_ROOT / cfg["clean_dir"] / f"{clean_split}_clean.csv"
        if not input_csv.exists():
            logging.warning(f"Input not found: {input_csv}")
            continue

        df          = pd.read_csv(str(input_csv))
        smiles_list = df[SMILES_COL].astype(str).tolist()

        base_cols = [SMILES_COL] + [c for c in cfg["labels"] if c in df.columns]
        out_df    = df[base_cols].copy()

        logging.info(
            f"  {dataset_key}/{out_split}: {len(smiles_list)} molecules"
        )

        for col_name, model in loaded_models.items():
            logging.info(f"    Extracting: {col_name}")
            embs      = extract_embeddings(model, smiles_list, batch_size=16)
            emb_strs  = [emb_to_str(row) for row in embs]
            out_df[col_name] = emb_strs
            logging.info(f"    Shape: {embs.shape}")

        out_path = out_dir / f"{cfg['file_prefix']}_{out_split}_embed.csv"
        out_df.to_csv(str(out_path), index=False)
        logging.info(f"  Saved: {out_path}")

    # Free GPU memory
    del loaded_models
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Uni-Mol embeddings")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default=None,
        help="Process a single dataset (default: all)",
    )
    args = parser.parse_args()

    setup_logging()
    logging.info("=" * 60)
    logging.info("Uni-Mol Embedding Extraction")
    logging.info(f"Output dim: {COMBINED_DIM} (512 Uni-Mol CLS + 2048 Morgan ECFP4)")
    logging.info("=" * 60)

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    for ds_key, cfg in targets.items():
        logging.info(f"\n{'='*40}")
        logging.info(f"Dataset: {ds_key}")
        process_dataset(ds_key, cfg)

    logging.info("\nAll Uni-Mol embeddings extracted.")


if __name__ == "__main__":
    main()
