"""
Universal Embedding + PC Feature Merge Script

This script:
1. Iterates over all datasets
2. Loads embedding CSVs
3. Loads PC feature CSVs
4. Merges them safely on Standardized SMILES
5. Avoids duplicate label columns
6. Saves combined feature files

Works for:
- BBBP
- ClinTox
- Flavor
- BACE
"""

import os
from pathlib import Path
import pandas as pd
import logging

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_EMB_ROOT = str(REPO_ROOT / "data" / "base_models")
FEATURE_ROOT  = str(REPO_ROOT / "data" / "features")
OUTPUT_ROOT   = str(REPO_ROOT / "data" / "base_models_features")
LOG_DIR       = str(REPO_ROOT / "logs")

DATASETS = [
    "bbbp_datasets",
    "clintox_datasets",
    "flavor_datasets",
    "bace_datasets",
]

SPLITS = ["train", "valid", "test"]
SMILES_COL = "Standardized SMILES"

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "merge_embeddings_features_all.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

for dataset in DATASETS:

    logging.info(f"========== DATASET: {dataset} ==========")

    dataset_emb_root = os.path.join(BASE_EMB_ROOT, dataset)
    dataset_feat_root = os.path.join(FEATURE_ROOT, dataset)
    dataset_out_root = os.path.join(OUTPUT_ROOT, dataset)

    os.makedirs(dataset_out_root, exist_ok=True)

    for split in SPLITS:

        emb_path = os.path.join(dataset_emb_root, f"{split}_embeddings.csv")
        feat_path = os.path.join(dataset_feat_root, f"{split}_features.csv")
        out_path = os.path.join(dataset_out_root, f"{split}_features.csv")

        if not os.path.exists(emb_path):
            logging.warning(f"Missing embeddings file: {emb_path}")
            continue

        if not os.path.exists(feat_path):
            logging.warning(f"Missing feature file: {feat_path}")
            continue

        logging.info(f"Merging {dataset} | {split}")

        emb_df = pd.read_csv(emb_path)
        feat_df = pd.read_csv(feat_path)

        # Remove duplicate columns except SMILES
        common_cols = set(emb_df.columns) & set(feat_df.columns)
        common_cols.discard(SMILES_COL)

        if common_cols:
            logging.info(f"{dataset} | {split} | Dropping duplicate cols: {common_cols}")
            emb_df = emb_df.drop(columns=list(common_cols))

        merged = feat_df.merge(emb_df, on=SMILES_COL, how="inner")

        logging.info(
            f"{dataset} | {split} | "
            f"feat={feat_df.shape}, emb={emb_df.shape}, merged={merged.shape}"
        )

        merged.to_csv(out_path, index=False)
        logging.info(f"Saved: {out_path}")

logging.info("All dataset merges completed successfully.")
