"""
BACE Classification — DeepChem-Equivalent Scaffold Split + Cleaning

Uses scaffold splitting (same logic as DeepChem)
and saves CSV files for classical ML pipelines.

Final columns:
- Standardized SMILES
- CID
- Class
"""

import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import deepchem as dc


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_FILE = str(REPO_ROOT / "data" / "raw" / "bace.csv")
OUT_DIR = str(REPO_ROOT / "data" / "clean" / "bace_datasets")
LOG_DIR = str(REPO_ROOT / "logs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "bace_data_cleaning.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def smiles_to_mol(smiles):
    if not isinstance(smiles, str):
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def standardise_smiles(smiles):
    try:
        return rdMolStandardize.StandardizeSmiles(smiles)
    except Exception:
        return None

def log_class_stats(df, prefix):
    total = len(df)
    logging.info(f"{prefix} | Total samples: {total}")

    counts = df["Class"].value_counts(dropna=False)
    for cls, cnt in counts.items():
        pct = (cnt / total) * 100 if total else 0
        logging.info(f"{prefix} | Class={cls} | Count={cnt} | Percent={pct:.2f}%")


def clean_bace(df):

    df = df.copy()

    df["mol_rdkit"] = df["mol"].apply(smiles_to_mol)
    df = df[df["mol_rdkit"].notnull()]

    df["Canonicalized SMILES"] = df["mol_rdkit"].apply(
        lambda m: Chem.MolToSmiles(m, canonical=True)
    )

    df["Standardized SMILES"] = df["Canonicalized SMILES"].apply(standardise_smiles)
    df = df[df["Standardized SMILES"].notnull()]

    df = df.drop_duplicates(subset=["Standardized SMILES", "Class"])

    df = df[["Standardized SMILES", "CID", "Class"]].reset_index(drop=True)

    return df


def main():

    logging.info("===== BACE SCAFFOLD CLEANING PIPELINE STARTED =====")

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(RAW_FILE)

    raw_df = pd.read_csv(RAW_FILE)

    logging.info("BEFORE CLEANING")
    logging.info(f"Total rows: {len(raw_df)}")
    log_class_stats(raw_df, "RAW")

    clean_df = clean_bace(raw_df)

    logging.info("AFTER CLEANING")
    logging.info(f"Total rows: {len(clean_df)}")
    log_class_stats(clean_df, "CLEAN")

    # ---------- DeepChem Scaffold Split ----------

    logging.info("Performing DeepChem scaffold split")

    dataset = dc.data.NumpyDataset(
        X=np.zeros((len(clean_df), 1)),
        y=clean_df["Class"].values,
        ids=clean_df["Standardized SMILES"].values
    )

    splitter = dc.splits.ScaffoldSplitter()
    train_ds, valid_ds, test_ds = splitter.train_valid_test_split(
        dataset,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=42
    )

    def subset_from_ids(ds):
        mask = clean_df["Standardized SMILES"].isin(ds.ids)
        return clean_df[mask].copy()

    train_df = subset_from_ids(train_ds)
    valid_df = subset_from_ids(valid_ds)
    test_df  = subset_from_ids(test_ds)

    # logging splits

    logging.info("SPLIT SIZES AFTER SCAFFOLD SPLIT")
    log_class_stats(train_df, "TRAIN")
    log_class_stats(valid_df, "VALID")
    log_class_stats(test_df,  "TEST")

    logging.info(
        f"Split sizes | Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}"
    )

    clean_df.to_csv(os.path.join(OUT_DIR, "bace_full_clean.csv"), index=False)
    train_df.to_csv(os.path.join(OUT_DIR, "train_clean.csv"), index=False)
    valid_df.to_csv(os.path.join(OUT_DIR, "valid_clean.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test_clean.csv"), index=False)

    logging.info("Saved files:")
    logging.info("bace_full_clean.csv")
    logging.info("train_clean.csv")
    logging.info("valid_clean.csv")
    logging.info("test_clean.csv")

    logging.info("===== PIPELINE FINISHED SUCCESSFULLY =====")


if __name__ == "__main__":
    main()
