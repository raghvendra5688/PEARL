"""
Data cleaning pipeline for EffiChem datasets (BBBP, ClinTox, Flavor).

This script:
1. Iterates over raw datasets and splits (train / test / valid)
2. Applies RDKit-based SMILES sanitization and standardization
3. Removes invalid molecules and duplicate (SMILES + label) combinations
4. Skips cleaning for datasets that are already curated (Flavor)
5. Saves cleaned datasets into a parallel directory structure
6. Logs detailed before/after statistics for reproducibility and auditing

Why this is needed:
- RDKit feature extraction and classical ML models are extremely sensitive
  to invalid SMILES, disconnected fragments, and inconsistent labeling.
- Duplicate SMILES with the same labels add no information and bias models.
- Logging statistics is critical for scientific transparency and debugging.
"""

import os
from pathlib import Path
import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_ROOT = str(REPO_ROOT / "data" / "raw")
CLEAN_ROOT = str(REPO_ROOT / "data" / "clean")
LOG_DIR = str(REPO_ROOT / "logs")

os.makedirs(CLEAN_ROOT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

"""
Logging to a file instead of stdout so that:
- dataset statistics are permanently recorded
- experiments are reproducible
- class imbalance changes are traceable
"""

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "data_cleaning.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

"""
Each dataset have:
- SMILES column name
- target columns

Flavor is already curated so no sanitization is applied.
"""

DATASET_CONFIG = {
    "bbbp_datasets": {
        "smiles_col": "smiles",
        "label_cols": ["p_np"],
        "clean": True,
    },
    "clintox_datasets": {
        "smiles_col": "smiles",
        "label_cols": ["FDA_APPROVED", "CT_TOX"],
        "clean": False,
    },
    "flavor_datasets": {
        "clean": False,
    },
}

SPLITS = ["train", "test", "valid"]


# Helper functions

def smiles_to_mol(smiles):
    """
    Converts a SMILES string to an RDKit Mol object.

    Why:
    - RDKit parsing is the strictest validity check for chemical structures.
    - Invalid SMILES must be removed before feature extraction.

    Returns None if parsing fails.
    """
    if not isinstance(smiles, str):
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def standardise_smiles(smiles):
    """
    Applies RDKit standardization.

    Why:
    - Ensures consistent representation (tautomers, charges, aromatics)
    - Prevents duplicate molecules appearing under different SMILES
    """
    try:
        return rdMolStandardize.StandardizeSmiles(smiles)
    except Exception:
        return None


def log_class_stats(df, label_cols, prefix):
    """
    Logs class counts and percentages for each label column.

    Why:
    - Cleaning can silently change class balance
    - Imbalance strongly affects ML model performance
    """
    total = len(df)
    logging.info(f"{prefix} | Total samples: {total}")

    for col in label_cols:
        counts = df[col].value_counts(dropna=False)
        for cls, cnt in counts.items():
            pct = (cnt / total) * 100 if total > 0 else 0
            logging.info(
                f"{prefix} | {col} = {cls} | count = {cnt} | percentage = {pct:.2f}%"
            )


def clean_dataset(df, smiles_col, label_cols):
    """
    Core cleaning routine.

    Steps:
    1. Convert SMILES into RDKit Mol (removes invalid SMILES)
    2. Canonicalize SMILES
    3. Standardize SMILES
    4. Drop duplicates based on (Standardized SMILES + labels)

    Important:
    - Duplicates are removed ONLY if both structure and labels match.
    - This avoids discarding conflicting supervision signals.
    """

    df = df.copy()

    df["mol"] = df[smiles_col].apply(smiles_to_mol)
    df = df[df["mol"].notnull()]

    df["Canonicalized SMILES"] = df["mol"].apply(
        lambda m: Chem.MolToSmiles(m, canonical=True)
    )

    df["Standardized SMILES"] = df["Canonicalized SMILES"].apply(standardise_smiles)
    df = df[df["Standardized SMILES"].notnull()]

    df = df.drop_duplicates(subset=["Standardized SMILES"] + label_cols)

    df = df[["Standardized SMILES"] + label_cols].reset_index(drop=True)

    return df


# Main execution

def main():
    logging.info("===== DATA CLEANING PIPELINE STARTED =====")

    for dataset_name, cfg in DATASET_CONFIG.items():
        raw_dir = os.path.join(RAW_ROOT, dataset_name)
        clean_dir = os.path.join(CLEAN_ROOT, dataset_name)
        os.makedirs(clean_dir, exist_ok=True)

        logging.info(f"Processing dataset: {dataset_name}")

        for split in SPLITS:
            input_path = os.path.join(raw_dir, f"{split}.csv")
            output_path = os.path.join(clean_dir, f"{split}_clean.csv")

            if not os.path.exists(input_path):
                logging.warning(f"Missing file: {input_path}")
                continue

            df = pd.read_csv(input_path)

            logging.info(f"{dataset_name} | {split} | BEFORE CLEANING")
            if cfg.get("clean", False):
                log_class_stats(df, cfg["label_cols"], f"{dataset_name} | {split}")

                smiles_col = cfg["smiles_col"]
                if smiles_col not in df.columns and "SMILES" in df.columns:
                    smiles_col = "SMILES"

                cleaned_df = clean_dataset(df, smiles_col, cfg["label_cols"])

                logging.info(f"{dataset_name} | {split} | AFTER CLEANING")
                log_class_stats(cleaned_df, cfg["label_cols"], f"{dataset_name} | {split}")

                removed = len(df) - len(cleaned_df)
                logging.info(
                    f"{dataset_name} | {split} | Rows removed: {removed}"
                )
            else:
                """
                Flavor dataset is already curated. We are not going ahead with any cleaning for Clintox
                We copy it unchanged to maintain a uniform pipeline.
                """
                cleaned_df = df.copy()
                logging.info(
                    f"{dataset_name} | {split} | No cleaning applied (preprocessed dataset)"
                )

            cleaned_df.to_csv(output_path, index=False)

    logging.info("===== DATA CLEANING PIPELINE FINISHED =====")


if __name__ == "__main__":
    main()
