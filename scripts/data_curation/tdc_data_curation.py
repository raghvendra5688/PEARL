"""
TDC / OpenADMET-adjacent Dataset Curation Pipeline

Addresses editor comment (see manuscript revision notes): PEARL's original four
benchmarks (BACE, BBBP, ClinTox, Flavor) are all MoleculeNet-era datasets. This
script pulls four additional, more rigorously curated ADMET endpoints from the
Therapeutics Data Commons (TDC) ADMET Benchmark Group -- the same curation
lineage the OpenADMET initiative builds on -- covering both classification and
regression:

    - hERG_Karim   (classification, cardiotoxicity / binding-affinity-like)
    - DILI         (classification, drug-induced liver injury)
    - Caco2_Wang   (regression, cell permeability)
    - Half_Life_Obach (regression, pharmacokinetic half-life)

Mirrors the conventions of scripts/data_curation/data_cleaning.py:
- Standardizes SMILES via RDKit (canonicalize + MolStandardize)
- Deduplicates on (Standardized SMILES + label)
- Uses TDC's own scaffold split (method="scaffold"), matching how
  BACE/BBBP/ClinTox are split in the original PEARL pipeline
- Writes raw pulls to data/raw/{dataset}_datasets/{split}.csv and cleaned
  splits to data/clean/{dataset}_datasets/{split}_clean.csv
- Logs before/after class or value statistics

Also writes data/clean/tdc_dataset_manifest.json recording, per dataset: task
type (binary/regression), SMILES column, label column, and split sizes, so
downstream scripts (PC-only baseline, GNN baselines, LoRA finetuning, RAFE)
can branch on task type without re-deriving this metadata.
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from tdc.single_pred import Tox, ADME

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw"
CLEAN_ROOT = REPO_ROOT / "data" / "clean"
TDC_CACHE = REPO_ROOT / "data" / "tdc_cache"
LOG_DIR = REPO_ROOT / "logs"
MANIFEST_PATH = CLEAN_ROOT / "tdc_dataset_manifest.json"

for d in (RAW_ROOT, CLEAN_ROOT, TDC_CACHE, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "tdc_data_curation.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

SPLITS = ["train", "valid", "test"]

# dataset_key -> (TDC loader class, TDC dataset name, task type, output label column)
DATASET_CONFIG = {
    "herg": {
        "loader": Tox,
        "tdc_name": "hERG_Karim",
        "task": "binary",
        "label_col": "hERG_Inhib",
    },
    "dili": {
        "loader": Tox,
        "tdc_name": "DILI",
        "task": "binary",
        "label_col": "DILI_Label",
    },
    "caco2": {
        "loader": ADME,
        "tdc_name": "Caco2_Wang",
        "task": "regression",
        "label_col": "Caco2_LogPapp",
    },
    "half_life": {
        "loader": ADME,
        "tdc_name": "Half_Life_Obach",
        "task": "regression",
        "label_col": "Half_Life_Hours",
        # Half-life is heavily right-skewed (train: mean=19h, max=1200h) -- a handful of
        # extreme, low-confidence PK measurements otherwise dominate RMSE-based training.
        "outlier_removal": "iqr_log1p",
    },
}


def compute_iqr_bounds_log1p(values: pd.Series, whisker: float = 1.5) -> tuple:
    """IQR bounds computed in log1p-space (appropriate for right-skewed, positive-only
    measurements like half-life), returned back-transformed to the original scale."""
    log_vals = np.log1p(values.astype(float))
    q1, q3 = log_vals.quantile(0.25), log_vals.quantile(0.75)
    iqr = q3 - q1
    lo_log = q1 - whisker * iqr
    hi_log = q3 + whisker * iqr
    return float(np.expm1(lo_log)), float(np.expm1(hi_log))


def remove_outliers(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame,
                     label_col: str, method: str) -> tuple:
    """Compute outlier bounds from TRAIN only, apply the same bounds to all splits so
    evaluation stays consistent with what the model is trained to predict."""
    if method != "iqr_log1p":
        raise ValueError(f"Unknown outlier removal method: {method}")

    lo, hi = compute_iqr_bounds_log1p(train_df[label_col])
    logging.info(f"Outlier bounds (log1p-IQR, back-transformed): [{lo:.3f}, {hi:.3f}] {label_col}")

    out = []
    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        before = len(df)
        filtered = df[(df[label_col] >= lo) & (df[label_col] <= hi)].reset_index(drop=True)
        removed = before - len(filtered)
        logging.info(f"{name} | outlier removal | removed {removed}/{before} rows outside [{lo:.3f}, {hi:.3f}]")
        out.append(filtered)
    return tuple(out)


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


def log_stats(df: pd.DataFrame, label_col: str, task: str, prefix: str) -> None:
    total = len(df)
    logging.info(f"{prefix} | Total samples: {total}")
    if task == "binary":
        counts = df[label_col].value_counts(dropna=False)
        for cls, cnt in counts.items():
            pct = (cnt / total) * 100 if total else 0
            logging.info(f"{prefix} | {label_col} = {cls} | count = {cnt} | pct = {pct:.2f}%")
    else:
        desc = df[label_col].describe()
        logging.info(f"{prefix} | {label_col} | mean={desc['mean']:.3f} std={desc['std']:.3f} "
                      f"min={desc['min']:.3f} max={desc['max']:.3f}")


def clean_split(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Standardize SMILES and drop invalid/duplicate rows, mirroring data_cleaning.py."""
    df = df.copy()
    df["mol"] = df["Drug"].apply(smiles_to_mol)
    df = df[df["mol"].notnull()]

    df["Canonicalized SMILES"] = df["mol"].apply(lambda m: Chem.MolToSmiles(m, canonical=True))
    df["Standardized SMILES"] = df["Canonicalized SMILES"].apply(standardise_smiles)
    df = df[df["Standardized SMILES"].notnull()]

    df = df.rename(columns={"Y": label_col})
    df = df.drop_duplicates(subset=["Standardized SMILES", label_col])
    df = df[["Standardized SMILES", label_col]].reset_index(drop=True)
    return df


def curate_dataset(dataset_key: str, cfg: dict) -> dict:
    logging.info("=" * 80)
    logging.info(f"Curating {dataset_key} ({cfg['tdc_name']}, task={cfg['task']})")
    logging.info("=" * 80)

    data = cfg["loader"](name=cfg["tdc_name"], path=str(TDC_CACHE))
    split = data.get_split(method="scaffold")

    raw_dir = RAW_ROOT / f"{dataset_key}_datasets"
    clean_dir = CLEAN_ROOT / f"{dataset_key}_datasets"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    cleaned = {}
    for split_name in SPLITS:
        raw_df = split[split_name]
        raw_df.to_csv(raw_dir / f"{split_name}.csv", index=False)

        log_stats(raw_df.rename(columns={"Y": cfg["label_col"]}), cfg["label_col"], cfg["task"],
                   f"{dataset_key} | {split_name} | BEFORE CLEANING")

        cleaned_df = clean_split(raw_df, cfg["label_col"])
        log_stats(cleaned_df, cfg["label_col"], cfg["task"], f"{dataset_key} | {split_name} | AFTER CLEANING")
        removed = len(raw_df) - len(cleaned_df)
        logging.info(f"{dataset_key} | {split_name} | Rows removed (cleaning): {removed}")
        cleaned[split_name] = cleaned_df

    if cfg.get("outlier_removal"):
        cleaned["train"], cleaned["valid"], cleaned["test"] = remove_outliers(
            cleaned["train"], cleaned["valid"], cleaned["test"], cfg["label_col"], cfg["outlier_removal"],
        )
        for split_name in SPLITS:
            log_stats(cleaned[split_name], cfg["label_col"], cfg["task"],
                       f"{dataset_key} | {split_name} | AFTER OUTLIER REMOVAL")

    split_sizes = {}
    for split_name in SPLITS:
        cleaned[split_name].to_csv(clean_dir / f"{split_name}_clean.csv", index=False)
        split_sizes[split_name] = len(cleaned[split_name])

    return {
        "tdc_name": cfg["tdc_name"],
        "task": cfg["task"],
        "smiles_col": "Standardized SMILES",
        "label_col": cfg["label_col"],
        "outlier_removal": cfg.get("outlier_removal"),
        "split_sizes": split_sizes,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TDC/OpenADMET-adjacent dataset curation")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()) + ["all"], default="all")
    args = parser.parse_args()

    logging.info("===== TDC DATASET CURATION STARTED =====")
    manifest = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

    dataset_keys = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]
    for dataset_key in dataset_keys:
        try:
            manifest[dataset_key] = curate_dataset(dataset_key, DATASET_CONFIG[dataset_key])
        except Exception as e:
            logging.error(f"Failed to curate {dataset_key}: {e}")
            raise

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Manifest written to {MANIFEST_PATH}")
    logging.info("===== TDC DATASET CURATION FINISHED =====")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
