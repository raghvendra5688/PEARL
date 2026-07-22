"""
Merge PC Features with Finetuned Embeddings — herg/dili/caco2/half_life

Mirrors pc_feature_extraction_ft_model_refactored.py (the original
bace/bbbp/clintox/flavor script) for the 4 new TDC datasets, covering both
the HF embedding modality (finetuned_model_embeddings_new_datasets.py) and
the Uni-Mol modality (unimol_embeddings_new_datasets.py).

Rather than duplicating the ~150 lines of RDKit/graph/fingerprint feature
code, this reuses pc_only_modelling.py's extract_pc_features() directly --
the exact same 473-dim PC feature vector used by the PC-only baseline (Phase
2) and every other PC feature consumer in this repo. Computed fresh from each
embedding CSV's own SMILES column (not read from the cached
data/pc_only_features/ CSVs) to guarantee row alignment -- pc_only's cache is
built from a dropna()'d frame and could in principle be a different length.

Usage:
    python pc_feature_merge_new_datasets.py
    python pc_feature_merge_new_datasets.py --modality hf --dataset herg
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "smiles" / "ml"))
from pc_only_modelling import extract_pc_features, sanitize_features  # noqa: E402

EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
LOG_DIR = REPO_ROOT / "logs"

SPLITS = ["train", "eval", "test"]

# modality -> (embedding root, output root, dataset-key -> (embed_dir, file_prefix, label_cols))
MODALITIES = {
    "hf": {
        "embed_root": EXTRAS_ROOT / "finetuned_embeddings",
        "output_root": EXTRAS_ROOT / "finetuned_pc_embeddings",
    },
    "unimol": {
        "embed_root": EXTRAS_ROOT / "unimol_embeddings",
        "output_root": EXTRAS_ROOT / "unimol_pc_embeddings",
    },
}

DATASETS = {
    "herg": {"embed_dir": "HERG_Embeddings", "file_prefix": "herg", "labels": ["hERG_Inhib"]},
    "dili": {"embed_dir": "DILI_Embeddings", "file_prefix": "dili", "labels": ["DILI_Label"]},
    "caco2": {"embed_dir": "CACO2_Embeddings", "file_prefix": "caco2", "labels": ["Caco2_LogPapp"]},
    "half_life": {"embed_dir": "HALF_LIFE_Embeddings", "file_prefix": "half_life", "labels": ["Half_Life_Hours"]},
}


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / "pc_feature_merge_new_datasets.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def process(modality: str, dataset_key: str) -> None:
    mcfg = MODALITIES[modality]
    dcfg = DATASETS[dataset_key]

    in_dir = mcfg["embed_root"] / dcfg["embed_dir"]
    out_dir = mcfg["output_root"] / dcfg["embed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        in_path = in_dir / f"{dcfg['file_prefix']}_{split}_embed.csv"
        out_path = out_dir / f"{dcfg['file_prefix']}_{split}_features.csv"

        if not in_path.exists():
            logging.warning(f"[{modality}/{dataset_key}/{split}] Missing embedding file: {in_path}")
            continue

        df = pd.read_csv(in_path)
        logging.info(f"[{modality}/{dataset_key}/{split}] Input shape: {df.shape}")

        pc_df = extract_pc_features(df["Standardized SMILES"])
        pc_df = sanitize_features(pc_df)

        embed_cols = [c for c in df.columns if c not in dcfg["labels"] + ["Standardized SMILES"]]

        merged = pd.concat(
            [
                df[["Standardized SMILES"]].reset_index(drop=True),
                df[embed_cols].reset_index(drop=True),
                pc_df.reset_index(drop=True),
                df[dcfg["labels"]].reset_index(drop=True),
            ],
            axis=1,
        )
        merged.to_csv(out_path, index=False)
        logging.info(f"[{modality}/{dataset_key}/{split}] Output shape: {merged.shape} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PC features with new-dataset embeddings")
    parser.add_argument("--modality", choices=list(MODALITIES.keys()), default=None)
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None)
    args = parser.parse_args()

    setup_logging()
    logging.info("=" * 60)
    logging.info("PC Feature Merge (herg/dili/caco2/half_life)")
    logging.info("=" * 60)

    modalities = [args.modality] if args.modality else list(MODALITIES.keys())
    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())

    for modality in modalities:
        for dataset_key in datasets:
            logging.info(f"\n{'='*40}\n{modality} / {dataset_key}")
            process(modality, dataset_key)

    logging.info("\nAll PC-feature merges complete.")


if __name__ == "__main__":
    main()
