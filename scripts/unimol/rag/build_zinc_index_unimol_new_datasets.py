"""
Build FAISS-GPU Indices for Uni-Mol ZINC-250k Embeddings — herg/dili/caco2/half_life

Mirrors build_zinc_index_unimol.py for the 4 new TDC datasets. Reads the .npy
files produced by embed_zinc250k_unimol_new_datasets.py, L2-normalises them
(cosine similarity via inner product), builds a GPU-accelerated FlatIP FAISS
index, then serialises the CPU version to disk.

Usage:
    python build_zinc_index_unimol_new_datasets.py --dataset herg
    python build_zinc_index_unimol_new_datasets.py --dataset all

Input  ($PEARL_EXTRAS_V2):
    zinc_embeddings_unimol/{dataset}/{col_name}.npy

Output ($PEARL_EXTRAS_V2):
    rag_indices_unimol/{dataset}/{col_name}.index
    rag_indices_unimol/{dataset}/meta.pkl
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
ZINC_CSV    = REPO_ROOT / "data" / "zinc250k" / "zinc250k_cleaned.csv"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
EMBED_ROOT  = EXTRAS_ROOT / "zinc_embeddings_unimol"
INDEX_ROOT  = EXTRAS_ROOT / "rag_indices_unimol"
LOG_DIR     = REPO_ROOT / "logs"

VALID_DATASETS = ["herg", "dili", "caco2", "half_life"]

DATASET_COL_NAMES = {
    "herg":      ["UniMol_FL", "UniMol_WL"],
    "dili":      ["UniMol_FL", "UniMol_WL"],
    "caco2":     ["UniMol_Huber"],
    "half_life": ["UniMol_Huber"],
}


def setup_logging(dataset: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / f"build_zinc_index_unimol_new_datasets_{dataset}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    logger = logging.getLogger()
    logger.handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    logger.addHandler(logging.StreamHandler())


def build_gpu_flatip_index(embeddings: np.ndarray, gpu_id: int = 0) -> faiss.Index:
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    res = faiss.StandardGpuResources()
    gpu_flat = faiss.GpuIndexFlatIP(res, d)
    gpu_flat.add(embeddings)
    cpu_index = faiss.index_gpu_to_cpu(gpu_flat)
    logging.info(f"  Index built: {cpu_index.ntotal} vectors, d={d}, metric=IP (cosine after L2-norm)")
    return cpu_index


def build_meta(zinc_df: pd.DataFrame) -> dict:
    return {
        "smiles": zinc_df["smiles"].tolist(),
        "logP":   zinc_df["logP"].values.astype(np.float32),
        "qed":    zinc_df["qed"].values.astype(np.float32),
        "SAS":    zinc_df["SAS"].values.astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS indices for new-dataset Uni-Mol ZINC-250k embeddings.")
    parser.add_argument("--dataset", choices=VALID_DATASETS + ["all"], default="all")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    datasets = VALID_DATASETS if args.dataset == "all" else [args.dataset]

    if not ZINC_CSV.exists():
        raise FileNotFoundError(f"ZINC CSV not found: {ZINC_CSV}")
    zinc_df = pd.read_csv(str(ZINC_CSV))

    for dataset in datasets:
        setup_logging(dataset)
        logging.info("=" * 60)
        logging.info(f"Build Uni-Mol ZINC FAISS Index | dataset={dataset} | GPU={args.gpu_id}")
        logging.info(f"Loaded {len(zinc_df)} ZINC-250k entries")
        logging.info("=" * 60)

        embed_dir = EMBED_ROOT / dataset
        index_dir = INDEX_ROOT / dataset
        index_dir.mkdir(parents=True, exist_ok=True)

        meta_path = index_dir / "meta.pkl"
        if not meta_path.exists():
            meta = build_meta(zinc_df)
            with open(str(meta_path), "wb") as f:
                pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Saved metadata: {meta_path}")
        else:
            logging.info(f"Metadata already exists: {meta_path}")

        for col_name in DATASET_COL_NAMES[dataset]:
            index_path = index_dir / f"{col_name}.index"
            if index_path.exists():
                logging.info(f"Index already exists, skipping: {index_path.name}")
                continue

            npy_path = embed_dir / f"{col_name}.npy"
            if not npy_path.exists():
                logging.warning(f"Embedding file not found, skipping: {npy_path}. Run embed_zinc250k_unimol_new_datasets.py first.")
                continue

            logging.info(f"\nBuilding index for: {col_name}")
            embeddings = np.load(str(npy_path)).astype(np.float32)
            logging.info(f"  Shape: {embeddings.shape}")

            cpu_index = build_gpu_flatip_index(embeddings, gpu_id=args.gpu_id)
            faiss.write_index(cpu_index, str(index_path))
            logging.info(f"  Saved index: {index_path}")

            del embeddings, cpu_index

        logging.info(f"\nDone. All Uni-Mol FAISS indices for {dataset} saved to: {index_dir}")


if __name__ == "__main__":
    main()
