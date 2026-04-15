"""
Build FAISS-GPU Indices for Uni-Mol ZINC-250k Embeddings

For a given dataset, loads the 2 Uni-Mol .npy embedding files (UniMol_FL,
UniMol_WL) produced by embed_zinc250k_unimol.py, L2-normalises them (cosine
similarity via inner product), builds a GPU-accelerated FlatIP FAISS index
(exact search, 100% recall), then serialises the CPU version to disk.

Embedding dimension: 2560 (512 Uni-Mol CLS + 2048 Morgan ECFP4)

Usage:
    python "Uni-Mol/RAG Pipeline/build_zinc_index_unimol.py" --dataset bace

Input  (EffiChem_Extras):
    zinc_embeddings_unimol/{dataset}/UniMol_FL.npy
    zinc_embeddings_unimol/{dataset}/UniMol_WL.npy

Output (EffiChem_Extras):
    rag_indices_unimol/{dataset}/UniMol_FL.index   — FAISS FlatIP (CPU, reloadable to GPU)
    rag_indices_unimol/{dataset}/UniMol_WL.index
    rag_indices_unimol/{dataset}/meta.pkl           — {smiles, logP, qed, SAS} numpy arrays
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
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
ZINC_CSV    = REPO_ROOT / "data" / "zinc250k" / "zinc250k_cleaned.csv"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
EMBED_ROOT  = EXTRAS_ROOT / "zinc_embeddings_unimol"
INDEX_ROOT  = EXTRAS_ROOT / "rag_indices_unimol"
LOG_DIR     = REPO_ROOT / "logs"

VALID_DATASETS = ["bace", "bbbp", "clintox", "flavor"]
MODEL_COL_NAMES = ["UniMol_FL", "UniMol_WL"]


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(dataset: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / f"build_zinc_index_unimol_{dataset}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


# ── Index builder ─────────────────────────────────────────────────────────────
def build_gpu_flatip_index(embeddings: np.ndarray, gpu_id: int = 0) -> faiss.Index:
    """
    L2-normalise embeddings in-place, build a GpuIndexFlatIP (exact cosine
    similarity via inner product), then return the equivalent CPU index for
    serialisation.
    """
    d = embeddings.shape[1]

    # In-place L2 normalisation → inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    # GPU resources
    res      = faiss.StandardGpuResources()
    gpu_flat = faiss.GpuIndexFlatIP(res, d)
    gpu_flat.add(embeddings)

    # Transfer back to CPU for serialisation
    cpu_index = faiss.index_gpu_to_cpu(gpu_flat)
    logging.info(
        f"  Index built: {cpu_index.ntotal} vectors, d={d}, "
        f"metric=IP (cosine after L2-norm)"
    )
    return cpu_index


# ── Metadata builder ──────────────────────────────────────────────────────────
def build_meta(zinc_df: pd.DataFrame) -> dict:
    """Extract SMILES, logP, QED, SAS arrays from ZINC dataframe."""
    meta = {
        "smiles": zinc_df["smiles"].tolist(),
        "logP":   zinc_df["logP"].values.astype(np.float32),
        "qed":    zinc_df["qed"].values.astype(np.float32),
        "SAS":    zinc_df["SAS"].values.astype(np.float32),
    }
    return meta


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS indices for Uni-Mol ZINC-250k embeddings."
    )
    parser.add_argument("--dataset", required=True, choices=VALID_DATASETS)
    parser.add_argument("--gpu-id",  type=int, default=0)
    args = parser.parse_args()
    dataset = args.dataset

    setup_logging(dataset)
    logging.info("=" * 60)
    logging.info(f"Build Uni-Mol ZINC FAISS Index | dataset={dataset} | GPU={args.gpu_id}")
    logging.info("=" * 60)

    if not ZINC_CSV.exists():
        raise FileNotFoundError(f"ZINC CSV not found: {ZINC_CSV}")
    zinc_df = pd.read_csv(str(ZINC_CSV))
    logging.info(f"Loaded {len(zinc_df)} ZINC-250k entries")

    embed_dir = EMBED_ROOT / dataset
    index_dir = INDEX_ROOT / dataset
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata once per dataset
    meta_path = index_dir / "meta.pkl"
    if not meta_path.exists():
        meta = build_meta(zinc_df)
        with open(str(meta_path), "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Saved metadata: {meta_path}")
    else:
        logging.info(f"Metadata already exists: {meta_path}")

    for col_name in MODEL_COL_NAMES:
        index_path = index_dir / f"{col_name}.index"
        if index_path.exists():
            logging.info(f"Index already exists, skipping: {index_path.name}")
            continue

        npy_path = embed_dir / f"{col_name}.npy"
        if not npy_path.exists():
            logging.warning(
                f"Embedding file not found, skipping: {npy_path}. "
                f"Run embed_zinc250k_unimol.py first."
            )
            continue

        logging.info(f"\nBuilding index for: {col_name}")
        logging.info(f"  Loading: {npy_path}")
        embeddings = np.load(str(npy_path)).astype(np.float32)
        logging.info(f"  Shape: {embeddings.shape}")

        cpu_index = build_gpu_flatip_index(embeddings, gpu_id=args.gpu_id)

        faiss.write_index(cpu_index, str(index_path))
        logging.info(f"  Saved index: {index_path}")

        del embeddings, cpu_index

    logging.info("\nDone. All Uni-Mol FAISS indices saved to:")
    logging.info(f"  {index_dir}")


if __name__ == "__main__":
    main()
