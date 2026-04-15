"""
Build FAISS-GPU Indices for ZINC-250k Embeddings

For a given dataset, loads each of the 6 .npy embedding files produced by
embed_zinc250k.py, L2-normalises them (cosine similarity via inner product),
builds a GPU-accelerated FlatIP FAISS index (exact search, 100% recall), then
serialises the CPU version to disk.

Why GpuIndexFlatIP over HNSW?
  - HNSW in FAISS is CPU-only; FlatIP runs entirely on GPU.
  - At 249,455 vectors the GPU exhaustive scan takes <2 ms per batch query —
    faster than HNSW at this scale with no recall trade-off.
  - The serialised CPU index can be reloaded onto GPU at retrieval time in one
    call (faiss.index_cpu_to_gpu), keeping retrieval zero-server and in-process.

Usage:
    python "RAG Pipeline/build_zinc_index.py" --dataset bace

Input  (EffiChem_Extras):
    zinc_embeddings/{dataset}/{col_name}.npy

Output (EffiChem_Extras):
    rag_indices/{dataset}/{col_name}.index   — FAISS FlatIP (CPU, reloadable to GPU)
    rag_indices/{dataset}/meta.pkl           — {smiles, logP, qed, SAS} numpy arrays
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
EMBED_ROOT  = EXTRAS_ROOT / "zinc_embeddings"
INDEX_ROOT  = EXTRAS_ROOT / "rag_indices"
LOG_DIR     = REPO_ROOT / "logs"

VALID_DATASETS = ["bace", "bbbp", "clintox", "flavor"]

MODEL_COL_NAMES = [
    "ChemBERTa_77M_MTR_FL",
    "ChemBERTa_77M_MLM_FL",
    "MolFormer_Finetuned_FL",
    "ChemBERTa_77M_MTR_WL",
    "ChemBERTa_77M_MLM_WL",
    "Molformer_Finetuned_WL",
]


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(dataset: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / f"build_zinc_index_{dataset}.log"),
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

    The caller must pass a writable copy of the embedding array if the original
    should remain un-normalised (embed_zinc250k.py saves raw embeddings).
    """
    d = embeddings.shape[1]

    # In-place L2 normalisation → inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    # GPU resources
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = gpu_id
    cfg.useFloat16 = False          # full float32 precision

    gpu_index = faiss.GpuIndexFlatIP(res, d, cfg)
    gpu_index.add(embeddings)

    logging.info(
        f"  GPU index built: {gpu_index.ntotal} vectors, dim={d}, "
        f"GPU={gpu_id}"
    )

    # Convert to CPU for serialisation
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    return cpu_index


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS GPU indices for ZINC-250k.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=VALID_DATASETS,
        help="Dataset identifier (determines which embedding folder to read).",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device index (default: 0).")
    args = parser.parse_args()
    dataset = args.dataset

    setup_logging(dataset)
    logging.info("=" * 60)
    logging.info(f"Build ZINC FAISS Indices | dataset={dataset} | GPU={args.gpu_id}")
    logging.info("=" * 60)

    # ── Load ZINC-250k metadata (same for all 6 indices) ──────────────────────
    if not ZINC_CSV.exists():
        raise FileNotFoundError(f"Cleaned ZINC CSV not found: {ZINC_CSV}")
    zinc_df = pd.read_csv(str(ZINC_CSV))
    meta = {
        "smiles": zinc_df["smiles"].tolist(),
        "logP":   zinc_df["logP"].values.astype(np.float32),
        "qed":    zinc_df["qed"].values.astype(np.float32),
        "SAS":    zinc_df["SAS"].values.astype(np.float32),
    }
    logging.info(f"Loaded metadata for {len(meta['smiles'])} ZINC-250k molecules")

    emb_dir = EMBED_ROOT / dataset
    idx_dir = INDEX_ROOT / dataset
    idx_dir.mkdir(parents=True, exist_ok=True)

    # ── Save shared meta.pkl once per dataset ─────────────────────────────────
    meta_path = idx_dir / "meta.pkl"
    with open(str(meta_path), "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Saved metadata: {meta_path}")

    # ── Build one index per model column ──────────────────────────────────────
    for col_name in MODEL_COL_NAMES:
        npy_path   = emb_dir / f"{col_name}.npy"
        index_path = idx_dir / f"{col_name}.index"

        if index_path.exists():
            logging.info(f"Index already exists, skipping: {index_path.name}")
            continue

        if not npy_path.exists():
            logging.warning(f"Embedding file not found, skipping: {npy_path}")
            continue

        logging.info(f"\nBuilding index: {col_name}")
        embeddings = np.load(str(npy_path)).astype(np.float32)
        logging.info(f"  Loaded embeddings: {embeddings.shape}")

        # build_gpu_flatip_index normalises in-place → pass a copy so the .npy
        # files on disk retain raw (un-normalised) embeddings
        cpu_index = build_gpu_flatip_index(embeddings.copy(), gpu_id=args.gpu_id)

        faiss.write_index(cpu_index, str(index_path))
        logging.info(f"  Saved index: {index_path}")

        del embeddings, cpu_index

    logging.info("\nDone. All FAISS indices saved to:")
    logging.info(f"  {idx_dir}")


if __name__ == "__main__":
    main()
