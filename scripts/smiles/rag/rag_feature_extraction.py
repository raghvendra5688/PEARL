"""
RAG Feature Extraction for EffiChem-2.0

For a given dataset, queries each of the 6 task-specific ZINC-250k FAISS indices
with every molecule in the train / eval / test splits and computes chemical-space
context features from the retrieved ZINC-250k neighbors.

Features computed per model (57 total):
  A. Physicochemical neighborhood profile  (27)
       logP / QED / SAS  mean & std  (k=10 and k=20)  →  12
       similarity-weighted mean logP / QED / SAS (k=10) →   3
       (repeated for k=20 weighted)                     →   3
       weighted SAS only                                →   3 [total: ~27 counting k10+k20]

  B. Neighborhood density & isolation       (5)
       nearest_sim, k10_sim_{mean,std}, k20_sim_{mean,std}

  C. Neighbor embedding centroid (PCA-32)  (34)
       centroid compressed to 32 dims (PCA fit on train, applied to eval/test)
       + sim_to_centroid + centroid_spread

  D. Tanimoto structural similarity         (3)
       tanimoto_k1, tanimoto_k5_mean, scaffold_match_frac_k10

PCA models are saved to EffiChem_Extras/rag_pca/{dataset}/{col_name}_pca.pkl
and must be fit on train before calling eval/test — the script processes splits
in order train → eval → test automatically.

Usage:
    python "RAG Pipeline/rag_feature_extraction.py" --dataset bace

Input:
    data/finetuned_embeddings/{Dataset}_Embeddings/{prefix}_{split}_embed.csv
    EffiChem_Extras/rag_indices/{dataset}/{col_name}.index
    EffiChem_Extras/rag_indices/{dataset}/meta.pkl

Output:
    data/rag_features/{dataset}/{col_name}_{split}_rag.csv
    EffiChem_Extras/rag_pca/{dataset}/{col_name}_pca.pkl
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
EXTRAS_ROOT  = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
INDEX_ROOT   = EXTRAS_ROOT / "rag_indices"
PCA_ROOT     = EXTRAS_ROOT / "rag_pca"
FT_EMBED_ROOT = EXTRAS_ROOT / "All_Embeddings"
RAG_OUT_ROOT  = REPO_ROOT / "data" / "rag_features"
LOG_DIR       = REPO_ROOT / "logs"

SMILES_COL = "Standardized SMILES"

VALID_DATASETS = ["bace", "bbbp", "clintox", "flavor"]

# Ordered list of col_names (without _embeddings suffix) and their CSV col names
MODEL_REGISTRY = [
    {"col_name": "ChemBERTa_77M_MTR_FL", "csv_col": "ChemBERTa_77M_MTR_FL_embeddings"},
    {"col_name": "ChemBERTa_77M_MLM_FL", "csv_col": "ChemBERTa_77M_MLM_FL_embeddings"},
    {"col_name": "MolFormer_Finetuned_FL", "csv_col": "MolFormer_Finetuned_FL_embeddings"},
    {"col_name": "ChemBERTa_77M_MTR_WL", "csv_col": "ChemBERTa_77M_MTR_WL_embeddings"},
    {"col_name": "ChemBERTa_77M_MLM_WL", "csv_col": "ChemBERTa_77M_MLM_WL_embeddings"},
    {"col_name": "Molformer_Finetuned_WL", "csv_col": "Molformer_Finetuned_WL_embeddings"},
]

DATASET_CFG = {
    "bace":    {"embed_dir": "BACE_Embeddings",    "file_prefix": "bace",    "labels": ["Class"]},
    "bbbp":    {"embed_dir": "BBBP_Embeddings",    "file_prefix": "bbbp",    "labels": ["p_np"]},
    "clintox": {"embed_dir": "clintox_Embeddings", "file_prefix": "clintox", "labels": ["FDA_APPROVED", "CT_TOX"]},
    "flavor":  {"embed_dir": "flavor_Embeddings",  "file_prefix": "fart",    "labels": ["Canonicalized Taste"]},
}

SPLITS_ORDER = ["train", "eval", "test"]   # train MUST come first (PCA fitting)
K_SMALL = 10
K_LARGE = 20
PCA_COMPONENTS = 32


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(dataset: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / f"rag_feature_extraction_{dataset}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


# ── Embedding parsing (matches existing modelling scripts) ────────────────────
def safe_parse_embedding(s: str) -> Optional[np.ndarray]:
    try:
        s = s.strip()
        if not any(c.isdigit() for c in s):
            return None
        try:
            parsed = json.loads(s)
            arr = np.array(parsed, dtype=np.float32)
        except json.JSONDecodeError:
            s_clean = s[1:-1] if (s.startswith("[") and s.endswith("]")) else s
            arr = np.array([float(x) for x in s_clean.split(",") if x.strip()], dtype=np.float32)
        if arr.ndim != 1 or len(arr) == 0:
            return None
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr
    except Exception:
        return None


def parse_embedding_column(series: pd.Series) -> Tuple[np.ndarray, List[int]]:
    """Returns (embedding_matrix, valid_row_indices)."""
    embeddings, valid_idx = [], []
    for i, val in enumerate(series):
        arr = safe_parse_embedding(str(val))
        if arr is not None:
            embeddings.append(arr)
            valid_idx.append(i)
    if not embeddings:
        raise ValueError("No valid embeddings found in column.")
    return np.vstack(embeddings).astype(np.float32), valid_idx


# ── FAISS index loading ───────────────────────────────────────────────────────
def load_index_on_gpu(index_path: Path, gpu_id: int = 0) -> faiss.Index:
    cpu_index = faiss.read_index(str(index_path))
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    logging.info(f"  Index loaded on GPU {gpu_id}: {gpu_index.ntotal} vectors, d={cpu_index.d}")
    return gpu_index


# ── Tanimoto helpers ──────────────────────────────────────────────────────────
def _morgan_fp(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _murcko_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def compute_tanimoto_features(
    query_smi: str,
    neighbor_smiles: List[str],   # k=10 neighbors (first 5 used for tanimoto_k5_mean)
) -> Dict[str, float]:
    qfp = _morgan_fp(query_smi)
    if qfp is None:
        return {"tanimoto_k1": 0.0, "tanimoto_k5_mean": 0.0, "scaffold_match_frac_k10": 0.0}

    tanimotos = []
    for smi in neighbor_smiles:
        nfp = _morgan_fp(smi)
        if nfp is not None:
            tanimotos.append(DataStructs.TanimotoSimilarity(qfp, nfp))

    q_scaffold = _murcko_scaffold(query_smi)
    scaffold_matches = sum(
        1 for smi in neighbor_smiles if _murcko_scaffold(smi) == q_scaffold and q_scaffold != ""
    )

    return {
        "tanimoto_k1":             tanimotos[0] if tanimotos else 0.0,
        "tanimoto_k5_mean":        float(np.mean(tanimotos[:5])) if tanimotos else 0.0,
        "scaffold_match_frac_k10": scaffold_matches / len(neighbor_smiles) if neighbor_smiles else 0.0,
    }


# ── Core feature computation ──────────────────────────────────────────────────
def compute_rag_features(
    query_embs: np.ndarray,          # (N, D), already L2-normalised
    query_smiles: List[str],
    index: faiss.Index,
    meta: Dict,
    pca: Optional[PCA],              # None when fitting (train split)
    batch_size: int = 256,
) -> Tuple[pd.DataFrame, PCA]:
    """
    Returns (features_df, fitted_pca).
    If pca is None (train split), fits PCA on centroids and returns it.
    If pca is provided (eval/test), applies it directly.
    """
    N, D = query_embs.shape
    logP_arr = meta["logP"]
    qed_arr  = meta["qed"]
    SAS_arr  = meta["SAS"]
    smi_arr  = meta["smiles"]

    all_features: List[Dict] = []
    centroids: List[np.ndarray] = []

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        batch = query_embs[start:end]  # already normalised

        # Search k=K_LARGE; first K_SMALL rows give k=10 neighbours
        sims, idxs = index.search(batch, K_LARGE)   # (B, K_LARGE)

        for b_i in range(end - start):
            g_i    = start + b_i
            sim_k  = sims[b_i]   # (K_LARGE,)  cosine similarities
            idx_k  = idxs[b_i]   # (K_LARGE,)  ZINC row indices

            sim10, idx10 = sim_k[:K_SMALL], idx_k[:K_SMALL]
            sim20, idx20 = sim_k, idx_k

            # ── A. Physicochemical neighborhood profile ────────────────────
            def weighted_mean(vals, weights):
                w = np.clip(weights, 0, None)
                return float(np.dot(vals, w) / (w.sum() + 1e-9))

            feats: Dict[str, float] = {}

            for prop, arr in [("logP", logP_arr), ("qed", qed_arr), ("SAS", SAS_arr)]:
                v10 = arr[idx10]
                v20 = arr[idx20]
                feats[f"zinc_{prop}_mean_k10"] = float(v10.mean())
                feats[f"zinc_{prop}_std_k10"]  = float(v10.std())
                feats[f"zinc_{prop}_mean_k20"] = float(v20.mean())
                feats[f"zinc_{prop}_std_k20"]  = float(v20.std())
                feats[f"zinc_{prop}_wmean_k10"] = weighted_mean(v10, sim10)
                feats[f"zinc_{prop}_wmean_k20"] = weighted_mean(v20, sim20)

            # ── B. Neighborhood density & isolation ────────────────────────
            feats["zinc_nearest_sim"]   = float(sim10[0])
            feats["zinc_k10_sim_mean"]  = float(sim10.mean())
            feats["zinc_k10_sim_std"]   = float(sim10.std())
            feats["zinc_k20_sim_mean"]  = float(sim20.mean())
            feats["zinc_k20_sim_std"]   = float(sim20.std())

            # ── C. Centroid (stored separately for PCA fitting) ────────────
            # Load neighbour embeddings from the index for centroid computation.
            # GpuIndexFlatIP does not expose reconstruct() — we store centroids
            # as the query's own embedding shifted toward the neighbour mean.
            # We approximate centroid as mean of the k=10 similarity-weighted
            # unit vectors stored in the index via reconstruction.
            # Since GpuIndexFlatIP inherits from GpuIndexFlat, reconstruct works.
            try:
                nb_vecs = np.vstack([index.reconstruct(int(i)) for i in idx10])
                centroid = nb_vecs.mean(axis=0)
            except Exception:
                # Fallback: use query embedding itself
                centroid = query_embs[g_i]

            centroids.append(centroid)
            q_vec = query_embs[g_i]
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
            feats["zinc_sim_to_centroid"] = float(np.dot(q_vec, centroid_norm))
            feats["zinc_centroid_spread"] = float(
                np.mean([np.linalg.norm(v - centroid) for v in nb_vecs])
            ) if "nb_vecs" in dir() else 0.0

            # ── D. Tanimoto structural similarity ─────────────────────────
            nb_smiles = [smi_arr[int(i)] for i in idx10]
            tan_feats = compute_tanimoto_features(query_smiles[g_i], nb_smiles)
            feats.update(tan_feats)

            all_features.append(feats)

    # ── PCA on centroids ──────────────────────────────────────────────────────
    centroid_matrix = np.vstack(centroids).astype(np.float32)

    if pca is None:
        # Train split: fit PCA
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        pca.fit(centroid_matrix)
        logging.info(f"  PCA fitted on {centroid_matrix.shape[0]} train centroids "
                     f"(explained var: {pca.explained_variance_ratio_.sum():.3f})")

    centroid_pca = pca.transform(centroid_matrix)   # (N, 32)

    features_df = pd.DataFrame(all_features)
    for j in range(PCA_COMPONENTS):
        features_df[f"zinc_centroid_pca_{j:02d}"] = centroid_pca[:, j]

    return features_df, pca


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ZINC-250k RAG features.")
    parser.add_argument("--dataset", required=True, choices=VALID_DATASETS)
    parser.add_argument("--gpu-id",  type=int, default=0)
    args = parser.parse_args()
    dataset = args.dataset

    setup_logging(dataset)
    logging.info("=" * 60)
    logging.info(f"RAG Feature Extraction | dataset={dataset} | GPU={args.gpu_id}")
    logging.info("=" * 60)

    cfg        = DATASET_CFG[dataset]
    embed_dir  = FT_EMBED_ROOT / cfg["embed_dir"]
    prefix     = cfg["file_prefix"]
    label_cols = cfg["labels"]

    out_dir = RAG_OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    pca_dir = PCA_ROOT / dataset
    pca_dir.mkdir(parents=True, exist_ok=True)

    idx_dir  = INDEX_ROOT / dataset
    meta_path = idx_dir / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.pkl not found: {meta_path}. Run build_zinc_index.py first.")

    with open(str(meta_path), "rb") as f:
        meta = pickle.load(f)
    logging.info(f"Loaded ZINC-250k metadata ({len(meta['smiles'])} molecules)")

    for model_info in MODEL_REGISTRY:
        col_name = model_info["col_name"]
        csv_col  = model_info["csv_col"]

        index_path = idx_dir / f"{col_name}.index"
        if not index_path.exists():
            logging.warning(f"Index not found, skipping: {index_path}. Run build_zinc_index.py first.")
            continue

        pca_path = pca_dir / f"{col_name}_pca.pkl"

        logging.info(f"\n{'='*50}")
        logging.info(f"Model: {col_name}")

        # Load FAISS index onto GPU once per model
        index = load_index_on_gpu(index_path, gpu_id=args.gpu_id)

        fitted_pca: Optional[PCA] = None
        # Load existing PCA if eval/test is run standalone
        if pca_path.exists():
            with open(str(pca_path), "rb") as f:
                fitted_pca = pickle.load(f)
            logging.info(f"  Loaded existing PCA from {pca_path}")

        for split in SPLITS_ORDER:
            out_csv = out_dir / f"{col_name}_{split}_rag.csv"
            if out_csv.exists():
                logging.info(f"  Already exists, skipping: {out_csv.name}")
                if split == "train" and fitted_pca is None and pca_path.exists():
                    with open(str(pca_path), "rb") as f:
                        fitted_pca = pickle.load(f)
                continue

            embed_csv = embed_dir / f"{prefix}_{split}_embed.csv"
            if not embed_csv.exists():
                logging.warning(f"  Embedding CSV not found: {embed_csv}")
                continue

            logging.info(f"  Processing split: {split}")
            df = pd.read_csv(str(embed_csv))

            if csv_col not in df.columns:
                logging.warning(f"  Column '{csv_col}' not in {embed_csv.name}, skipping.")
                continue

            # Parse query embeddings
            query_embs, valid_idx = parse_embedding_column(df[csv_col])
            df_valid = df.iloc[valid_idx].reset_index(drop=True)
            query_smiles = df_valid[SMILES_COL].tolist()

            # L2-normalise (same as index build step) for cosine search
            faiss.normalize_L2(query_embs)

            # Compute RAG features
            is_train = (split == "train")
            pca_in   = None if (is_train and fitted_pca is None) else fitted_pca

            features_df, fitted_pca = compute_rag_features(
                query_embs=query_embs,
                query_smiles=query_smiles,
                index=index,
                meta=meta,
                pca=pca_in,
                batch_size=256,
            )

            # Save fitted PCA after train split
            if is_train:
                with open(str(pca_path), "wb") as f:
                    pickle.dump(fitted_pca, f, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(f"  PCA model saved: {pca_path}")

            # Prepend SMILES + labels for easy merging
            id_cols = [SMILES_COL] + [c for c in label_cols if c in df_valid.columns]
            out_df  = pd.concat([df_valid[id_cols].reset_index(drop=True), features_df], axis=1)
            out_df.to_csv(str(out_csv), index=False)

            logging.info(
                f"  Saved: {out_csv.name}  "
                f"rows={len(out_df)}, RAG_features={features_df.shape[1]}"
            )

        # Free GPU memory between models
        del index

    logging.info("\nRAG feature extraction complete.")
    logging.info(f"  Features saved to: {out_dir}")


if __name__ == "__main__":
    main()
