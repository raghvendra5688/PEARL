"""
RAG Feature Extraction (Uni-Mol) — herg/dili/caco2/half_life

Mirrors rag_feature_extraction_unimol.py for the 4 new TDC datasets. Queries
each task-specific Uni-Mol ZINC-250k FAISS index (built by
build_zinc_index_unimol_new_datasets.py) with every molecule in the
train/eval/test splits and computes the same 57 chemical-space context
features per model as the HF pipeline.

herg/dili use 2 models (UniMol_FL, UniMol_WL); caco2/half_life use 1
(UniMol_Huber).

Query embeddings are loaded from:
    $PEARL_EXTRAS_V2/unimol_embeddings/{Dataset}_Embeddings/{prefix}_{split}_embed.csv

PCA models are saved to $PEARL_EXTRAS_V2/rag_pca_unimol/{dataset}/{col_name}_pca.pkl.

Usage:
    python rag_feature_extraction_unimol_new_datasets.py --dataset herg

Output:
    data/rag_features_unimol/{dataset}/{col_name}_{split}_rag.csv
    $PEARL_EXTRAS_V2/rag_pca_unimol/{dataset}/{col_name}_pca.pkl
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
REPO_ROOT    = Path(__file__).resolve().parent.parent.parent.parent
EXTRAS_ROOT  = Path(os.environ.get("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
INDEX_ROOT   = EXTRAS_ROOT / "rag_indices_unimol"
PCA_ROOT     = EXTRAS_ROOT / "rag_pca_unimol"
EMBED_ROOT   = EXTRAS_ROOT / "unimol_embeddings"
RAG_OUT_ROOT = REPO_ROOT / "data" / "rag_features_unimol"
LOG_DIR      = REPO_ROOT / "logs"

SMILES_COL = "Standardized SMILES"
VALID_DATASETS = ["herg", "dili", "caco2", "half_life"]

MODEL_REGISTRY = {
    "herg":  [{"col_name": "UniMol_FL", "csv_col": "UniMol_FL_embeddings"},
              {"col_name": "UniMol_WL", "csv_col": "UniMol_WL_embeddings"}],
    "caco2": [{"col_name": "UniMol_Huber", "csv_col": "UniMol_Huber_embeddings"}],
}
MODEL_REGISTRY["dili"] = MODEL_REGISTRY["herg"]
MODEL_REGISTRY["half_life"] = MODEL_REGISTRY["caco2"]

DATASET_CFG = {
    "herg":      {"embed_dir": "HERG_Embeddings",      "file_prefix": "herg",      "labels": ["hERG_Inhib"]},
    "dili":      {"embed_dir": "DILI_Embeddings",      "file_prefix": "dili",      "labels": ["DILI_Label"]},
    "caco2":     {"embed_dir": "CACO2_Embeddings",     "file_prefix": "caco2",     "labels": ["Caco2_LogPapp"]},
    "half_life": {"embed_dir": "HALF_LIFE_Embeddings", "file_prefix": "half_life", "labels": ["Half_Life_Hours"]},
}

SPLITS_ORDER = ["train", "eval", "test"]
K_SMALL = 10
K_LARGE = 20
PCA_COMPONENTS = 32


def setup_logging(dataset: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / f"rag_feature_extraction_unimol_new_datasets_{dataset}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    logger = logging.getLogger()
    logger.handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    logger.addHandler(logging.StreamHandler())


def safe_parse_embedding(s: str) -> Optional[np.ndarray]:
    try:
        s = s.strip()
        if not any(c.isdigit() for c in s):
            return None
        try:
            arr = np.array(json.loads(s), dtype=np.float32)
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
    embeddings, valid_idx = [], []
    for i, val in enumerate(series):
        arr = safe_parse_embedding(str(val))
        if arr is not None:
            embeddings.append(arr)
            valid_idx.append(i)
    if not embeddings:
        raise ValueError("No valid embeddings found in column.")
    return np.vstack(embeddings).astype(np.float32), valid_idx


def load_index_on_gpu(index_path: Path, gpu_id: int = 0, use_gpu: bool = True) -> faiss.Index:
    cpu_index = faiss.read_index(str(index_path))
    if not use_gpu:
        logging.info(f"  Index loaded on CPU (--no-gpu): {cpu_index.ntotal} vectors, d={cpu_index.d}")
        return cpu_index
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    logging.info(f"  Index loaded on GPU {gpu_id}: {gpu_index.ntotal} vectors, d={cpu_index.d}")
    return gpu_index


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


def compute_tanimoto_features(query_smi: str, neighbor_smiles: List[str]) -> Dict[str, float]:
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


def compute_rag_features(
    query_embs: np.ndarray,
    query_smiles: List[str],
    index: faiss.Index,
    meta: Dict,
    pca: Optional[PCA],
    batch_size: int = 256,
) -> Tuple[pd.DataFrame, PCA]:
    N, D = query_embs.shape
    logP_arr = meta["logP"]
    qed_arr  = meta["qed"]
    SAS_arr  = meta["SAS"]
    smi_arr  = meta["smiles"]

    all_features: List[Dict] = []
    centroids: List[np.ndarray] = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = query_embs[start:end]

        sims, idxs = index.search(batch, K_LARGE)

        for b_i in range(end - start):
            g_i = start + b_i
            sim_k = sims[b_i]
            idx_k = idxs[b_i]

            sim10, idx10 = sim_k[:K_SMALL], idx_k[:K_SMALL]
            sim20, idx20 = sim_k, idx_k

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

            feats["zinc_nearest_sim"]  = float(sim10[0])
            feats["zinc_k10_sim_mean"] = float(sim10.mean())
            feats["zinc_k10_sim_std"]  = float(sim10.std())
            feats["zinc_k20_sim_mean"] = float(sim20.mean())
            feats["zinc_k20_sim_std"]  = float(sim20.std())

            try:
                nb_vecs = np.vstack([index.reconstruct(int(i)) for i in idx10])
                centroid = nb_vecs.mean(axis=0)
            except Exception:
                nb_vecs = None
                centroid = query_embs[g_i]

            centroids.append(centroid)
            q_vec = query_embs[g_i]
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
            feats["zinc_sim_to_centroid"] = float(np.dot(q_vec, centroid_norm))
            feats["zinc_centroid_spread"] = float(
                np.mean([np.linalg.norm(v - centroid) for v in nb_vecs])
            ) if nb_vecs is not None else 0.0

            nb_smiles = [smi_arr[int(i)] for i in idx10]
            tan_feats = compute_tanimoto_features(query_smiles[g_i], nb_smiles)
            feats.update(tan_feats)

            all_features.append(feats)

    centroid_matrix = np.vstack(centroids).astype(np.float32)

    if pca is None:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        pca.fit(centroid_matrix)
        logging.info(f"  PCA fitted on {centroid_matrix.shape[0]} train centroids "
                     f"(explained var: {pca.explained_variance_ratio_.sum():.3f})")

    centroid_pca = pca.transform(centroid_matrix)

    features_df = pd.DataFrame(all_features)
    for j in range(PCA_COMPONENTS):
        features_df[f"zinc_centroid_pca_{j:02d}"] = centroid_pca[:, j]

    return features_df, pca


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract new-dataset ZINC-250k Uni-Mol RAG features.")
    parser.add_argument("--dataset", required=True, choices=VALID_DATASETS)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU Faiss index (avoids CUDA kernel issues)")
    args = parser.parse_args()
    dataset = args.dataset

    setup_logging(dataset)
    logging.info("=" * 60)
    logging.info(f"Uni-Mol RAG Feature Extraction (new datasets) | dataset={dataset} | GPU={args.gpu_id}")
    logging.info("=" * 60)

    cfg = DATASET_CFG[dataset]
    embed_dir = EMBED_ROOT / cfg["embed_dir"]
    prefix = cfg["file_prefix"]
    label_cols = cfg["labels"]

    out_dir = RAG_OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    pca_dir = PCA_ROOT / dataset
    pca_dir.mkdir(parents=True, exist_ok=True)

    idx_dir = INDEX_ROOT / dataset
    meta_path = idx_dir / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.pkl not found: {meta_path}. Run build_zinc_index_unimol_new_datasets.py first.")

    with open(str(meta_path), "rb") as f:
        meta = pickle.load(f)
    logging.info(f"Loaded ZINC-250k metadata ({len(meta['smiles'])} molecules)")

    for model_info in MODEL_REGISTRY[dataset]:
        col_name = model_info["col_name"]
        csv_col = model_info["csv_col"]

        index_path = idx_dir / f"{col_name}.index"
        if not index_path.exists():
            logging.warning(f"Index not found, skipping: {index_path}. Run build_zinc_index_unimol_new_datasets.py first.")
            continue

        pca_path = pca_dir / f"{col_name}_pca.pkl"

        logging.info(f"\n{'='*50}")
        logging.info(f"Model: {col_name}")

        index = load_index_on_gpu(index_path, gpu_id=args.gpu_id, use_gpu=not args.no_gpu)

        fitted_pca: Optional[PCA] = None
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

            query_embs, valid_idx = parse_embedding_column(df[csv_col])
            df_valid = df.iloc[valid_idx].reset_index(drop=True)
            query_smiles = df_valid[SMILES_COL].tolist()

            faiss.normalize_L2(query_embs)

            is_train = (split == "train")
            pca_in = None if (is_train and fitted_pca is None) else fitted_pca

            features_df, fitted_pca = compute_rag_features(
                query_embs=query_embs,
                query_smiles=query_smiles,
                index=index,
                meta=meta,
                pca=pca_in,
                batch_size=256,
            )

            if is_train:
                with open(str(pca_path), "wb") as f:
                    pickle.dump(fitted_pca, f, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(f"  PCA model saved: {pca_path}")

            id_cols = [SMILES_COL] + [c for c in label_cols if c in df_valid.columns]
            out_df = pd.concat([df_valid[id_cols].reset_index(drop=True), features_df], axis=1)
            out_df.to_csv(str(out_csv), index=False)

            logging.info(f"  Saved: {out_csv.name}  rows={len(out_df)}, RAG_features={features_df.shape[1]}")

        del index

    logging.info("\nUni-Mol RAG feature extraction (new datasets) complete.")
    logging.info(f"  Features saved to: {out_dir}")


if __name__ == "__main__":
    main()
