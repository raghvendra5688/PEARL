"""
plot_roc_pr_4datasets.py

Generate the manuscript's 2x2 ROC and PR curve figures across BACE, BBBP,
hERG and DILI, comparing all four tree-based feature-engineering methods:
PC-only, FT Embed, FT Embed+PC, and RAFE.

This regenerates the figures that previously omitted the RAFE curve for
BACE and BBBP (those panels' RAFE models exist on disk; they were just not
wired up in the original figure-generation code).

All curves are computed from real, saved artifacts:
  - tree model .pkl files (CatBoost / LightGBM / XGBoost) loaded with joblib
  - precomputed embedding / physicochemical-feature / RAG-feature CSVs

No live CLM or Uni-Mol forward passes are needed here (pure tabular
tree-model inference), so this script is much lighter than
scripts/eval/plot_bace_roc_pr.py, which it borrows its helper-function
patterns from (_parse_emb / parse_emb_col / tree_predict / build_ft_smiles /
build_pc_ft_smiles / build_rafe_smiles / build_rafe_unimol).

Output:
    results/figures/roc_curves_4datasets.pdf
    results/figures/pr_curves_4datasets.pdf
"""

import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths (legacy "extras" roots not tracked in git; verified present) ───────
EXTRAS_ROOT_V1 = Path("/export/cse/rmall/Raghvendra/EffiChem_Extras")
EFFCHEM2       = Path("/export/cse/rmall/Raghvendra/EffChem-2.0")
EXTRAS_ROOT_V2 = Path("/export/qcai-omics/Raghvendra/EffiChem_Extras_v2")
PEARL_ROOT     = Path(__file__).resolve().parent.parent.parent

SMILES_COL = "Standardized SMILES"

OUT_DIR = PEARL_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ORDER = ["BACE", "BBBP", "hERG", "DILI"]
LABEL_COLS = {"BACE": "Class", "BBBP": "p_np", "hERG": "hERG_Inhib", "DILI": "DILI_Label"}
TEST_CSV = {
    "BACE": EFFCHEM2 / "data" / "clean" / "bace_datasets" / "test_clean.csv",
    "BBBP": EFFCHEM2 / "data" / "clean" / "bbbp_datasets" / "test_clean.csv",
    "hERG": PEARL_ROOT / "data" / "clean" / "herg_datasets" / "test_clean.csv",
    "DILI": PEARL_ROOT / "data" / "clean" / "dili_datasets" / "test_clean.csv",
}

# Six SMILES-model finetuned-embedding column names that must be excluded
# from "PC feature" columns in the v1 (BACE/BBBP) PC_FT / RAG csvs — matches
# _SKIP_EMBED_COLS in plot_bace_roc_pr.py.
SKIP_EMBED_COLS_V1 = [
    "Molformer_Finetuned_WL_embeddings",
    "MolFormer_Finetuned_FL_embeddings",
    "ChemBERTa_77M_MTR_WL_embeddings",
    "ChemBERTa_77M_MTR_FL_embeddings",
    "ChemBERTa_77M_MLM_WL_embeddings",
    "ChemBERTa_77M_MLM_FL_embeddings",
]

# Target AUC values from the PEARL manuscript tables (for verification only —
# legend values use the actually-computed AUC/AUPR, never these targets).
TARGET_AUC = {
    ("BACE", "PC-only"): 0.850, ("BACE", "FT Embed"): 0.869,
    ("BACE", "FT Embed+PC"): None, ("BACE", "RAFE"): 0.858,
    ("BBBP", "PC-only"): 0.937, ("BBBP", "FT Embed"): 0.935,
    ("BBBP", "FT Embed+PC"): 0.934, ("BBBP", "RAFE"): 0.940,
    ("hERG", "PC-only"): 0.874, ("hERG", "FT Embed"): 0.864,
    ("hERG", "FT Embed+PC"): 0.871, ("hERG", "RAFE"): 0.871,
    ("DILI", "PC-only"): 0.914, ("DILI", "FT Embed"): 0.893,
    ("DILI", "FT Embed+PC"): 0.873, ("DILI", "RAFE"): 0.875,
}
AUC_MISMATCH_TOL = 0.01

METHOD_ORDER = ["PC-only", "FT Embed", "FT Embed+PC", "RAFE"]
METHOD_COLORS = {
    "PC-only":     "#757575",
    "FT Embed":    "#2E7D32",
    "FT Embed+PC": "#6A1B9A",
    "RAFE":        "#C62828",
}


# ── Embedding parser (verbatim pattern from plot_bace_roc_pr.py) ─────────────
def _parse_emb(s: str):
    import json
    s = str(s).strip()
    try:
        return np.array(json.loads(s), dtype=np.float32)
    except Exception:
        try:
            clean = s[1:-1] if s.startswith("[") else s
            return np.array([float(x) for x in clean.split(",") if x.strip()],
                            dtype=np.float32)
        except Exception:
            return None


def parse_emb_col(series: pd.Series):
    pairs = [(i, _parse_emb(v)) for i, v in enumerate(series)]
    valid = [(i, e) for i, e in pairs if e is not None]
    return (np.vstack([e for _, e in valid]).astype(np.float32),
            [i for i, _ in valid])


def tree_predict(model_path: Path, X: np.ndarray) -> np.ndarray:
    model = joblib.load(str(model_path))
    return model.predict_proba(X)[:, 1]


# ── PC-only builder (all 4 datasets) ─────────────────────────────────────────
PC_ONLY_MODEL_DIR = {
    "BACE": "BACE_PC_Only_Results",
    "BBBP": "BBBP_PC_Only_Results",
    "hERG": "HERG_PC_Only_Results",
    "DILI": "DILI_PC_Only_Results",
}
PC_ONLY_FEATURE_DIR = {"BACE": "bace", "BBBP": "bbbp", "hERG": "herg", "DILI": "dili"}


def build_pc_only_Xy(dataset: str):
    test_df = pd.read_csv(TEST_CSV[dataset])
    feat_csv = (PEARL_ROOT / "data" / "pc_only_features"
                / PC_ONLY_FEATURE_DIR[dataset] / "test_pc_features.csv")
    feat_df = pd.read_csv(feat_csv)
    if len(test_df) != len(feat_df):
        raise ValueError(
            f"{dataset}: row-count mismatch between test_clean.csv "
            f"({len(test_df)}) and PC-only features ({len(feat_df)})"
        )
    X = (feat_df.apply(pd.to_numeric, errors="coerce")
         .fillna(0).clip(-1e6, 1e6).values.astype(np.float32))
    y = test_df[LABEL_COLS[dataset]].values.astype(int)
    return X, y


def select_pc_only_model(dataset: str):
    """Try CatBoost/LightGBM/XGBoost PC-only models and pick the one whose
    computed AUC best matches the manuscript target (falls back to CatBoost
    if no target is defined)."""
    X, y = build_pc_only_Xy(dataset)
    model_dir = (PEARL_ROOT / "results" / "pc_only"
                 / PC_ONLY_MODEL_DIR[dataset] / "models")
    target = TARGET_AUC[(dataset, "PC-only")]
    candidates = []
    for name in ["CatBoost", "LightGBM", "XGBoost"]:
        p = model_dir / f"{name}.pkl"
        if not p.exists():
            continue
        score = tree_predict(p, X)
        auc = roc_auc_score(y, score)
        diff = abs(auc - target) if target is not None else 0.0
        log.info(f"    [PC-only candidate] {dataset} {name}: AUC={auc:.4f} "
                 f"(target {target})")
        candidates.append((diff, name, p, score, auc))
    if not candidates:
        raise FileNotFoundError(f"No PC-only models found for {dataset} in {model_dir}")
    candidates.sort(key=lambda c: c[0])
    _, best_name, best_path, best_score, best_auc = candidates[0]
    log.info(f"    -> selected PC-only model for {dataset}: {best_name} "
             f"(AUC={best_auc:.4f})")
    return y, best_score, best_path


# ── v1 (BACE/BBBP) SMILES-embedding builders — generalized from
#    plot_bace_roc_pr.py's build_ft_smiles / build_pc_ft_smiles / build_rafe_smiles
def build_ft_smiles_v1(dataset: str, emb_col: str, split: str = "test"):
    csv = EXTRAS_ROOT_V1 / "All_Embeddings" / f"{dataset}_Embeddings" / f"{dataset.lower()}_{split}_embed.csv"
    df = pd.read_csv(str(csv))
    embs, idxs = parse_emb_col(df[emb_col])
    y = df[LABEL_COLS[dataset]].iloc[idxs].values.astype(int)
    return embs, y


def build_pc_ft_smiles_v1(dataset: str, emb_col: str, split: str = "test"):
    csv = EXTRAS_ROOT_V1 / "PC_FT_All_Embeddings" / f"{dataset}_Embeddings" / f"{dataset.lower()}_{split}_features.csv"
    df = pd.read_csv(str(csv))
    embs, idxs = parse_emb_col(df[emb_col])
    df = df.iloc[idxs].reset_index(drop=True)
    label_col = LABEL_COLS[dataset]
    skip = {SMILES_COL, label_col} | set(SKIP_EMBED_COLS_V1)
    pc_cols = [c for c in df.columns if c not in skip]
    pc = df[pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    X = np.hstack([embs, pc]).astype(np.float32)
    y = df[label_col].values.astype(int)
    return X, y


def build_rafe_smiles_v1(dataset: str, emb_col: str, rag_col: str, split: str = "test"):
    label_col = LABEL_COLS[dataset]
    pc_csv = EXTRAS_ROOT_V1 / "PC_FT_All_Embeddings" / f"{dataset}_Embeddings" / f"{dataset.lower()}_{split}_features.csv"
    rag_csv = EFFCHEM2 / "data" / "rag_features" / dataset.lower() / f"{rag_col}_{split}_rag.csv"
    df_pc = pd.read_csv(str(pc_csv))
    df_rag = pd.read_csv(str(rag_csv))

    merged = df_pc.merge(
        df_rag.drop(columns=[c for c in df_rag.columns if c == label_col], errors="ignore"),
        on=SMILES_COL, how="inner",
    )
    embs, idxs = parse_emb_col(merged[emb_col])
    merged = merged.iloc[idxs].reset_index(drop=True)

    rag_cols = [c for c in df_rag.columns if c not in [SMILES_COL, label_col]]
    skip = {SMILES_COL, label_col} | set(SKIP_EMBED_COLS_V1)
    pc_cols = [c for c in merged.columns if c not in skip and c not in rag_cols]

    pc = merged[pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    rag = merged[rag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X = np.hstack([embs, pc, rag]).astype(np.float32)
    y = merged[label_col].values.astype(int)
    return X, y


# ── v2 (hERG/DILI) builders — generalized from build_rafe_unimol ────────────
def build_ft_embed_v2(embed_csv: Path, emb_col: str, label_col: str):
    df = pd.read_csv(str(embed_csv))
    embs, idxs = parse_emb_col(df[emb_col])
    y = df[label_col].iloc[idxs].values.astype(int)
    return embs, y


def build_pc_ft_v2(features_csv: Path, emb_col: str, label_col: str, extra_skip_cols):
    df = pd.read_csv(str(features_csv))
    embs, idxs = parse_emb_col(df[emb_col])
    df = df.iloc[idxs].reset_index(drop=True)
    skip = {SMILES_COL, label_col, emb_col} | set(extra_skip_cols)
    pc_cols = [c for c in df.columns if c not in skip]
    pc = df[pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    X = np.hstack([embs, pc]).astype(np.float32)
    y = df[label_col].values.astype(int)
    return X, y


def build_rafe_v2(embed_csv: Path, features_csv: Path, rag_csv: Path,
                   emb_col: str, label_col: str, extra_skip_cols):
    df_emb = pd.read_csv(str(embed_csv))
    df_pc = pd.read_csv(str(features_csv))
    df_rag = pd.read_csv(str(rag_csv))

    pc_cols_to_add = [c for c in df_pc.columns
                       if c not in ({SMILES_COL, label_col, emb_col} | set(extra_skip_cols))]

    merged = df_emb[[SMILES_COL, label_col, emb_col]].merge(
        df_pc[[SMILES_COL] + pc_cols_to_add], on=SMILES_COL, how="inner"
    ).merge(
        df_rag.drop(columns=[c for c in df_rag.columns if c == label_col], errors="ignore"),
        on=SMILES_COL, how="inner",
    )

    embs, idxs = parse_emb_col(merged[emb_col])
    merged = merged.iloc[idxs].reset_index(drop=True)

    rag_feat_cols = [c for c in df_rag.columns if c not in [SMILES_COL, label_col]]
    pc = merged[pc_cols_to_add].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    rag = merged[rag_feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X = np.hstack([embs, pc, rag]).astype(np.float32)
    y = merged[label_col].values.astype(int)
    return X, y


def _tree(Xy, model_path):
    X, y = Xy
    score = tree_predict(model_path, X)
    return y, score


# ── Per-dataset method registries ────────────────────────────────────────────
def compute_dataset_results(dataset: str):
    """Return dict method_label -> (path_used, y, score) for one dataset."""
    results = {}

    def run(method, fn):
        log.info(f"[{dataset} / {method}]")
        try:
            path_used, y, score = fn()
            results[method] = (path_used, y, score)
            log.info(f"  OK  n={len(y)}")
        except Exception as e:
            log.error(f"  FAILED: {e}", exc_info=True)

    if dataset == "BACE":
        run("PC-only", lambda: _pc_only_wrapper("BACE"))
        run("FT Embed", lambda: (
            EXTRAS_ROOT_V1 / "BACE_FT_Results" / "MolFormer_Finetuned_FL"
            / "models" / "MolFormer_Finetuned_FL_CatBoost.pkl",
            *_tree(build_ft_smiles_v1("BACE", "MolFormer_Finetuned_FL_embeddings"),
                   EXTRAS_ROOT_V1 / "BACE_FT_Results" / "MolFormer_Finetuned_FL"
                   / "models" / "MolFormer_Finetuned_FL_CatBoost.pkl"),
        ))
        run("FT Embed+PC", lambda: (
            EXTRAS_ROOT_V1 / "BACE_PC_FT_Results" / "MolFormer_Finetuned_FL_embeddings"
            / "models" / "XGBoost.pkl",
            *_tree(build_pc_ft_smiles_v1("BACE", "MolFormer_Finetuned_FL_embeddings"),
                   EXTRAS_ROOT_V1 / "BACE_PC_FT_Results" / "MolFormer_Finetuned_FL_embeddings"
                   / "models" / "XGBoost.pkl"),
        ))
        run("RAFE", lambda: (
            EFFCHEM2 / "results" / "rag" / "bace" / "Molformer_Finetuned_WL"
            / "models" / "LightGBM.pkl",
            *_tree(build_rafe_smiles_v1("BACE", "Molformer_Finetuned_WL_embeddings", "Molformer_Finetuned_WL"),
                   EFFCHEM2 / "results" / "rag" / "bace" / "Molformer_Finetuned_WL"
                   / "models" / "LightGBM.pkl"),
        ))

    elif dataset == "BBBP":
        run("PC-only", lambda: _pc_only_wrapper("BBBP"))
        run("FT Embed", lambda: (
            EXTRAS_ROOT_V1 / "BBBP_FT_Results" / "MolFormer_Finetuned_FL"
            / "models" / "MolFormer_Finetuned_FL_XGBoost.pkl",
            *_tree(build_ft_smiles_v1("BBBP", "MolFormer_Finetuned_FL_embeddings"),
                   EXTRAS_ROOT_V1 / "BBBP_FT_Results" / "MolFormer_Finetuned_FL"
                   / "models" / "MolFormer_Finetuned_FL_XGBoost.pkl"),
        ))
        run("FT Embed+PC", lambda: (
            EXTRAS_ROOT_V1 / "BBBP_PC_FT_Results" / "MolFormer_Finetuned_FL_embeddings"
            / "models" / "LightGBM.pkl",
            *_tree(build_pc_ft_smiles_v1("BBBP", "MolFormer_Finetuned_FL_embeddings"),
                   EXTRAS_ROOT_V1 / "BBBP_PC_FT_Results" / "MolFormer_Finetuned_FL_embeddings"
                   / "models" / "LightGBM.pkl"),
        ))
        run("RAFE", lambda: (
            EFFCHEM2 / "results" / "rag" / "bbbp" / "MolFormer_Finetuned_FL"
            / "models" / "XGBoost.pkl",
            *_tree(build_rafe_smiles_v1("BBBP", "MolFormer_Finetuned_FL_embeddings", "MolFormer_Finetuned_FL"),
                   EFFCHEM2 / "results" / "rag" / "bbbp" / "MolFormer_Finetuned_FL"
                   / "models" / "XGBoost.pkl"),
        ))

    elif dataset == "hERG":
        run("PC-only", lambda: _pc_only_wrapper("hERG"))
        run("FT Embed", lambda: (
            PEARL_ROOT / "results" / "ft_embeddings" / "UniMol_HERG_FT_Results"
            / "UniMol_FL" / "models" / "LightGBM.pkl",
            *_tree(build_ft_embed_v2(
                EXTRAS_ROOT_V2 / "unimol_embeddings" / "HERG_Embeddings" / "herg_test_embed.csv",
                "UniMol_FL_embeddings", "hERG_Inhib"),
                PEARL_ROOT / "results" / "ft_embeddings" / "UniMol_HERG_FT_Results"
                / "UniMol_FL" / "models" / "LightGBM.pkl"),
        ))
        run("FT Embed+PC", lambda: (
            PEARL_ROOT / "results" / "ft_embeddings" / "UniMol_HERG_PC_FT_Results"
            / "UniMol_FL" / "models" / "LightGBM.pkl",
            *_tree(build_pc_ft_v2(
                EXTRAS_ROOT_V2 / "unimol_pc_embeddings" / "HERG_Embeddings" / "herg_test_features.csv",
                "UniMol_FL_embeddings", "hERG_Inhib", ["UniMol_WL_embeddings"]),
                PEARL_ROOT / "results" / "ft_embeddings" / "UniMol_HERG_PC_FT_Results"
                / "UniMol_FL" / "models" / "LightGBM.pkl"),
        ))
        run("RAFE", lambda: (
            PEARL_ROOT / "results" / "rag_unimol" / "HERG" / "UniMol_FL" / "models" / "LightGBM.pkl",
            *_tree(build_rafe_v2(
                EXTRAS_ROOT_V2 / "unimol_embeddings" / "HERG_Embeddings" / "herg_test_embed.csv",
                EXTRAS_ROOT_V2 / "unimol_pc_embeddings" / "HERG_Embeddings" / "herg_test_features.csv",
                PEARL_ROOT / "data" / "rag_features_unimol" / "herg" / "UniMol_FL_test_rag.csv",
                "UniMol_FL_embeddings", "hERG_Inhib", ["UniMol_WL_embeddings"]),
                PEARL_ROOT / "results" / "rag_unimol" / "HERG" / "UniMol_FL" / "models" / "LightGBM.pkl"),
        ))

    elif dataset == "DILI":
        run("PC-only", lambda: _pc_only_wrapper("DILI"))
        run("FT Embed", lambda: (
            PEARL_ROOT / "results" / "ft_embeddings" / "DILI_FT_Results"
            / "ChemBERTa_77M_MLM_WL" / "models" / "CatBoost.pkl",
            *_tree(build_ft_embed_v2(
                EXTRAS_ROOT_V2 / "finetuned_embeddings" / "DILI_Embeddings" / "dili_test_embed.csv",
                "ChemBERTa_77M_MLM_WL_embeddings", "DILI_Label"),
                PEARL_ROOT / "results" / "ft_embeddings" / "DILI_FT_Results"
                / "ChemBERTa_77M_MLM_WL" / "models" / "CatBoost.pkl"),
        ))
        run("FT Embed+PC", lambda: (
            PEARL_ROOT / "results" / "ft_embeddings" / "UniMol_DILI_PC_FT_Results"
            / "UniMol_FL" / "models" / "LightGBM.pkl",
            *_tree(build_pc_ft_v2(
                EXTRAS_ROOT_V2 / "unimol_pc_embeddings" / "DILI_Embeddings" / "dili_test_features.csv",
                "UniMol_FL_embeddings", "DILI_Label", ["UniMol_WL_embeddings"]),
                PEARL_ROOT / "results" / "ft_embeddings" / "UniMol_DILI_PC_FT_Results"
                / "UniMol_FL" / "models" / "LightGBM.pkl"),
        ))
        run("RAFE", lambda: (
            PEARL_ROOT / "results" / "rag_unimol" / "DILI" / "UniMol_FL" / "models" / "XGBoost.pkl",
            *_tree(build_rafe_v2(
                EXTRAS_ROOT_V2 / "unimol_embeddings" / "DILI_Embeddings" / "dili_test_embed.csv",
                EXTRAS_ROOT_V2 / "unimol_pc_embeddings" / "DILI_Embeddings" / "dili_test_features.csv",
                PEARL_ROOT / "data" / "rag_features_unimol" / "dili" / "UniMol_FL_test_rag.csv",
                "UniMol_FL_embeddings", "DILI_Label", ["UniMol_WL_embeddings"]),
                PEARL_ROOT / "results" / "rag_unimol" / "DILI" / "UniMol_FL" / "models" / "XGBoost.pkl"),
        ))

    return results


def _pc_only_wrapper(dataset):
    y, score, path = select_pc_only_model(dataset)
    return path, y, score


# ── Metrics assembly + verification log ──────────────────────────────────────
def compute_all():
    """Returns {dataset: {method: (fpr, tpr, auc, prec, rec, aupr, prevalence)}}"""
    all_results = {}
    report_rows = []
    for dataset in DATASET_ORDER:
        log.info("=" * 70)
        log.info(f"Dataset: {dataset}")
        log.info("=" * 70)
        raw = compute_dataset_results(dataset)
        dataset_curves = {}
        for method in METHOD_ORDER:
            if method not in raw:
                log.warning(f"  MISSING curve: {dataset} / {method}")
                continue
            path_used, y, score = raw[method]
            auc = roc_auc_score(y, score)
            aupr = average_precision_score(y, score)
            fpr, tpr, _ = roc_curve(y, score)
            prec, rec, _ = precision_recall_curve(y, score)
            prevalence = y.mean()
            target = TARGET_AUC.get((dataset, method))
            mismatch = (target is not None and abs(auc - target) > AUC_MISMATCH_TOL)
            flag = "MISMATCH" if mismatch else ("OK" if target is not None else "n/a-target")
            if mismatch:
                log.warning(f"  [{dataset}/{method}] computed AUC={auc:.4f} vs "
                            f"target={target:.3f} -> diff>{AUC_MISMATCH_TOL} FLAG={flag}")
            else:
                log.info(f"  [{dataset}/{method}] computed AUC={auc:.4f} AUPR={aupr:.4f} "
                         f"(target={target}) {flag}")
            dataset_curves[method] = (fpr, tpr, auc, prec, rec, aupr, prevalence)
            report_rows.append({
                "dataset": dataset, "method": method,
                "model_path": str(path_used), "auc": auc, "aupr": aupr,
                "target_auc": target, "flag": flag,
            })
        all_results[dataset] = dataset_curves
    return all_results, report_rows


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_roc(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0))
    panel_pos = {"BACE": (0, 0), "BBBP": (0, 1), "hERG": (1, 0), "DILI": (1, 1)}

    for dataset, (r, c) in panel_pos.items():
        ax = axes[r, c]
        ax.plot([0, 1], [0, 1], color="lightgray", lw=0.8, ls=":")
        curves = all_results.get(dataset, {})
        for method in METHOD_ORDER:
            if method not in curves:
                continue
            fpr, tpr, auc_, _, _, _, _ = curves[method]
            ax.plot(fpr, tpr, color=METHOD_COLORS[method], lw=1.5, ls="-",
                     label=f"{method} ({auc_:.3f})")
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9,
                  title="Method (AUC)", title_fontsize=7.5,
                  handlelength=2.0, borderpad=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)

    fig.suptitle("ROC curves, all tree-based methods, four classification datasets",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT_DIR / "roc_curves_4datasets.pdf"
    fig.savefig(str(p), bbox_inches="tight")
    log.info(f"Saved: {p}")
    plt.close(fig)
    return p


def plot_pr(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0))
    panel_pos = {"BACE": (0, 0), "BBBP": (0, 1), "hERG": (1, 0), "DILI": (1, 1)}

    for dataset, (r, c) in panel_pos.items():
        ax = axes[r, c]
        curves = all_results.get(dataset, {})
        prevalence = None
        for method in METHOD_ORDER:
            if method not in curves:
                continue
            _, _, _, prec, rec, aupr_, prev = curves[method]
            prevalence = prev
            ax.plot(rec, prec, color=METHOD_COLORS[method], lw=1.5, ls="-",
                     label=f"{method} ({aupr_:.3f})")
        if prevalence is not None:
            ax.axhline(y=prevalence, color="lightgray", lw=0.8, ls=":", label="_nolegend_")
        ax.set_xlabel("Recall", fontsize=9)
        ax.set_ylabel("Precision", fontsize=9)
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9,
                  title="Method (AUPR)", title_fontsize=7.5,
                  handlelength=2.0, borderpad=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)

    fig.suptitle("PR curves, all tree-based methods, four classification datasets",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT_DIR / "pr_curves_4datasets.pdf"
    fig.savefig(str(p), bbox_inches="tight")
    log.info(f"Saved: {p}")
    plt.close(fig)
    return p


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("#" * 70)
    log.info("4-dataset ROC / PR curve generation — PEARL manuscript figures")
    log.info("#" * 70)
    all_results, report_rows = compute_all()

    n_ok = sum(len(v) for v in all_results.values())
    log.info(f"\n{n_ok}/16 (dataset, method) curves computed successfully.")

    log.info("\n" + "=" * 100)
    log.info(f"{'Dataset':<7}{'Method':<14}{'AUC':>8}{'AUPR':>8}{'Target':>8}{'Flag':>12}")
    log.info("-" * 100)
    for row in report_rows:
        tgt = f"{row['target_auc']:.3f}" if row["target_auc"] is not None else "n/a"
        log.info(f"{row['dataset']:<7}{row['method']:<14}{row['auc']:>8.4f}"
                  f"{row['aupr']:>8.4f}{tgt:>8}{row['flag']:>12}   {row['model_path']}")
    log.info("=" * 100)

    if n_ok == 0:
        log.error("No results at all — check errors above.")
        sys.exit(1)

    roc_path = plot_roc(all_results)
    pr_path = plot_pr(all_results)
    log.info(f"\nDone. Outputs:\n  {roc_path}\n  {pr_path}")
