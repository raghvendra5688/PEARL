"""
FT-Embed(+PC) Modelling — herg/dili/caco2/half_life, HF and Uni-Mol embeddings

Phase 7 of the editor-response revision (see manuscript/editor_response_suggestions.md):
trains XGBoost/LightGBM/CatBoost on finetuned-CLM embeddings (FT-only) and on
embeddings+PC-features combined (FT+PC), for both the HF path
(finetuned_model_embeddings_new_datasets.py) and the Uni-Mol path
(unimol_embeddings_new_datasets.py). One script covers all 4 combinations of
{modality} x {mode} because both embedding CSV families share the same shape
(SMILES + N columns ending in "_embeddings" + label column(s)) -- the training
loop is agnostic to which modality produced the embedding.

Reuses pc_only_modelling.py's optimize_model/run_optimization/build_model
(pure numeric functions, no PC-specific logic) and bootstrap_ci.py, so this
new phase is on the exact same statistical footing as every other baseline.

Usage:
    python ft_modelling_new_datasets.py --modality hf --mode ft_only --dataset herg
    python ft_modelling_new_datasets.py --modality unimol --mode ft_pc --dataset all
    python ft_modelling_new_datasets.py --modality hf --mode ft_pc --dataset caco2 --trials 20
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "smiles" / "ml"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from pc_only_modelling import (  # noqa: E402
    apply_target_transform, build_model, invert_target_transform,
    run_optimization,
)
from bootstrap_ci import bootstrap_ci  # noqa: E402

EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
RESULTS_ROOT = REPO_ROOT / "results" / "ft_embeddings"
LOG_DIR = REPO_ROOT / "logs"

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
N_JOBS = int(os.getenv("N_JOBS", str(min(os.cpu_count() or 1, 60))))
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "20"))

DATASETS = {
    "herg": {"file_prefix": "herg", "label_col": "hERG_Inhib", "task": "binary"},
    "dili": {"file_prefix": "dili", "label_col": "DILI_Label", "task": "binary"},
    "caco2": {"file_prefix": "caco2", "label_col": "Caco2_LogPapp", "task": "regression", "target_transform": None},
    "half_life": {"file_prefix": "half_life", "label_col": "Half_Life_Hours", "task": "regression", "target_transform": "log1p"},
}

EMBED_DIR_NAME = {"herg": "HERG_Embeddings", "dili": "DILI_Embeddings", "caco2": "CACO2_Embeddings", "half_life": "HALF_LIFE_Embeddings"}
RESULT_DIR_NAME = {"herg": "HERG", "dili": "DILI", "caco2": "CACO2", "half_life": "HALF_LIFE"}

# modality -> (ft_only embed root, ft_pc embed root, ft_only file suffix, ft_pc file suffix, results-name prefix)
MODALITY_CFG = {
    "hf": {
        "ft_only_root": EXTRAS_ROOT / "finetuned_embeddings",
        "ft_pc_root": EXTRAS_ROOT / "finetuned_pc_embeddings",
        "results_tag": "",  # e.g. results/ft_embeddings/HERG_FT_Results
    },
    "unimol": {
        "ft_only_root": EXTRAS_ROOT / "unimol_embeddings",
        "ft_pc_root": EXTRAS_ROOT / "unimol_pc_embeddings",
        "results_tag": "UniMol_",  # e.g. results/ft_embeddings/UniMol_HERG_FT_Results
    },
}

MODEL_TYPES = {"XGBoost": "xgb", "LightGBM": "lgb", "CatBoost": "cb"}


def setup_logging(log_dir: Path, name: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / f"{name}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    logger = logging.getLogger()
    logger.handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    logger.addHandler(logging.StreamHandler())


def safe_parse_embedding(s: str) -> Optional[np.ndarray]:
    try:
        arr = np.array(json.loads(s), dtype=np.float32)
        if arr.ndim != 1 or len(arr) == 0:
            return None
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr
    except Exception:
        return None


def embedding_columns(df: pd.DataFrame, label_col: str) -> List[str]:
    return [c for c in df.columns if c.endswith("_embeddings")]


def pc_columns(df: pd.DataFrame, label_col: str, all_embed_cols: List[str]) -> List[str]:
    return [c for c in df.columns if c not in all_embed_cols + [label_col, "Standardized SMILES"]]


def build_feature_matrix(df: pd.DataFrame, embed_col: str, pc_cols: List[str]) -> np.ndarray:
    parsed = df[embed_col].apply(safe_parse_embedding)
    valid = parsed.notna()
    emb_mat = np.vstack(parsed[valid].values)
    if pc_cols:
        pc_mat = df.loc[valid, pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
        pc_mat = np.nan_to_num(pc_mat, nan=0.0, posinf=1e6, neginf=-1e6)
        X = np.concatenate([emb_mat, pc_mat], axis=1)
    else:
        X = emb_mat
    return X, valid


def evaluate_and_ci(
    name: str, model: Any, X_train, y_train, X_test, y_test, sample_weights,
    task: str, n_classes: int, out_dir: Path, target_transform: Optional[str],
) -> Dict[str, Any]:
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_test)

    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "models" / f"{name}.pkl")

    if task == "regression":
        y_pred_orig = invert_target_transform(y_pred, target_transform)
        metrics = {
            "R2": round(float(r2_score(y_test, y_pred_orig)), 4),
            "MAE": round(float(mean_absolute_error(y_test, y_pred_orig)), 4),
            "Spearman": round(float(spearmanr(y_test, y_pred_orig).correlation), 4),
        }
        ci = {
            "R2": bootstrap_ci(y_test, y_pred_orig, r2_score, stratified=False),
            "Spearman": bootstrap_ci(y_test, y_pred_orig, lambda a, b: spearmanr(a, b).correlation, stratified=False),
        }
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
            "AUC": round(float(roc_auc_score(y_test, y_proba)), 4),
            "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "Precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "F1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        }
        ci = {
            "MCC": bootstrap_ci(y_test, y_pred, matthews_corrcoef, stratified=True),
            "AUC": bootstrap_ci(y_test, y_proba, roc_auc_score, stratified=True),
        }

    with open(out_dir / "metrics" / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "metrics" / f"{name}_ci_metrics.json", "w") as f:
        json.dump(ci, f, indent=2)

    logging.info(f"  {name}: {metrics}")
    return metrics


def process_embedding_column(
    dataset_key: str, embed_col: str, pc_cols: List[str],
    train_df, val_df, test_df, dcfg: Dict, out_dir: Path,
) -> None:
    label_col = dcfg["label_col"]
    task_full = dcfg["task"]  # "binary" or "regression"
    task_for_common = "regression" if task_full == "regression" else "binary"
    target_transform = dcfg.get("target_transform")

    X_train, valid_train = build_feature_matrix(train_df, embed_col, pc_cols)
    X_val, valid_val = build_feature_matrix(val_df, embed_col, pc_cols)
    X_test, valid_test = build_feature_matrix(test_df, embed_col, pc_cols)

    if task_full == "regression":
        y_train = apply_target_transform(train_df.loc[valid_train, label_col].astype(float).values, target_transform)
        y_test = test_df.loc[valid_test, label_col].astype(float).values
        n_classes = 0
        sample_weights = np.ones(len(y_train), dtype=np.float32)
    else:
        y_train = train_df.loc[valid_train, label_col].astype(int).values
        y_test = test_df.loc[valid_test, label_col].astype(int).values
        n_classes = 2
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    logging.info(f"[{dataset_key} | {embed_col}] X_train={X_train.shape}, X_test={X_test.shape}")

    best_params = {}
    for clf_name, mt in MODEL_TYPES.items():
        best_params[clf_name] = run_optimization(mt, X_train, y_train, sample_weights, task_for_common, n_classes)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics" / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    for clf_name, mt in MODEL_TYPES.items():
        model = build_model(clf_name, task_for_common, n_classes, best_params[clf_name])
        evaluate_and_ci(
            clf_name, model, X_train, y_train, X_test, y_test, sample_weights,
            task_for_common, n_classes, out_dir, target_transform,
        )


def run(modality: str, mode: str, dataset_key: str) -> None:
    dcfg = DATASETS[dataset_key]
    mcfg = MODALITY_CFG[modality]
    embed_root = mcfg["ft_only_root"] if mode == "ft_only" else mcfg["ft_pc_root"]
    embed_dir = embed_root / EMBED_DIR_NAME[dataset_key]

    suffix = "embed" if mode == "ft_only" else "features"
    file_prefix = dcfg["file_prefix"]

    def _load(split):
        p = embed_dir / f"{file_prefix}_{split}_{suffix}.csv"
        if not p.exists():
            return None
        return pd.read_csv(p)

    train_df, val_df, test_df = _load("train"), _load("eval"), _load("test")
    if train_df is None or val_df is None or test_df is None:
        logging.warning(f"[{modality}/{mode}/{dataset_key}] missing split file(s) under {embed_dir}, skipping")
        return

    all_embed_cols = embedding_columns(train_df, dcfg["label_col"])
    if not all_embed_cols:
        logging.warning(f"[{modality}/{mode}/{dataset_key}] no embedding columns found, skipping")
        return
    pc_cols = pc_columns(train_df, dcfg["label_col"], all_embed_cols) if mode == "ft_pc" else []

    results_name = f"{mcfg['results_tag']}{RESULT_DIR_NAME[dataset_key]}_{'PC_FT' if mode == 'ft_pc' else 'FT'}_Results"
    out_root = RESULTS_ROOT / results_name

    for embed_col in all_embed_cols:
        tag = embed_col.replace("_embeddings", "")
        out_dir = out_root / tag
        setup_logging(out_dir / "logs", f"{modality}_{mode}_{dataset_key}_{tag}")
        logging.info("=" * 60)
        logging.info(f"FT Modelling | modality={modality} mode={mode} dataset={dataset_key} embed={tag}")
        logging.info("=" * 60)
        process_embedding_column(dataset_key, embed_col, pc_cols, train_df, val_df, test_df, dcfg, out_dir)
        logging.info(f"Done -> {out_dir}")


def main() -> None:
    global OPTUNA_TRIALS
    parser = argparse.ArgumentParser(description="FT-Embed(+PC) modelling for herg/dili/caco2/half_life")
    parser.add_argument("--modality", choices=list(MODALITY_CFG.keys()), required=True)
    parser.add_argument("--mode", choices=["ft_only", "ft_pc"], required=True)
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    if args.trials is not None:
        OPTUNA_TRIALS = args.trials
        import pc_only_modelling as pcm
        pcm.OPTUNA_TRIALS = args.trials

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run(args.modality, args.mode, ds)


if __name__ == "__main__":
    main()
