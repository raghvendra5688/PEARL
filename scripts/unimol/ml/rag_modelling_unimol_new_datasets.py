"""
RAG-Augmented Modelling (Uni-Mol) — herg/dili/caco2/half_life

Uni-Mol counterpart of rag_modelling_new_datasets.py. Trains XGBoost/LightGBM/
CatBoost on [Uni-Mol embedding + PC features + ZINC-250k RAFE features] for
both classification (herg/dili) and regression (caco2/half_life).

Usage:
    python rag_modelling_unimol_new_datasets.py --dataset herg
    python rag_modelling_unimol_new_datasets.py --dataset all --trials 20

Input:
    $PEARL_EXTRAS_V2/unimol_pc_embeddings/{Dataset}_Embeddings/{prefix}_{split}_features.csv
    data/rag_features_unimol/{dataset}/{col_name}_{split}_rag.csv

Output:
    results/rag_unimol/{DATASET}/{col_name}/ — metrics, models
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
PC_ROOT = EXTRAS_ROOT / "unimol_pc_embeddings"
RAG_ROOT = REPO_ROOT / "data" / "rag_features_unimol"
RESULTS_ROOT = REPO_ROOT / "results" / "rag_unimol"
LOG_DIR = REPO_ROOT / "logs"

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
N_JOBS = int(os.getenv("N_JOBS", str(min(os.cpu_count() or 1, 60))))
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "20"))

SMILES_COL = "Standardized SMILES"

DATASETS = {
    "herg": {"file_prefix": "herg", "label_col": "hERG_Inhib", "task": "binary"},
    "dili": {"file_prefix": "dili", "label_col": "DILI_Label", "task": "binary"},
    "caco2": {"file_prefix": "caco2", "label_col": "Caco2_LogPapp", "task": "regression", "target_transform": None},
    "half_life": {"file_prefix": "half_life", "label_col": "Half_Life_Hours", "task": "regression", "target_transform": "log1p"},
}

EMBED_DIR_NAME = {"herg": "HERG_Embeddings", "dili": "DILI_Embeddings", "caco2": "CACO2_Embeddings", "half_life": "HALF_LIFE_Embeddings"}
RESULT_DIR_NAME = {"herg": "HERG", "dili": "DILI", "caco2": "CACO2", "half_life": "HALF_LIFE"}

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


def embedding_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("_embeddings")]


def build_feature_matrix(
    df_pc: pd.DataFrame, df_rag: pd.DataFrame, embed_col: str, label_col: str, all_embed_cols: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    rag_drop = [c for c in df_rag.columns if c == label_col]
    merged = df_pc.merge(df_rag.drop(columns=rag_drop, errors="ignore"), on=SMILES_COL, how="inner")

    parsed = merged[embed_col].apply(safe_parse_embedding)
    valid = parsed.notna()
    merged = merged.loc[valid].reset_index(drop=True)
    parsed = parsed.loc[valid].reset_index(drop=True)
    emb = np.vstack(parsed.values)

    rag_cols = [c for c in df_rag.columns if c not in [SMILES_COL, label_col]]
    skip = [SMILES_COL, label_col] + all_embed_cols + rag_cols
    pc_cols = [c for c in merged.columns if c not in skip]

    pc_feats = merged[pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(-1e6, 1e6).values.astype(np.float32)
    rag_feats = merged[rag_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)

    X = np.hstack([emb, pc_feats, rag_feats]).astype(np.float32)
    logging.info(f"  Feature matrix: {X.shape}  (emb={emb.shape[1]}, PC={pc_feats.shape[1]}, RAG={rag_feats.shape[1]})")
    return X, merged


def evaluate_and_ci(
    name: str, model: Any, X_train, y_train, X_test, y_test, sample_weights,
    task: str, out_dir: Path, target_transform: Optional[str],
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
    dataset_key: str, embed_col: str, all_embed_cols: List[str],
    train_pc, val_pc, test_pc, train_rag, val_rag, test_rag, dcfg: Dict, out_dir: Path,
) -> None:
    label_col = dcfg["label_col"]
    task_full = dcfg["task"]
    task_for_common = "regression" if task_full == "regression" else "binary"
    target_transform = dcfg.get("target_transform")

    X_train, merged_train = build_feature_matrix(train_pc, train_rag, embed_col, label_col, all_embed_cols)
    X_val, merged_val = build_feature_matrix(val_pc, val_rag, embed_col, label_col, all_embed_cols)
    X_test, merged_test = build_feature_matrix(test_pc, test_rag, embed_col, label_col, all_embed_cols)

    if task_full == "regression":
        y_train = apply_target_transform(merged_train[label_col].astype(float).values, target_transform)
        y_test = merged_test[label_col].astype(float).values
        n_classes = 0
        sample_weights = np.ones(len(y_train), dtype=np.float32)
    else:
        y_train = merged_train[label_col].astype(int).values
        y_test = merged_test[label_col].astype(int).values
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
        evaluate_and_ci(clf_name, model, X_train, y_train, X_test, y_test, sample_weights,
                        task_for_common, out_dir, target_transform)


def run(dataset_key: str) -> None:
    dcfg = DATASETS[dataset_key]
    pc_dir = PC_ROOT / EMBED_DIR_NAME[dataset_key]
    rag_dir = RAG_ROOT / dataset_key
    file_prefix = dcfg["file_prefix"]

    def _load_pc(split):
        p = pc_dir / f"{file_prefix}_{split}_features.csv"
        return pd.read_csv(p) if p.exists() else None

    train_pc, val_pc, test_pc = _load_pc("train"), _load_pc("eval"), _load_pc("test")
    if train_pc is None or val_pc is None or test_pc is None:
        logging.warning(f"[{dataset_key}] missing PC+embedding split file(s) under {pc_dir}, skipping")
        return

    all_embed_cols = embedding_columns(train_pc)
    if not all_embed_cols:
        logging.warning(f"[{dataset_key}] no embedding columns found, skipping")
        return

    out_root = RESULTS_ROOT / RESULT_DIR_NAME[dataset_key]

    for embed_col in all_embed_cols:
        col_name = embed_col.replace("_embeddings", "")

        def _load_rag(split):
            p = rag_dir / f"{col_name}_{split}_rag.csv"
            return pd.read_csv(p) if p.exists() else None

        train_rag, val_rag, test_rag = _load_rag("train"), _load_rag("eval"), _load_rag("test")
        if train_rag is None or test_rag is None:
            logging.warning(f"[{dataset_key} | {col_name}] RAG features missing under {rag_dir}, "
                            "run rag_feature_extraction_unimol_new_datasets.py first, skipping")
            continue

        out_dir = out_root / col_name
        setup_logging(out_dir / "logs", f"rag_unimol_{dataset_key}_{col_name}")
        logging.info("=" * 60)
        logging.info(f"RAG Modelling (Uni-Mol) | dataset={dataset_key} embed={col_name}")
        logging.info("=" * 60)
        process_embedding_column(dataset_key, embed_col, all_embed_cols,
                                  train_pc, val_pc, test_pc, train_rag, val_rag, test_rag, dcfg, out_dir)
        logging.info(f"Done -> {out_dir}")


def main() -> None:
    global OPTUNA_TRIALS
    parser = argparse.ArgumentParser(description="RAG-augmented Uni-Mol modelling for herg/dili/caco2/half_life")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    if args.trials is not None:
        OPTUNA_TRIALS = args.trials
        import pc_only_modelling as pcm
        pcm.OPTUNA_TRIALS = args.trials

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run(ds)


if __name__ == "__main__":
    main()
