"""
RAG-Augmented Modelling Pipeline — Flavor (Multiclass)

Multiclass classification (target: Canonicalized Taste) using:
    [Finetuned CLM embedding] + [PC features] + [ZINC-250k RAG features]

Metrics: Accuracy, F1 (macro/micro), MCC, per-class AUPR.

Input:
    data/finetuned_pc_embeddings/flavor_Embeddings/fart_{split}_features.csv
    data/rag_features/flavor/{col_name}_{split}_rag.csv

Output:
    results/rag/flavor/{col_name}/ — metrics, models, confusion matrix
"""

import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    matthews_corrcoef, make_scorer, roc_auc_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))

LABEL_COL   = "Canonicalized Taste"
SMILES_COL  = "Standardized SMILES"
FILE_PREFIX = "fart"
EMBED_COLS  = [
    "ChemBERTa_77M_MTR_FL_embeddings", "ChemBERTa_77M_MLM_FL_embeddings",
    "MolFormer_Finetuned_FL_embeddings", "ChemBERTa_77M_MTR_WL_embeddings",
    "ChemBERTa_77M_MLM_WL_embeddings",  "Molformer_Finetuned_WL_embeddings",
]
EMBED_TO_RAG = {
    "ChemBERTa_77M_MTR_FL_embeddings":  "ChemBERTa_77M_MTR_FL",
    "ChemBERTa_77M_MLM_FL_embeddings":  "ChemBERTa_77M_MLM_FL",
    "MolFormer_Finetuned_FL_embeddings":"MolFormer_Finetuned_FL",
    "ChemBERTa_77M_MTR_WL_embeddings":  "ChemBERTa_77M_MTR_WL",
    "ChemBERTa_77M_MLM_WL_embeddings":  "ChemBERTa_77M_MLM_WL",
    "Molformer_Finetuned_WL_embeddings":"Molformer_Finetuned_WL",
}
MODEL_COLORS = {"XGBoost": "tab:blue", "LightGBM": "tab:green", "CatBoost": "tab:red"}
RANDOM_SEED   = int(os.getenv("RANDOM_SEED", "42"))
N_JOBS        = int(os.getenv("N_JOBS", str(min(os.cpu_count() or 1, 60))))
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "20"))

DATA_ROOT   = EXTRAS_ROOT / "PC_FT_All_Embeddings" / "flavor_Embeddings"
RAG_ROOT    = REPO_ROOT / "data" / "rag_features" / "flavor"
OUTPUT_ROOT = REPO_ROOT / "results" / "rag" / "flavor"
LOG_DIR     = OUTPUT_ROOT / "logs"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_DIR / "rag_flavor.log", mode="a"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    root = logging.getLogger(); root.setLevel(logging.INFO)
    root.addHandler(fh); root.addHandler(ch)


def safe_parse_embedding(s: str) -> Optional[np.ndarray]:
    try:
        s = s.strip()
        if not any(c.isdigit() for c in s): return None
        try:
            arr = np.array(json.loads(s), dtype=np.float32)
        except json.JSONDecodeError:
            clean = s[1:-1] if s.startswith("[") else s
            arr = np.array([float(x) for x in clean.split(",") if x.strip()], dtype=np.float32)
        if arr.ndim != 1 or len(arr) == 0: return None
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    except Exception:
        return None


def parse_embedding_col(series: pd.Series) -> np.ndarray:
    valid = [safe_parse_embedding(str(v)) for v in series]
    valid = [e for e in valid if e is not None]
    if not valid: raise ValueError("No valid embeddings.")
    return np.vstack(valid).astype(np.float32)


def load_split(split: str) -> pd.DataFrame:
    path = DATA_ROOT / f"{FILE_PREFIX}_{split}_features.csv"
    if not path.exists(): raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(str(path))


def load_rag(col_name: str, split: str) -> Optional[pd.DataFrame]:
    path = RAG_ROOT / f"{col_name}_{split}_rag.csv"
    if not path.exists():
        logging.warning(f"RAG features missing: {path}"); return None
    return pd.read_csv(str(path))


def build_feature_matrix(
    df_pc, df_rag, emb_col, le: LabelEncoder
) -> Tuple[np.ndarray, np.ndarray]:
    merged = df_pc.merge(
        df_rag.drop(columns=[c for c in df_rag.columns if c == LABEL_COL], errors="ignore"),
        on=SMILES_COL, how="inner",
    )
    emb = parse_embedding_col(merged[emb_col])
    skip = [SMILES_COL, LABEL_COL] + EMBED_COLS
    rag_cols = [c for c in df_rag.columns if c not in [SMILES_COL, LABEL_COL]]
    pc_cols  = [c for c in merged.columns if c not in skip + rag_cols]
    pc_feats  = merged[pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    rag_feats = merged[rag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X = np.hstack([emb, pc_feats, rag_feats]).astype(np.float32)
    y = le.transform(merged[LABEL_COL].astype(str).values)
    logging.info(f"  Matrix: {X.shape}  (emb={emb.shape[1]}, PC={pc_feats.shape[1]}, RAG={rag_feats.shape[1]})")
    return X, y


def _objective(trial, model_type, X, y, n_classes):
    params = {
        "max_depth":     trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "n_estimators":  trial.suggest_int("n_estimators", 100, 600),
    }
    if model_type == "xgb":
        m = xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes,
                               eval_metric="mlogloss", random_state=RANDOM_SEED,
                               tree_method="hist", n_jobs=N_JOBS, **params)
    elif model_type == "lgb":
        m = lgb.LGBMClassifier(objective="multiclass", num_class=n_classes,
                                class_weight="balanced", random_state=RANDOM_SEED,
                                n_jobs=N_JOBS, verbosity=-1, **params)
    else:
        m = cb.CatBoostClassifier(loss_function="MultiClass", auto_class_weights="Balanced",
                                  random_seed=RANDOM_SEED, verbose=0,
                                  thread_count=N_JOBS, **params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    return float(np.mean(cross_val_score(m, X, y, scoring=make_scorer(matthews_corrcoef), cv=cv, n_jobs=1)))


def run_optimisation(model_type, X, y, n_classes):
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda t: _objective(t, model_type, X, y, n_classes), n_trials=OPTUNA_TRIALS)
    logging.info(f"  {model_type} best MCC={study.best_value:.4f}")
    return study.best_params


def train_and_evaluate(name, model, X_tr, y_tr, X_te, y_te, out_dirs, tag, n_classes):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)   # (N, n_classes)
    metrics = {
        "Accuracy":  round(float(accuracy_score(y_te, y_pred)), 3),
        "Precision": round(float(precision_score(y_te, y_pred, average="macro", zero_division=0)), 3),
        "Recall":    round(float(recall_score(y_te, y_pred, average="macro", zero_division=0)), 3),
        "F1_macro": round(float(f1_score(y_te, y_pred, average="macro", zero_division=0)), 3),
        "F1_micro": round(float(f1_score(y_te, y_pred, average="micro", zero_division=0)), 3),
        "MCC":      round(float(matthews_corrcoef(y_te, y_pred)), 3),
    }
    # Macro one-vs-rest AUC (matches pc_only_modelling.py's multiclass convention)
    try:
        metrics["AUC"] = round(float(roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")), 3)
    except ValueError:
        pass
    # Per-class AUPR (one-vs-rest)
    for cls_i in range(n_classes):
        y_bin = (y_te == cls_i).astype(int)
        if y_bin.sum() > 0:
            metrics[f"AUPR_class{cls_i}"] = round(float(average_precision_score(y_bin, y_prob[:, cls_i])), 3)

    logging.info(f"  [Flavor|{tag}] {name}: acc={metrics['Accuracy']} MCC={metrics['MCC']} F1_macro={metrics['F1_macro']}")
    joblib.dump(model, out_dirs["models"] / f"{name}.pkl")
    with open(out_dirs["metrics"] / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return y_prob


def run_for_embedding(emb_col, train_df, val_df, test_df, le: LabelEncoder):
    col_name = EMBED_TO_RAG[emb_col]
    n_classes = len(le.classes_)
    logging.info(f"\n{'='*20} {col_name} {'='*20}  (n_classes={n_classes})")

    rag_tr = load_rag(col_name, "train")
    rag_va = load_rag(col_name, "eval")
    rag_te = load_rag(col_name, "test")
    if rag_tr is None or rag_te is None:
        logging.warning(f"  Skipping {col_name}."); return

    try:
        X_tr, y_tr = build_feature_matrix(train_df, rag_tr, emb_col, le)
        X_va, y_va = build_feature_matrix(val_df,   rag_va, emb_col, le)
        X_te, y_te = build_feature_matrix(test_df,  rag_te, emb_col, le)
    except Exception as e:
        logging.error(f"  Feature matrix failed: {e}"); return

    out_root = OUTPUT_ROOT / col_name
    out_dirs = {k: out_root / k for k in ["models", "metrics", "plots"]}
    for d in out_dirs.values(): d.mkdir(parents=True, exist_ok=True)

    best = {
        "XGBoost":  run_optimisation("xgb", X_tr, y_tr, n_classes),
        "LightGBM": run_optimisation("lgb", X_tr, y_tr, n_classes),
        "CatBoost": run_optimisation("cb",  X_tr, y_tr, n_classes),
    }
    with open(out_dirs["metrics"] / "best_params.json", "w") as f:
        json.dump(best, f, indent=2)

    tag = col_name
    train_and_evaluate(
        "XGBoost", xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", random_state=RANDOM_SEED, tree_method="hist",
            n_jobs=N_JOBS, **best["XGBoost"]),
        X_tr, y_tr, X_te, y_te, out_dirs, tag, n_classes)
    train_and_evaluate(
        "LightGBM", lgb.LGBMClassifier(objective="multiclass", num_class=n_classes,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=N_JOBS,
            verbosity=-1, **best["LightGBM"]),
        X_tr, y_tr, X_te, y_te, out_dirs, tag, n_classes)
    train_and_evaluate(
        "CatBoost", cb.CatBoostClassifier(loss_function="MultiClass",
            auto_class_weights="Balanced", random_seed=RANDOM_SEED, verbose=0,
            thread_count=N_JOBS, **best["CatBoost"]),
        X_tr, y_tr, X_te, y_te, out_dirs, tag, n_classes)


def main():
    setup_logging()
    logging.info("=" * 70)
    logging.info("RAG-Augmented Flavor Modelling (multiclass, emb + PC + ZINC RAG features)")
    logging.info(f"N_JOBS={N_JOBS}  OPTUNA_TRIALS={OPTUNA_TRIALS}")
    logging.info("=" * 70)

    train_df = load_split("train")
    val_df   = load_split("eval")
    test_df  = load_split("test")

    # Fit label encoder on training labels only
    le = LabelEncoder()
    le.fit(train_df[LABEL_COL].astype(str).values)
    logging.info(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    for emb_col in EMBED_COLS:
        try:
            run_for_embedding(emb_col, train_df, val_df, test_df, le)
        except Exception as e:
            logging.error(f"Failed for {emb_col}: {e}", exc_info=True)

    logging.info("\nFlavor RAG modelling complete.")


if __name__ == "__main__":
    main()
