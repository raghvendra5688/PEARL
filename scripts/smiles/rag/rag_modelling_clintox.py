"""
RAG-Augmented Modelling Pipeline — ClinTox

ClinTox has two binary targets: FDA_APPROVED and CT_TOX.
The full pipeline (Optuna + XGB/LGB/CatBoost) is run independently for each label.

Input:
    data/finetuned_pc_embeddings/clintox_Embeddings/clintox_{split}_features.csv
    data/rag_features/clintox/{col_name}_{split}_rag.csv

Output:
    results/rag/clintox/{label}/{col_name}/ — metrics, models, ROC/PR curves
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
    matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve, make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))

LABEL_COLS  = ["FDA_APPROVED", "CT_TOX"]
SMILES_COL  = "Standardized SMILES"
FILE_PREFIX = "clintox"
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

DATA_ROOT   = EXTRAS_ROOT / "PC_FT_All_Embeddings" / "clintox_Embeddings"
RAG_ROOT    = REPO_ROOT / "data" / "rag_features" / "clintox"
OUTPUT_ROOT = REPO_ROOT / "results" / "rag" / "clintox"
LOG_DIR     = OUTPUT_ROOT / "logs"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_DIR / "rag_clintox.log", mode="a"); fh.setFormatter(fmt)
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


def build_feature_matrix(df_pc, df_rag, emb_col, label_col) -> Tuple[np.ndarray, np.ndarray]:
    merged = df_pc.merge(
        df_rag.drop(columns=[c for c in df_rag.columns if c in LABEL_COLS], errors="ignore"),
        on=SMILES_COL, how="inner",
    )
    emb = parse_embedding_col(merged[emb_col])
    skip = [SMILES_COL, label_col] + LABEL_COLS + EMBED_COLS
    rag_cols = [c for c in df_rag.columns if c not in [SMILES_COL] + LABEL_COLS]
    pc_cols  = [c for c in merged.columns if c not in skip + rag_cols]
    pc_feats  = merged[pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    rag_feats = merged[rag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X = np.hstack([emb, pc_feats, rag_feats]).astype(np.float32)
    y = merged[label_col].astype(int).values
    logging.info(f"    Matrix: {X.shape}  (emb={emb.shape[1]}, PC={pc_feats.shape[1]}, RAG={rag_feats.shape[1]})")
    return X, y


def _objective(trial, model_type, X, y, spw):
    params = {
        "max_depth":     trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "n_estimators":  trial.suggest_int("n_estimators", 100, 600),
    }
    if model_type == "xgb":
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                               scale_pos_weight=spw, random_state=RANDOM_SEED,
                               tree_method="hist", n_jobs=N_JOBS, **params)
    elif model_type == "lgb":
        m = lgb.LGBMClassifier(class_weight="balanced", random_state=RANDOM_SEED,
                                n_jobs=N_JOBS, verbosity=-1, **params)
    else:
        m = cb.CatBoostClassifier(auto_class_weights="Balanced", loss_function="Logloss",
                                  random_seed=RANDOM_SEED, verbose=0, thread_count=N_JOBS, **params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    return float(np.mean(cross_val_score(m, X, y, scoring=make_scorer(matthews_corrcoef), cv=cv, n_jobs=1)))


def run_optimisation(model_type, X, y, spw):
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda t: _objective(t, model_type, X, y, spw), n_trials=OPTUNA_TRIALS)
    logging.info(f"    {model_type} best MCC={study.best_value:.4f}")
    return study.best_params


@contextlib.contextmanager
def _plot_ctx():
    try: yield
    finally: plt.close("all")


def train_and_evaluate(name, model, X_tr, y_tr, X_te, y_te, out_dirs, tag):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    metrics = {
        "Accuracy":  round(float(accuracy_score(y_te, y_pred)), 3),
        "AUC":       round(float(roc_auc_score(y_te, y_prob)), 3),
        "AUPR":      round(float(average_precision_score(y_te, y_prob)), 3),
        "Precision": round(float(precision_score(y_te, y_pred, average="macro", zero_division=0)), 3),
        "Recall":    round(float(recall_score(y_te, y_pred, average="macro", zero_division=0)), 3),
        "F1_macro":  round(float(f1_score(y_te, y_pred, average="macro")), 3),
        "F1_micro":  round(float(f1_score(y_te, y_pred, average="micro")), 3),
        "MCC":       round(float(matthews_corrcoef(y_te, y_pred)), 3),
    }
    logging.info(f"    [ClinTox|{tag}] {name}: {metrics}")
    joblib.dump(model, out_dirs["models"] / f"{name}.pkl")
    with open(out_dirs["metrics"] / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return y_prob


def plot_curves(preds, y_te, out_dirs, tag):
    with _plot_ctx():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, prob in preds.items():
            fpr, tpr, _ = roc_curve(y_te, prob)
            axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, prob):.3f})",
                         color=MODEL_COLORS[name], linewidth=2)
            prec, rec, _ = precision_recall_curve(y_te, prob)
            axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y_te, prob):.3f})",
                         color=MODEL_COLORS[name], linewidth=2)
        axes[0].plot([0,1],[0,1],"k--"); axes[0].set_title(f"ClinTox ROC — {tag}")
        axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()
        axes[1].set_title(f"ClinTox PR — {tag}")
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()
        plt.tight_layout()
        plt.savefig(out_dirs["plots"] / "roc_pr_curves.pdf", dpi=300, bbox_inches="tight")


def run_for_label(label_col, train_df, val_df, test_df):
    logging.info(f"\n{'#'*60}")
    logging.info(f"ClinTox label: {label_col}")

    for emb_col in EMBED_COLS:
        col_name = EMBED_TO_RAG[emb_col]
        logging.info(f"\n  {'='*20} {col_name} {'='*20}")
        rag_tr = load_rag(col_name, "train")
        rag_va = load_rag(col_name, "eval")
        rag_te = load_rag(col_name, "test")
        if rag_tr is None or rag_te is None:
            logging.warning(f"  Skipping {col_name}."); continue

        try:
            X_tr, y_tr = build_feature_matrix(train_df, rag_tr, emb_col, label_col)
            X_va, y_va = build_feature_matrix(val_df,   rag_va, emb_col, label_col)
            X_te, y_te = build_feature_matrix(test_df,  rag_te, emb_col, label_col)
        except Exception as e:
            logging.error(f"  Feature matrix failed: {e}"); continue

        spw = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)
        out_root = OUTPUT_ROOT / label_col / col_name
        out_dirs = {k: out_root / k for k in ["models", "metrics", "plots"]}
        for d in out_dirs.values(): d.mkdir(parents=True, exist_ok=True)

        best = {
            "XGBoost":  run_optimisation("xgb", X_tr, y_tr, spw),
            "LightGBM": run_optimisation("lgb", X_tr, y_tr, spw),
            "CatBoost": run_optimisation("cb",  X_tr, y_tr, spw),
        }
        with open(out_dirs["metrics"] / "best_params.json", "w") as f:
            json.dump(best, f, indent=2)

        tag = f"{label_col}_{col_name}"
        preds = {}
        preds["XGBoost"] = train_and_evaluate(
            "XGBoost", xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=spw, random_state=RANDOM_SEED, tree_method="hist",
                n_jobs=N_JOBS, **best["XGBoost"]),
            X_tr, y_tr, X_te, y_te, out_dirs, tag)
        preds["LightGBM"] = train_and_evaluate(
            "LightGBM", lgb.LGBMClassifier(class_weight="balanced", random_state=RANDOM_SEED,
                n_jobs=N_JOBS, verbosity=-1, **best["LightGBM"]),
            X_tr, y_tr, X_te, y_te, out_dirs, tag)
        preds["CatBoost"] = train_and_evaluate(
            "CatBoost", cb.CatBoostClassifier(auto_class_weights="Balanced", loss_function="Logloss",
                random_seed=RANDOM_SEED, verbose=0, thread_count=N_JOBS, **best["CatBoost"]),
            X_tr, y_tr, X_te, y_te, out_dirs, tag)
        plot_curves(preds, y_te, out_dirs, tag)


def main():
    setup_logging()
    logging.info("=" * 70)
    logging.info("RAG-Augmented ClinTox Modelling (emb + PC + ZINC RAG features)")
    logging.info(f"Labels: {LABEL_COLS}  |  N_JOBS={N_JOBS}  OPTUNA_TRIALS={OPTUNA_TRIALS}")
    logging.info("=" * 70)
    train_df = load_split("train")
    val_df   = load_split("eval")
    test_df  = load_split("test")
    for label_col in LABEL_COLS:
        try:
            run_for_label(label_col, train_df, val_df, test_df)
        except Exception as e:
            logging.error(f"Failed for label {label_col}: {e}", exc_info=True)
    logging.info("\nClinTox RAG modelling complete.")


if __name__ == "__main__":
    main()
