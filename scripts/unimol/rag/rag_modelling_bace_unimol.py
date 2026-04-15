"""
Uni-Mol RAG-Augmented Modelling Pipeline — BACE

Binary classification (target: Class) using:
    [Uni-Mol 2560-dim embedding] + [PC features] + [ZINC-250k RAG features]

For each of 2 Uni-Mol model variants (UniMol_FL, UniMol_WL), merges Uni-Mol
embeddings with existing PC features and RAG features, then runs Optuna-tuned
XGBoost / LightGBM / CatBoost (same hyperparameter space and CV strategy as
the rest of EffiChem-2.0).

Input:
    data/unimol_embeddings/BACE_Embeddings/bace_{split}_embed.csv
    data/finetuned_pc_embeddings/BACE_Embeddings/bace_{split}_features.csv   (PC features)
    data/rag_features_unimol/bace/{col_name}_{split}_rag.csv

Output:
    results/rag_unimol/bace/{col_name}/ — metrics, models, ROC/PR curves
"""

import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))

# Columns from ChemBERTa/MolFormer embedding files to exclude when extracting PC cols
_OTHER_EMB_COLS = [
    "ChemBERTa_77M_MTR_FL_embeddings", "ChemBERTa_77M_MLM_FL_embeddings",
    "MolFormer_Finetuned_FL_embeddings", "ChemBERTa_77M_MTR_WL_embeddings",
    "ChemBERTa_77M_MLM_WL_embeddings",  "Molformer_Finetuned_WL_embeddings",
]


class Config:
    UNIMOL_EMBED_ROOT = EXTRAS_ROOT / "unimol_embeddings" / "BACE_Embeddings"
    PC_EMBED_ROOT     = EXTRAS_ROOT / "PC_FT_All_Embeddings" / "BACE_Embeddings"
    RAG_ROOT          = REPO_ROOT / "data" / "rag_features_unimol" / "bace"
    OUTPUT_ROOT       = REPO_ROOT / "results" / "rag_unimol" / "bace"
    LOG_DIR           = OUTPUT_ROOT / "logs"

    LABEL_COL     = "Class"
    SMILES_COL    = "Standardized SMILES"
    FILE_PREFIX   = "bace"
    RANDOM_SEED   = int(os.getenv("RANDOM_SEED", "42"))
    N_JOBS        = int(os.getenv("N_JOBS", str(min(os.cpu_count() or 1, 60))))
    OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "20"))

    EMBED_COLS = [
        "UniMol_FL_embeddings",
        "UniMol_WL_embeddings",
    ]
    EMBED_TO_RAG = {
        "UniMol_FL_embeddings": "UniMol_FL",
        "UniMol_WL_embeddings": "UniMol_WL",
    }
    MODEL_COLORS = {"XGBoost": "tab:blue", "LightGBM": "tab:green", "CatBoost": "tab:red"}

    def __init__(self):
        self.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(config: Config) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh  = logging.FileHandler(config.LOG_DIR / "rag_unimol_bace.log", mode="a")
    fh.setFormatter(fmt)
    ch  = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)


# ── Embedding parsing ─────────────────────────────────────────────────────────
def safe_parse_embedding(s: str) -> Optional[np.ndarray]:
    try:
        s = s.strip()
        if not any(c.isdigit() for c in s):
            return None
        try:
            arr = np.array(json.loads(s), dtype=np.float32)
        except json.JSONDecodeError:
            clean = s[1:-1] if s.startswith("[") else s
            arr = np.array(
                [float(x) for x in clean.split(",") if x.strip()], dtype=np.float32
            )
        if arr.ndim != 1 or len(arr) == 0:
            return None
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    except Exception:
        return None


def parse_embedding_col(series: pd.Series) -> np.ndarray:
    embs = [safe_parse_embedding(str(v)) for v in series]
    valid = [e for e in embs if e is not None]
    if not valid:
        raise ValueError("No valid embeddings found.")
    return np.vstack(valid).astype(np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_split(config: Config, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (unimol_df, pc_df). pc_df may be None if file missing."""
    unimol_path = config.UNIMOL_EMBED_ROOT / f"{config.FILE_PREFIX}_{split}_embed.csv"
    if not unimol_path.exists():
        raise FileNotFoundError(f"Uni-Mol embedding CSV not found: {unimol_path}")
    unimol_df = pd.read_csv(str(unimol_path))

    pc_path = config.PC_EMBED_ROOT / f"{config.FILE_PREFIX}_{split}_features.csv"
    pc_df = pd.read_csv(str(pc_path)) if pc_path.exists() else None
    if pc_df is None:
        logging.warning(f"PC features not found: {pc_path} — using embeddings only")
    return unimol_df, pc_df


def load_rag_features(config: Config, col_name: str, split: str) -> Optional[pd.DataFrame]:
    path = config.RAG_ROOT / f"{col_name}_{split}_rag.csv"
    if not path.exists():
        logging.warning(f"RAG features not found: {path}")
        return None
    return pd.read_csv(str(path))


def build_feature_matrix(
    unimol_df: pd.DataFrame,
    pc_df: Optional[pd.DataFrame],
    df_rag: pd.DataFrame,
    emb_col: str,
    config: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge Uni-Mol embed df with PC features (if available) and RAG df on SMILES.
    Returns X = [unimol_emb | PC | RAG], y = labels.
    """
    # Start with Uni-Mol embeddings
    merged = unimol_df.copy()

    # Merge PC features on SMILES
    if pc_df is not None:
        pc_cols_to_add = [
            c for c in pc_df.columns
            if c not in [config.SMILES_COL, config.LABEL_COL] + _OTHER_EMB_COLS
        ]
        merged = merged.merge(
            pc_df[[config.SMILES_COL] + pc_cols_to_add],
            on=config.SMILES_COL,
            how="inner",
        )

    # Merge RAG features
    rag_drop = [c for c in df_rag.columns if c == config.LABEL_COL]
    merged = merged.merge(
        df_rag.drop(columns=rag_drop, errors="ignore"),
        on=config.SMILES_COL,
        how="inner",
    )

    # Parse Uni-Mol embedding
    emb = parse_embedding_col(merged[emb_col])

    # PC columns: numeric cols that aren't meta/embed/label/rag
    rag_cols  = [c for c in df_rag.columns if c not in [config.SMILES_COL, config.LABEL_COL]]
    skip_cols = [config.SMILES_COL, config.LABEL_COL] + config.EMBED_COLS + _OTHER_EMB_COLS
    pc_cols   = [c for c in merged.columns if c not in skip_cols + rag_cols]

    pc_feats  = merged[pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    rag_feats = merged[rag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

    X = np.hstack([emb, pc_feats, rag_feats]).astype(np.float32)
    y = merged[config.LABEL_COL].astype(int).values
    logging.info(
        f"  Feature matrix: {X.shape}  "
        f"(emb={emb.shape[1]}, PC={pc_feats.shape[1]}, RAG={rag_feats.shape[1]})"
    )
    return X, y


# ── Optuna optimisation ───────────────────────────────────────────────────────
def _objective(trial, model_type, X, y, spw, config):
    params = {
        "max_depth":     trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "n_estimators":  trial.suggest_int("n_estimators", 100, 600),
    }
    if model_type == "xgb":
        m = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=spw, random_state=config.RANDOM_SEED,
            tree_method="hist", n_jobs=config.N_JOBS, **params,
        )
    elif model_type == "lgb":
        m = lgb.LGBMClassifier(
            class_weight="balanced", random_state=config.RANDOM_SEED,
            n_jobs=config.N_JOBS, verbosity=-1, **params,
        )
    else:
        m = cb.CatBoostClassifier(
            auto_class_weights="Balanced", loss_function="Logloss",
            random_seed=config.RANDOM_SEED, verbose=0,
            thread_count=config.N_JOBS, **params,
        )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    scores = cross_val_score(
        m, X, y, scoring=make_scorer(matthews_corrcoef), cv=cv, n_jobs=1
    )
    return float(np.mean(scores))


def run_optimisation(model_type, X, y, spw, config):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
    )
    study.optimize(
        lambda t: _objective(t, model_type, X, y, spw, config),
        n_trials=config.OPTUNA_TRIALS,
    )
    logging.info(
        f"  {model_type} best MCC={study.best_value:.4f}  params={study.best_params}"
    )
    return study.best_params


# ── Training & evaluation ─────────────────────────────────────────────────────
@contextlib.contextmanager
def _plot_ctx():
    try:
        yield
    finally:
        plt.close("all")


def train_and_evaluate(name, model, X_tr, y_tr, X_te, y_te, out_dirs, tag, config):
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
    logging.info(f"  [BACE|{tag}] {name}: {metrics}")
    joblib.dump(model, out_dirs["models"] / f"{name}.pkl")
    with open(out_dirs["metrics"] / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return y_prob


def plot_curves(preds, y_te, out_dirs, tag, config):
    with _plot_ctx():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, prob in preds.items():
            fpr, tpr, _ = roc_curve(y_te, prob)
            auc = roc_auc_score(y_te, prob)
            axes[0].plot(
                fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=config.MODEL_COLORS[name], linewidth=2,
            )
            prec, rec, _ = precision_recall_curve(y_te, prob)
            ap = average_precision_score(y_te, prob)
            axes[1].plot(
                rec, prec, label=f"{name} (AP={ap:.3f})",
                color=config.MODEL_COLORS[name], linewidth=2,
            )
        axes[0].plot([0, 1], [0, 1], "k--")
        axes[0].set_title(f"BACE ROC — {tag}")
        axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()
        axes[1].set_title(f"BACE PR — {tag}")
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()
        plt.tight_layout()
        plt.savefig(out_dirs["plots"] / "roc_pr_curves.pdf", dpi=300, bbox_inches="tight")


# ── Per-embedding pipeline ────────────────────────────────────────────────────
def run_for_embedding(emb_col, train_unimol, train_pc, val_unimol, val_pc, test_unimol, test_pc, config):
    col_name = config.EMBED_TO_RAG[emb_col]
    logging.info(f"\n{'='*20} {col_name} {'='*20}")

    rag_train = load_rag_features(config, col_name, "train")
    rag_val   = load_rag_features(config, col_name, "eval")
    rag_test  = load_rag_features(config, col_name, "test")
    if rag_train is None or rag_test is None:
        logging.warning(f"  RAG features missing for {col_name}, skipping.")
        return

    try:
        X_tr, y_tr = build_feature_matrix(train_unimol, train_pc, rag_train, emb_col, config)
        X_va, y_va = build_feature_matrix(val_unimol,   val_pc,   rag_val,   emb_col, config)
        X_te, y_te = build_feature_matrix(test_unimol,  test_pc,  rag_test,  emb_col, config)
    except Exception as e:
        logging.error(f"  Feature matrix failed for {col_name}: {e}")
        return

    spw = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)

    out_root = config.OUTPUT_ROOT / col_name
    out_dirs = {k: out_root / k for k in ["models", "metrics", "plots"]}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    best = {
        "XGBoost":  run_optimisation("xgb", X_tr, y_tr, spw, config),
        "LightGBM": run_optimisation("lgb", X_tr, y_tr, spw, config),
        "CatBoost": run_optimisation("cb",  X_tr, y_tr, spw, config),
    }
    with open(out_dirs["metrics"] / "best_params.json", "w") as f:
        json.dump(best, f, indent=2)

    preds = {}
    preds["XGBoost"] = train_and_evaluate(
        "XGBoost",
        xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=spw, random_state=config.RANDOM_SEED,
            tree_method="hist", n_jobs=config.N_JOBS, **best["XGBoost"],
        ),
        X_tr, y_tr, X_te, y_te, out_dirs, col_name, config,
    )
    preds["LightGBM"] = train_and_evaluate(
        "LightGBM",
        lgb.LGBMClassifier(
            class_weight="balanced", random_state=config.RANDOM_SEED,
            n_jobs=config.N_JOBS, verbosity=-1, **best["LightGBM"],
        ),
        X_tr, y_tr, X_te, y_te, out_dirs, col_name, config,
    )
    preds["CatBoost"] = train_and_evaluate(
        "CatBoost",
        cb.CatBoostClassifier(
            auto_class_weights="Balanced", loss_function="Logloss",
            random_seed=config.RANDOM_SEED, verbose=0,
            thread_count=config.N_JOBS, **best["CatBoost"],
        ),
        X_tr, y_tr, X_te, y_te, out_dirs, col_name, config,
    )

    plot_curves(preds, y_te, out_dirs, col_name, config)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    config = Config()
    setup_logging(config)

    logging.info("=" * 70)
    logging.info("Uni-Mol RAG-Augmented BACE Modelling (UniMol emb + PC + ZINC RAG)")
    logging.info(f"N_JOBS={config.N_JOBS}  OPTUNA_TRIALS={config.OPTUNA_TRIALS}")
    logging.info("=" * 70)

    train_unimol, train_pc = load_split(config, "train")
    val_unimol,   val_pc   = load_split(config, "eval")
    test_unimol,  test_pc  = load_split(config, "test")

    for emb_col in config.EMBED_COLS:
        try:
            run_for_embedding(
                emb_col,
                train_unimol, train_pc,
                val_unimol, val_pc,
                test_unimol, test_pc,
                config,
            )
        except Exception as e:
            logging.error(f"Failed for {emb_col}: {e}", exc_info=True)

    logging.info("\nBACE Uni-Mol RAG modelling complete.")


if __name__ == "__main__":
    main()
