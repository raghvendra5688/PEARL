"""
Uni-Mol RAG-Augmented Modelling Pipeline — Flavor (Multiclass)

Multiclass classification (target: Canonicalized Taste) using:
    [Uni-Mol 2560-dim embedding] + [PC features] + [ZINC-250k RAG features]

Metrics: Accuracy, F1 (macro/micro), MCC, per-class AUPR.

Input:
    data/unimol_embeddings/flavor_Embeddings/fart_{split}_embed.csv
    data/finetuned_pc_embeddings/flavor_Embeddings/fart_{split}_features.csv
    data/rag_features_unimol/flavor/{col_name}_{split}_rag.csv

Output:
    results/rag_unimol/flavor/{col_name}/ — metrics, models
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    matthews_corrcoef, make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))

_OTHER_EMB_COLS = [
    "ChemBERTa_77M_MTR_FL_embeddings", "ChemBERTa_77M_MLM_FL_embeddings",
    "MolFormer_Finetuned_FL_embeddings", "ChemBERTa_77M_MTR_WL_embeddings",
    "ChemBERTa_77M_MLM_WL_embeddings",  "Molformer_Finetuned_WL_embeddings",
]

LABEL_COL   = "Canonicalized Taste"
SMILES_COL  = "Standardized SMILES"
FILE_PREFIX = "fart"

EMBED_COLS = ["UniMol_FL_embeddings", "UniMol_WL_embeddings"]
EMBED_TO_RAG = {
    "UniMol_FL_embeddings": "UniMol_FL",
    "UniMol_WL_embeddings": "UniMol_WL",
}

RANDOM_SEED   = int(os.getenv("RANDOM_SEED", "42"))
N_JOBS        = int(os.getenv("N_JOBS", str(min(os.cpu_count() or 1, 60))))
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "20"))

UNIMOL_EMBED_ROOT = EXTRAS_ROOT / "unimol_embeddings" / "flavor_Embeddings"
PC_EMBED_ROOT     = EXTRAS_ROOT / "PC_FT_All_Embeddings" / "flavor_Embeddings"
RAG_ROOT          = REPO_ROOT / "data" / "rag_features_unimol" / "flavor"
OUTPUT_ROOT       = REPO_ROOT / "results" / "rag_unimol" / "flavor"
LOG_DIR           = OUTPUT_ROOT / "logs"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_DIR / "rag_unimol_flavor.log", mode="a"); fh.setFormatter(fmt)
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


def load_split(split: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    unimol_path = UNIMOL_EMBED_ROOT / f"{FILE_PREFIX}_{split}_embed.csv"
    if not unimol_path.exists():
        raise FileNotFoundError(f"Uni-Mol embedding CSV not found: {unimol_path}")
    unimol_df = pd.read_csv(str(unimol_path))
    pc_path = PC_EMBED_ROOT / f"{FILE_PREFIX}_{split}_features.csv"
    pc_df = pd.read_csv(str(pc_path)) if pc_path.exists() else None
    if pc_df is None:
        logging.warning(f"PC features not found: {pc_path} — using embeddings only")
    return unimol_df, pc_df


def load_rag(col_name: str, split: str) -> Optional[pd.DataFrame]:
    path = RAG_ROOT / f"{col_name}_{split}_rag.csv"
    if not path.exists():
        logging.warning(f"RAG features missing: {path}"); return None
    return pd.read_csv(str(path))


def build_feature_matrix(
    unimol_df: pd.DataFrame,
    pc_df: Optional[pd.DataFrame],
    df_rag: pd.DataFrame,
    emb_col: str,
    le: LabelEncoder,
) -> Tuple[np.ndarray, np.ndarray]:
    merged = unimol_df.copy()
    if pc_df is not None:
        pc_cols_to_add = [
            c for c in pc_df.columns
            if c not in [SMILES_COL, LABEL_COL] + _OTHER_EMB_COLS
        ]
        merged = merged.merge(
            pc_df[[SMILES_COL] + pc_cols_to_add], on=SMILES_COL, how="inner"
        )
    rag_drop = [c for c in df_rag.columns if c == LABEL_COL]
    merged = merged.merge(df_rag.drop(columns=rag_drop, errors="ignore"), on=SMILES_COL, how="inner")

    emb = parse_embedding_col(merged[emb_col])
    rag_cols  = [c for c in df_rag.columns if c not in [SMILES_COL, LABEL_COL]]
    skip_cols = [SMILES_COL, LABEL_COL] + EMBED_COLS + _OTHER_EMB_COLS
    pc_cols   = [c for c in merged.columns if c not in skip_cols + rag_cols]

    pc_feats  = merged[pc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e6, 1e6).values
    rag_feats = merged[rag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X = np.hstack([emb, pc_feats, rag_feats]).astype(np.float32)
    y = le.transform(merged[LABEL_COL].astype(str).values)
    logging.info(
        f"  Matrix: {X.shape}  "
        f"(emb={emb.shape[1]}, PC={pc_feats.shape[1]}, RAG={rag_feats.shape[1]})"
    )
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
    y_prob = model.predict_proba(X_te)
    metrics = {
        "Accuracy": round(float(accuracy_score(y_te, y_pred)), 3),
        "F1_macro": round(float(f1_score(y_te, y_pred, average="macro", zero_division=0)), 3),
        "F1_micro": round(float(f1_score(y_te, y_pred, average="micro", zero_division=0)), 3),
        "MCC":      round(float(matthews_corrcoef(y_te, y_pred)), 3),
    }
    for cls_i in range(n_classes):
        y_bin = (y_te == cls_i).astype(int)
        if y_bin.sum() > 0:
            metrics[f"AUPR_class{cls_i}"] = round(
                float(average_precision_score(y_bin, y_prob[:, cls_i])), 3
            )
    logging.info(
        f"  [Flavor|{tag}] {name}: acc={metrics['Accuracy']} "
        f"MCC={metrics['MCC']} F1_macro={metrics['F1_macro']}"
    )
    joblib.dump(model, out_dirs["models"] / f"{name}.pkl")
    with open(out_dirs["metrics"] / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return y_prob


def run_for_embedding(
    emb_col: str,
    train_u: pd.DataFrame, train_pc: Optional[pd.DataFrame],
    val_u:   pd.DataFrame, val_pc:   Optional[pd.DataFrame],
    test_u:  pd.DataFrame, test_pc:  Optional[pd.DataFrame],
    le: LabelEncoder,
) -> None:
    col_name  = EMBED_TO_RAG[emb_col]
    n_classes = len(le.classes_)
    logging.info(f"\n{'='*20} {col_name} {'='*20}  (n_classes={n_classes})")

    rag_tr = load_rag(col_name, "train")
    rag_va = load_rag(col_name, "eval")
    rag_te = load_rag(col_name, "test")
    if rag_tr is None or rag_te is None:
        logging.warning(f"  Skipping {col_name}."); return

    try:
        X_tr, y_tr = build_feature_matrix(train_u, train_pc, rag_tr, emb_col, le)
        X_va, y_va = build_feature_matrix(val_u,   val_pc,   rag_va, emb_col, le)
        X_te, y_te = build_feature_matrix(test_u,  test_pc,  rag_te, emb_col, le)
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
        "XGBoost",
        xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes,
                           eval_metric="mlogloss", random_state=RANDOM_SEED,
                           tree_method="hist", n_jobs=N_JOBS, **best["XGBoost"]),
        X_tr, y_tr, X_te, y_te, out_dirs, tag, n_classes,
    )
    train_and_evaluate(
        "LightGBM",
        lgb.LGBMClassifier(objective="multiclass", num_class=n_classes,
                            class_weight="balanced", random_state=RANDOM_SEED,
                            n_jobs=N_JOBS, verbosity=-1, **best["LightGBM"]),
        X_tr, y_tr, X_te, y_te, out_dirs, tag, n_classes,
    )
    train_and_evaluate(
        "CatBoost",
        cb.CatBoostClassifier(loss_function="MultiClass", auto_class_weights="Balanced",
                               random_seed=RANDOM_SEED, verbose=0,
                               thread_count=N_JOBS, **best["CatBoost"]),
        X_tr, y_tr, X_te, y_te, out_dirs, tag, n_classes,
    )


def main():
    setup_logging()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logging.info("=" * 70)
    logging.info("Uni-Mol RAG-Augmented Flavor Modelling (multiclass, UniMol emb + PC + ZINC RAG)")
    logging.info(f"N_JOBS={N_JOBS}  OPTUNA_TRIALS={OPTUNA_TRIALS}")
    logging.info("=" * 70)

    train_u, train_pc = load_split("train")
    val_u,   val_pc   = load_split("eval")
    test_u,  test_pc  = load_split("test")

    le = LabelEncoder()
    le.fit(train_u[LABEL_COL].astype(str).values)
    logging.info(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    for emb_col in EMBED_COLS:
        try:
            run_for_embedding(
                emb_col,
                train_u, train_pc,
                val_u, val_pc,
                test_u, test_pc,
                le,
            )
        except Exception as e:
            logging.error(f"Failed for {emb_col}: {e}", exc_info=True)

    logging.info("\nFlavor Uni-Mol RAG modelling complete.")


if __name__ == "__main__":
    main()
