"""
BACE PC-only Modeling Pipeline

Binary classification for:
- Class (BACE)

This script performs supervised binary classification for the BACE dataset
using Physicochemical features only (no embeddings).
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    make_scorer,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURE_ROOT = str(REPO_ROOT / "data" / "features" / "bace_datasets")
OUTPUT_ROOT  = str(REPO_ROOT / "results" / "PC" / "bace_dataset")
LOG_DIR      = str(REPO_ROOT / "logs")

ROC_DIR    = os.path.join(OUTPUT_ROOT, "ROC_Curves")
PR_DIR     = os.path.join(OUTPUT_ROOT, "PR_Curves")
MODEL_DIR  = os.path.join(OUTPUT_ROOT, "models")
METRIC_DIR = os.path.join(OUTPUT_ROOT, "metrics")

for d in [ROC_DIR, PR_DIR, MODEL_DIR, METRIC_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# logging utils
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "bace_PC_features.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

RANDOM_SEED = 42
TOTAL_CORES = os.cpu_count()
N_JOBS = max(1, TOTAL_CORES // 2)

LABEL_COL = "Class"
META_COLS = ["Standardized SMILES"]

MODEL_COLORS = {
    "XGBoost": "tab:blue",
    "LightGBM": "tab:green",
    "CatBoost": "tab:red"
}

# load data

def load_split(split_name):
    path = os.path.join(FEATURE_ROOT, f"{split_name}_features.csv")
    logging.info(f"Loading {split_name} split from {path}")
    return pd.read_csv(path)

train_df = load_split("train")
val_df   = load_split("valid")
test_df  = load_split("test")

def split_xy(df):
    X = df.drop(columns=META_COLS + [LABEL_COL])
    y = df[LABEL_COL].astype(int)
    return X, y

def sanitize_features(X, split_name):

    logging.info(f"Sanitizing features for {split_name} split")

    nan_before = X.isna().sum().sum()
    inf_before = np.isinf(X.values).sum()

    logging.info(
        f"{split_name} BEFORE | NaNs={nan_before}, Infs={inf_before}"
    )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.clip(lower=-1e6, upper=1e6)

    nan_after = X.isna().sum().sum()
    inf_after = np.isinf(X.values).sum()

    logging.info(
        f"{split_name} AFTER | NaNs={nan_after}, Infs={inf_after}"
    )

    return X

X_train, y_train = split_xy(train_df)
X_val, y_val     = split_xy(val_df)
X_test, y_test   = split_xy(test_df)

X_train = sanitize_features(X_train, "train")
X_val   = sanitize_features(X_val, "validation")
X_test  = sanitize_features(X_test, "test")

logging.info(f"Train shape: {X_train.shape}")
logging.info(f"Validation shape: {X_val.shape}")
logging.info(f"Test shape: {X_test.shape}")

# handling imabalanced data
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
scale_pos_weight = n_neg / n_pos

logging.info(
    f"BACE class dist | 0={n_neg}, 1={n_pos}, scale_pos_weight={scale_pos_weight:.3f}"
)

# Optuna optimization
def optimize_model(trial, model_type, X_train, y_train, scale_pos_weight):

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
    }

    if model_type == "xgb":
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            tree_method="hist",
            n_jobs=N_JOBS,
            **params
        )

    elif model_type == "lgb":
        model = lgb.LGBMClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            verbosity=-1,
            **params
        )

    else:
        model = cb.CatBoostClassifier(
            auto_class_weights="Balanced",
            loss_function="Logloss",
            random_seed=RANDOM_SEED,
            verbose=0,
            thread_count=N_JOBS,
            **params
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    mcc_scorer = make_scorer(matthews_corrcoef)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring=mcc_scorer,
        cv=cv,
        n_jobs=1
    )

    return float(np.mean(scores))


def run_optimization(model_type):
    logging.info(f"Running Optuna for {model_type}")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda t: optimize_model(t, model_type, X_train, y_train, scale_pos_weight),
        n_trials=10
    )

    return study.best_params


best_params = {
    "XGBoost": run_optimization("xgb"),
    "LightGBM": run_optimization("lgb"),
    "CatBoost": run_optimization("cb")
}

with open(os.path.join(METRIC_DIR, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=4)

# training and evaluation
def train_and_evaluate(name, model):

    logging.info(f"Training {name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1_macro": f1_score(y_test, y_pred, average="macro"),
        "F1_micro": f1_score(y_test, y_pred, average="micro"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
    np.save(os.path.join(METRIC_DIR, f"{name}_metrics.npy"), metrics)

    logging.info(f"{name} metrics: {metrics}")

    return y_proba


predictions = {}

predictions["XGBoost"] = train_and_evaluate(
    "XGBoost",
    xgb.XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        tree_method="hist",
        n_jobs=N_JOBS,
        **best_params["XGBoost"]
    )
)

predictions["LightGBM"] = train_and_evaluate(
    "LightGBM",
    lgb.LGBMClassifier(
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        **best_params["LightGBM"]
    )
)

predictions["CatBoost"] = train_and_evaluate(
    "CatBoost",
    cb.CatBoostClassifier(
        auto_class_weights="Balanced",
        loss_function="Logloss",
        random_seed=RANDOM_SEED,
        verbose=0,
        thread_count=N_JOBS,
        **best_params["CatBoost"]
    )
)

# PR and ROC curves
plt.figure(figsize=(8, 6))
for model_name, y_proba in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", color=MODEL_COLORS[model_name])

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("BACE ROC Curves (PC Features)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROC_DIR, "roc_all_models.pdf"))
plt.close()


plt.figure(figsize=(8, 6))
for model_name, y_proba in predictions.items():
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.plot(rec, prec, label=f"{model_name} (AP={ap:.3f})", color=MODEL_COLORS[model_name])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("BACE Precision–Recall Curves (PC Features)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PR_DIR, "pr_all_models.pdf"))
plt.close()

logging.info("BACE PC-only modeling completed successfully.")
