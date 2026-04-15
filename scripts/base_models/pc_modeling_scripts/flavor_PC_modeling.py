"""
Flavor PC-only Modeling Pipeline (Multiclass)

This script performs supervised multiclass classification for the Flavor dataset
using physiochemical molecular features only (no embeddings).

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

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    make_scorer
)
from sklearn.model_selection import StratifiedKFold


# Paths

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURE_ROOT = str(REPO_ROOT / "data" / "features" / "flavor_datasets")
OUTPUT_ROOT  = str(REPO_ROOT / "results" / "PC" / "flavor_dataset")
LOG_DIR      = str(REPO_ROOT / "logs")

ROC_DIR    = os.path.join(OUTPUT_ROOT, "ROC_Curves")
PR_DIR     = os.path.join(OUTPUT_ROOT, "PR_Curves")
MODEL_DIR  = os.path.join(OUTPUT_ROOT, "models")
METRIC_DIR = os.path.join(OUTPUT_ROOT, "metrics")

for d in [ROC_DIR, PR_DIR, MODEL_DIR, METRIC_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


logging.basicConfig(
    filename=os.path.join(LOG_DIR, "flavor_PC_features.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

RANDOM_SEED = 42

TOTAL_CORES = os.cpu_count()
N_JOBS = max(1, TOTAL_CORES // 2)

LABEL_COL = "Canonicalized Taste"
META_COLS = ["Standardized SMILES"]

MODEL_COLORS = {
    "XGBoost": "tab:blue",
    "LightGBM": "tab:green",
    "CatBoost": "tab:red"
}

# Data Loading
def load_split(split_name):
    path = os.path.join(FEATURE_ROOT, f"{split_name}_features.csv")
    logging.info(f"Loading {split_name} split from {path}")
    return pd.read_csv(path)

train_df = load_split("train")
val_df   = load_split("valid")
test_df  = load_split("test")

# Feature / Label Split

label_encoder = LabelEncoder()

def split_xy(df, fit_encoder=False):
    X = df.drop(columns=META_COLS + [LABEL_COL])
    y_raw = df[LABEL_COL]

    if fit_encoder:
        y = label_encoder.fit_transform(y_raw)
    else:
        y = label_encoder.transform(y_raw)

    return X, y

X_train, y_train = split_xy(train_df, fit_encoder=True)
X_val,   y_val   = split_xy(val_df)
X_test,  y_test  = split_xy(test_df)

# Feature Sanitization
def sanitize_features(X, split_name):
    """
    Ensures numerical stability for tree-based models.
    """
    nan_before = X.isna().sum().sum()
    inf_before = np.isinf(X.values).sum()

    logging.info(
        f"{split_name} BEFORE sanitization | "
        f"NaNs={nan_before}, Infs={inf_before}"
    )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.clip(lower=-1e6, upper=1e6)

    nan_after = X.isna().sum().sum()
    inf_after = np.isinf(X.values).sum()

    logging.info(
        f"{split_name} AFTER sanitization | "
        f"NaNs={nan_after}, Infs={inf_after}"
    )

    return X

X_train = sanitize_features(X_train, "train")
X_val   = sanitize_features(X_val, "validation")
X_test  = sanitize_features(X_test, "test")

# Sample Weights
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

logging.info("Sample weights computed using balanced strategy")

# Optuna Optimization
def optimize_model(trial, model_type):

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
    }

    if model_type == "xgb":
        base_model = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            tree_method="hist",
            n_jobs=N_JOBS,
            **params
        )

    elif model_type == "lgb":
        base_model = lgb.LGBMClassifier(
            objective="multiclass",
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            **params
        )

    else:  # CatBoost
        base_model = cb.CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=RANDOM_SEED,
            verbose=0,
            thread_count=N_JOBS,
            **params
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    mcc_scores = []

    for tr_idx, va_idx in cv.split(X_train, y_train):

        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        w_tr = sample_weights[tr_idx]

        # fresh model per fold
        if model_type == "xgb":
            model = xgb.XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                tree_method="hist",
                n_jobs=N_JOBS,
                **params
            )

        elif model_type == "lgb":
            model = lgb.LGBMClassifier(
                objective="multiclass",
                random_state=RANDOM_SEED,
                n_jobs=N_JOBS,
                **params
            )

        else:
            model = cb.CatBoostClassifier(
                loss_function="MultiClass",
                random_seed=RANDOM_SEED,
                verbose=0,
                thread_count=N_JOBS,
                **params
            )

        model.fit(X_tr, y_tr, sample_weight=w_tr)
        preds = model.predict(X_va)

        mcc = matthews_corrcoef(y_va, preds)
        mcc_scores.append(mcc)

    return float(np.mean(mcc_scores))

def run_optimization(model_type):
    logging.info(f"Running Optuna for {model_type}")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(lambda t: optimize_model(t, model_type), n_trials=10)

    logging.info(f"{model_type} best MCC: {study.best_value:.4f}")
    return study.best_params

best_params = {
    "XGBoost":  run_optimization("xgb"),
    "LightGBM": run_optimization("lgb"),
    "CatBoost": run_optimization("cb")
}

with open(os.path.join(METRIC_DIR, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=4)

# Training & Evaluation
predictions = {}

def train_and_evaluate(name, model):
    logging.info(f"Training {name}")

    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc      = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_micro = f1_score(y_test, y_pred, average="micro")
    mcc      = matthews_corrcoef(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_proba, multi_class="ovr")

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1_macro": f1_macro,
        "F1_micro": f1_micro,
        "MCC": mcc,
        "AUC_macro": auc
    }

    logging.info(f"{name} metrics: {metrics}")

    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
    np.save(os.path.join(METRIC_DIR, f"{name}_metrics.npy"), metrics)

    return y_proba

predictions["XGBoost"] = train_and_evaluate(
    "XGBoost",
    xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        tree_method="hist",
        n_jobs=N_JOBS,
        **best_params["XGBoost"]
    )
)

predictions["LightGBM"] = train_and_evaluate(
    "LightGBM",
    lgb.LGBMClassifier(
        objective="multiclass",
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        **best_params["LightGBM"]
    )
)

predictions["CatBoost"] = train_and_evaluate(
    "CatBoost",
    cb.CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=RANDOM_SEED,
        verbose=0,
        thread_count=N_JOBS,
        **best_params["CatBoost"]
    )
)

# ROC & PR Curves (Macro)
n_classes = len(label_encoder.classes_)
y_test_bin = label_binarize(y_test, classes=range(n_classes))

# ROC
plt.figure(figsize=(8, 6))

for name, y_proba in predictions.items():
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=MODEL_COLORS[name])

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Flavor ROC Curves (PC Features)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(ROC_DIR, "roc_all_models.pdf"))
plt.close()

# PR
plt.figure(figsize=(8, 6))

for name, y_proba in predictions.items():
    ap = average_precision_score(y_test_bin, y_proba, average="macro")
    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_proba.ravel())
    plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=MODEL_COLORS[name])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Flavor Precision–Recall Curves (PC Features)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(PR_DIR, "pr_all_models.pdf"))
plt.close()

logging.info("Flavor PC-only multiclass modeling completed successfully.")
