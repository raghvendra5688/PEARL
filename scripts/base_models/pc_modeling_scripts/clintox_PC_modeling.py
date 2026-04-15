"""
ClinTox PC-only Modeling Pipeline

This script performs supervised binary classification for the ClinTox dataset
using physiochemical molecular features only (no embeddings).

Two targets are modeled independently:
1. FDA_APPROVED
2. CT_TOX
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
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    make_scorer
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURE_ROOT = str(REPO_ROOT / "data" / "features" / "clintox_datasets")
OUTPUT_ROOT = str(REPO_ROOT / "results" / "PC" / "clintox_dataset")
LOG_DIR = str(REPO_ROOT / "logs")

MODEL_COLORS = {
    "XGBoost": "tab:blue",
    "LightGBM": "tab:green",
    "CatBoost": "tab:red"
}

LABELS = ["FDA_APPROVED"]
META_COLS = ["Standardized SMILES"]
RANDOM_SEED = 42
TOTAL_CORES = os.cpu_count()
N_JOBS = max(1, TOTAL_CORES // 2)

for label in LABELS:
    for sub in ["ROC_Curves", "PR_Curves", "models", "metrics"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, label, sub), exist_ok=True)

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "clintox_PC_features.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def load_split(split_name):
    path = os.path.join(FEATURE_ROOT, f"{split_name}_features.csv")
    logging.info(f"Loading {split_name} split from {path}")
    return pd.read_csv(path)

def split_xy(df, label_col):
    X = df.drop(columns=META_COLS + [label_col])
    y = df[label_col].astype(int)
    return X, y

def sanitize_features(X, split_name, label):
    logging.info(f"[{label}] Sanitizing {split_name} split")

    nan_before = X.isna().sum().sum()
    inf_before = np.isinf(X.values).sum()

    logging.info(
        f"[{label}] {split_name} BEFORE | "
        f"NaNs={nan_before}, Infs={inf_before}"
    )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.clip(lower=-1e6, upper=1e6)

    nan_after = X.isna().sum().sum()
    inf_after = np.isinf(X.values).sum()

    logging.info(
        f"[{label}] {split_name} AFTER | "
        f"NaNs={nan_after}, Infs={inf_after}"
    )

    return X

train_df = load_split("train")
val_df   = load_split("valid")
test_df  = load_split("test")

for df in [train_df, val_df, test_df]:
    if "CT_TOX" in df.columns:
        df.drop(columns=["CT_TOX"], inplace=True)

# Main loop per label
for LABEL_COL in LABELS:

    logging.info(f"========== Starting ClinTox task: {LABEL_COL} ==========")

    X_train, y_train = split_xy(train_df, LABEL_COL)
    X_val, y_val     = split_xy(val_df, LABEL_COL)
    X_test, y_test   = split_xy(test_df, LABEL_COL)

    X_train = sanitize_features(X_train, "train", LABEL_COL)
    X_val   = sanitize_features(X_val, "validation", LABEL_COL)
    X_test  = sanitize_features(X_test, "test", LABEL_COL)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos

    logging.info(
        f"[{LABEL_COL}] Class distribution | "
        f"0: {n_neg} ({n_neg/len(y_train)*100:.2f}%), "
        f"1: {n_pos} ({n_pos/len(y_train)*100:.2f}%), "
        f"scale_pos_weight={scale_pos_weight:.3f}"
    )

    # Optuna optimization
    def optimize_model(trial, model_type):

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
                missing=np.nan,
                n_jobs=N_JOBS,
                **params
            )

        elif model_type == "lgb":
            model = lgb.LGBMClassifier(
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=N_JOBS,
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
        sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(lambda t: optimize_model(t, model_type), n_trials=10)
        return study.best_params

    best_params = {
        "XGBoost": run_optimization("xgb"),
        "LightGBM": run_optimization("lgb"),
        "CatBoost": run_optimization("cb")
    }

    with open(os.path.join(OUTPUT_ROOT, LABEL_COL, "metrics", "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    # Train and evaluate
    predictions = {}

    def train_and_evaluate(name, model):
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

        logging.info(f"[{LABEL_COL}] {name} metrics: {metrics}")

        joblib.dump(
            model,
            os.path.join(OUTPUT_ROOT, LABEL_COL, "models", f"{name}.pkl")
        )

        np.save(
            os.path.join(OUTPUT_ROOT, LABEL_COL, "metrics", f"{name}_metrics.npy"),
            metrics
        )

        return y_proba

    predictions["XGBoost"] = train_and_evaluate(
        "XGBoost",
        xgb.XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            tree_method="hist",
            missing=np.nan,
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

    # Combined ROC
    plt.figure(figsize=(8, 6))

    for model_name, y_proba in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)

        plt.plot(
            fpr, tpr,
            label=f"{model_name} (AUC={auc_score:.3f})",
            color=MODEL_COLORS[model_name],
            linewidth=2
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ClinTox ROC Curves – {LABEL_COL}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, LABEL_COL, "ROC_Curves", "roc_all_models.pdf"))
    plt.close()

    # Combined PR
    plt.figure(figsize=(8, 6))

    for model_name, y_proba in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)

        plt.plot(
            recall, precision,
            label=f"{model_name} (AP={ap:.3f})",
            color=MODEL_COLORS[model_name],
            linewidth=2
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"ClinTox Precision–Recall Curves – {LABEL_COL}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, LABEL_COL, "PR_Curves", "pr_all_models.pdf"))
    plt.close()

    logging.info(f"========== Completed ClinTox task: {LABEL_COL} ==========")

logging.info("ClinTox PC-only modeling completed successfully.")
