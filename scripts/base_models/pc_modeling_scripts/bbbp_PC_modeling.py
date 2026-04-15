"""
BBBP RDKit-only Modeling Pipeline

This script performs supervised binary classification for the BBBP dataset
using rdkit molecular features only (no embeddings).

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
FEATURE_ROOT = str(REPO_ROOT / "data" / "features" / "bbbp_datasets")
OUTPUT_ROOT = str(REPO_ROOT / "results" / "PC" / "bbbp_dataset")
LOG_DIR = str(REPO_ROOT / "logs")

ROC_DIR = os.path.join(OUTPUT_ROOT, "ROC_Curves")
PR_DIR = os.path.join(OUTPUT_ROOT, "PR_Curves")
MODEL_DIR = os.path.join(OUTPUT_ROOT, "models")
METRIC_DIR = os.path.join(OUTPUT_ROOT, "metrics")


MODEL_COLORS = {
    "XGBoost": "tab:blue",
    "LightGBM": "tab:green",
    "CatBoost": "tab:red"
}

for d in [ROC_DIR, PR_DIR, MODEL_DIR, METRIC_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "bbbp_rdkit_features.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

RANDOM_SEED = 42
TOTAL_CORES = os.cpu_count()
N_JOBS = max(1, TOTAL_CORES // 2)
LABEL_COL = "p_np"

# SMILES column is intentionally removed from training to avoid information leakage
META_COLS = ["Standardized SMILES"]

def load_split(split_name):
    """
    Loads a feature CSV corresponding to a data split.
    """
    path = os.path.join(FEATURE_ROOT, f"{split_name}_features.csv")
    logging.info(f"Loading {split_name} split from {path}")
    return pd.read_csv(path)

train_df = load_split("train")
val_df   = load_split("valid")
test_df  = load_split("test")

def split_xy(df):
    """
    Separates input features (X) and labels (y).
    SMILES is removed because it is not a numeric feature.
    """
    X = df.drop(columns=META_COLS + [LABEL_COL])
    y = df[LABEL_COL].astype(int)
    return X, y


def sanitize_features(X, split_name):
    """
    Ensures that the feature matrix contains only finite numeric values.

    Tree-based models (especially XGBoost) cannot handle NaN or infinite values.
    These may arise from RDKit descriptors or graph-based features.

    Strategy:
    - Replace +inf / -inf with NaN
    - Impute NaN values using column-wise median
    """

    logging.info(f"Sanitizing features for {split_name} split")

    n_rows, n_cols = X.shape

    nan_before = X.isna().sum().sum()
    inf_before = np.isinf(X.values).sum()

    logging.info(
        f"{split_name} BEFORE sanitization | "
        f"Rows={n_rows}, Cols={n_cols}, "
        f"NaNs={nan_before}, Infs={inf_before}"
    )

    # Replace infinities with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Median imputation (safe for tree models)
    X = X.fillna(X.median())
    X = X.clip(lower=-1e6, upper=1e6)

    nan_after = X.isna().sum().sum()
    inf_after = np.isinf(X.values).sum()

    logging.info(
        f"{split_name} AFTER sanitization | "
        f"NaNs={nan_after}, Infs={inf_after}"
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

# Class imbalance handling:
# In BBBP, class 1 is the majority and class 0 is the minority.
# XGBoost expects scale_pos_weight = (#negatives / #positives).

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()

scale_pos_weight = n_neg / n_pos

logging.info(
    f"BBBP class distribution | "
    f"0 (minority): {n_neg} ({n_neg/len(y_train)*100:.2f}%), "
    f"1 (majority): {n_pos} ({n_pos/len(y_train)*100:.2f}%), "
    f"scale_pos_weight: {scale_pos_weight:.3f}"
)

# Hyperparameter optimization using Optuna

def optimize_model(
    trial: optuna.Trial,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float
) -> float:
    """
    Optimize model hyperparameters using Optuna with stratified 5-fold CV
    and MCC as objective metric.
    """

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

    else:  # CatBoost
        model = cb.CatBoostClassifier(
            auto_class_weights="Balanced",
            loss_function="Logloss",
            random_seed=RANDOM_SEED,
            verbose=0,
            thread_count=N_JOBS,
            **params
        )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    mcc_scorer = make_scorer(matthews_corrcoef)

    # IMPORTANT:
    # n_jobs must be 1 here because CatBoost/XGBoost already use threads internally.
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
    logging.info(f"Running Optuna optimization for {model_type}")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda t: optimize_model(
            t,
            model_type,
            X_train,
            y_train,
            X_val,
            y_val,
            scale_pos_weight
        ),
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

# Training and evaluation
def train_and_evaluate(name, model):
    """
    Trains a model on the training set and evaluates on the test set.
    Saves metrics, ROC curves, PR curves, and trained models.
    """
    logging.info(f"Training {name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_micro = f1_score(y_test, y_pred, average="micro")

    mcc = matthews_corrcoef(y_test, y_pred)

    metrics = {
        "Accuracy": acc,
        "AUC": auc_score,
        "Precision": precision,
        "Recall": recall,
        "F1_macro": f1_macro,
        "F1_micro": f1_micro,
        "MCC": mcc
    }

    logging.info(
        f"{name} | "
        f"Acc={acc:.4f}, AUC={auc_score:.4f}, "
        f"Precision={precision:.4f}, Recall={recall:.4f}, "
        f"F1_macro={f1_macro:.4f}, F1_micro={f1_micro:.4f}, "
        f"MCC={mcc:.4f}"
    )

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(os.path.join(ROC_DIR, f"roc_{name}.png"))
    plt.close()

    # Precision–Recall curve
    pr, rc, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.figure()
    plt.plot(rc, pr, label=f"{name} (AP={ap:.3f})")
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(PR_DIR, f"pr_{name}.png"))
    plt.close()

    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
    np.save(os.path.join(METRIC_DIR, f"{name}_metrics.npy"), metrics)

    return y_pred, y_proba


# Final model training
predictions = {}

predictions["XGBoost"] = train_and_evaluate(
    "XGBoost",
    xgb.XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        tree_method="hist",
        missing=np.nan,
        n_jobs = N_JOBS,
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

plt.figure(figsize=(8, 6))

for model_name, (_, y_proba) in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc_score:.3f})",
        color=MODEL_COLORS[model_name],
        linewidth=2
    )

plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("BBBP ROC Curves (RDKit Features)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ROC_DIR, "roc_all_models.pdf"))
plt.close()


plt.figure(figsize=(8, 6))

for model_name, (_, y_proba) in predictions.items():
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    plt.plot(
        recall,
        precision,
        label=f"{model_name} (AP = {ap:.3f})",
        color=MODEL_COLORS[model_name],
        linewidth=2
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("BBBP Precision–Recall Curves (RDKit Features)")
plt.legend(loc="best")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PR_DIR, "pr_all_models.pdf"))
plt.close()

logging.info("BBBP RDKit-only modeling completed successfully.")
# nohup python bbbp_rdkit_modeling.py > bbbp_nohup.out 2>&1 &
