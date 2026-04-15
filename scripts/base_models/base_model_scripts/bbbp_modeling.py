"""
BBBP Base-Model-Embedding Modeling Pipeline

This script performs supervised binary classification for the BBBP dataset
using pretrained base-model embeddings as features (no RDKit features).
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
import ast

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
BASE_MODEL_ROOT = str(REPO_ROOT / "data" / "base_models" / "bbbp_datasets")
OUTPUT_ROOT = str(REPO_ROOT / "results" / "base_models" / "bbbp_dataset")
LOG_DIR = str(REPO_ROOT / "logs")

RANDOM_SEED = 42
TOTAL_CORES = os.cpu_count()
N_JOBS = max(1, TOTAL_CORES // 2)
LABEL_COL = "p_np"
SMILES_COL = "Standardized SMILES"

# Embedding model name prefixes in CSV
EMBEDDING_MODELS = [
    "ChemBERTa_77M_MTR_Base",
    "ChemBERTa_77M_MLM_Base",
    "MolFormer_Base"
]

MODEL_COLORS = {
    "XGBoost": "tab:blue",
    "LightGBM": "tab:green",
    "CatBoost": "tab:red"
}


os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "bbbp_base_model_ml.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def extract_embeddings(df, emb_col):
        X_list = df[emb_col].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
        X = np.vstack(X_list.values)
        y = df[LABEL_COL].astype(int).values
        return X, y

def load_split(split_name):
    path = os.path.join(BASE_MODEL_ROOT, f"{split_name}_embeddings.csv")
    logging.info(f"Loading {split_name} split from {path}")
    return pd.read_csv(path)

train_df = load_split("train")
val_df   = load_split("valid")
test_df  = load_split("test")


# IMBALANCE HANDLING
y_train_full = train_df[LABEL_COL].astype(int)
n_pos = (y_train_full == 1).sum()
n_neg = (y_train_full == 0).sum()
scale_pos_weight = n_neg / n_pos

logging.info(
    f"BBBP class distribution | 0: {n_neg}, 1: {n_pos}, scale_pos_weight={scale_pos_weight:.3f}"
)

# OPTIMIZATION USING OPTUNA
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

def run_optimization(model_type, X_train, y_train, X_val, y_val):
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

# TRAINING & EVALUATION
def train_and_evaluate(name, model, X_train, y_train, X_test, y_test,
                       ROC_DIR, PR_DIR, MODEL_DIR, METRIC_DIR, embedding_tag):

    logging.info(f"Training {name} for {embedding_tag}")

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

    logging.info(f"{embedding_tag} | {name} | {metrics}")

    joblib.dump(model, os.path.join(MODEL_DIR, f"{embedding_tag}_{name}.pkl"))
    np.save(os.path.join(METRIC_DIR, f"{embedding_tag}_{name}_metrics.npy"), metrics)

    return y_proba


# MAIN EXECUTION
for emb_name in EMBEDDING_MODELS:

    logging.info(f"==================== EMBEDDING: {emb_name}===============")

    EMB_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, emb_name)
    ROC_DIR = os.path.join(EMB_OUTPUT_ROOT, "ROC_Curves")
    PR_DIR = os.path.join(EMB_OUTPUT_ROOT, "PR_Curves")
    MODEL_DIR = os.path.join(EMB_OUTPUT_ROOT, "models")
    METRIC_DIR = os.path.join(EMB_OUTPUT_ROOT, "metrics")

    for d in [ROC_DIR, PR_DIR, MODEL_DIR, METRIC_DIR]:
        os.makedirs(d, exist_ok=True)


    X_train, y_train = extract_embeddings(train_df, emb_name)
    X_val, y_val     = extract_embeddings(val_df, emb_name)
    X_test, y_test   = extract_embeddings(test_df, emb_name)

    logging.info(f"[{LABEL_COL} | {emb_name}] Shapes |"
    f"Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")


    X_train = np.nan_to_num(X_train)
    X_val   = np.nan_to_num(X_val)
    X_test  = np.nan_to_num(X_test)

    best_params = {
        "XGBoost":  run_optimization("xgb", X_train, y_train, X_val, y_val),
        "LightGBM": run_optimization("lgb", X_train, y_train, X_val, y_val),
        "CatBoost": run_optimization("cb",  X_train, y_train, X_val, y_val)
    }

    with open(os.path.join(METRIC_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    predictions = {}

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
        ),
        X_train, y_train, X_test, y_test,
        ROC_DIR, PR_DIR, MODEL_DIR, METRIC_DIR, emb_name
    )

    predictions["LightGBM"] = train_and_evaluate(
        "LightGBM",
        lgb.LGBMClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            **best_params["LightGBM"]
        ),
        X_train, y_train, X_test, y_test,
        ROC_DIR, PR_DIR, MODEL_DIR, METRIC_DIR, emb_name
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
        ),
        X_train, y_train, X_test, y_test,
        ROC_DIR, PR_DIR, MODEL_DIR, METRIC_DIR, emb_name
    )

    # ROC (all ML models)
    plt.figure(figsize=(8, 6))

    for model_name, y_proba in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})",
                 color=MODEL_COLORS[model_name], linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"BBBP ROC Curves — {emb_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ROC_DIR, "roc_all_models.pdf"))
    plt.close()

    # PR (all ML models)
    plt.figure(figsize=(8, 6))

    for model_name, y_proba in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})",
                 color=MODEL_COLORS[model_name], linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"BBBP Precision–Recall Curves — {emb_name}")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PR_DIR, "pr_all_models.pdf"))
    plt.close()

    logging.info(f"Completed BBBP modeling for embedding: {emb_name}")

logging.info("All BBBP base-model embedding experiments completed successfully.")
# nohup python bbbp_modeling.py > bbbp_base_models_nohup.out 2>&1 &