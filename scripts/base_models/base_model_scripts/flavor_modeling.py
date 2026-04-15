"""
Flavor Base-Model-Embeddings Modeling Pipeline (Multiclass)

This script performs supervised multiclass classification for the Flavor dataset
using base model embeddings only (no RDKit, no SMILES).
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
    balanced_accuracy_score,
    make_scorer
)
from sklearn.model_selection import StratifiedKFold

# PATHS
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_MODEL_ROOT = str(REPO_ROOT / "data" / "base_models" / "flavor_datasets")
OUTPUT_ROOT = str(REPO_ROOT / "results" / "base_models" / "flavor_dataset")
LOG_DIR = str(REPO_ROOT / "logs")

LABEL_COL = "Canonicalized Taste"
META_COLS = ["Standardized SMILES"]

EMBED_MODELS = [
    "ChemBERTa_77M_MTR_Base",
    "ChemBERTa_77M_MLM_Base",
    "MolFormer_Base"
]

MODEL_COLORS = {
    "XGBoost": "tab:blue",
    "LightGBM": "tab:green",
    "CatBoost": "tab:red"
}

RANDOM_SEED = 42

TOTAL_CORES = os.cpu_count()
N_JOBS = max(1, TOTAL_CORES // 2)

for emb in EMBED_MODELS:
    for sub in ["ROC_Curves", "PR_Curves", "models", "metrics"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, emb, sub), exist_ok=True)

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "flavor_base_model_ml.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# DATA

def load_split(split_name):
    path = os.path.join(BASE_MODEL_ROOT, f"{split_name}_embeddings.csv")
    logging.info(f"Loading {split_name} split from {path}")
    return pd.read_csv(path)

train_df = load_split("train")
val_df   = load_split("valid")
test_df  = load_split("test")

label_encoder = LabelEncoder()
label_encoder.fit(train_df[LABEL_COL])

def extract_embeddings(df, emb_col):
    X_list = df[emb_col].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
    X = np.vstack(X_list.values)
    y = label_encoder.transform(df[LABEL_COL])
    return X, y

# MAIN EXECUTION
for EMB_NAME in EMBED_MODELS:

    logging.info(f"========== Flavor using {EMB_NAME} embeddings ==========")

    X_train, y_train = extract_embeddings(train_df, EMB_NAME)
    X_val, y_val     = extract_embeddings(val_df, EMB_NAME)
    X_test, y_test   = extract_embeddings(test_df, EMB_NAME)

    logging.info(f"[{LABEL_COL} | {EMB_NAME}] Shapes |"
    f"Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    X_train = np.nan_to_num(X_train)
    X_val   = np.nan_to_num(X_val)
    X_test  = np.nan_to_num(X_test)

    # Sample weights (multiclass)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # OPTIMIZATION USING OPTUNA
    def optimize_model(trial, model_type):

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        mcc_scores = []

        for tr_idx, va_idx in cv.split(X_train, y_train):

            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
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

    with open(
        os.path.join(OUTPUT_ROOT, EMB_NAME, "metrics", "best_params.json"),
        "w"
    ) as f:
        json.dump(best_params, f, indent=4)

    # TRAINING & EVALUATION
    predictions = {}

    def train_and_evaluate(name, model):

        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1_macro": f1_score(y_test, y_pred, average="macro"),
            "F1_micro": f1_score(y_test, y_pred, average="micro"),
            "MCC": matthews_corrcoef(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_proba, multi_class="ovr")
        }

        logging.info(f"[{EMB_NAME}] {name} metrics: {metrics}")

        joblib.dump(model, os.path.join(OUTPUT_ROOT, EMB_NAME, "models", f"{name}.pkl"))
        np.save(os.path.join(OUTPUT_ROOT, EMB_NAME, "metrics", f"{name}_metrics.npy"), metrics)

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

    # ROC & PR
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
    plt.title(f"Flavor ROC – {EMB_NAME}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, EMB_NAME, "ROC_Curves", "roc_all_models.pdf"))
    plt.close()

    # PR
    plt.figure(figsize=(8, 6))

    for name, y_proba in predictions.items():
        ap = average_precision_score(y_test_bin, y_proba, average="macro")
        precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_proba.ravel())
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=MODEL_COLORS[name])

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Flavor PR – {EMB_NAME}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, EMB_NAME, "PR_Curves", "pr_all_models.pdf"))
    plt.close()

    logging.info(f"Completed Flavor | {EMB_NAME}")

logging.info("Flavor Base-Model Embedding ML modeling completed successfully.")
# nohup python flavor_modeling.py > flavor_base_models_nohup.out 2>&1 &
