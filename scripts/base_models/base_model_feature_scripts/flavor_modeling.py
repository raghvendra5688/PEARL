"""
Flavor Embedding + RDKit Feature Modeling Pipeline (Multiclass)

For each embedding model:
- Features = [Embedding Vector] + [RDKit + Graph + Fingerprints]
- ML models trained independently per embedding (no embedding concatenation)

Target:
- Canonicalized Taste (multiclass)
"""

import os
import ast
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
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = str(REPO_ROOT / "data" / "base_models_features" / "flavor_datasets")
OUTPUT_ROOT = str(REPO_ROOT / "results" / "base_models_features" / "flavor_dataset")
LOG_DIR = str(REPO_ROOT / "logs")

LABEL_COL = "Canonicalized Taste"

EMBED_COLS = [
    "ChemBERTa_77M_MTR_Base",
    "ChemBERTa_77M_MLM_Base",
    "MolFormer_Base"
]

META_COLS = ["Standardized SMILES"]
RANDOM_SEED = 42
TOTAL_CORES = os.cpu_count()
N_JOBS = max(1, TOTAL_CORES // 2)

MODEL_COLORS = {
    "XGBoost": "tab:blue",
    "LightGBM": "tab:green",
    "CatBoost": "tab:red"
}

for emb in EMBED_COLS:
    for sub in ["ROC_Curves", "PR_Curves", "models", "metrics"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, emb, sub), exist_ok=True)

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "flavor_base_model_feature_ml.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Loading and Sanitization Utilities
def load_split(split):
    path = os.path.join(DATA_ROOT, f"{split}_features.csv")
    logging.info(f"Loading {split}: {path}")
    return pd.read_csv(path)

def parse_embedding_column(series):
    return np.vstack(
        series.apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)).values
    )

def sanitize_features(X: pd.DataFrame, split_name: str) -> pd.DataFrame:
    logging.info(f"Sanitizing chemical features: {split_name}")

    X = X.apply(pd.to_numeric, errors="coerce")

    nan_before = X.isna().sum().sum()
    inf_before = np.isinf(X.values).sum()

    logging.info(f"{split_name} BEFORE | NaN={nan_before}, Inf={inf_before}")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.clip(lower=-1e6, upper=1e6)

    nan_after = X.isna().sum().sum()
    inf_after = np.isinf(X.values).sum()

    logging.info(f"{split_name} AFTER | NaN={nan_after}, Inf={inf_after}")

    return X


def build_feature_matrix(df, emb_col, label_encoder):
    # ---- embeddings (NOT sanitized) ----
    emb = parse_embedding_column(df[emb_col])

    # ---- chemical features only ----
    other_feats = df.drop(columns=META_COLS + EMBED_COLS + [LABEL_COL])
    other_feats = sanitize_features(other_feats, f"{emb_col}")

    X = np.hstack([emb, other_feats.values])
    y = label_encoder.transform(df[LABEL_COL])

    return X, y

train_df = load_split("train")
val_df   = load_split("valid")
test_df  = load_split("test")

label_encoder = LabelEncoder()
label_encoder.fit(train_df[LABEL_COL])

n_classes = len(label_encoder.classes_)

# Main execution loop for each embedding
for EMB_NAME in EMBED_COLS:

    logging.info(f"========== Flavor | Embedding: {EMB_NAME} ==========")

    X_train, y_train = build_feature_matrix(train_df, EMB_NAME, label_encoder)
    X_val,   y_val   = build_feature_matrix(val_df,   EMB_NAME, label_encoder)
    X_test,  y_test  = build_feature_matrix(test_df,  EMB_NAME, label_encoder)

    logging.info(
        f"Shapes | Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
    )

    # sample weights for class imbalance
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # Hyperparameter Optimization
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

    def run_opt(model_type):
        logging.info(f"Running Optuna for {model_type}")

        sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(lambda t: optimize_model(t, model_type), n_trials=10)

        logging.info(f"{model_type} best MCC: {study.best_value:.4f}")
        return study.best_params

    best_params = {
        "XGBoost":  run_opt("xgb"),
        "LightGBM": run_opt("lgb"),
        "CatBoost": run_opt("cb")
    }

    with open(os.path.join(OUTPUT_ROOT, EMB_NAME, "metrics", "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    # Training and Evaluation
    predictions = {}

    def train_eval(name, model):

        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1_macro": f1_score(y_test, y_pred, average="macro"),
            "F1_micro": f1_score(y_test, y_pred, average="micro"),
            "MCC": matthews_corrcoef(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob, multi_class="ovr")
        }

        joblib.dump(model, os.path.join(OUTPUT_ROOT, EMB_NAME, "models", f"{name}.pkl"))
        np.save(os.path.join(OUTPUT_ROOT, EMB_NAME, "metrics", f"{name}_metrics.npy"), metrics)

        logging.info(f"[{EMB_NAME}] {name} metrics: {metrics}")

        return y_prob

    predictions["XGBoost"] = train_eval(
        "XGBoost",
        xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=n_classes,
            random_state=RANDOM_SEED,
            tree_method="hist",
            n_jobs=N_JOBS,
            **best_params["XGBoost"]
        )
    )

    predictions["LightGBM"] = train_eval(
        "LightGBM",
        lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            **best_params["LightGBM"]
        )
    )

    predictions["CatBoost"] = train_eval(
        "CatBoost",
        cb.CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=RANDOM_SEED,
            verbose=0,
            thread_count=N_JOBS,
            **best_params["CatBoost"]
        )
    )

    # ROC
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    plt.figure(figsize=(8, 6))
    for name, prob in predictions.items():
        auc = roc_auc_score(y_test, prob, multi_class="ovr")
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), prob.ravel())
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=MODEL_COLORS[name])

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"Flavor ROC — {EMB_NAME}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, EMB_NAME, "ROC_Curves", "roc_all_models.pdf"))
    plt.close()

    # PR
    plt.figure(figsize=(8, 6))
    for name, prob in predictions.items():
        ap = average_precision_score(y_test_bin, prob, average="macro")
        prec, rec, _ = precision_recall_curve(y_test_bin.ravel(), prob.ravel())
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=MODEL_COLORS[name])

    plt.title(f"Flavor PR — {EMB_NAME}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, EMB_NAME, "PR_Curves", "pr_all_models.pdf"))
    plt.close()

    logging.info(f"Completed Flavor | {EMB_NAME}")

logging.info("Flavor base model with feature modeling completed successfully.")

# nohup python flavor_modeling.py > flavor_base_models_features_nohup.out 2>&1 &