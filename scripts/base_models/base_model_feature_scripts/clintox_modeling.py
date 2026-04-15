"""
ClinTox Embedding + RDKit Feature Modeling Pipeline

Binary classification for:
1. FDA_APPROVED
2. CT_TOX

For each task:
- Run ML separately for each embedding model
- Features = [Embedding vector] + [RDKit + Graph + Fingerprints]
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

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, matthews_corrcoef,
    roc_curve, precision_recall_curve, average_precision_score, make_scorer
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = str(REPO_ROOT / "data" / "base_models_features" / "clintox_datasets")
OUTPUT_ROOT = str(REPO_ROOT / "results" / "base_models_features" / "clintox_dataset")
LOG_DIR = str(REPO_ROOT / "logs")

LABELS = ["FDA_APPROVED"]

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

for label in LABELS:
    for emb in EMBED_COLS:
        for sub in ["ROC_Curves", "PR_Curves", "models", "metrics"]:
            os.makedirs(os.path.join(OUTPUT_ROOT, label, emb, sub), exist_ok=True)

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "clintox_base_model_feature_ml.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Loading and Sanitization Utilities
def load_split(split):
    path = os.path.join(DATA_ROOT, f"{split}_features.csv")
    logging.info(f"Loading {split}: {path}")
    return pd.read_csv(path)


def parse_embedding_column(series):
    return np.vstack(series.apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)).values)


def sanitize_features(X: pd.DataFrame, split_name: str) -> pd.DataFrame:
    logging.info(f"Sanitizing features: {split_name}")

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


def build_feature_matrix(df, emb_col, label_col):
    emb = parse_embedding_column(df[emb_col])

    other_feats = df.drop(columns=META_COLS + EMBED_COLS + [label_col])
    other_feats = sanitize_features(other_feats, f"{emb_col}")

    X = np.hstack([emb, other_feats.values])
    y = df[label_col].astype(int).values

    return X, y

# Load Data
train_df = load_split("train")
val_df   = load_split("valid")
test_df  = load_split("test")

for df in [train_df, val_df, test_df]:
    if "CT_TOX" in df.columns:
        df.drop(columns=["CT_TOX"], inplace=True)

# Main execution loop for each task and embedding
for LABEL_COL in LABELS:

    logging.info(f"========== TASK: {LABEL_COL} ==========")

    for EMB_NAME in EMBED_COLS:

        logging.info(f"---- Embedding: {EMB_NAME} ----")

        X_train, y_train = build_feature_matrix(train_df, EMB_NAME, LABEL_COL)
        X_val,   y_val   = build_feature_matrix(val_df,   EMB_NAME, LABEL_COL)
        X_test,  y_test  = build_feature_matrix(test_df,  EMB_NAME, LABEL_COL)

        logging.info(f"Shapes | Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        scale_pos_weight = n_neg / n_pos

        logging.info(f"Class dist | 0={n_neg}, 1={n_pos}, spw={scale_pos_weight:.3f}")

        # Hyperparameter Optimization
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

        def run_opt(model_type):
            sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
            study = optuna.create_study(direction="maximize", sampler=sampler)

            study.optimize(lambda t: optimize_model(t, model_type), n_trials=10)
            return study.best_params

        best_params = {
            "XGBoost": run_opt("xgb"),
            "LightGBM": run_opt("lgb"),
            "CatBoost": run_opt("cb")
        }

        with open(os.path.join(OUTPUT_ROOT, LABEL_COL, EMB_NAME, "metrics", "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)

        # Training and Evaluation
        predictions = {}

        def train_eval(name, model):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_prob),
                "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "F1_macro": f1_score(y_test, y_pred, average="macro"),
                "F1_micro": f1_score(y_test, y_pred, average="micro"),
                "MCC": matthews_corrcoef(y_test, y_pred),
            }

            joblib.dump(model, os.path.join(OUTPUT_ROOT, LABEL_COL, EMB_NAME, "models", f"{name}.pkl"))
            np.save(os.path.join(OUTPUT_ROOT, LABEL_COL, EMB_NAME, "metrics", f"{name}_metrics.npy"), metrics)

            logging.info(f"[{LABEL_COL} | {EMB_NAME}] {name} metrics: {metrics}")

            return y_prob

        predictions["XGBoost"] = train_eval(
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

        predictions["LightGBM"] = train_eval(
            "LightGBM",
            lgb.LGBMClassifier(
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=N_JOBS,
                **best_params["LightGBM"]
            )
        )

        predictions["CatBoost"] = train_eval(
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

        # -------- ROC --------

        plt.figure(figsize=(8, 6))
        for name, prob in predictions.items():
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=MODEL_COLORS[name])

        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ClinTox ROC — {LABEL_COL} — {EMB_NAME}")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_ROOT, LABEL_COL, EMB_NAME, "ROC_Curves", "roc_all_models.pdf"))
        plt.close()

        # -------- PR --------

        plt.figure(figsize=(8, 6))
        for name, prob in predictions.items():
            prec, rec, _ = precision_recall_curve(y_test, prob)
            ap = average_precision_score(y_test, prob)
            plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=MODEL_COLORS[name])

        plt.title(f"ClinTox PR — {LABEL_COL} — {EMB_NAME}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_ROOT, LABEL_COL, EMB_NAME, "PR_Curves", "pr_all_models.pdf"))
        plt.close()

        logging.info(f"Completed | {LABEL_COL} | {EMB_NAME}")

logging.info("ClinTox base model with features modeling completed successfully.")
# nohup python clintox_modeling.py > clintox_base_models_features_nohup.out 2>&1 &
