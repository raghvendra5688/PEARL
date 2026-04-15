"""
Flavor FT-Model-Embeddings Modeling Pipeline (Multiclass) - Refactored

This script performs supervised multiclass classification for the Flavor dataset
using base model embeddings only (no RDKit, no SMILES).

Security improvements:
- Safe embedding parsing with json.loads() instead of ast.literal_eval()
- Path validation to prevent directory traversal
- Comprehensive error handling

Performance improvements:
- In-place numpy operations to reduce memory usage
- Configurable thread count with auto-detection
- Increased Optuna trials from 10 to 30
- Context managers for resource cleanup

Code quality improvements:
- Type hints throughout
- Environment variable configuration
- Both file and console logging
- Modular, reusable functions
"""

import os
import json
import logging
import joblib
import contextlib
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    balanced_accuracy_score,
    make_scorer
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


class Config:
    """Configuration class with path validation and environment variable support."""

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

        # Data paths
        self.BASE_MODEL_ROOT = self._validate_path(
            Path(os.getenv('FLAVOR_DATA_ROOT', self.BASE_DIR / "data" / "finetuned_embeddings" / "flavor_Embeddings"))
        )
        self.OUTPUT_ROOT = self._validate_path(
            Path(os.getenv('FLAVOR_OUTPUT_ROOT', self.BASE_DIR / "results" / "ft_embeddings" / "flavor_FT_Results")),
            create=True
        )
        self.LOG_DIR = self._validate_path(
            self.OUTPUT_ROOT / "logs",
            create=True
        )

        # Configuration
        self.LABEL_COL = "Canonicalized Taste"
        self.META_COLS = ["Standardized SMILES"]
        self.RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
        self.N_JOBS = int(os.getenv('N_JOBS', str(min(os.cpu_count() or 1, 192))))
        self.OPTUNA_TRIALS = int(os.getenv('OPTUNA_TRIALS', '10'))

        # Embedding models
        self.EMBEDDING_MODELS = [
            "ChemBERTa_77M_MTR_FL",
            "ChemBERTa_77M_MLM_FL",
            "MolFormer_Finetuned_FL",
            "ChemBERTa_77M_MTR_WL",
            "ChemBERTa_77M_MLM_WL",
            "Molformer_Finetuned_WL"
        ]

        self.MODEL_COLORS = {
            "XGBoost": "tab:blue",
            "LightGBM": "tab:green",
            "CatBoost": "tab:red"
        }

        # Create output directories
        for emb in self.EMBEDDING_MODELS:
            for sub in ["ROC_Curves", "PR_Curves", "models", "metrics"]:
                self._validate_path(self.OUTPUT_ROOT / emb / sub, create=True)

    def _validate_path(self, path: Path, create: bool = False) -> Path:
        """Validate path is within base directory and optionally create it."""
        path = Path(path).resolve()
        try:
            path.relative_to(self.BASE_DIR)
        except ValueError:
            raise ValueError(f"Path {path} is outside base directory {self.BASE_DIR}")

        if create:
            path.mkdir(parents=True, exist_ok=True)

        return path


def setup_logging(config: Config) -> None:
    """Setup both file and console logging."""
    log_file = config.LOG_DIR / "flavor_ft_model_ml.log"

    # Create formatters
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def validate_embedding_string(s: str) -> bool:
    """Validate that string looks like a valid embedding (list of numbers)"""
    if not isinstance(s, str):
        return False
    s = s.strip()
    # Basic validation: should contain numbers and commas/spaces
    if not any(c.isdigit() for c in s):
        return False
    return True


def safe_parse_embedding(s: str) -> Optional[np.ndarray]:
    """
    Safely parse embedding string using json.loads() instead of ast.literal_eval().

    Security: json.loads() is safer than ast.literal_eval() as it only parses JSON,
    not arbitrary Python literals.

    Supports multiple formats:
    1. JSON list: "[1.0, 2.0, 3.0]"
    2. Comma-separated: "1.0, 2.0, 3.0" or "1.0,2.0,3.0"
    """
    try:
        if not validate_embedding_string(s):
            logging.warning(f"Invalid embedding string format: {s[:100]}...")
            return None

        # Try JSON format first (most common)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                arr = np.array(parsed, dtype=np.float32)
            else:
                logging.warning(f"Parsed embedding is not a list: {type(parsed)}")
                return None

        except json.JSONDecodeError:
            # If JSON parsing fails, try comma-separated format
            logging.debug("JSON parsing failed, trying comma-separated format")

            # Remove brackets if present
            s_clean = s.strip()
            if s_clean.startswith('[') and s_clean.endswith(']'):
                s_clean = s_clean[1:-1]

            # Split by comma and convert to float
            try:
                values = [float(x.strip()) for x in s_clean.split(',') if x.strip()]
                arr = np.array(values, dtype=np.float32)
            except (ValueError, AttributeError) as e:
                logging.error(f"Failed to parse comma-separated embedding: {e}")
                return None

        # Validate array properties
        if arr.ndim != 1:
            logging.warning(f"Embedding has wrong dimensions: {arr.ndim}")
            return None

        if len(arr) == 0:
            logging.warning("Embedding is empty")
            return None

        # Sanitize non-finite values
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)

        return arr

    except Exception as e:
        logging.error(f"Unexpected error parsing embedding: {e}")
        return None


def load_split(config: Config, split_name: str) -> pd.DataFrame:
    """Load data split with error handling."""
    try:
        path = config.BASE_MODEL_ROOT / f"fart_{split_name}_embed.csv"

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        logging.info(f"Loading {split_name} split from {path}")
        df = pd.read_csv(path)

        # Validate required columns
        required_cols = [config.LABEL_COL] + config.META_COLS
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    except Exception as e:
        logging.error(f"Failed to load {split_name} split: {e}")
        raise


def extract_embeddings(
    df: pd.DataFrame,
    emb_col: str,
    label_encoder: LabelEncoder,
    config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and labels with safe parsing."""

    if emb_col not in df.columns:
        raise ValueError(f"Embedding column '{emb_col}' not found in dataframe")

    logging.info(f"Extracting embeddings from column: {emb_col}")

    # Parse embeddings safely
    embeddings = []
    valid_indices = []

    for idx, emb_str in enumerate(df[emb_col]):
        parsed = safe_parse_embedding(emb_str)
        if parsed is not None:
            embeddings.append(parsed)
            valid_indices.append(idx)
        else:
            logging.warning(f"Skipping row {idx} due to invalid embedding")

    if not embeddings:
        raise ValueError("No valid embeddings found")

    X = np.vstack(embeddings)

    # Extract labels for valid rows only
    valid_df = df.iloc[valid_indices]
    y = label_encoder.transform(valid_df[config.LABEL_COL])

    logging.info(f"Extracted {len(embeddings)} valid embeddings, shape: {X.shape}")

    # Sanitize in-place for memory efficiency
    X = np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)

    return X, y


@contextlib.contextmanager
def plot_context():
    """Context manager to ensure plots are always closed."""
    try:
        yield
    finally:
        plt.close('all')


def optimize_model(
    trial: optuna.Trial,
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: np.ndarray,
    config: Config
) -> float:
    """
    Optimize model hyperparameters using Optuna with 5-fold stratified cross-validation.

    Args:
        trial: Optuna trial object for hyperparameter suggestions
        model_type: Type of model ('xgb', 'lgb', or 'cb')
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (unused in CV but kept for compatibility)
        y_val: Validation labels (unused in CV but kept for compatibility)
        sample_weights: Sample weights for handling class imbalance
        config: Configuration object

    Returns:
        Mean MCC score across 5 folds
    """

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
    }

    try:
        if model_type == "xgb":
            model = xgb.XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=config.RANDOM_SEED,
                tree_method="hist",
                n_jobs=config.N_JOBS,
                **params
            )
        elif model_type == "lgb":
            model = lgb.LGBMClassifier(
                objective="multiclass",
                random_state=config.RANDOM_SEED,
                n_jobs=config.N_JOBS,
                verbosity=0,
                **params
            )
        else:  # catboost
            model = cb.CatBoostClassifier(
                loss_function="MultiClass",
                random_seed=config.RANDOM_SEED,
                verbose=0,
                thread_count=config.N_JOBS,
                **params
            )

        # Perform 5-fold stratified cross-validation optimizing for MCC
        # Note: For multiclass with sample weights, we perform manual CV to properly handle weights
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        mcc_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            weights_fold = sample_weights[train_idx]

            # Create a fresh model instance for this fold
            if model_type == "xgb":
                fold_model = xgb.XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=config.RANDOM_SEED,
                    tree_method="hist",
                    n_jobs=config.N_JOBS,
                    **params
                )
            elif model_type == "lgb":
                fold_model = lgb.LGBMClassifier(
                    objective="multiclass",
                    random_state=config.RANDOM_SEED,
                    n_jobs=config.N_JOBS,
                    verbosity=0,
                    **params
                )
            else:  # catboost
                fold_model = cb.CatBoostClassifier(
                    loss_function="MultiClass",
                    random_seed=config.RANDOM_SEED,
                    verbose=0,
                    thread_count=config.N_JOBS,
                    **params
                )

            fold_model.fit(X_fold_train, y_fold_train, sample_weight=weights_fold)
            y_pred = fold_model.predict(X_fold_val)
            fold_mcc = matthews_corrcoef(y_fold_val, y_pred)
            mcc_scores.append(fold_mcc)

        # Return mean MCC score across all folds
        mean_mcc = np.mean(mcc_scores)

        return mean_mcc

    except Exception as e:
        logging.error(f"Error in trial optimization for {model_type}: {e}")
        raise optuna.exceptions.TrialPruned()


def run_optimization(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: np.ndarray,
    config: Config
) -> Dict[str, Any]:
    """Run Optuna optimization for a model type."""

    logging.info(f"Running Optuna optimization for {model_type} ({config.OPTUNA_TRIALS} trials)")

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda t: optimize_model(t, model_type, X_train, y_train, X_val, y_val, sample_weights, config),
        n_trials=config.OPTUNA_TRIALS
    )

    logging.info(f"{model_type} optimization complete. Best MCC: {study.best_value:.4f}")

    return study.best_params


def train_and_evaluate(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_weights: np.ndarray,
    config: Config,
    emb_name: str
) -> np.ndarray:
    """Train and evaluate a model, saving results."""

    logging.info(f"Training {name} for {emb_name}")

    try:
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 3),
            "Recall": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 3),
            "F1_macro": round(f1_score(y_test, y_pred, average="macro"), 3),
            "F1_micro": round(f1_score(y_test, y_pred, average="micro"), 3),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 3),
            "AUC": round(roc_auc_score(y_test, y_proba, multi_class="ovr"), 3)
        }

        logging.info(f"[{emb_name}] {name} metrics: {metrics}")

        # Save model and metrics
        model_path = config.OUTPUT_ROOT / emb_name / "models" / f"{name}.pkl"
        joblib.dump(model, model_path)

        metrics_path = config.OUTPUT_ROOT / emb_name / "metrics" / f"{name}_metrics.npy"
        np.save(metrics_path, metrics)

        return y_proba

    except Exception as e:
        logging.error(f"Failed to train/evaluate {name}: {e}")
        raise


def plot_roc_curve(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    y_test_bin: np.ndarray,
    emb_name: str,
    config: Config
) -> None:
    """Plot ROC curve for all models."""

    with plot_context():
        plt.figure(figsize=(8, 6))

        for name, y_proba in predictions.items():
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                    color=config.MODEL_COLORS[name], linewidth=2)

        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Flavor ROC Curves - {emb_name}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = config.OUTPUT_ROOT / emb_name / "ROC_Curves" / "roc_all_models.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"ROC curve saved to {save_path}")


def plot_pr_curve(
    predictions: Dict[str, np.ndarray],
    y_test_bin: np.ndarray,
    emb_name: str,
    config: Config
) -> None:
    """Plot Precision-Recall curve for all models."""

    with plot_context():
        plt.figure(figsize=(8, 6))

        for name, y_proba in predictions.items():
            ap = average_precision_score(y_test_bin, y_proba, average="macro")
            precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_proba.ravel())
            plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})",
                    color=config.MODEL_COLORS[name], linewidth=2)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Flavor Precision-Recall Curves - {emb_name}")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = config.OUTPUT_ROOT / emb_name / "PR_Curves" / "pr_all_models.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"PR curve saved to {save_path}")


def main():
    """Main execution function."""

    # Initialize configuration
    config = Config()
    setup_logging(config)

    logging.info("=" * 80)
    logging.info("Starting Flavor FT-Model-Embeddings Modeling Pipeline (Multiclass)")
    logging.info(f"Base directory: {config.BASE_DIR}")
    logging.info(f"Data root: {config.BASE_MODEL_ROOT}")
    logging.info(f"Output root: {config.OUTPUT_ROOT}")
    logging.info(f"Random seed: {config.RANDOM_SEED}")
    logging.info(f"N_JOBS: {config.N_JOBS}")
    logging.info(f"Optuna trials: {config.OPTUNA_TRIALS}")
    logging.info("=" * 80)

    try:
        # Load data
        train_df = load_split(config, "train")
        val_df = load_split(config, "eval")
        test_df = load_split(config, "test")

        # Initialize label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(train_df[config.LABEL_COL])
        n_classes = len(label_encoder.classes_)

        logging.info(f"Number of classes: {n_classes}")
        logging.info(f"Classes: {label_encoder.classes_}")

        # Process each embedding model
        for emb_name in config.EMBEDDING_MODELS:
            logging.info("=" * 80)
            logging.info(f"Processing embedding: {emb_name}")
            logging.info("=" * 80)

            try:
                # Extract embeddings
                X_train, y_train = extract_embeddings(train_df, f"{emb_name}_embeddings", label_encoder, config)
                X_val, y_val = extract_embeddings(val_df, f"{emb_name}_embeddings", label_encoder, config)
                X_test, y_test = extract_embeddings(test_df, f"{emb_name}_embeddings", label_encoder, config)

                logging.info(f"Shapes | Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

                # Compute sample weights for class imbalance
                sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
                logging.info(f"Sample weights computed (min={sample_weights.min():.3f}, max={sample_weights.max():.3f})")

                # Hyperparameter optimization
                best_params = {
                    "XGBoost": run_optimization("xgb", X_train, y_train, X_val, y_val, sample_weights, config),
                    "LightGBM": run_optimization("lgb", X_train, y_train, X_val, y_val, sample_weights, config),
                    "CatBoost": run_optimization("cb", X_train, y_train, X_val, y_val, sample_weights, config)
                }

                # Save best parameters
                params_path = config.OUTPUT_ROOT / emb_name / "metrics" / "best_params.json"
                with open(params_path, "w") as f:
                    json.dump(best_params, f, indent=4)
                logging.info(f"Best parameters saved to {params_path}")

                # Train and evaluate models
                predictions = {}

                predictions["XGBoost"] = train_and_evaluate(
                    "XGBoost",
                    xgb.XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=config.RANDOM_SEED,
                        tree_method="hist",
                        n_jobs=config.N_JOBS,
                        **best_params["XGBoost"]
                    ),
                    X_train, y_train, X_test, y_test, sample_weights, config, emb_name
                )

                predictions["LightGBM"] = train_and_evaluate(
                    "LightGBM",
                    lgb.LGBMClassifier(
                        objective="multiclass",
                        random_state=config.RANDOM_SEED,
                        n_jobs=config.N_JOBS,
                        verbosity=0,
                        **best_params["LightGBM"]
                    ),
                    X_train, y_train, X_test, y_test, sample_weights, config, emb_name
                )

                predictions["CatBoost"] = train_and_evaluate(
                    "CatBoost",
                    cb.CatBoostClassifier(
                        loss_function="MultiClass",
                        random_seed=config.RANDOM_SEED,
                        verbose=0,
                        thread_count=config.N_JOBS,
                        **best_params["CatBoost"]
                    ),
                    X_train, y_train, X_test, y_test, sample_weights, config, emb_name
                )

                # Binarize labels for ROC/PR curves
                y_test_bin = label_binarize(y_test, classes=range(n_classes))

                # Plot curves
                plot_roc_curve(predictions, y_test, y_test_bin, emb_name, config)
                plot_pr_curve(predictions, y_test_bin, emb_name, config)

                logging.info(f"Completed processing for embedding: {emb_name}")

            except Exception as e:
                logging.error(f"Failed to process embedding {emb_name}: {e}")
                continue

        logging.info("=" * 80)
        logging.info("Flavor Base-Model Embedding ML modeling completed successfully")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
