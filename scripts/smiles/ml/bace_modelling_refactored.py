"""
BACE FT-Model-Embedding Modeling Pipeline (REFACTORED)

This script performs supervised binary classification for the BACE dataset
using finetuned-model embeddings as features (no extra PC features).

Security and performance improvements:
- Input validation and error handling
- Configurable parameters via environment variables
- Optimized data processing
- Proper resource management
- Path traversal protection
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import contextlib

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

# ==================== CONFIGURATION ====================

class Config:
    """Configuration with environment variable support and validation"""

    def __init__(self):
        # Get base directory and validate it exists
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

        # Paths with validation
        self.BASE_MODEL_ROOT = self._validate_path(
            os.getenv('BASE_MODEL_ROOT', self.BASE_DIR / 'data' / 'finetuned_embeddings' / 'BACE_Embeddings')
        )
        self.OUTPUT_ROOT = self._validate_path(
            os.getenv('OUTPUT_ROOT', self.BASE_DIR / 'results' / 'ft_embeddings' / 'BACE_FT_Results'),
            create=True
        )
        self.LOG_DIR = self._validate_path(
            os.getenv('LOG_DIR', self.BASE_DIR / 'results' / 'ft_embeddings' / 'BACE_FT_Results' / 'logs'),
            create=True
        )

        # Model configuration
        self.RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
        self.N_JOBS = int(os.getenv('N_JOBS', str(min(os.cpu_count() or 1, 60))))
        self.OPTUNA_TRIALS = int(os.getenv('OPTUNA_TRIALS', '20'))

        # Data columns
        self.LABEL_COL = "Class"
        self.SMILES_COL = "Standardized SMILES"

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

    def _validate_path(self, path: Path, create: bool = False) -> Path:
        """Validate and normalize path, optionally create if needed"""
        path = Path(path).resolve()

        # Security: Ensure path is within base directory
        try:
            path.relative_to(self.BASE_DIR)
        except ValueError:
            raise ValueError(f"Path {path} is outside base directory {self.BASE_DIR}")

        if create:
            path.mkdir(parents=True, exist_ok=True)
        elif not path.exists() and not create:
            logging.warning(f"Path does not exist: {path}")

        return path


# ==================== LOGGING SETUP ====================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging with both file and console handlers"""
    log_file = log_dir / "bace_ft_model_ml.log"

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(funcName)s | %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(levelname)s | %(message)s"
    )

    # File handler with rotation support
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ==================== DATA LOADING & PROCESSING ====================

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
    Safely parse embedding string with validation.

    Supports multiple formats:
    1. JSON list: "[1.0, 2.0, 3.0]"
    2. Comma-separated: "1.0, 2.0, 3.0" or "1.0,2.0,3.0"
    """
    try:
        if not validate_embedding_string(s):
            logging.warning(f"Invalid embedding format: {s[:50]}...")
            return None

        # Try JSON format first (most common)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                arr = np.array(parsed, dtype=np.float32)
            else:
                logging.warning(f"Embedding is not a list: {type(parsed)}")
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

        if not np.isfinite(arr).all():
            logging.warning("Embedding contains non-finite values")
            # Replace non-finite values instead of rejecting
            arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)

        return arr

    except Exception as e:
        logging.error(f"Unexpected error parsing embedding: {e}")
        return None


def extract_embeddings(df: pd.DataFrame, emb_col: str, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings and labels from dataframe with proper error handling

    Args:
        df: Input dataframe
        emb_col: Column name containing embeddings
        label_col: Column name containing labels

    Returns:
        Tuple of (X, y) as numpy arrays

    Raises:
        ValueError: If data validation fails
    """
    if emb_col not in df.columns:
        raise ValueError(f"Embedding column '{emb_col}' not found in dataframe")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")

    # Parse embeddings with validation
    embeddings = []
    valid_indices = []

    for idx, emb_str in enumerate(df[emb_col]):
        parsed = safe_parse_embedding(emb_str)
        if parsed is not None:
            embeddings.append(parsed)
            valid_indices.append(idx)

    if not embeddings:
        raise ValueError("No valid embeddings found")

    # Check all embeddings have same dimension
    emb_dims = [len(e) for e in embeddings]
    if len(set(emb_dims)) > 1:
        raise ValueError(f"Inconsistent embedding dimensions: {set(emb_dims)}")

    X = np.vstack(embeddings)
    y = df.iloc[valid_indices][label_col].astype(int).values

    logging.info(f"Extracted {len(embeddings)} valid embeddings from {len(df)} total samples")

    if len(valid_indices) < len(df):
        logging.warning(f"Dropped {len(df) - len(valid_indices)} invalid samples")

    return X, y


def load_split(split_name: str, base_path: Path) -> pd.DataFrame:
    """Load data split with error handling"""
    path = base_path / f"bace_{split_name}_embed.csv"

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    logging.info(f"Loading {split_name} split from {path}")

    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded {len(df)} samples from {split_name} split")
        return df
    except Exception as e:
        logging.error(f"Failed to load {split_name} split: {e}")
        raise


# ==================== MODEL OPTIMIZATION ====================

def optimize_model(
    trial: optuna.Trial,
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
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
        scale_pos_weight: Class weight scaling factor for XGBoost
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
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=config.RANDOM_SEED,
                tree_method="hist",
                missing=np.nan,
                n_jobs=config.N_JOBS,
                **params
            )

        elif model_type == "lgb":
            model = lgb.LGBMClassifier(
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
                n_jobs=config.N_JOBS,
                verbose=-1,  # Suppress warnings
                **params
            )

        else:  # catboost
            model = cb.CatBoostClassifier(
                auto_class_weights="Balanced",
                loss_function="Logloss",
                random_seed=config.RANDOM_SEED,
                thread_count=config.N_JOBS,
                verbose=0,
                **params
            )

        # Perform 5-fold stratified cross-validation optimizing for MCC
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        mcc_scorer = make_scorer(matthews_corrcoef)
        scores = cross_val_score(model, X_train, y_train, scoring=mcc_scorer, cv=cv, n_jobs=1)

        # Return mean MCC score across all folds
        mean_mcc = np.mean(scores)

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
    scale_pos_weight: float,
    config: Config
) -> Dict[str, Any]:
    """Run Optuna optimization for a model"""

    logging.info(f"Running Optuna optimization for {model_type}")

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Suppress Optuna's verbose output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        lambda t: optimize_model(
            t, model_type, X_train, y_train, X_val, y_val, scale_pos_weight, config
        ),
        n_trials=config.OPTUNA_TRIALS,
        show_progress_bar=True
    )

    logging.info(f"{model_type} optimization complete. Best MCC: {study.best_value:.4f}")

    return study.best_params


# ==================== MODEL TRAINING & EVALUATION ====================

def train_and_evaluate(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dirs: Dict[str, Path],
    embedding_tag: str,
    config: Config
) -> np.ndarray:
    """Train and evaluate a model with comprehensive error handling"""

    logging.info(f"Training {name} for {embedding_tag}")

    try:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "Accuracy": float(np.round(accuracy_score(y_test, y_pred), 3)),
            "AUC": float(np.round(roc_auc_score(y_test, y_proba), 3)),
            "Precision": float(np.round(precision_score(y_test, y_pred, average="macro"), 3)),
            "Recall": float(np.round(recall_score(y_test, y_pred, average="macro"), 3)),
            "F1_macro": float(np.round(f1_score(y_test, y_pred, average="macro"), 3)),
            "F1_micro": float(np.round(f1_score(y_test, y_pred, average="micro"), 3)),
            "MCC": float(np.round(matthews_corrcoef(y_test, y_pred), 3))
        }

        logging.info(f"{embedding_tag} | {name} | {metrics}")

        # Save model and metrics with error handling
        model_path = output_dirs['models'] / f"{embedding_tag}_{name}.pkl"
        metrics_path = output_dirs['metrics'] / f"{embedding_tag}_{name}_metrics.npy"

        try:
            joblib.dump(model, model_path)
            np.save(metrics_path, metrics)
            logging.info(f"Saved model to {model_path}")
        except Exception as e:
            logging.error(f"Failed to save model/metrics: {e}")
            raise

        return y_proba

    except Exception as e:
        logging.error(f"Error training {name}: {e}")
        raise


# ==================== VISUALIZATION ====================

@contextlib.contextmanager
def plot_context():
    """Context manager for matplotlib plots to ensure cleanup"""
    try:
        yield
    finally:
        plt.close('all')


def create_roc_curve(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    output_path: Path,
    emb_name: str,
    model_colors: Dict[str, str]
) -> None:
    """Create and save ROC curve with error handling"""

    with plot_context():
        try:
            plt.figure(figsize=(8, 6))

            for model_name, y_proba in predictions.items():
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc_score = roc_auc_score(y_test, y_proba)
                plt.plot(
                    fpr, tpr,
                    label=f"{model_name} (AUC={auc_score:.3f})",
                    color=model_colors[model_name],
                    linewidth=2
                )

            plt.plot([0, 1], [0, 1], "k--", linewidth=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"BACE ROC Curves - {emb_name}")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved ROC curve to {output_path}")

        except Exception as e:
            logging.error(f"Failed to create ROC curve: {e}")
            raise


def create_pr_curve(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    output_path: Path,
    emb_name: str,
    model_colors: Dict[str, str]
) -> None:
    """Create and save Precision-Recall curve with error handling"""

    with plot_context():
        try:
            plt.figure(figsize=(8, 6))

            for model_name, y_proba in predictions.items():
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                ap = average_precision_score(y_test, y_proba)
                plt.plot(
                    recall, precision,
                    label=f"{model_name} (AP={ap:.3f})",
                    color=model_colors[model_name],
                    linewidth=2
                )

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"BACE Precision-Recall Curves - {emb_name}")
            plt.legend(loc="best")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved PR curve to {output_path}")

        except Exception as e:
            logging.error(f"Failed to create PR curve: {e}")
            raise


# ==================== MAIN EXECUTION ====================

def process_embedding(
    emb_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale_pos_weight: float,
    config: Config,
    logger: logging.Logger
) -> None:
    """Process a single embedding model"""

    logger.info(f"{'='*20} EMBEDDING: {emb_name} {'='*15}")

    # Create output directories
    emb_output_root = config.OUTPUT_ROOT / emb_name
    output_dirs = {
        'roc': emb_output_root / 'ROC_Curves',
        'pr': emb_output_root / 'PR_Curves',
        'models': emb_output_root / 'models',
        'metrics': emb_output_root / 'metrics'
    }

    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Extract embeddings
    try:
        X_train, y_train = extract_embeddings(train_df, f"{emb_name}_embeddings", config.LABEL_COL)
        X_val, y_val = extract_embeddings(val_df, f"{emb_name}_embeddings", config.LABEL_COL)
        X_test, y_test = extract_embeddings(test_df, f"{emb_name}_embeddings", config.LABEL_COL)
    except Exception as e:
        logger.error(f"Failed to extract embeddings for {emb_name}: {e}")
        return

    logger.info(
        f"[{config.LABEL_COL} | {emb_name}] Shapes | "
        f"Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
    )

    # Sanitize data (in-place for memory efficiency)
    X_train = np.nan_to_num(X_train, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    X_val = np.nan_to_num(X_val, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)

    # Optimize hyperparameters
    best_params = {
        "XGBoost": run_optimization("xgb", X_train, y_train, X_val, y_val, scale_pos_weight, config),
        "LightGBM": run_optimization("lgb", X_train, y_train, X_val, y_val, scale_pos_weight, config),
        "CatBoost": run_optimization("cb", X_train, y_train, X_val, y_val, scale_pos_weight, config)
    }

    # Save best parameters
    params_path = output_dirs['metrics'] / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=4)

    # Train models
    predictions = {}

    predictions["XGBoost"] = train_and_evaluate(
        "XGBoost",
        xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=config.RANDOM_SEED,
            tree_method="hist",
            missing=np.nan,
            n_jobs=config.N_JOBS,
            **best_params["XGBoost"]
        ),
        X_train, y_train, X_test, y_test,
        output_dirs, emb_name, config
    )

    predictions["LightGBM"] = train_and_evaluate(
        "LightGBM",
        lgb.LGBMClassifier(
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=config.N_JOBS,
            verbose=-1,
            **best_params["LightGBM"]
        ),
        X_train, y_train, X_test, y_test,
        output_dirs, emb_name, config
    )

    predictions["CatBoost"] = train_and_evaluate(
        "CatBoost",
        cb.CatBoostClassifier(
            auto_class_weights="Balanced",
            loss_function="Logloss",
            random_seed=config.RANDOM_SEED,
            verbose=0,
            thread_count=config.N_JOBS,
            **best_params["CatBoost"]
        ),
        X_train, y_train, X_test, y_test,
        output_dirs, emb_name, config
    )

    # Create visualizations
    create_roc_curve(
        predictions, y_test,
        output_dirs['roc'] / "roc_all_models.pdf",
        emb_name, config.MODEL_COLORS
    )

    create_pr_curve(
        predictions, y_test,
        output_dirs['pr'] / "pr_all_models.pdf",
        emb_name, config.MODEL_COLORS
    )

    logger.info(f"Completed BACE modeling for embedding: {emb_name}")


def main():
    """Main execution function"""

    # Initialize configuration
    try:
        config = Config()
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Setup logging
    logger = setup_logging(config.LOG_DIR)
    logger.info("="*60)
    logger.info("BACE FT-Model-Embedding Modeling Pipeline (REFACTORED)")
    logger.info("="*60)
    logger.info(f"Configuration: N_JOBS={config.N_JOBS}, OPTUNA_TRIALS={config.OPTUNA_TRIALS}")

    # Load data
    try:
        train_df = load_split("train", config.BASE_MODEL_ROOT)
        val_df = load_split("eval", config.BASE_MODEL_ROOT)
        test_df = load_split("test", config.BASE_MODEL_ROOT)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Calculate class imbalance
    y_train_full = train_df[config.LABEL_COL].astype(int)
    n_pos = (y_train_full == 1).sum()
    n_neg = (y_train_full == 0).sum()
    scale_pos_weight = n_neg / n_pos

    logger.info(
        f"BACE class distribution | 0: {n_neg}, 1: {n_pos}, "
        f"scale_pos_weight={scale_pos_weight:.3f}"
    )

    # Process each embedding
    for emb_name in config.EMBEDDING_MODELS:
        try:
            process_embedding(
                emb_name, train_df, val_df, test_df,
                scale_pos_weight, config, logger
            )
        except Exception as e:
            logger.error(f"Failed to process embedding {emb_name}: {e}", exc_info=True)
            continue

    logger.info("All BACE base-model embedding experiments completed successfully.")


if __name__ == "__main__":
    main()
