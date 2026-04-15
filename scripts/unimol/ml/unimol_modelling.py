"""
Uni-Mol Embedding Modelling Pipeline

Trains XGBoost / LightGBM / CatBoost classifiers on Uni-Mol LoRA embeddings
(UniMol_FL or UniMol_WL) for any of the four EffiChem datasets.

Embeddings are read from:
    EffiChem_Extras/unimol_embeddings/{DATASET}_Embeddings/{dataset}_{split}_embed.csv

Results are written to:
    results/unimol_modelling/{dataset}/UniMol_{FL|WL}/
        metrics/{clf}_metrics.json
        models/{clf}.pkl
        plots/roc_pr_curves.pdf

Usage:
    # Single dataset + single config
    python unimol_modelling.py --dataset bace --config fl
    python unimol_modelling.py --dataset flavor --config wl

    # All datasets × both configs
    python unimol_modelling.py --dataset all --config both

    # Control Optuna trials and CPU threads
    python unimol_modelling.py --dataset clintox --config fl --trials 30 --jobs 16
"""

import argparse
import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# ── Constants ──────────────────────────────────────────────────────────────────

EXTRAS_ROOT  = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
EMBED_ROOT   = EXTRAS_ROOT / "unimol_embeddings"
REPO_ROOT    = Path(__file__).resolve().parent.parent.parent  # EffChem-2.0/
# Mirrors the finetuned-embedding results layout used by the other ml-scripts
# e.g. results/finetuned/BACE_FT_Results/ for ChemBERTa/MolFormer experiments.
# Pattern here: results/finetuned/UniMol_{DATASET}_FT_Results/UniMol_{FL|WL}/
RESULTS_ROOT = REPO_ROOT / "results" / "finetuned"

DATASET_REGISTRY: Dict[str, Dict] = {
    "bace": {
        "embed_dir":   "BACE_Embeddings",
        "file_prefix": "bace",
        "label_col":   "Class",
        "multiclass":  False,
    },
    "bbbp": {
        "embed_dir":   "BBBP_Embeddings",
        "file_prefix": "bbbp",
        "label_col":   "p_np",
        "multiclass":  False,
    },
    "clintox": {
        "embed_dir":   "clintox_Embeddings",
        "file_prefix": "clintox",
        "label_col":   "FDA_APPROVED",
        "multiclass":  False,
    },
    "flavor": {
        "embed_dir":   "flavor_Embeddings",
        "file_prefix": "fart",       # actual filenames: fart_{train|eval|test}_embed.csv
        "label_col":   "Canonicalized Taste",
        "multiclass":  True,
    },
}

# Embedding column name in the CSV for each loss config
CONFIG_TO_COL: Dict[str, str] = {
    "fl": "UniMol_FL_embeddings",
    "wl": "UniMol_WL_embeddings",
}

CONFIG_TAG: Dict[str, str] = {
    "fl": "UniMol_FL",
    "wl": "UniMol_WL",
}

MODEL_COLORS: Dict[str, str] = {
    "XGBoost":  "tab:blue",
    "LightGBM": "tab:green",
    "CatBoost": "tab:red",
}


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, dataset: str, config: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"unimol_{dataset}_{config}_ml.log"

    logger = logging.getLogger(f"unimol.{dataset}.{config}")
    if logger.handlers:
        return logger  # already configured (happens when --config both)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(funcName)s | %(message)s"
    ))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Embedding Parsing ──────────────────────────────────────────────────────────

def safe_parse_embedding(s: str) -> Optional[np.ndarray]:
    """Parse a comma-separated or JSON embedding string into a float32 array."""
    if not isinstance(s, str) or not any(c.isdigit() for c in s):
        return None
    try:
        # Try JSON list first
        try:
            parsed = json.loads(s)
            arr = np.array(parsed, dtype=np.float32)
        except (json.JSONDecodeError, ValueError):
            s_clean = s.strip().lstrip("[").rstrip("]")
            arr = np.array([float(x) for x in s_clean.split(",") if x.strip()],
                           dtype=np.float32)

        if arr.ndim != 1 or len(arr) == 0:
            return None
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr
    except Exception:
        return None


def extract_embeddings(
    df: pd.DataFrame,
    emb_col: str,
    label_col: str,
    le: Optional[LabelEncoder] = None,
    fit_le: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse embeddings and return (X, y).

    For multiclass datasets pass a LabelEncoder; set fit_le=True on the
    training split and False on val/test splits.
    """
    if emb_col not in df.columns:
        raise ValueError(f"Column '{emb_col}' not found. Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    embeddings, valid_idx = [], []
    for i, s in enumerate(df[emb_col]):
        arr = safe_parse_embedding(str(s))
        if arr is not None:
            embeddings.append(arr)
            valid_idx.append(i)

    if not embeddings:
        raise ValueError(f"No valid embeddings in column '{emb_col}'")

    dims = {len(e) for e in embeddings}
    if len(dims) > 1:
        raise ValueError(f"Inconsistent embedding dims: {dims}")

    X = np.vstack(embeddings)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    raw_labels = df.iloc[valid_idx][label_col]
    if le is not None:
        y = le.fit_transform(raw_labels.astype(str)) if fit_le \
            else le.transform(raw_labels.astype(str))
    else:
        y = raw_labels.astype(int).values

    return X, y


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_splits(
    dataset: str,
    reg: Dict,
    emb_col: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, Optional[LabelEncoder]]:
    """Load train/val/test splits and extract embeddings."""
    embed_dir  = EMBED_ROOT / reg["embed_dir"]
    prefix     = reg["file_prefix"]
    label_col  = reg["label_col"]
    multiclass = reg["multiclass"]

    def _read(split: str) -> pd.DataFrame:
        path = embed_dir / f"{prefix}_{split}_embed.csv"
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")
        logger.info(f"  Loading {split} split from {path}")
        return pd.read_csv(path)

    train_df = _read("train")
    val_df   = _read("eval")
    test_df  = _read("test")

    le = LabelEncoder() if multiclass else None

    X_train, y_train = extract_embeddings(train_df, emb_col, label_col, le, fit_le=True)
    X_val,   y_val   = extract_embeddings(val_df,   emb_col, label_col, le, fit_le=False)
    X_test,  y_test  = extract_embeddings(test_df,  emb_col, label_col, le, fit_le=False)

    logger.info(
        f"  Shapes — train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, le


# ── Optuna Optimisation ────────────────────────────────────────────────────────

def _build_model(
    model_type: str,
    params: Dict,
    multiclass: bool,
    n_classes: int,
    scale_pos_weight: float,
    seed: int,
    n_jobs: int,
) -> Any:
    """Instantiate a classifier with given hyperparams."""
    if model_type == "xgb":
        if multiclass:
            return xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=seed,
                tree_method="hist",
                n_jobs=n_jobs,
                **params,
            )
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            tree_method="hist",
            missing=np.nan,
            n_jobs=n_jobs,
            **params,
        )

    if model_type == "lgb":
        if multiclass:
            return lgb.LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                class_weight="balanced",
                random_state=seed,
                n_jobs=n_jobs,
                verbose=-1,
                **params,
            )
        return lgb.LGBMClassifier(
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
            **params,
        )

    # catboost
    if multiclass:
        return cb.CatBoostClassifier(
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=seed,
            thread_count=n_jobs,
            verbose=0,
            **params,
        )
    return cb.CatBoostClassifier(
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=seed,
        thread_count=n_jobs,
        verbose=0,
        **params,
    )


def _suggest_params(trial: optuna.Trial) -> Dict:
    return {
        "max_depth":     trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "n_estimators":  trial.suggest_int("n_estimators", 100, 600),
    }


def optimize_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    multiclass: bool,
    n_classes: int,
    scale_pos_weight: float,
    n_trials: int,
    seed: int,
    n_jobs: int,
    logger: logging.Logger,
) -> Dict:
    """Run Optuna TPE search; returns best hyperparameters."""
    logger.info(f"  Optimising {model_type} ({n_trials} trials)…")

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        model  = _build_model(model_type, params, multiclass, n_classes,
                               scale_pos_weight, seed, n_jobs)
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(
            model, X_train, y_train,
            scoring=make_scorer(matthews_corrcoef),
            cv=cv, n_jobs=1,
        )
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"    Best CV MCC={study.best_value:.4f} params={study.best_params}")
    return study.best_params


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    multiclass: bool,
) -> Dict[str, float]:
    mcc      = float(matthews_corrcoef(y_test, y_pred))
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro",  zero_division=0))
    f1_micro = float(f1_score(y_test, y_pred, average="micro",  zero_division=0))
    prec     = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec      = float(recall_score(y_test, y_pred, average="macro",    zero_division=0))

    if not multiclass:
        try:
            auc  = float(roc_auc_score(y_test, y_proba[:, 1]))
        except ValueError:
            auc  = float("nan")
        try:
            aupr = float(average_precision_score(y_test, y_proba[:, 1]))
        except ValueError:
            aupr = float("nan")
    else:
        try:
            auc = float(roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="macro"
            ))
        except ValueError:
            auc = float("nan")
        n_cls     = y_proba.shape[1]
        aupr_vals = [
            average_precision_score((y_test == c).astype(int), y_proba[:, c])
            for c in range(n_cls)
            if (y_test == c).sum() > 0
        ]
        aupr = float(np.mean(aupr_vals)) if aupr_vals else float("nan")

    return dict(
        Accuracy=round(accuracy, 4),
        AUC=round(auc,      4),
        AUPR=round(aupr,    4),
        Precision=round(prec,    4),
        Recall=round(rec,     4),
        F1_macro=round(f1_macro, 4),
        F1_micro=round(f1_micro, 4),
        MCC=round(mcc,      4),
    )


# ── Training & Evaluation ──────────────────────────────────────────────────────

def train_and_evaluate(
    clf_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    multiclass: bool,
    output_dirs: Dict[str, Path],
    config_tag: str,
    logger: logging.Logger,
) -> np.ndarray:
    """Fit model, compute metrics, persist model + metrics. Returns y_proba."""
    logger.info(f"  Training {clf_name}…")

    # For multiclass XGBoost, pass sample weights to balance classes
    fit_kwargs: Dict = {}
    if multiclass and isinstance(model, xgb.XGBClassifier):
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)

    model.fit(X_train, y_train, **fit_kwargs)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)           # (N, n_classes)

    metrics = compute_metrics(y_test, y_pred, y_proba, multiclass)
    logger.info(f"  {clf_name} | {metrics}")

    # Persist
    joblib.dump(model, output_dirs["models"] / f"{clf_name}.pkl")
    with open(output_dirs["metrics"] / f"{clf_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return y_proba


# ── Visualisation ──────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _plot_ctx():
    try:
        yield
    finally:
        plt.close("all")


def create_roc_curve(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    multiclass: bool,
    output_path: Path,
    dataset: str,
    config_tag: str,
) -> None:
    with _plot_ctx():
        plt.figure(figsize=(8, 6))
        for clf_name, y_proba in predictions.items():
            color = MODEL_COLORS.get(clf_name, "gray")
            if not multiclass:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                # Macro-average OvR
                n_cls = y_proba.shape[1]
                fpr_all, tpr_all = [], []
                for c in range(n_cls):
                    f, t, _ = roc_curve((y_test == c).astype(int), y_proba[:, c])
                    fpr_all.append(f); tpr_all.append(t)
                mean_fpr = np.linspace(0, 1, 200)
                mean_tpr = np.mean(
                    [np.interp(mean_fpr, f, t) for f, t in zip(fpr_all, tpr_all)], axis=0
                )
                fpr, tpr = mean_fpr, mean_tpr
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            plt.plot(fpr, tpr, label=f"{clf_name} (AUC={auc:.3f})",
                     color=color, linewidth=2)

        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{dataset.upper()} ROC Curves — {config_tag}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")


def create_pr_curve(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    multiclass: bool,
    output_path: Path,
    dataset: str,
    config_tag: str,
) -> None:
    with _plot_ctx():
        plt.figure(figsize=(8, 6))
        for clf_name, y_proba in predictions.items():
            color = MODEL_COLORS.get(clf_name, "gray")
            if not multiclass:
                prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba[:, 1])
                ap = average_precision_score(y_test, y_proba[:, 1])
                plt.plot(rec_arr, prec_arr, label=f"{clf_name} (AP={ap:.3f})",
                         color=color, linewidth=2)
            else:
                # Macro-average over classes
                n_cls = y_proba.shape[1]
                ap_vals = []
                for c in range(n_cls):
                    y_bin = (y_test == c).astype(int)
                    if y_bin.sum() == 0:
                        continue
                    p_c, r_c, _ = precision_recall_curve(y_bin, y_proba[:, c])
                    ap_vals.append(average_precision_score(y_bin, y_proba[:, c]))
                ap = float(np.mean(ap_vals)) if ap_vals else float("nan")
                plt.plot([], [], label=f"{clf_name} (macro AP={ap:.3f})",
                         color=color, linewidth=2)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{dataset.upper()} Precision-Recall Curves — {config_tag}")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(
    dataset: str,
    config: str,
    n_trials: int,
    seed: int,
    n_jobs: int,
) -> None:
    reg        = DATASET_REGISTRY[dataset]
    multiclass = reg["multiclass"]
    emb_col    = CONFIG_TO_COL[config]
    tag        = CONFIG_TAG[config]

    # e.g. results/finetuned/UniMol_BACE_FT_Results/UniMol_FL/
    out_root = RESULTS_ROOT / f"UniMol_{dataset.upper()}_FT_Results" / tag
    log_dir  = out_root / "logs"
    output_dirs = {
        "models":  out_root / "models",
        "metrics": out_root / "metrics",
        "plots":   out_root / "plots",
    }
    for d in [log_dir, *output_dirs.values()]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir, dataset, config)
    logger.info("=" * 60)
    logger.info(f"Uni-Mol Modelling | dataset={dataset.upper()} | config={tag}")
    logger.info("=" * 60)

    # ── Load data ──
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, le = load_splits(
            dataset, reg, emb_col, logger
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Run the embedding extraction step first (run_unimol_embeddings.sh).")
        return

    n_classes = int(y_train.max()) + 1 if multiclass else 2

    # Class imbalance factor for binary XGBoost
    if not multiclass:
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = n_neg / n_pos if n_pos else 1.0
        logger.info(f"  Class dist — 0:{n_neg}, 1:{n_pos}, "
                    f"scale_pos_weight={scale_pos_weight:.3f}")
    else:
        scale_pos_weight = 1.0
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"  Class dist — {dict(zip(unique.tolist(), counts.tolist()))}")

    # ── Hyperparameter search ──
    model_types = {"XGBoost": "xgb", "LightGBM": "lgb", "CatBoost": "cb"}
    best_params: Dict[str, Dict] = {}
    for clf_name, mt in model_types.items():
        best_params[clf_name] = optimize_model(
            mt, X_train, y_train, multiclass, n_classes,
            scale_pos_weight, n_trials, seed, n_jobs, logger,
        )

    (output_dirs["metrics"] / "best_params.json").write_text(
        json.dumps(best_params, indent=4)
    )

    # ── Train final models ──
    predictions: Dict[str, np.ndarray] = {}
    for clf_name, mt in model_types.items():
        model = _build_model(
            mt, best_params[clf_name], multiclass, n_classes,
            scale_pos_weight, seed, n_jobs,
        )
        predictions[clf_name] = train_and_evaluate(
            clf_name, model,
            X_train, y_train, X_test, y_test,
            multiclass, output_dirs, tag, logger,
        )

    # ── Plots ──
    create_roc_curve(
        predictions, y_test, multiclass,
        output_dirs["plots"] / "roc_curves.pdf",
        dataset, tag,
    )
    create_pr_curve(
        predictions, y_test, multiclass,
        output_dirs["plots"] / "pr_curves.pdf",
        dataset, tag,
    )

    logger.info(f"Done. Results saved to {out_root}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Uni-Mol embedding modelling: XGBoost / LightGBM / CatBoost"
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_REGISTRY.keys(), "all"],
        default="all",
        help="Dataset to run (default: all)",
    )
    parser.add_argument(
        "--config",
        choices=["fl", "wl", "both"],
        default="both",
        help="Embedding config to use — fl (Focal Loss), wl (Weighted Loss), or both (default: both)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=int(os.getenv("OPTUNA_TRIALS", "20")),
        help="Optuna trials per model (default: 20)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.getenv("N_JOBS", str(min(os.cpu_count() or 1, 60)))),
        help="CPU threads (default: min(cpu_count, 60))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("RANDOM_SEED", "42")),
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    datasets = list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    configs  = ["fl", "wl"] if args.config == "both" else [args.config]

    for dataset in datasets:
        for config in configs:
            try:
                run_pipeline(
                    dataset=dataset,
                    config=config,
                    n_trials=args.trials,
                    seed=args.seed,
                    n_jobs=args.jobs,
                )
            except Exception as e:
                logging.getLogger("unimol").error(
                    f"Pipeline failed for {dataset}/{config}: {e}", exc_info=True
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
