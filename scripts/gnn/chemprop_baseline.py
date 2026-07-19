"""
Chemprop (D-MPNN) GNN Baseline

Addresses editor comment (see manuscript revision notes, Section 2 of
editor_response_suggestions.md): PEARL's manuscript only ever cited Chemprop
results *from the original Chemprop paper* (different splits, different
protocol) rather than running it in-house on PEARL's own scaffold splits. This
script trains and evaluates Chemprop directly on the same data/clean/ splits
used by every other PEARL baseline (CLM E2E LoRA, PC-only, RAFE), for all 8
datasets (the original 4 MoleculeNet-era sets plus the 4 new TDC ADMET sets),
so the "does a graph model beat/match the CLM pipeline" comparison is finally
a controlled, in-house experiment instead of a cross-paper citation.

Pipeline per dataset:
1. Write temp 2-column (SMILES, label) CSVs for train/valid/test -- label-
   encoded for multiclass (Flavor), target-transformed for skewed regression
   targets (Half_Life_Obach) -- so chemprop always sees a clean numeric target.
2. Optuna search (default 20 trials, matching the tree-classifier baselines'
   rigor) over depth/hidden_size/ffn_num_layers/
   dropout, each trial a short chemprop_train run; validation metric parsed
   from chemprop's own stdout.
3. Final chemprop_train run with the best config and more epochs, evaluated on
   the held-out test split via --separate_test_path.
4. Predictions are read back from chemprop's test_preds.csv (row order is
   preserved from the input CSV) and scored with the *same* sklearn metric
   functions used by pc_only_modelling.py, so numbers are directly comparable
   across baselines rather than trusting chemprop's own printed metrics.

Usage:
    python chemprop_baseline.py --dataset {bace,bbbp,clintox,flavor,herg,dili,caco2,half_life,all}
"""

import os
import re
import json
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    matthews_corrcoef, mean_squared_error, mean_absolute_error, r2_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEAN_ROOT = BASE_DIR / "data" / "clean"
TMP_ROOT = BASE_DIR / "data" / "gnn_tmp" / "chemprop"
OUTPUT_ROOT = BASE_DIR / "results" / "gnn" / "chemprop"
# Note: the original PEARL_EXTRAS path (/export/cse/rmall/Raghvendra/EffiChem_Extras)
# was found to be read-only from this host; new GNN-baseline artifacts go to a
# separate, confirmed-writable location instead. Override with $PEARL_EXTRAS_V2.
PEARL_EXTRAS = Path(os.getenv("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
OPTUNA_TRIALS = int(os.getenv("CHEMPROP_OPTUNA_TRIALS", "20"))
SEARCH_EPOCHS = int(os.getenv("CHEMPROP_SEARCH_EPOCHS", "20"))
FINAL_EPOCHS = int(os.getenv("CHEMPROP_FINAL_EPOCHS", "50"))
GPU_ID = os.getenv("CHEMPROP_GPU", "0")

SPLITS = ["train", "valid", "test"]

DATASET_CONFIG = {
    "bace": {"clean_dir": CLEAN_ROOT / "bace_datasets", "smiles_col": "Standardized SMILES",
             "label_col": "Class", "task": "binary"},
    "bbbp": {"clean_dir": CLEAN_ROOT / "bbbp_datasets", "smiles_col": "Standardized SMILES",
             "label_col": "p_np", "task": "binary"},
    "clintox": {"clean_dir": CLEAN_ROOT / "clintox_datasets", "smiles_col": "Standardized SMILES",
                "label_col": "FDA_APPROVED", "task": "binary"},
    "flavor": {"clean_dir": CLEAN_ROOT / "flavor_datasets", "smiles_col": "Standardized SMILES",
               "label_col": "Canonicalized Taste", "task": "multiclass"},
    "herg": {"clean_dir": CLEAN_ROOT / "herg_datasets", "smiles_col": "Standardized SMILES",
             "label_col": "hERG_Inhib", "task": "binary"},
    "dili": {"clean_dir": CLEAN_ROOT / "dili_datasets", "smiles_col": "Standardized SMILES",
             "label_col": "DILI_Label", "task": "binary"},
    "caco2": {"clean_dir": CLEAN_ROOT / "caco2_datasets", "smiles_col": "Standardized SMILES",
              "label_col": "Caco2_LogPapp", "task": "regression", "target_transform": None},
    "half_life": {"clean_dir": CLEAN_ROOT / "half_life_datasets", "smiles_col": "Standardized SMILES",
                  "label_col": "Half_Life_Hours", "task": "regression", "target_transform": "log1p"},
}

CHEMPROP_SMILES_COL = "smiles"
CHEMPROP_LABEL_COL = "target"


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(log_dir / "chemprop_baseline.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def apply_target_transform(y: np.ndarray, transform: Optional[str]) -> np.ndarray:
    return np.log1p(y) if transform == "log1p" else y


def invert_target_transform(y: np.ndarray, transform: Optional[str]) -> np.ndarray:
    return np.expm1(y) if transform == "log1p" else y


def prepare_chemprop_splits(dataset: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Write temp 2-column CSVs chemprop can consume; returns encoding metadata."""
    tmp_dir = TMP_ROOT / dataset
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw = {s: pd.read_csv(cfg["clean_dir"] / f"{s}_clean.csv") for s in SPLITS}

    label_encoder = None
    n_classes = 0
    if cfg["task"] == "multiclass":
        label_encoder = LabelEncoder()
        label_encoder.fit(pd.concat([raw[s][cfg["label_col"]] for s in SPLITS]))
        n_classes = len(label_encoder.classes_)

    paths = {}
    for s in SPLITS:
        df = raw[s]
        smiles = df[cfg["smiles_col"]]
        if cfg["task"] == "multiclass":
            label = label_encoder.transform(df[cfg["label_col"]])
        elif cfg["task"] == "regression":
            label = apply_target_transform(df[cfg["label_col"]].astype(float).values, cfg.get("target_transform"))
        else:
            label = df[cfg["label_col"]].astype(int).values

        out_df = pd.DataFrame({CHEMPROP_SMILES_COL: smiles.values, CHEMPROP_LABEL_COL: label})
        out_path = tmp_dir / f"{s}.csv"
        out_df.to_csv(out_path, index=False)
        paths[s] = out_path

    return {"paths": paths, "label_encoder": label_encoder, "n_classes": n_classes,
            "y_test_orig": raw["test"][cfg["label_col"]].values}


def chemprop_dataset_type(task: str) -> str:
    return {"binary": "classification", "multiclass": "multiclass", "regression": "regression"}[task]


def chemprop_metrics(task: str) -> Dict[str, Any]:
    if task == "binary":
        return {"metric": "mcc", "extra": ["auc", "accuracy", "f1"], "direction": "maximize"}
    if task == "multiclass":
        return {"metric": "cross_entropy", "extra": ["accuracy", "f1"], "direction": "minimize"}
    return {"metric": "rmse", "extra": ["mae", "r2"], "direction": "minimize"}


def run_chemprop_train(splits: Dict[str, Any], cfg: Dict[str, Any], save_dir: Path,
                        hparams: Dict[str, Any], epochs: int) -> Optional[str]:
    """Runs chemprop_train as a subprocess; returns raw stdout+stderr text."""
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_cfg = chemprop_metrics(cfg["task"])

    cmd = [
        "chemprop_train",
        "--data_path", str(splits["paths"]["train"]),
        "--separate_val_path", str(splits["paths"]["valid"]),
        "--separate_test_path", str(splits["paths"]["test"]),
        "--smiles_columns", CHEMPROP_SMILES_COL,
        "--target_columns", CHEMPROP_LABEL_COL,
        "--dataset_type", chemprop_dataset_type(cfg["task"]),
        "--metric", metrics_cfg["metric"],
        "--extra_metrics", *metrics_cfg["extra"],
        "--epochs", str(epochs),
        "--depth", str(hparams["depth"]),
        "--hidden_size", str(hparams["hidden_size"]),
        "--ffn_num_layers", str(hparams["ffn_num_layers"]),
        "--dropout", str(hparams["dropout"]),
        "--save_dir", str(save_dir),
        "--gpu", GPU_ID,
        "--seed", str(RANDOM_SEED),
        "--pytorch_seed", str(RANDOM_SEED),
        "--save_preds",
        "--quiet",
    ]
    if cfg["task"] == "multiclass":
        cmd += ["--multiclass_num_classes", str(splits["n_classes"])]
    if cfg["task"] == "binary":
        cmd += ["--class_balance"]

    # cwd pinned explicitly so behavior is identical regardless of the caller's
    # own working directory (e.g. invoked from repo root or from scripts/gnn/ itself)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
    if result.returncode != 0:
        logging.error(f"chemprop_train failed: {result.stderr[-3000:]}")
        return None
    return result.stdout + result.stderr


def parse_validation_score(log_text: str, metric: str) -> Optional[float]:
    match = re.search(rf"best validation {re.escape(metric)} = ([\-\d.]+)", log_text)
    return float(match.group(1)) if match else None


def optuna_objective(trial: optuna.Trial, splits: Dict[str, Any], cfg: Dict[str, Any], trial_root: Path) -> float:
    hparams = {
        "depth": trial.suggest_int("depth", 2, 5),
        "hidden_size": trial.suggest_categorical("hidden_size", [300, 600, 900]),
        "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }
    metrics_cfg = chemprop_metrics(cfg["task"])
    save_dir = trial_root / f"trial_{trial.number}"
    log_text = run_chemprop_train(splits, cfg, save_dir, hparams, SEARCH_EPOCHS)
    shutil.rmtree(save_dir, ignore_errors=True)  # trial checkpoints are disposable

    if log_text is None:
        raise optuna.exceptions.TrialPruned()
    score = parse_validation_score(log_text, metrics_cfg["metric"])
    if score is None:
        raise optuna.exceptions.TrialPruned()
    return score if metrics_cfg["direction"] == "maximize" else -score


def move_large_files_to_extras(save_dir: Path, dataset: str) -> None:
    """Move any >50MB artifact to $PEARL_EXTRAS_V2 (see path note above), leaving a symlink behind.

    Defensive by design: the original PEARL_EXTRAS path was found to be
    read-only from this host, so failures here must not crash the whole run --
    large files are simply left in place (with a loud warning) if the move fails.
    """
    for f in save_dir.rglob("*"):
        if not f.is_file() or f.stat().st_size <= LARGE_FILE_THRESHOLD_BYTES:
            continue
        size_mb = f.stat().st_size / 1e6
        try:
            dest_dir = PEARL_EXTRAS / "gnn_checkpoints" / "chemprop" / dataset
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f.name
            shutil.move(str(f), str(dest))
            f.symlink_to(dest)
            logging.info(f"Moved large file ({size_mb:.1f}MB) to {dest}, left symlink at {f}")
        except OSError as e:
            logging.warning(
                f"Could not move large file ({size_mb:.1f}MB) at {f} to PEARL_EXTRAS "
                f"({PEARL_EXTRAS}): {e}. Left in place under results/ -- PEARL_EXTRAS may "
                f"be read-only from this host; move it manually once write access is available."
            )


def compute_metrics(task: str, y_test_orig: np.ndarray, y_pred: np.ndarray,
                     n_classes: int, target_transform: Optional[str]) -> Dict[str, float]:
    if task == "regression":
        y_pred_orig = invert_target_transform(y_pred, target_transform)
        return {
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))), 4),
            "MAE": round(float(mean_absolute_error(y_test_orig, y_pred_orig)), 4),
            "R2": round(float(r2_score(y_test_orig, y_pred_orig)), 4),
            "Spearman": round(float(spearmanr(y_test_orig, y_pred_orig).correlation), 4),
        }

    if task == "multiclass":
        y_pred_class = y_pred.argmax(axis=1)
        # test_preds.csv for multiclass gives per-class probabilities in class-index columns
        auc = roc_auc_score(y_test_orig, y_pred, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    else:
        y_pred_class = (y_pred >= 0.5).astype(int)
        auc = roc_auc_score(y_test_orig, y_pred)

    return {
        "Accuracy": round(accuracy_score(y_test_orig, y_pred_class), 3),
        "AUC": round(float(auc), 3),
        "Precision": round(precision_score(y_test_orig, y_pred_class, average="macro", zero_division=0), 3),
        "Recall": round(recall_score(y_test_orig, y_pred_class, average="macro", zero_division=0), 3),
        "F1_macro": round(f1_score(y_test_orig, y_pred_class, average="macro"), 3),
        "F1_micro": round(f1_score(y_test_orig, y_pred_class, average="micro"), 3),
        "MCC": round(matthews_corrcoef(y_test_orig, y_pred_class), 3),
    }


def read_predictions(save_dir: Path, task: str, n_classes: int) -> np.ndarray:
    preds_path = save_dir / "test_preds.csv"
    df = pd.read_csv(preds_path)
    value_cols = [c for c in df.columns if c != CHEMPROP_SMILES_COL]
    if task == "multiclass":
        # chemprop writes a single column holding the stringified per-class probability list
        return np.array([json.loads(v) for v in df[value_cols[0]]])
    return df[value_cols[0]].values


def run_dataset(dataset: str) -> Dict[str, Any]:
    cfg = DATASET_CONFIG[dataset]
    out_dir = OUTPUT_ROOT / f"{dataset.upper()}_Chemprop_Results"
    setup_logging(out_dir / "logs")

    logging.info("=" * 80)
    logging.info(f"Chemprop baseline: {dataset} (task={cfg['task']})")
    logging.info("=" * 80)

    splits = prepare_chemprop_splits(dataset, cfg)

    trial_root = TMP_ROOT / dataset / "optuna_trials"
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda t: optuna_objective(t, splits, cfg, trial_root),
        n_trials=OPTUNA_TRIALS,
    )
    shutil.rmtree(trial_root, ignore_errors=True)

    best_hparams = study.best_params
    logging.info(f"Best hyperparameters: {best_hparams} (val score {study.best_value:.4f})")
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics" / "best_params.json", "w") as f:
        json.dump(best_hparams, f, indent=2)

    final_save_dir = out_dir / "final_model"
    log_text = run_chemprop_train(splits, cfg, final_save_dir, best_hparams, FINAL_EPOCHS)
    if log_text is None:
        raise RuntimeError(f"Final chemprop_train run failed for {dataset}")

    y_pred = read_predictions(final_save_dir, cfg["task"], splits["n_classes"])
    y_test_orig = splits["y_test_orig"]
    if cfg["task"] == "multiclass":
        y_test_orig = splits["label_encoder"].transform(y_test_orig)

    metrics = compute_metrics(cfg["task"], y_test_orig, y_pred, splits["n_classes"], cfg.get("target_transform"))
    logging.info(f"[{dataset} | Chemprop] metrics: {metrics}")
    with open(out_dir / "metrics" / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    move_large_files_to_extras(final_save_dir, dataset)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Chemprop (D-MPNN) GNN baseline, in-house on PEARL splits")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()) + ["all"], default="all")
    args = parser.parse_args()

    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    summary = {}
    for dataset in datasets:
        try:
            summary[dataset] = run_dataset(dataset)
        except Exception as e:
            logging.error(f"Failed dataset {dataset}: {e}")
            raise

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [{"Dataset": dataset, **metrics} for dataset, metrics in summary.items()]
    new_df = pd.DataFrame(rows)
    summary_path = OUTPUT_ROOT / "chemprop_summary.csv"
    if summary_path.exists():
        existing_df = pd.read_csv(summary_path)
        existing_df = existing_df[~existing_df["Dataset"].isin(datasets)]
        summary_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        summary_df = new_df
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
