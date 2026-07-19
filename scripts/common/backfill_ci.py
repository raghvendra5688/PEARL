"""
Backfill bootstrapped 95% CIs (matching PEARL_paper.tex's exact protocol -- see
bootstrap_ci.py) onto PC-only and Chemprop results that were already produced
without them. Does NOT retrain anything: reloads saved models/predictions and
recomputes predictions or reads them from disk, which is cheap.

- PC-only: reloads results/pc_only/{DATASET}_PC_Only_Results/models/{model}.pkl
  + the cached test feature CSV, re-predicts (fast, no retraining), computes CI.
- Chemprop: reloads results/gnn/chemprop/{DATASET}_Chemprop_Results/final_model/
  test_preds.csv (already-saved predictions) + the original test_clean.csv labels.

Writes a `ci_metrics.json` alongside each existing `metrics.json` /
`test_metrics.json`, and prints a combined comparison table.

Usage:
    python backfill_ci.py --which {pc_only,chemprop,all} --dataset {name,all}
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef, roc_auc_score, r2_score
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts" / "smiles" / "ml"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "gnn"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "common"))

import pc_only_modelling as pcm  # noqa: E402
import chemprop_baseline as cpb  # noqa: E402
from bootstrap_ci import bootstrap_ci  # noqa: E402

PC_ONLY_RESULTS = BASE_DIR / "results" / "pc_only"
CHEMPROP_RESULTS = BASE_DIR / "results" / "gnn" / "chemprop"


def spearman_metric(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation


def backfill_pc_only(dataset: str) -> dict:
    cfg = pcm.DATASET_CONFIG[dataset]
    task = cfg["task"]
    out_dir = PC_ONLY_RESULTS / f"{dataset.upper()}_PC_Only_Results"
    if not (out_dir / "models").exists():
        print(f"[pc_only] {dataset}: no models found, skipping")
        return {}

    test_df = pcm.load_or_compute_features(dataset, "test", cfg)
    if task == "multiclass":
        train_df = pcm.load_or_compute_features(dataset, "train", cfg)
        valid_df = pcm.load_or_compute_features(dataset, "valid", cfg)
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(pd.concat([train_df[cfg["label_col"]], valid_df[cfg["label_col"]], test_df[cfg["label_col"]]]))
        y_test = label_encoder.transform(test_df[cfg["label_col"]])
        n_classes = len(label_encoder.classes_)
    elif task == "regression":
        y_test = test_df[cfg["label_col"]].astype(float).values
        n_classes = 0
    else:
        y_test = test_df[cfg["label_col"]].astype(int).values
        n_classes = 2

    feature_cols = [c for c in test_df.columns if c != cfg["label_col"]]
    X_test = test_df[feature_cols].values.astype(np.float32)

    results = {}
    for model_name in ["XGBoost", "LightGBM", "CatBoost"]:
        model_path = out_dir / "models" / f"{model_name}.pkl"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        if task == "regression":
            y_pred_orig = pcm.invert_target_transform(y_pred, cfg.get("target_transform"))
            ci = {
                "R2": bootstrap_ci(y_test, y_pred_orig, r2_score, stratified=False),
                "Spearman": bootstrap_ci(y_test, y_pred_orig, spearman_metric, stratified=False),
            }
        else:
            ci = {"MCC": bootstrap_ci(y_test, y_pred, matthews_corrcoef, stratified=True)}
            y_proba = model.predict_proba(X_test)
            if task == "binary":
                ci["AUC"] = bootstrap_ci(y_test, y_proba[:, 1], roc_auc_score, stratified=True)

        with open(out_dir / "metrics" / f"{model_name}_ci_metrics.json", "w") as f:
            json.dump(ci, f, indent=2)
        results[model_name] = ci
        print(f"[pc_only] {dataset} | {model_name}: {ci}")

    return results


def backfill_chemprop(dataset: str) -> dict:
    cfg = cpb.DATASET_CONFIG[dataset]
    task = cfg["task"]
    out_dir = CHEMPROP_RESULTS / f"{dataset.upper()}_Chemprop_Results"
    preds_path = out_dir / "final_model" / "test_preds.csv"
    if not preds_path.exists():
        print(f"[chemprop] {dataset}: no test_preds.csv found, skipping")
        return {}

    test_df = pd.read_csv(cfg["clean_dir"] / "test_clean.csv")
    y_test_raw = test_df[cfg["label_col"]].values

    n_classes = 0
    if task == "multiclass":
        train_df = pd.read_csv(cfg["clean_dir"] / "train_clean.csv")
        valid_df = pd.read_csv(cfg["clean_dir"] / "valid_clean.csv")
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(pd.concat([train_df[cfg["label_col"]], valid_df[cfg["label_col"]], test_df[cfg["label_col"]]]))
        y_test = label_encoder.transform(y_test_raw)
        n_classes = len(label_encoder.classes_)
    else:
        y_test = y_test_raw

    y_pred = cpb.read_predictions(out_dir / "final_model", task, n_classes)

    if task == "regression":
        y_pred_orig = cpb.invert_target_transform(y_pred, cfg.get("target_transform"))
        ci = {
            "R2": bootstrap_ci(y_test.astype(float), y_pred_orig, r2_score, stratified=False),
            "Spearman": bootstrap_ci(y_test.astype(float), y_pred_orig, spearman_metric, stratified=False),
        }
    elif task == "multiclass":
        y_pred_class = y_pred.argmax(axis=1)
        ci = {"MCC": bootstrap_ci(y_test, y_pred_class, matthews_corrcoef, stratified=True)}
    else:
        y_pred_class = (y_pred >= 0.5).astype(int)
        ci = {
            "MCC": bootstrap_ci(y_test, y_pred_class, matthews_corrcoef, stratified=True),
            "AUC": bootstrap_ci(y_test, y_pred, roc_auc_score, stratified=True),
        }

    with open(out_dir / "metrics" / "ci_metrics.json", "w") as f:
        json.dump(ci, f, indent=2)
    print(f"[chemprop] {dataset}: {ci}")
    return ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["pc_only", "chemprop", "all"], default="all")
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    all_datasets = list(pcm.DATASET_CONFIG.keys())
    datasets = all_datasets if args.dataset == "all" else [args.dataset]

    if args.which in ("pc_only", "all"):
        for ds in datasets:
            try:
                backfill_pc_only(ds)
            except Exception as e:
                print(f"[pc_only] {ds}: FAILED ({e})")

    if args.which in ("chemprop", "all"):
        for ds in datasets:
            try:
                backfill_chemprop(ds)
            except Exception as e:
                print(f"[chemprop] {ds}: FAILED ({e})")


if __name__ == "__main__":
    main()
