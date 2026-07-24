"""
Aggregate RAG Results — herg/dili/caco2/half_life

Collects all *_metrics.json (point estimates) and *_ci_metrics.json (bootstrap
CI, mean +/- SE) files produced by rag_modelling_new_datasets.py (HF) and
rag_modelling_unimol_new_datasets.py (Uni-Mol) for the 4 new TDC datasets, and
produces one combined CSV in the same mean/SE long format used throughout
this revision (results/summary/classification_mean_se_by_method.csv and
regression_mean_se_by_method.csv).

Expected result layout:
    results/rag/{DATASET}/{col_name}/metrics/{ml_model}_metrics.json
    results/rag/{DATASET}/{col_name}/metrics/{ml_model}_ci_metrics.json
    results/rag_unimol/{DATASET}/{col_name}/metrics/{ml_model}_metrics.json
    results/rag_unimol/{DATASET}/{col_name}/metrics/{ml_model}_ci_metrics.json

Output:
    results/rag/aggregated/new_datasets_rag_results.csv        (one row per model/embedding/dataset)
    results/rag/aggregated/new_datasets_rag_mean_se.csv         (mean/SE long format, best model per dataset+modality)

Usage:
    python aggregate_rag_results_new_datasets.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_ROOTS = {"hf": REPO_ROOT / "results" / "rag", "unimol": REPO_ROOT / "results" / "rag_unimol"}
OUT_DIR = REPO_ROOT / "results" / "rag" / "aggregated"

DATASETS = ["herg", "dili", "caco2", "half_life"]
RESULT_DIR_NAME = {"herg": "HERG", "dili": "DILI", "caco2": "CACO2", "half_life": "HALF_LIFE"}
TASK = {"herg": "classification", "dili": "classification", "caco2": "regression", "half_life": "regression"}
# "best" model selection metric per task -- Spearman for regression (not R2),
# matching the session-wide convention adopted for every other method.
SELECTION_METRIC = {"classification": "MCC", "regression": "Spearman"}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])


def load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load {path}: {e}")
        return None


def collect_records(modality: str, dataset: str) -> List[Dict]:
    dataset_dir = RESULTS_ROOTS[modality] / RESULT_DIR_NAME[dataset]
    if not dataset_dir.exists():
        return []

    records = []
    for metrics_file in sorted(dataset_dir.rglob("*_metrics.json")):
        if metrics_file.name in ("best_params.json",) or metrics_file.stem.endswith("_ci_metrics"):
            continue

        metrics = load_json(metrics_file)
        if metrics is None:
            continue

        embedding_model = metrics_file.parent.parent.name  # …/{col_name}/metrics/{ml_model}_metrics.json
        ml_model = metrics_file.stem[: -len("_metrics")]

        ci_file = metrics_file.parent / f"{ml_model}_ci_metrics.json"
        ci = load_json(ci_file) if ci_file.exists() else {}

        record: Dict = {
            "dataset": dataset, "modality": modality,
            "embedding_model": embedding_model, "ml_model": ml_model,
        }
        record.update(metrics)
        for metric_name, ci_vals in ci.items():
            record[f"{metric_name}_se"] = ci_vals.get("se")
            record[f"{metric_name}_ci_lo"] = ci_vals.get("ci_lo")
            record[f"{metric_name}_ci_hi"] = ci_vals.get("ci_hi")
        records.append(record)

    return records


def main() -> int:
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict] = []
    for dataset in DATASETS:
        for modality in ("hf", "unimol"):
            recs = collect_records(modality, dataset)
            logging.info(f"{dataset} / {modality}: {len(recs)} records")
            all_records.extend(recs)

    if not all_records:
        logging.error("No RAG results found for any new dataset -- run rag_modelling_(unimol_)new_datasets.py first.")
        return 1

    df = pd.DataFrame(all_records)
    full_path = OUT_DIR / "new_datasets_rag_results.csv"
    df.to_csv(full_path, index=False)
    logging.info(f"Saved {len(df)} rows -> {full_path}")

    # Best model+embedding per dataset x modality, by the task-appropriate metric
    mean_se_rows = []
    for dataset in DATASETS:
        task = TASK[dataset]
        sel_metric = SELECTION_METRIC[task]
        for modality in ("hf", "unimol"):
            sub = df[(df["dataset"] == dataset) & (df["modality"] == modality)]
            sub = sub[sub[sel_metric].notna()]
            if sub.empty:
                continue
            best = sub.loc[sub[sel_metric].idxmax()]
            metrics = ["MCC", "AUC"] if task == "classification" else ["R2", "Spearman"]
            for metric in metrics:
                if metric not in best or pd.isna(best[metric]):
                    continue
                mean_se_rows.append({
                    "Dataset": dataset.upper(),
                    "Modality": modality,
                    "EmbeddingModel": best["embedding_model"],
                    "MLModel": best["ml_model"],
                    "Metric": metric,
                    "Mean": best[metric],
                    "SE": best.get(f"{metric}_se"),
                })

    mean_se_df = pd.DataFrame(mean_se_rows)
    mean_se_path = OUT_DIR / "new_datasets_rag_mean_se.csv"
    mean_se_df.to_csv(mean_se_path, index=False)
    logging.info(f"Saved {len(mean_se_df)} rows -> {mean_se_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
