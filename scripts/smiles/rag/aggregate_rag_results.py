"""
Aggregate RAG Results

Collects all *_metrics.json files produced by the RAG modelling scripts
(rag_modelling_bace.py, rag_modelling_bbbp.py, rag_modelling_clintox.py,
rag_modelling_flavor.py) and produces one CSV per dataset.

Expected result layout:
    results/rag/{dataset}/{embedding_model}/metrics/{ml_model}_metrics.json

Output:
    results/rag/aggregated/{dataset}_rag_results.csv   (one per dataset)
    results/rag/aggregated/all_rag_results.csv          (combined)

Usage:
    python aggregate_rag_results.py
    python aggregate_rag_results.py --output-dir /some/other/dir
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "rag"

DATASETS = ["bace", "bbbp", "clintox", "flavor"]


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Core helpers ──────────────────────────────────────────────────────────────
def load_json_metrics(path: Path) -> Optional[Dict]:
    """Load a *_metrics.json file and return as dict, or None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load {path}: {e}")
        return None


def collect_dataset_records(dataset: str) -> List[Dict]:
    """
    Walk results/rag/{dataset}/**/metrics/*_metrics.json and build a list of
    flat record dicts ready for a DataFrame row.

    Supported path structures:
        Standard:  results/rag/{dataset}/{embedding_model}/metrics/{ml_model}_metrics.json
        ClinTox:   results/rag/{dataset}/{label}/{embedding_model}/metrics/{ml_model}_metrics.json

    The embedding model is always the directory immediately above "metrics/".
    An optional "label" column is added when an extra level exists (e.g. CT_TOX, FDA_APPROVED).
    """
    dataset_dir = RESULTS_ROOT / dataset
    if not dataset_dir.exists():
        logging.warning(f"  Directory does not exist: {dataset_dir}")
        return []

    records = []
    for metrics_file in sorted(dataset_dir.rglob("*_metrics.json")):
        if "best_params" in metrics_file.name:
            continue

        metrics = load_json_metrics(metrics_file)
        if metrics is None:
            continue

        # The metrics file lives inside a "metrics" directory.
        # Its parent is the embedding model folder; any directories between
        # dataset_dir and the embedding model folder are label sub-directories.
        metrics_dir   = metrics_file.parent          # …/metrics
        embedding_dir = metrics_dir.parent           # …/{embedding_model}
        embedding_model = embedding_dir.name

        # Collect any intermediate path parts as the label (empty for most datasets)
        try:
            rel_to_dataset = embedding_dir.relative_to(dataset_dir)
            label_parts = rel_to_dataset.parts[:-1]  # everything above embedding_model
            label = "/".join(label_parts) if label_parts else None
        except ValueError:
            label = None

        stem = metrics_file.stem  # e.g. "XGBoost_metrics"
        ml_model = stem[: -len("_metrics")] if stem.endswith("_metrics") else stem

        record: Dict = {"dataset": dataset}
        if label:
            record["label"] = label
        record["embedding_model"] = embedding_model
        record["ml_model"]        = ml_model
        record.update(metrics)
        records.append(record)

    return records


# ── Per-dataset aggregation ───────────────────────────────────────────────────
def aggregate_dataset(dataset: str) -> pd.DataFrame:
    """Return a DataFrame with all RAG results for one dataset."""
    records = collect_dataset_records(dataset)
    logging.info(f"  {dataset}: {len(records)} records found")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Metadata columns first, then metric columns sorted alphabetically
    meta_cols = ["dataset"]
    if "label" in df.columns:
        meta_cols.append("label")
    meta_cols += ["embedding_model", "ml_model"]
    metric_cols = sorted(c for c in df.columns if c not in meta_cols)
    df = df[meta_cols + metric_cols]

    sort_cols = (["label"] if "label" in df.columns else []) + ["embedding_model", "ml_model"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate RAG modelling results into per-dataset CSV tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write CSV files (default: results/rag/aggregated/)",
    )
    args = parser.parse_args()

    setup_logging()

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else RESULTS_ROOT / "aggregated"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 70)
    logging.info("RAG Results Aggregation")
    logging.info(f"Repo root  : {REPO_ROOT}")
    logging.info(f"Results dir: {RESULTS_ROOT}")
    logging.info(f"Output dir : {output_dir}")
    logging.info("=" * 70)

    all_frames: List[pd.DataFrame] = []

    for dataset in DATASETS:
        logging.info(f"\nDataset: {dataset.upper()}")
        logging.info("-" * 50)

        df = aggregate_dataset(dataset)

        if df.empty:
            logging.warning(f"  No results found for {dataset} — skipping.")
            continue

        out_path = output_dir / f"{dataset}_rag_results.csv"
        df.to_csv(out_path, index=False)
        logging.info(f"  Saved {len(df)} rows → {out_path}")

        if "MCC" in df.columns:
            print(f"\n{dataset.upper()} — MCC summary:")
            summary_cols = ["embedding_model", "ml_model", "MCC"]
            for col in ["AUC", "AUPR", "Accuracy", "F1_macro"]:
                if col in df.columns:
                    summary_cols.append(col)
            print(df[summary_cols].to_string(index=False))

        all_frames.append(df)

    if not all_frames:
        logging.error("No results found in any dataset directory.")
        return 1

    combined = pd.concat(all_frames, ignore_index=True)

    # Ensure label column is always present and placed after dataset
    if "label" not in combined.columns:
        combined.insert(1, "label", None)
    else:
        # Move label to position right after dataset
        cols = list(combined.columns)
        cols.remove("label")
        cols.insert(1, "label")
        combined = combined[cols]

    # Fill missing labels (non-clintox rows) with empty string for clarity
    combined["label"] = combined["label"].fillna("")

    combined_path = output_dir / "all_rag_results.csv"
    combined.to_csv(combined_path, index=False)

    logging.info("\n" + "=" * 70)
    logging.info("Summary")
    logging.info(f"  Total rows      : {len(combined)}")
    logging.info(f"  Datasets        : {combined['dataset'].nunique()}")
    logging.info(f"  Embedding models: {combined['embedding_model'].nunique()}")
    logging.info(f"  ML models       : {combined['ml_model'].nunique()}")
    logging.info(f"  Combined output : {combined_path}")
    logging.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
