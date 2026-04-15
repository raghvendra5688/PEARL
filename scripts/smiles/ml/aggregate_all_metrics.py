"""
Aggregate All Metrics Script

This script automatically finds all *_Results directories and aggregates metrics from each.

Usage:
    python aggregate_all_metrics.py
    python aggregate_all_metrics.py --output-dir ../aggregated_metrics
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )


def find_results_directories(base_dir: Path) -> List[Path]:
    """
    Find all directories ending with '_Results' or '_FT_Results'.

    Args:
        base_dir: Base directory to search

    Returns:
        List of paths to results directories
    """
    results_dirs = []

    for item in base_dir.iterdir():
        if item.is_dir() and ("_Results" in item.name or "_FT_Results" in item.name):
            results_dirs.append(item)

    logging.info(f"Found {len(results_dirs)} results directories")
    return sorted(results_dirs)


def aggregate_metrics_from_task(task_dir: Path) -> pd.DataFrame:
    """
    Aggregate all metrics from a task directory into a DataFrame.

    Args:
        task_dir: Path to the task results directory

    Returns:
        DataFrame with all metrics
    """
    metrics_files = list(task_dir.rglob("*_metrics.npy"))

    if not metrics_files:
        logging.warning(f"No metrics files found in {task_dir.name}")
        return pd.DataFrame()

    all_metrics = []

    for metrics_path in sorted(metrics_files):
        try:
            # Get relative path from task_dir
            rel_path = metrics_path.relative_to(task_dir)
            parts = rel_path.parts

            # Extract embedding model (first directory)
            embedding_model = parts[0] if len(parts) > 0 else "Unknown"

            # Extract ML model name from filename
            filename = metrics_path.stem  # Remove .npy extension

            if filename.endswith("_metrics"):
                model_part = filename[:-8]  # Remove "_metrics"

                # Check if model_part contains the embedding name
                if embedding_model in model_part:
                    ml_model = model_part.replace(embedding_model + "_", "")
                else:
                    ml_model = model_part
            else:
                ml_model = filename

            # Extract task name
            task_name = task_dir.name

            # Load metrics
            metrics = np.load(metrics_path, allow_pickle=True)

            if isinstance(metrics, np.ndarray) and metrics.shape == ():
                metrics_dict = metrics.item()
            elif isinstance(metrics, dict):
                metrics_dict = metrics
            else:
                logging.warning(f"Unexpected metrics format in {metrics_path}")
                continue

            # Combine metadata and metrics
            row = {
                "task": task_name,
                "embedding_model": embedding_model,
                "ml_model": ml_model,
                **metrics_dict
            }
            all_metrics.append(row)

        except Exception as e:
            logging.error(f"Error processing {metrics_path}: {e}")
            continue

    if not all_metrics:
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics)

    # Reorder columns
    metadata_cols = ["task", "embedding_model", "ml_model"]
    metric_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + sorted(metric_cols)]

    return df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from all *_Results directories",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory containing results folders (default: repo root results/finetuned)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV files (default: same as base-dir)"
    )

    args = parser.parse_args()

    setup_logging()

    # Construct paths
    if args.base_dir is not None:
        base_dir = Path(args.base_dir).resolve()
    else:
        base_dir = Path(__file__).resolve().parent.parent.parent / "results" / "finetuned"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else base_dir

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 80)
    logging.info("Aggregate All Metrics Script")
    logging.info(f"Base directory: {base_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("=" * 80)

    # Find all results directories
    results_dirs = find_results_directories(base_dir)

    if not results_dirs:
        logging.error("No results directories found")
        return 1

    # Aggregate metrics from each task
    all_task_data = []

    for task_dir in results_dirs:
        logging.info(f"\nProcessing: {task_dir.name}")
        logging.info("-" * 80)

        df = aggregate_metrics_from_task(task_dir)

        if df.empty:
            logging.warning(f"No metrics found for {task_dir.name}")
            continue

        # Save individual task CSV
        task_output = output_dir / f"{task_dir.name}_metrics.csv"
        df.to_csv(task_output, index=False)
        logging.info(f"Saved {len(df)} records to {task_output.name}")

        # Add to combined data
        all_task_data.append(df)

    if not all_task_data:
        logging.error("No metrics to aggregate")
        return 1

    # Create combined CSV with all tasks
    combined_df = pd.concat(all_task_data, ignore_index=True)
    combined_output = output_dir / "all_metrics_combined.csv"
    combined_df.to_csv(combined_output, index=False)

    logging.info("=" * 80)
    logging.info("Summary:")
    logging.info(f"  Total records: {len(combined_df)}")
    logging.info(f"  Tasks: {combined_df['task'].nunique()}")
    logging.info(f"  Embedding models: {combined_df['embedding_model'].nunique()}")
    logging.info(f"  ML models: {combined_df['ml_model'].nunique()}")
    logging.info(f"  Combined output: {combined_output}")
    logging.info("=" * 80)

    # Display summary by task
    print("\nRecords per task:")
    print(combined_df.groupby('task').size().to_string())

    return 0


if __name__ == "__main__":
    exit(main())
