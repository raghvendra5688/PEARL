"""
Aggregate Metrics Collection Script

This script collects all *_metrics.npy files from a specified results directory
and combines them into a comprehensive CSV table.

Usage:
    python aggregate_metrics.py --task BBBP_FT_Results --output bbbp_metrics.csv
    python aggregate_metrics.py --task flavor_FT_Results --output flavor_metrics.csv
    python aggregate_metrics.py --task BBBP_PC_FT_Results --output bbbp_pc_metrics.csv
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )


def find_metrics_files(task_dir: Path) -> List[Path]:
    """
    Find all *_metrics.npy files in the task directory.

    Args:
        task_dir: Path to the task results directory

    Returns:
        List of paths to metrics files
    """
    metrics_files = list(task_dir.rglob("*_metrics.npy"))
    logging.info(f"Found {len(metrics_files)} metrics files in {task_dir}")
    return metrics_files


def parse_metrics_path(metrics_path: Path, task_dir: Path) -> Dict[str, str]:
    """
    Parse metadata from the metrics file path.

    Expected structure:
    - task_dir/EmbeddingModel/metrics/ModelName_metrics.npy

    Args:
        metrics_path: Path to the metrics file
        task_dir: Root task directory

    Returns:
        Dictionary with metadata (embedding_model, ml_model, task_name)
    """
    try:
        # Get relative path from task_dir
        rel_path = metrics_path.relative_to(task_dir)
        parts = rel_path.parts

        # Extract embedding model (first directory)
        embedding_model = parts[0] if len(parts) > 0 else "Unknown"

        # Extract ML model name from filename
        filename = metrics_path.stem  # Remove .npy extension

        # Handle different naming patterns
        # Pattern 1: "ModelName_metrics" (e.g., XGBoost_metrics)
        # Pattern 2: "EmbeddingName_ModelName_metrics" (e.g., ChemBERTa_77M_MLM_WL_XGBoost_metrics)

        if filename.endswith("_metrics"):
            model_part = filename[:-8]  # Remove "_metrics"

            # Check if model_part contains the embedding name
            if embedding_model in model_part:
                # Remove embedding name to get ML model
                ml_model = model_part.replace(embedding_model + "_", "")
            else:
                ml_model = model_part
        else:
            ml_model = filename

        # Extract task name from task_dir
        task_name = task_dir.name.replace("_Results", "").replace("_FT", "").replace("_PC", "")

        return {
            "task": task_name,
            "embedding_model": embedding_model,
            "ml_model": ml_model
        }

    except Exception as e:
        logging.error(f"Error parsing path {metrics_path}: {e}")
        return {
            "task": "Unknown",
            "embedding_model": "Unknown",
            "ml_model": "Unknown"
        }


def load_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load metrics from .npy file.

    Args:
        metrics_path: Path to the metrics file

    Returns:
        Dictionary of metrics or None if loading fails
    """
    try:
        metrics = np.load(metrics_path, allow_pickle=True)

        # Handle different formats
        if isinstance(metrics, np.ndarray):
            if metrics.shape == ():
                # Scalar array containing a dictionary
                metrics_dict = metrics.item()
            else:
                logging.warning(f"Unexpected metrics format in {metrics_path}")
                return None
        elif isinstance(metrics, dict):
            metrics_dict = metrics
        else:
            logging.warning(f"Unknown metrics type in {metrics_path}: {type(metrics)}")
            return None

        return metrics_dict

    except Exception as e:
        logging.error(f"Error loading metrics from {metrics_path}: {e}")
        return None


def aggregate_metrics(task_dir: Path) -> pd.DataFrame:
    """
    Aggregate all metrics from a task directory into a DataFrame.

    Args:
        task_dir: Path to the task results directory

    Returns:
        DataFrame with all metrics
    """
    metrics_files = find_metrics_files(task_dir)

    if not metrics_files:
        logging.warning(f"No metrics files found in {task_dir}")
        return pd.DataFrame()

    all_metrics = []

    for metrics_path in sorted(metrics_files):
        logging.info(f"Processing: {metrics_path.relative_to(task_dir)}")

        # Parse metadata from path
        metadata = parse_metrics_path(metrics_path, task_dir)

        # Load metrics
        metrics_dict = load_metrics(metrics_path)

        if metrics_dict is None:
            continue

        # Combine metadata and metrics
        row = {**metadata, **metrics_dict}
        all_metrics.append(row)

    # Create DataFrame
    if not all_metrics:
        logging.warning("No valid metrics loaded")
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics)

    # Reorder columns: metadata first, then metrics
    metadata_cols = ["task", "embedding_model", "ml_model"]
    metric_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + sorted(metric_cols)]

    logging.info(f"Aggregated {len(df)} metric records")

    return df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from .npy files into a CSV table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aggregate_metrics.py --task BBBP_FT_Results --output bbbp_metrics.csv
  python aggregate_metrics.py --task flavor_FT_Results --output flavor_metrics.csv
  python aggregate_metrics.py --task BBBP_PC_FT_Results --output bbbp_pc_metrics.csv
  python aggregate_metrics.py --task clintox_PC_FT_Results --output clintox_pc_metrics.csv
  python aggregate_metrics.py --task flavor_PC_FT_Results --output flavor_pc_metrics.csv
        """
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task directory name (e.g., BBBP_FT_Results, flavor_FT_Results)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file name (e.g., bbbp_metrics.csv)"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory containing task folders (default: repo root)"
    )

    args = parser.parse_args()

    setup_logging()

    # Construct paths
    if args.base_dir is not None:
        base_dir = Path(args.base_dir).resolve()
    else:
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
    task_dir = base_dir / "results" / "ft_embeddings" / args.task
    output_path = base_dir / "results" / "ft_embeddings" / args.output

    logging.info("=" * 80)
    logging.info("Metrics Aggregation Script")
    logging.info(f"Base directory: {base_dir}")
    logging.info(f"Task directory: {task_dir}")
    logging.info(f"Output file: {output_path}")
    logging.info("=" * 80)

    # Check if task directory exists
    if not task_dir.exists():
        logging.error(f"Task directory does not exist: {task_dir}")
        return 1

    if not task_dir.is_dir():
        logging.error(f"Task path is not a directory: {task_dir}")
        return 1

    # Aggregate metrics
    df = aggregate_metrics(task_dir)

    if df.empty:
        logging.error("No metrics to save")
        return 1

    # Save to CSV
    df.to_csv(output_path, index=False)
    logging.info(f"Saved {len(df)} records to {output_path}")

    # Display summary
    logging.info("=" * 80)
    logging.info("Summary:")
    logging.info(f"  Total records: {len(df)}")
    logging.info(f"  Embedding models: {df['embedding_model'].nunique()}")
    logging.info(f"  ML models: {df['ml_model'].nunique()}")
    logging.info(f"  Columns: {', '.join(df.columns)}")
    logging.info("=" * 80)

    # Display first few rows
    print("\nFirst 5 rows of the aggregated metrics:")
    print(df.head().to_string())

    return 0


if __name__ == "__main__":
    exit(main())
