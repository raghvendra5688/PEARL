
import os
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_RESULTS_DIR = str(REPO_ROOT / "results" / "PC")

OUTPUT_EXCEL_PATH = os.path.join(
    BASE_RESULTS_DIR, "PC_all_results.xlsx"
)

MODELS = ["XGBoost", "LightGBM", "CatBoost"]

DATASETS = {
    "BBBP": {
        "path": "bbbp_dataset/metrics",
        "sheet": "BBBP"
    },
    "Flavor": {
        "path": "flavor_dataset/metrics",
        "sheet": "Flavor"
    },
    "ClinTox_FDA_APPROVED": {
        "path": "clintox_dataset/FDA_APPROVED/metrics",
        "sheet": "ClinTox_FDA_APPROVED"
    },
    "BACE": {
        "path": "bace_dataset/metrics",
        "sheet": "BACE"
    }
}

def load_metrics(metrics_dir):
    """
    Load metrics for all models from a metrics directory.
    Returns a DataFrame.
    """
    rows = []

    for model in MODELS:
        metric_file = os.path.join(metrics_dir, f"{model}_metrics.npy")

        if not os.path.exists(metric_file):
            print(f"[WARNING] Missing: {metric_file}")
            continue

        metrics = np.load(metric_file, allow_pickle=True).item()
        metrics["Model"] = model
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Model")

    # Sort columns alphabetically for clean presentation
    df = df.reindex(sorted(df.columns), axis=1)

    return df


with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine="xlsxwriter") as writer:

    for dataset_name, cfg in DATASETS.items():
        metrics_path = os.path.join(BASE_RESULTS_DIR, cfg["path"])

        if not os.path.isdir(metrics_path):
            print(f"[WARNING] Directory not found: {metrics_path}")
            continue

        df = load_metrics(metrics_path)

        if df.empty:
            print(f"[WARNING] No metrics found for {dataset_name}")
            continue

        df.to_excel(writer, sheet_name=cfg["sheet"])

print("\nPC results successfully exported to:")
print(OUTPUT_EXCEL_PATH)
