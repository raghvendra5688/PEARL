import os
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_RESULTS_DIR = str(REPO_ROOT / "results" / "base_models_features")

OUTPUT_EXCEL_PATH = os.path.join(
    BASE_RESULTS_DIR, "base_model_with_features_results.xlsx"
)

ML_MODELS = ["XGBoost", "LightGBM", "CatBoost"]

EMB_MODELS = [
    "ChemBERTa_77M_MLM_Base",
    "ChemBERTa_77M_MTR_Base",
    "MolFormer_Base"
]

DATASETS = {
    "BBBP": {
        "type": "simple",
        "path": "bbbp_dataset"
    },
    "Flavor": {
        "type": "simple",
        "path": "flavor_dataset"
    },
    "ClinTox_FDA_APPROVED": {
        "type": "label",
        "path": "clintox_dataset/FDA_APPROVED"
    },
    "BACE": {
        "type": "simple",
        "path": "bace_dataset"
    }
}

def load_metrics(metrics_dir):
    rows = []

    for ml_model in ML_MODELS:
        metric_file = os.path.join(metrics_dir, f"{ml_model}_metrics.npy")

        if not os.path.exists(metric_file):
            print(f"[WARNING] Missing: {metric_file}")
            continue

        metrics = np.load(metric_file, allow_pickle=True).item()
        metrics["ML_Model"] = ml_model
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.set_index("ML_Model")

    df = df.reindex(sorted(df.columns), axis=1)
    return df


with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine="xlsxwriter") as writer:

    for sheet_name, cfg in DATASETS.items():

        print(f"Processing {sheet_name}")

        all_rows = []

        base_path = os.path.join(BASE_RESULTS_DIR, cfg["path"])

        if not os.path.isdir(base_path):
            print(f"[WARNING] Directory not found: {base_path}")
            continue

        for emb in EMB_MODELS:

            metrics_dir = os.path.join(base_path, emb, "metrics")

            if not os.path.isdir(metrics_dir):
                print(f"[WARNING] Missing metrics dir: {metrics_dir}")
                continue

            df = load_metrics(metrics_dir)

            if df.empty:
                print(f"[WARNING] No metrics for {sheet_name} | {emb}")
                continue

            df.insert(0, "Embedding_Model", emb)
            df.insert(1, "Dataset", sheet_name)

            all_rows.append(df.reset_index())

        if not all_rows:
            print(f"[WARNING] No results found for {sheet_name}")
            continue

        final_df = pd.concat(all_rows, ignore_index=True)

        # Excel sheet name must be <= 31 chars
        final_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

print("\nBase-model with features ML results successfully exported to:")
print(OUTPUT_EXCEL_PATH)
