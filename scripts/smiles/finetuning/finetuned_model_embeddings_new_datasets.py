"""
Generate Embeddings from LoRA-Finetuned Models — herg/dili/caco2/half_life

Mirrors finetuned_model_embeddings.py (the original bace/bbbp/clintox/flavor
script) for the 4 new TDC datasets added in Phase 1-6 of the editor-response
revision (see manuscript/editor_response_suggestions.md). Kept as a separate
script rather than folding into the original because these datasets:
  - live under $PEARL_EXTRAS_V2, not the original $PEARL_EXTRAS
  - herg/dili use FL+WL (like the originals) but caco2/half_life use Huber
    only (a regression target, not a classification one)

Usage:
    python finetuned_model_embeddings_new_datasets.py

Input:
    - Finetuned models: $PEARL_EXTRAS_V2/{loss}_{DATASET}/{model}_LoRA_Finetuned/
    - Clean data:       data/clean/{dataset}_datasets/{split}_clean.csv

Output:
    - Embeddings CSVs:  data/finetuned_embeddings/{Dataset}_Embeddings/{prefix}_{split}_embed.csv
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch

from transformers import AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from safe_model_loading import load_tokenizer_safe  # noqa: E402

CLEAN_ROOT = REPO_ROOT / "data" / "clean"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
# New-artifacts convention (see editor_response_suggestions.md): embeddings are
# large per-molecule CSVs (hundreds of MB for herg's ~13k molecules) and must
# not live in the git repo -- mirrors unimol_embeddings.py's own
# EXTRAS_ROOT/"unimol_embeddings" pattern, in a sibling dir so the two
# extraction scripts never write to the same file.
OUTPUT_ROOT = EXTRAS_ROOT / "finetuned_embeddings"
LOG_DIR = REPO_ROOT / "logs"

os.makedirs(str(OUTPUT_ROOT), exist_ok=True)
os.makedirs(str(LOG_DIR), exist_ok=True)

# Dataset configuration: maps dataset key to clean data folder, output folder,
# file prefix for embedding CSVs, label column(s), loss variants, and the
# $PEARL_EXTRAS_V2 directory-name fragment used for each loss.
DATASETS = {
    "herg": {
        "clean_dir": "herg_datasets",
        "output_dir": "HERG_Embeddings",
        "file_prefix": "herg",
        "labels": ["hERG_Inhib"],
        "splits": {"train": "train", "eval": "valid", "test": "test"},
        "losses": {"focal_loss": "focal_loss_HERG", "weighted_loss": "weighted_loss_HERG"},
    },
    "dili": {
        "clean_dir": "dili_datasets",
        "output_dir": "DILI_Embeddings",
        "file_prefix": "dili",
        "labels": ["DILI_Label"],
        "splits": {"train": "train", "eval": "valid", "test": "test"},
        "losses": {"focal_loss": "focal_loss_DILI", "weighted_loss": "weighted_loss_DILI"},
    },
    "caco2": {
        "clean_dir": "caco2_datasets",
        "output_dir": "CACO2_Embeddings",
        "file_prefix": "caco2",
        "labels": ["Caco2_LogPapp"],
        "splits": {"train": "train", "eval": "valid", "test": "test"},
        "losses": {"huber_loss": "huber_loss_CACO2"},
    },
    "half_life": {
        "clean_dir": "half_life_datasets",
        "output_dir": "HALF_LIFE_Embeddings",
        "file_prefix": "half_life",
        "labels": ["Half_Life_Hours"],
        "splits": {"train": "train", "eval": "valid", "test": "test"},
        "losses": {"huber_loss": "huber_loss_HALF_LIFE"},
    },
}

# Folder-name (from LoRA-merge safe_model_name()) -> (embedding column suffix, tokenizer source)
MODEL_INFO = {
    "DeepChem__ChemBERTa__77M__MTR_LoRA_Finetuned": ("ChemBERTa_77M_MTR", "DeepChem/ChemBERTa-77M-MTR"),
    "DeepChem__ChemBERTa__77M__MLM_LoRA_Finetuned": ("ChemBERTa_77M_MLM", "DeepChem/ChemBERTa-77M-MLM"),
    "ibm__MoLFormer__XL__both__10pct_LoRA_Finetuned": ("MolFormer_Finetuned", "ibm/MoLFormer-XL-both-10pct"),
}

LOSS_SUFFIX = {"focal_loss": "FL", "weighted_loss": "WL", "huber_loss": "Huber"}

logging.basicConfig(
    filename=str(LOG_DIR / "finetuned_model_embeddings_new_datasets.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)


def get_embeddings(smiles_list, tokenizer, model, batch_size=64):
    """Extract embeddings from a model by batching SMILES strings (mean-pooled
    last_hidden_state -- pooler_output is randomly initialised when AutoModel
    loads a *ForSequenceClassification checkpoint, so it must not be used)."""
    all_embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        with torch.no_grad():
            output = model(**tokens)
            emb = output.last_hidden_state.mean(dim=1)
        all_embeddings.append(emb.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def main():
    logging.info("=" * 60)
    logging.info("Starting finetuned model embedding extraction (new TDC datasets)")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Models root: {EXTRAS_ROOT}")
    logging.info(f"Output root: {OUTPUT_ROOT}")

    for dataset_key, cfg in DATASETS.items():
        logging.info(f"\n{'='*40}")
        logging.info(f"Processing dataset: {dataset_key}")

        dataset_out_dir = OUTPUT_ROOT / cfg["output_dir"]
        os.makedirs(str(dataset_out_dir), exist_ok=True)

        model_info_list = []
        for loss_key, extras_dirname in cfg["losses"].items():
            for folder_name, (col_suffix, tokenizer_name) in MODEL_INFO.items():
                model_path = EXTRAS_ROOT / extras_dirname / folder_name
                if not model_path.exists():
                    logging.warning(f"Model not found, skipping: {model_path}")
                    continue
                col_name = f"{col_suffix}_{LOSS_SUFFIX[loss_key]}_embeddings"
                model_info_list.append({
                    "path": model_path,
                    "col_name": col_name,
                    "tokenizer_name": tokenizer_name,
                    "loss_key": loss_key,
                })

        if not model_info_list:
            logging.warning(f"No finetuned models found for {dataset_key}, skipping")
            continue

        loaded_models = {}
        for mi in model_info_list:
            logging.info(f"Loading model: {mi['col_name']} from {mi['path']}")
            tokenizer = load_tokenizer_safe(mi["tokenizer_name"], trust_remote_code=True)
            model = AutoModel.from_pretrained(str(mi["path"]), trust_remote_code=True)
            model.to(DEVICE)
            model.eval()
            loaded_models[mi["col_name"]] = (tokenizer, model)
            logging.info("  Loaded successfully")

        for out_split, clean_split in cfg["splits"].items():
            input_path = CLEAN_ROOT / cfg["clean_dir"] / f"{clean_split}_clean.csv"
            if not input_path.exists():
                logging.warning(f"Input file not found: {input_path}")
                continue

            logging.info(f"Loading {clean_split} split from {input_path}")
            df = pd.read_csv(str(input_path))
            smiles = df["Standardized SMILES"].astype(str).tolist()

            base_cols = ["Standardized SMILES"] + [c for c in cfg["labels"] if c in df.columns]
            out_df = df[base_cols].copy()

            logging.info(f"  {dataset_key} | {out_split} | Samples={len(smiles)}")

            for col_name, (tokenizer, model) in loaded_models.items():
                logging.info(f"  Generating embeddings: {col_name}")
                emb = get_embeddings(smiles, tokenizer, model)
                emb_as_str = [np.array2string(e, separator=",", precision=8) for e in emb]
                out_df[col_name] = emb_as_str
                logging.info(f"    Shape: {emb.shape[0]} x {emb.shape[1]}")

            output_path = dataset_out_dir / f"{cfg['file_prefix']}_{out_split}_embed.csv"
            out_df.to_csv(str(output_path), index=False)
            logging.info(f"  Saved: {output_path}")

        del loaded_models
        torch.cuda.empty_cache()

    logging.info("\nAll new-dataset embedding extraction completed successfully.")


if __name__ == "__main__":
    main()
