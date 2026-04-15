"""
Generate Embeddings from LoRA-Finetuned Models

Extracts embeddings from finetuned CLMs (merged LoRA + base models) for all
datasets and saves them in the format expected by the downstream ML modeling
scripts in Finetuned Model Scripts/ml-scripts/.

Usage:
    python "Finetuned Model Scripts/lora-finetuning-scripts/finetuned_model_embeddings.py"

Input:
    - Finetuned models:  models/lora_finetuned/{dataset}/{loss_type}/{model}_LoRA_Finetuned/
    - Clean data:        data/clean/{dataset}_datasets/{split}_clean.csv

Output:
    - Embeddings CSVs:   data/finetuned_embeddings/{Dataset}_Embeddings/{prefix}_{split}_embed.csv
"""

import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CLEAN_ROOT = REPO_ROOT / "data" / "clean"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
OUTPUT_ROOT = REPO_ROOT / "data" / "finetuned_embeddings"
LOG_DIR = REPO_ROOT / "logs"

os.makedirs(str(OUTPUT_ROOT), exist_ok=True)
os.makedirs(str(LOG_DIR), exist_ok=True)

SMILES_COL = "Standardized SMILES"

# Dataset configuration: maps dataset key to clean data folder, output folder,
# file prefix for embedding CSVs, label columns, and split names.
DATASETS = {
    "bace": {
        "clean_dir": "bace_datasets",
        "output_dir": "BACE_Embeddings",
        "model_dir":  "BACE",
        "file_prefix": "bace",
        "labels": ["Class"],
        "splits": {"train": "train", "eval": "valid", "test": "test"},
    },
    "bbbp": {
        "clean_dir": "bbbp_datasets",
        "output_dir": "BBBP_Embeddings",
        "model_dir":  "BBBP",
        "file_prefix": "bbbp",
        "labels": ["p_np"],
        "splits": {"train": "train", "eval": "valid", "test": "test"},
    },
    "clintox": {
        "clean_dir": "clintox_datasets",
        "output_dir": "clintox_Embeddings",
        "model_dir":  "clintox",
        "file_prefix": "clintox",
        "labels": ["FDA_APPROVED", "CT_TOX"],
        "splits": {"train": "train", "eval": "valid", "test": "test"},
    },
    "flavor": {
        "clean_dir": "flavor_datasets",
        "output_dir": "flavor_Embeddings",
        "model_dir":  "flavor",
        "file_prefix": "fart",
        "labels": ["Canonicalized Taste"],
        "splits": {"train": "train", "eval": "valid", "test": "test"},
    },
}

# Mapping from finetuned model folder names to the embedding column names
# expected by the downstream ML scripts, and the HuggingFace tokenizer source.
#
# Folder names are produced by safe_model_name() in the finetuning scripts:
#   re.sub(r"[^a-zA-Z0-9]", "__", model_name)
FINETUNED_MODELS = {
    "focal_loss": {
        "DeepChem__ChemBERTa__77M__MTR_LoRA_Finetuned": {
            "col_name": "ChemBERTa_77M_MTR_FL_embeddings",
            "tokenizer": "DeepChem/ChemBERTa-77M-MTR",
        },
        "DeepChem__ChemBERTa__77M__MLM_LoRA_Finetuned": {
            "col_name": "ChemBERTa_77M_MLM_FL_embeddings",
            "tokenizer": "DeepChem/ChemBERTa-77M-MLM",
        },
        "ibm__MoLFormer__XL__both__10pct_LoRA_Finetuned": {
            "col_name": "MolFormer_Finetuned_FL_embeddings",
            "tokenizer": "ibm/MoLFormer-XL-both-10pct",
        },
    },
    "weighted_loss": {
        "DeepChem__ChemBERTa__77M__MTR_LoRA_Finetuned": {
            "col_name": "ChemBERTa_77M_MTR_WL_embeddings",
            "tokenizer": "DeepChem/ChemBERTa-77M-MTR",
        },
        "DeepChem__ChemBERTa__77M__MLM_LoRA_Finetuned": {
            "col_name": "ChemBERTa_77M_MLM_WL_embeddings",
            "tokenizer": "DeepChem/ChemBERTa-77M-MLM",
        },
        "ibm__MoLFormer__XL__both__10pct_LoRA_Finetuned": {
            "col_name": "Molformer_Finetuned_WL_embeddings",
            "tokenizer": "ibm/MoLFormer-XL-both-10pct",
        },
    },
}

logging.basicConfig(
    filename=str(LOG_DIR / "finetuned_model_embeddings.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)


def get_embeddings(smiles_list, tokenizer, model, batch_size=64):
    """Extract embeddings from a model by batching SMILES strings."""
    all_embeddings = []

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]

        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        with torch.no_grad():
            output = model(**tokens)
            # Use mean of last_hidden_state — pooler_output is randomly initialised
            # when AutoModel loads a RobertaForSequenceClassification checkpoint.
            emb = output.last_hidden_state.mean(dim=1)

        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    logging.info("=" * 60)
    logging.info("Starting finetuned model embedding extraction")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Models root: {EXTRAS_ROOT}")
    logging.info(f"Output root: {OUTPUT_ROOT}")

    for dataset_key, cfg in DATASETS.items():
        logging.info(f"\n{'='*40}")
        logging.info(f"Processing dataset: {dataset_key}")

        dataset_out_dir = OUTPUT_ROOT / cfg["output_dir"]
        os.makedirs(str(dataset_out_dir), exist_ok=True)

        # Collect all model info for this dataset
        model_info_list = []
        for loss_type, models_dict in FINETUNED_MODELS.items():
            for folder_name, info in models_dict.items():
                model_path = EXTRAS_ROOT / f"{loss_type}_{cfg['model_dir']}" / folder_name
                if not model_path.exists():
                    logging.warning(
                        f"Model not found, skipping: {model_path}"
                    )
                    continue
                model_info_list.append({
                    "path": model_path,
                    "col_name": info["col_name"],
                    "tokenizer_name": info["tokenizer"],
                    "loss_type": loss_type,
                    "folder_name": folder_name,
                })

        if not model_info_list:
            logging.warning(f"No finetuned models found for {dataset_key}, skipping")
            continue

        # Load all models for this dataset
        loaded_models = {}
        for mi in model_info_list:
            logging.info(f"Loading model: {mi['col_name']} from {mi['path']}")
            tokenizer = AutoTokenizer.from_pretrained(
                mi["tokenizer_name"], trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                str(mi["path"]), trust_remote_code=True
            )
            model.to(DEVICE)
            model.eval()
            loaded_models[mi["col_name"]] = (tokenizer, model)
            logging.info(f"  Loaded successfully")

        # Process each split
        for out_split, clean_split in cfg["splits"].items():
            input_path = CLEAN_ROOT / cfg["clean_dir"] / f"{clean_split}_clean.csv"

            if not input_path.exists():
                logging.warning(f"Input file not found: {input_path}")
                continue

            logging.info(f"Loading {clean_split} split from {input_path}")
            df = pd.read_csv(str(input_path))

            smiles = df[SMILES_COL].astype(str).tolist()

            # Start output dataframe with SMILES + labels
            base_cols = [SMILES_COL] + [c for c in cfg["labels"] if c in df.columns]
            out_df = df[base_cols].copy()

            logging.info(f"  {dataset_key} | {out_split} | Samples={len(smiles)}")

            # Generate embeddings for each model
            for col_name, (tokenizer, model) in loaded_models.items():
                logging.info(f"  Generating embeddings: {col_name}")

                emb = get_embeddings(smiles, tokenizer, model)

                # Store as comma-separated string arrays (matching base_model_embeddings.py format)
                emb_as_str = [
                    np.array2string(e, separator=",", precision=8) for e in emb
                ]
                out_df[col_name] = emb_as_str

                logging.info(
                    f"    Shape: {emb.shape[0]} x {emb.shape[1]}"
                )

            # Save output CSV
            output_path = dataset_out_dir / f"{cfg['file_prefix']}_{out_split}_embed.csv"
            out_df.to_csv(str(output_path), index=False)
            logging.info(f"  Saved: {output_path}")

        # Free GPU memory
        del loaded_models
        torch.cuda.empty_cache()

    logging.info("\nAll datasets processed successfully.")


if __name__ == "__main__":
    main()
