"""
Generate Base Model Embeddings for BBBP, ClinTox, Flavor (clean datasets)
"""

import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_ROOT = str(REPO_ROOT / "data" / "clean")
OUTPUT_ROOT = str(REPO_ROOT / "data" / "base_models")
LOG_DIR = str(REPO_ROOT / "logs")

SMILES_COL = "Standardized SMILES"

DATASETS = {
    "bbbp_datasets": {
        "labels": ["p_np"]
    },
    "clintox_datasets": {
        "labels": ["FDA_APPROVED", "CT_TOX"]
    },
    "flavor_datasets": {
        "labels": ["Canonicalized Taste"]
    },
    "bace_datasets": {
        "labels": ["Class"]
    }
}

# Base Models Embeddings information

models_info = {
    "ChemBERTa_77M_MTR_Base": {
        "tokenizer": "DeepChem/ChemBERTa-77M-MTR",
        "model": "DeepChem/ChemBERTa-77M-MTR"
    },
    "ChemBERTa_77M_MLM_Base": {
        "tokenizer": "DeepChem/ChemBERTa-77M-MLM",
        "model": "DeepChem/ChemBERTa-77M-MLM"
    },
    "MolFormer_Base": {
        "tokenizer": "ibm/MoLFormer-XL-both-10pct",
        "model": "ibm/MoLFormer-XL-both-10pct"
    }
}

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "base_model_embeddings.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# loading the model
def load_model(model_info):
    tokenizer = AutoTokenizer.from_pretrained(
        model_info["tokenizer"],
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_info["model"],
        trust_remote_code=True
    )

    model.to(DEVICE)
    model.eval()

    logging.info(f"Loaded model: {model_info['model']}")
    return tokenizer, model


tokenizers = {}
models = {}

logging.info("Loading base models...")

for name, info in models_info.items():
    tok, mdl = load_model(info)
    tokenizers[name] = tok
    models[name] = mdl

logging.info("All base models loaded successfully")

# embedding generation function
def get_embeddings(smiles_list, tokenizer, model, model_name):

    tokens = tokenizer(
        smiles_list,
        return_tensors="pt",
        padding=True,
        truncation=False
    )

    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    with torch.no_grad():
        output = model(**tokens)

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            emb = output.pooler_output
        else:
            emb = output.last_hidden_state.mean(dim=1)

    final_emb = emb.cpu().numpy()

    logging.info(
        f"Embeddings generated | Model={model_name} | "
        f"Samples={final_emb.shape[0]} | Dim={final_emb.shape[1]}"
    )

    return final_emb

# main execution
for dataset_name, cfg in DATASETS.items():

    logging.info(f"Processing dataset: {dataset_name}")

    dataset_clean_dir = os.path.join(CLEAN_ROOT, dataset_name)
    dataset_out_dir = os.path.join(OUTPUT_ROOT, dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:

        input_path = os.path.join(dataset_clean_dir, f"{split}_clean.csv")
        output_path = os.path.join(dataset_out_dir, f"{split}_embeddings.csv")

        logging.info(f"Loading: {input_path}")
        df = pd.read_csv(input_path)

        smiles = df[SMILES_COL].astype(str).tolist()

        base_cols = [SMILES_COL] + cfg["labels"]
        out_df = df[base_cols].copy()

        logging.info(
            f"{dataset_name} | {split} | Samples={len(smiles)}"
        )

        for model_name in models_info:

            logging.info(
                f"Embedding started | Dataset={dataset_name} | Split={split} | Model={model_name}"
            )

            emb = get_embeddings(
                smiles,
                tokenizers[model_name],
                models[model_name],
                model_name
            )

            emb_as_str = [np.array2string(e, separator=",", precision=8) for e in emb]
            out_df[f"{model_name}"] = emb_as_str

        out_df.to_csv(output_path, index=False)
        logging.info(f"Saved embeddings CSV: {output_path}")

logging.info("All datasets and splits processed successfully.")
