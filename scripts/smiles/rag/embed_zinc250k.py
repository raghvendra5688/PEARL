"""
Embed ZINC-250k Using LoRA-Finetuned CLMs

For a given dataset, loads each of the 6 task-specific finetuned models and embeds
all 249,455 ZINC-250k molecules. Produces 6 .npy files per dataset (24 total across
all datasets), stored outside the git repo to avoid >50 MB file limits.

Usage:
    python "RAG Pipeline/embed_zinc250k.py" --dataset bace
    python "RAG Pipeline/embed_zinc250k.py" --dataset bbbp
    python "RAG Pipeline/embed_zinc250k.py" --dataset clintox
    python "RAG Pipeline/embed_zinc250k.py" --dataset flavor

Input:
    data/zinc250k/zinc250k_cleaned.csv
    /export/cse/rmall/Raghvendra/EffiChem_Extras/{loss_type}_{DATASET}/{model_folder}/

Output (outside git repo):
    /export/cse/rmall/Raghvendra/EffiChem_Extras/zinc_embeddings/{dataset}/{col_name}.npy
    Shape: (249455, 384) for ChemBERTa models, (249455, 768) for MolFormer
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
ZINC_CSV    = REPO_ROOT / "data" / "zinc250k" / "zinc250k_cleaned.csv"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
EMBED_ROOT  = EXTRAS_ROOT / "zinc_embeddings"
LOG_DIR     = REPO_ROOT / "logs"

# Maps --dataset arg to the directory name suffix used in EffiChem_Extras
# e.g. focal_loss_BACE, weighted_loss_clintox, etc.
DATASET_DIR_NAME = {
    "bace":    "BACE",
    "bbbp":    "BBBP",
    "clintox": "clintox",
    "flavor":  "flavor",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model registry (mirrors finetuned_model_embeddings.py) ────────────────────
# col_name becomes the .npy filename (without _embeddings suffix used in CSV cols)
FINETUNED_MODELS = {
    "focal_loss": {
        "DeepChem__ChemBERTa__77M__MTR_LoRA_Finetuned": {
            "col_name":  "ChemBERTa_77M_MTR_FL",
            "tokenizer": "DeepChem/ChemBERTa-77M-MTR",
        },
        "DeepChem__ChemBERTa__77M__MLM_LoRA_Finetuned": {
            "col_name":  "ChemBERTa_77M_MLM_FL",
            "tokenizer": "DeepChem/ChemBERTa-77M-MLM",
        },
        "ibm__MoLFormer__XL__both__10pct_LoRA_Finetuned": {
            "col_name":  "MolFormer_Finetuned_FL",
            "tokenizer": "ibm/MoLFormer-XL-both-10pct",
        },
    },
    "weighted_loss": {
        "DeepChem__ChemBERTa__77M__MTR_LoRA_Finetuned": {
            "col_name":  "ChemBERTa_77M_MTR_WL",
            "tokenizer": "DeepChem/ChemBERTa-77M-MTR",
        },
        "DeepChem__ChemBERTa__77M__MLM_LoRA_Finetuned": {
            "col_name":  "ChemBERTa_77M_MLM_WL",
            "tokenizer": "DeepChem/ChemBERTa-77M-MLM",
        },
        "ibm__MoLFormer__XL__both__10pct_LoRA_Finetuned": {
            "col_name":  "Molformer_Finetuned_WL",
            "tokenizer": "ibm/MoLFormer-XL-both-10pct",
        },
    },
}

VALID_DATASETS = ["bace", "bbbp", "clintox", "flavor"]


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(dataset: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / f"embed_zinc250k_{dataset}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


# ── Embedding ─────────────────────────────────────────────────────────────────
def get_embeddings(
    smiles_list: list,
    tokenizer,
    model,
    batch_size: int = 128,
) -> np.ndarray:
    """Batch-extract embeddings from a CLM. Identical pattern to
    finetuned_model_embeddings.py::get_embeddings()."""
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

        if (i // batch_size) % 50 == 0:
            logging.info(
                f"  Embedded {min(i + batch_size, len(smiles_list))}/{len(smiles_list)} molecules"
            )

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Embed ZINC-250k with finetuned CLMs.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=VALID_DATASETS,
        help="Dataset whose finetuned models to use for embedding.",
    )
    args = parser.parse_args()
    dataset = args.dataset

    setup_logging(dataset)
    logging.info("=" * 60)
    logging.info(f"ZINC-250k Embedding | dataset={dataset} | device={DEVICE}")
    logging.info("=" * 60)

    # Load ZINC-250k SMILES
    if not ZINC_CSV.exists():
        raise FileNotFoundError(f"Cleaned ZINC CSV not found: {ZINC_CSV}")
    zinc_df = pd.read_csv(str(ZINC_CSV))
    smiles_list = zinc_df["smiles"].tolist()
    logging.info(f"Loaded {len(smiles_list)} ZINC-250k SMILES from {ZINC_CSV}")

    # Output directory for this dataset
    out_dir = EMBED_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for loss_type, models_dict in FINETUNED_MODELS.items():
        for folder_name, info in models_dict.items():
            col_name   = info["col_name"]
            tokenizer_name = info["tokenizer"]
            model_path = EXTRAS_ROOT / f"{loss_type}_{DATASET_DIR_NAME[dataset]}" / folder_name
            out_path   = out_dir / f"{col_name}.npy"

            if out_path.exists():
                logging.info(f"Already exists, skipping: {out_path.name}")
                continue

            if not model_path.exists():
                logging.warning(f"Model not found, skipping: {model_path}")
                continue

            logging.info(f"\nLoading model: {col_name}  ({loss_type})")
            logging.info(f"  Model path : {model_path}")
            logging.info(f"  Output     : {out_path}")

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
            model.to(DEVICE)
            model.eval()

            embeddings = get_embeddings(smiles_list, tokenizer, model, batch_size=128)
            np.save(str(out_path), embeddings)

            logging.info(f"  Saved: shape={embeddings.shape}, dtype={embeddings.dtype}")

            del tokenizer, model, embeddings
            torch.cuda.empty_cache()

    logging.info("\nDone. All ZINC-250k embeddings saved to:")
    logging.info(f"  {out_dir}")


if __name__ == "__main__":
    main()
