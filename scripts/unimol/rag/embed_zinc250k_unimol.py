"""
Embed ZINC-250k Using Finetuned Uni-Mol LoRA Models

For a given dataset, loads the 2 task-specific finetuned Uni-Mol models (focal loss
and weighted loss) and embeds all 249,455 ZINC-250k molecules into 2560-dim vectors
(512-dim Uni-Mol CLS + 2048-dim Morgan ECFP4).

Produces 2 .npy files per dataset (8 total across all datasets), stored outside
the git repo to avoid >50 MB file limits.

Usage:
    python "Uni-Mol/RAG Pipeline/embed_zinc250k_unimol.py" --dataset bace
    python "Uni-Mol/RAG Pipeline/embed_zinc250k_unimol.py" --dataset bbbp
    python "Uni-Mol/RAG Pipeline/embed_zinc250k_unimol.py" --dataset clintox
    python "Uni-Mol/RAG Pipeline/embed_zinc250k_unimol.py" --dataset flavor

Input:
    data/zinc250k/zinc250k_cleaned.csv
    /export/cse/rmall/Raghvendra/EffiChem_Extras/{loss_type}_{DATASET}/dptech__Uni__Mol_LoRA_Finetuned/

Output (outside git repo):
    /export/cse/rmall/Raghvendra/EffiChem_Extras/zinc_embeddings_unimol/{dataset}/UniMol_FL.npy
    /export/cse/rmall/Raghvendra/EffiChem_Extras/zinc_embeddings_unimol/{dataset}/UniMol_WL.npy
    Shape: (249455, 2560)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Project imports ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAINER_DIR = _SCRIPT_DIR.parent / "finetuning"
sys.path.insert(0, str(_TRAINER_DIR))
from unimol_lora_trainer import load_finetuned_unimol, UniMolLoRAClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
ZINC_CSV    = REPO_ROOT / "data" / "zinc250k" / "zinc250k_cleaned.csv"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS", "/export/cse/rmall/Raghvendra/EffiChem_Extras"))
EMBED_ROOT  = EXTRAS_ROOT / "zinc_embeddings_unimol"
LOG_DIR     = REPO_ROOT / "logs"

# Maps --dataset arg to the directory name suffix used in EffiChem_Extras
DATASET_DIR_NAME = {
    "bace":    "BACE",
    "bbbp":    "BBBP",
    "clintox": "clintox",
    "flavor":  "flavor",
}

DATASET_NUM_CLASSES = {
    "bace":    2,
    "bbbp":    2,
    "clintox": 2,
    "flavor":  None,   # read from config.json saved with the checkpoint
}

MODEL_FOLDER = "dptech__Uni__Mol_LoRA_Finetuned"
VALID_DATASETS = ["bace", "bbbp", "clintox", "flavor"]

# col_name → loss_type folder prefix
LOSS_VARIANTS = {
    "UniMol_FL": "focal_loss",
    "UniMol_WL": "weighted_loss",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(dataset: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / f"embed_zinc250k_unimol_{dataset}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_smiles(
    model: UniMolLoRAClassifier,
    smiles_list: list,
    batch_size: int = 16,
    log_every: int = 50,
) -> np.ndarray:
    """Returns (N, 2560) float32 array."""
    all_embs = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]
        embs  = model.get_embeddings(batch, batch_size=len(batch))
        all_embs.append(embs)
        if (i // batch_size) % log_every == 0:
            logging.info(
                f"  Embedded {min(i + batch_size, len(smiles_list))}/{len(smiles_list)} molecules"
            )
    return np.concatenate(all_embs, axis=0).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed ZINC-250k with finetuned Uni-Mol LoRA models."
    )
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
    logging.info(
        f"ZINC-250k Uni-Mol Embedding | dataset={dataset} | device={DEVICE}"
    )
    logging.info("=" * 60)

    if not ZINC_CSV.exists():
        raise FileNotFoundError(f"Cleaned ZINC CSV not found: {ZINC_CSV}")
    zinc_df     = pd.read_csv(str(ZINC_CSV))
    smiles_list = zinc_df["smiles"].tolist()
    logging.info(f"Loaded {len(smiles_list)} ZINC-250k SMILES from {ZINC_CSV}")

    out_dir = EMBED_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    dir_name     = DATASET_DIR_NAME[dataset]
    num_classes  = DATASET_NUM_CLASSES[dataset]

    for col_name, loss_type in LOSS_VARIANTS.items():
        out_path   = out_dir / f"{col_name}.npy"
        model_path = EXTRAS_ROOT / f"{loss_type}_{dir_name}" / MODEL_FOLDER

        if out_path.exists():
            logging.info(f"Already exists, skipping: {out_path.name}")
            continue

        if not model_path.exists():
            logging.warning(f"Model not found, skipping: {model_path}")
            continue

        logging.info(f"\nLoading model: {col_name}  ({loss_type})")
        logging.info(f"  Model path : {model_path}")
        logging.info(f"  Output     : {out_path}")

        try:
            model = load_finetuned_unimol(model_path, num_classes=num_classes)
            model.to(DEVICE)
            model.eval()
        except Exception as e:
            logging.error(f"  Failed to load model: {e}")
            continue

        embeddings = embed_smiles(model, smiles_list, batch_size=16)
        np.save(str(out_path), embeddings)
        logging.info(
            f"  Saved: shape={embeddings.shape}, dtype={embeddings.dtype}"
        )

        del model, embeddings
        torch.cuda.empty_cache()

    logging.info("\nDone. All Uni-Mol ZINC-250k embeddings saved to:")
    logging.info(f"  {out_dir}")


if __name__ == "__main__":
    main()
