"""
Embed ZINC-250k Using Finetuned Uni-Mol LoRA Models — herg/dili/caco2/half_life

Mirrors embed_zinc250k_unimol.py (the original bace/bbbp/clintox/flavor script)
for the 4 new TDC datasets. Separate script because these live under
$PEARL_EXTRAS_V2 (not $PEARL_EXTRAS) and caco2/half_life use a single Huber
loss variant (regression) rather than FL+WL.

Produces 2 .npy files for herg/dili (UniMol_FL, UniMol_WL) and 1 for
caco2/half_life (UniMol_Huber), each (249455, 2560) -- 512-dim Uni-Mol CLS +
2048-dim Morgan ECFP4.

Usage:
    python embed_zinc250k_unimol_new_datasets.py --dataset herg
    python embed_zinc250k_unimol_new_datasets.py --dataset all

Input:
    data/zinc250k/zinc250k_cleaned.csv
    $PEARL_EXTRAS_V2/{loss}_{DATASET}/dptech__Uni__Mol_LoRA_Finetuned/

Output (outside git repo):
    $PEARL_EXTRAS_V2/zinc_embeddings_unimol/{dataset}/UniMol_FL.npy      (herg/dili)
    $PEARL_EXTRAS_V2/zinc_embeddings_unimol/{dataset}/UniMol_WL.npy      (herg/dili)
    $PEARL_EXTRAS_V2/zinc_embeddings_unimol/{dataset}/UniMol_Huber.npy   (caco2/half_life)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAINER_DIR = _SCRIPT_DIR.parent / "finetuning"
sys.path.insert(0, str(_TRAINER_DIR))
from unimol_lora_trainer import load_finetuned_unimol  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
ZINC_CSV    = REPO_ROOT / "data" / "zinc250k" / "zinc250k_cleaned.csv"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
EMBED_ROOT  = EXTRAS_ROOT / "zinc_embeddings_unimol"
RAG_OUT_ROOT = REPO_ROOT / "data" / "rag_features_unimol"
LOG_DIR     = REPO_ROOT / "logs"

MODEL_FOLDER = "dptech__Uni__Mol_LoRA_Finetuned"
VALID_DATASETS = ["herg", "dili", "caco2", "half_life"]
RAG_SPLITS = ["train", "eval", "test"]

LOSS_SUFFIX = {"focal_loss": "FL", "weighted_loss": "WL", "huber_loss": "Huber"}

# dataset -> (EffiChem_Extras_v2 dir-name fragment, loss keys used).
# num_classes/task_type are read from each checkpoint's config.json by
# load_finetuned_unimol() -- no need to hardcode them here.
DATASET_CFG = {
    "herg":      {"extras_suffix": "HERG",      "losses": ["focal_loss", "weighted_loss"]},
    "dili":      {"extras_suffix": "DILI",      "losses": ["focal_loss", "weighted_loss"]},
    "caco2":     {"extras_suffix": "CACO2",     "losses": ["huber_loss"]},
    "half_life": {"extras_suffix": "HALF_LIFE", "losses": ["huber_loss"]},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(dataset: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / f"embed_zinc250k_unimol_new_datasets_{dataset}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    logger = logging.getLogger()
    logger.handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    logger.addHandler(logging.StreamHandler())


def rag_features_already_extracted(dataset: str, col_name: str) -> bool:
    """True if rag_feature_extraction_unimol_new_datasets.py has already produced
    all 3 split CSVs for this (dataset, col_name) -- in that case the raw ZINC
    embedding is no longer needed downstream, so re-embedding would be wasted work."""
    rag_dir = RAG_OUT_ROOT / dataset
    return all((rag_dir / f"{col_name}_{split}_rag.csv").exists() for split in RAG_SPLITS)


def embed_smiles(model, smiles_list: list, batch_size: int = 64, log_every: int = 50) -> np.ndarray:
    all_embs = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]
        embs = model.get_embeddings(batch, batch_size=len(batch))
        all_embs.append(embs)
        if (i // batch_size) % log_every == 0:
            logging.info(f"  Embedded {min(i + batch_size, len(smiles_list))}/{len(smiles_list)} molecules")
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed ZINC-250k with new-dataset finetuned Uni-Mol LoRA models.")
    parser.add_argument("--dataset", choices=VALID_DATASETS + ["all"], default="all")
    args = parser.parse_args()
    datasets = VALID_DATASETS if args.dataset == "all" else [args.dataset]

    if not ZINC_CSV.exists():
        raise FileNotFoundError(f"Cleaned ZINC CSV not found: {ZINC_CSV}")
    zinc_df = pd.read_csv(str(ZINC_CSV))
    smiles_list = zinc_df["smiles"].tolist()

    for dataset in datasets:
        setup_logging(dataset)
        logging.info("=" * 60)
        logging.info(f"ZINC-250k Uni-Mol Embedding | dataset={dataset} | device={DEVICE}")
        logging.info(f"Loaded {len(smiles_list)} ZINC-250k SMILES from {ZINC_CSV}")
        logging.info("=" * 60)

        cfg = DATASET_CFG[dataset]
        out_dir = EMBED_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        for loss_key in cfg["losses"]:
            col_name = f"UniMol_{LOSS_SUFFIX[loss_key]}"
            out_path = out_dir / f"{col_name}.npy"
            if out_path.exists():
                logging.info(f"Already exists, skipping: {out_path.name}")
                continue

            if rag_features_already_extracted(dataset, col_name):
                logging.info(f"RAG features already extracted for {col_name}, skipping ZINC re-embedding.")
                continue

            model_path = EXTRAS_ROOT / f"{loss_key}_{cfg['extras_suffix']}" / MODEL_FOLDER
            if not model_path.exists():
                logging.warning(f"Model not found, skipping: {model_path}")
                continue

            logging.info(f"\nLoading model: {col_name}  ({model_path})")
            try:
                model = load_finetuned_unimol(model_path)
                model.to(DEVICE)
                model.eval()
            except Exception as e:
                logging.error(f"  Failed to load model: {e}")
                continue

            embeddings = embed_smiles(model, smiles_list, batch_size=64)
            np.save(str(out_path), embeddings)
            logging.info(f"  Saved: {out_path}  shape={embeddings.shape}, dtype={embeddings.dtype}")

            del model, embeddings
            torch.cuda.empty_cache()

        logging.info(f"\nDone. All Uni-Mol ZINC-250k embeddings for {dataset} saved to: {out_dir}")


if __name__ == "__main__":
    main()
