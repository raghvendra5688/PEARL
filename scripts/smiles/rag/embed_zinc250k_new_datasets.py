"""
Embed ZINC-250k Using Finetuned HF LoRA Models — herg/dili/caco2/half_life

Mirrors embed_zinc250k.py (the original bace/bbbp/clintox/flavor script) for
the 4 new TDC datasets added in Phase 1-6 of the editor-response revision
(see manuscript/editor_response_suggestions.md). Kept as a separate script
because these datasets:
  - live under $PEARL_EXTRAS_V2, not the original $PEARL_EXTRAS
  - herg/dili use focal_loss + weighted_loss (like the originals) but
    caco2/half_life use huber_loss only (a regression target)

Produces one .npy file per (model, loss) combination: 6 for herg/dili
(3 base models x FL/WL), 3 for caco2/half_life (3 base models x Huber).

Usage:
    python embed_zinc250k_new_datasets.py --dataset herg
    python embed_zinc250k_new_datasets.py --dataset all

Input:
    data/zinc250k/zinc250k_cleaned.csv
    $PEARL_EXTRAS_V2/{loss}_{DATASET}/{model}_LoRA_Finetuned/

Output (outside git repo, per "big embeddings/indices -> EffiChem_Extras_v2"
convention):
    $PEARL_EXTRAS_V2/zinc_embeddings/{dataset}/{col_name}.npy
    Shape: (249455, hidden_dim)
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
ZINC_CSV    = REPO_ROOT / "data" / "zinc250k" / "zinc250k_cleaned.csv"
EXTRAS_ROOT = Path(os.environ.get("PEARL_EXTRAS_V2", "/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"))
EMBED_ROOT  = EXTRAS_ROOT / "zinc_embeddings"
RAG_OUT_ROOT = REPO_ROOT / "data" / "rag_features"
LOG_DIR     = REPO_ROOT / "logs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALID_DATASETS = ["herg", "dili", "caco2", "half_life"]
RAG_SPLITS = ["train", "eval", "test"]

# Folder-name (from LoRA-merge safe_model_name()) -> (col_name suffix, tokenizer source).
# Same 3 base models across all 4 new datasets (matches
# finetuned_model_embeddings_new_datasets.py's MODEL_INFO exactly).
MODEL_INFO = {
    "DeepChem__ChemBERTa__77M__MTR_LoRA_Finetuned": ("ChemBERTa_77M_MTR", "DeepChem/ChemBERTa-77M-MTR"),
    "DeepChem__ChemBERTa__77M__MLM_LoRA_Finetuned": ("ChemBERTa_77M_MLM", "DeepChem/ChemBERTa-77M-MLM"),
    "ibm__MoLFormer__XL__both__10pct_LoRA_Finetuned": ("MolFormer_Finetuned", "ibm/MoLFormer-XL-both-10pct"),
}

LOSS_SUFFIX = {"focal_loss": "FL", "weighted_loss": "WL", "huber_loss": "Huber"}

# dataset -> (EffiChem_Extras_v2 dir-name fragment, loss keys used)
DATASET_CFG = {
    "herg":      {"extras_suffix": "HERG",      "losses": ["focal_loss", "weighted_loss"]},
    "dili":      {"extras_suffix": "DILI",      "losses": ["focal_loss", "weighted_loss"]},
    "caco2":     {"extras_suffix": "CACO2",     "losses": ["huber_loss"]},
    "half_life": {"extras_suffix": "HALF_LIFE", "losses": ["huber_loss"]},
}


def setup_logging(dataset: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / f"embed_zinc250k_new_datasets_{dataset}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    logger = logging.getLogger()
    logger.handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    logger.addHandler(logging.StreamHandler())


def rag_features_already_extracted(dataset: str, col_name: str) -> bool:
    """True if rag_feature_extraction_new_datasets.py has already produced all
    3 split CSVs for this (dataset, col_name) -- in that case the raw ZINC
    embedding is no longer needed downstream, so re-embedding would be wasted work."""
    rag_dir = RAG_OUT_ROOT / dataset
    return all((rag_dir / f"{col_name}_{split}_rag.csv").exists() for split in RAG_SPLITS)


def load_tokenizer_safe(tokenizer_name: str):
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
    from safe_model_loading import load_tokenizer_safe as _load
    return _load(tokenizer_name, trust_remote_code=True)


def embed_smiles(model, tokenizer, smiles_list, batch_size: int = 128, log_every: int = 200) -> np.ndarray:
    """Mean-pooled last_hidden_state (matches finetuned_model_embeddings_new_datasets.py --
    pooler_output is randomly initialised when AutoModel loads a
    *ForSequenceClassification checkpoint, so it must not be used)."""
    all_embs = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        with torch.no_grad():
            output = model(**tokens)
            emb = output.last_hidden_state.mean(dim=1)
        all_embs.append(emb.cpu().numpy())
        if (i // batch_size) % log_every == 0:
            logging.info(f"  Embedded {min(i + batch_size, len(smiles_list))}/{len(smiles_list)} molecules")
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed ZINC-250k with new-dataset finetuned HF LoRA models.")
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
        logging.info(f"ZINC-250k HF Embedding | dataset={dataset} | device={DEVICE}")
        logging.info(f"Loaded {len(smiles_list)} ZINC-250k SMILES from {ZINC_CSV}")
        logging.info("=" * 60)

        cfg = DATASET_CFG[dataset]
        out_dir = EMBED_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        for loss_key in cfg["losses"]:
            loss_dir = f"{loss_key}_{cfg['extras_suffix']}"
            for folder_name, (col_suffix, tokenizer_name) in MODEL_INFO.items():
                col_name = f"{col_suffix}_{LOSS_SUFFIX[loss_key]}"
                out_path = out_dir / f"{col_name}.npy"
                if out_path.exists():
                    logging.info(f"Already exists, skipping: {out_path.name}")
                    continue

                if rag_features_already_extracted(dataset, col_name):
                    logging.info(f"RAG features already extracted for {col_name}, skipping ZINC re-embedding.")
                    continue

                model_path = EXTRAS_ROOT / loss_dir / folder_name
                if not model_path.exists():
                    logging.warning(f"Model not found, skipping: {model_path}")
                    continue

                logging.info(f"\nLoading model: {col_name}  ({model_path})")
                tokenizer = load_tokenizer_safe(tokenizer_name)
                model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
                model.to(DEVICE)
                model.eval()

                embeddings = embed_smiles(model, tokenizer, smiles_list, batch_size=128)
                np.save(str(out_path), embeddings)
                logging.info(f"  Saved: {out_path}  shape={embeddings.shape}, dtype={embeddings.dtype}")

                del model, tokenizer, embeddings
                torch.cuda.empty_cache()

        logging.info(f"\nDone. All HF ZINC-250k embeddings for {dataset} saved to: {out_dir}")


if __name__ == "__main__":
    main()
