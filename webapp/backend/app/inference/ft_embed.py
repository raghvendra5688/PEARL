"""Inference wrapper for Half-Life's rank1 model: "FT Embed" -- a CatBoost regressor
fed the raw pooled MolFormer (Huber-loss finetuned) embedding vector directly, with
no physicochemical features concatenated (that's a different rank, not rank1).

Unlike BACE/HERG's Uni-Mol checkpoint (a single self-contained folder) or RAFE's
two-folder split, Half-Life's embedding backbone is bundled inside its own rank1
folder (rank1_ft-embed/encoder_checkpoint/), so model_registry.py's normal
single-hf_subfolder download already covers it.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from ..vendor.finetuned_model_embeddings import DEVICE, get_embeddings
from ..vendor.pc_only_modelling import invert_target_transform
from ..vendor.safe_model_loading import load_tokenizer_safe
from ._molformer_compat import ensure_transformers_onnx_shim

MOLFORMER_MODEL_ID = "ibm/MoLFormer-XL-both-10pct"


class FtEmbedBundle:
    def __init__(self, catboost_model, tokenizer, encoder):
        self.catboost_model = catboost_model
        self.tokenizer = tokenizer
        self.encoder = encoder


def load_model(save_dir: Path) -> FtEmbedBundle:
    from transformers import AutoModel

    save_dir = Path(save_dir)
    catboost_model = joblib.load(str(save_dir / "CatBoost.pkl"))

    encoder_dir = save_dir / "encoder_checkpoint"
    # revision="main" overrides safe_model_loading.py's pinned revision, which is
    # broken against this webapp's newer transformers -- see ARCHITECTURE.md.
    tokenizer = load_tokenizer_safe(MOLFORMER_MODEL_ID, trust_remote_code=True, revision="main")

    # The checkpoint's own bundled configuration_molformer.py has the same
    # transformers.onnx import problem regardless of the revision override above.
    ensure_transformers_onnx_shim()
    encoder = AutoModel.from_pretrained(str(encoder_dir), trust_remote_code=True)
    encoder.to(DEVICE)
    encoder.eval()

    return FtEmbedBundle(catboost_model, tokenizer, encoder)


def predict(bundle: FtEmbedBundle, smiles: str) -> dict:
    embedding = get_embeddings([smiles], bundle.tokenizer, bundle.encoder)
    raw = float(bundle.catboost_model.predict(embedding)[0])
    value = float(invert_target_transform(np.array([raw]), "log1p")[0])
    return {"value": value}
