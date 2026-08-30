"""Inference wrapper for BBBP/DILI's rank1 model: RAFE (Retrieval-Augmented Feature
Enhancement) -- "FT Embed+PC further augmented with ZINC-250k chemical-neighbourhood
features" (PLAN.md). The feature vector is the embedding, the physicochemical
features, AND the retrieval features, concatenated in that order: 2560-dim embedding
+ 473-dim PC + 60-dim RAG = 3093, matching DILI's training script
(`rag_modelling_unimol_new_datasets.py::build_feature_matrix()`:
`X = np.hstack([emb, pc_feats, rag_feats])`). PC features are cleaned the same way
as Caco-2's in `pc_only.py`, reused here rather than duplicated.

Two HF folders are needed per dataset (unlike every other inference module): the
retrieval artifacts (rank1_rafe/: XGBoost.pkl, the FAISS index, zinc250k_metadata.pkl,
pca_model.pkl) and a separate folder holding the embedding backbone that produces
the query vector used to search that index. model_registry.py downloads both and
passes their paths here.

BBBP uses MolFormer embeddings (768-dim, raw pooled vector -- matches BBBP's
pca_model.pkl, mean_.shape == (768,)). DILI uses Uni-Mol embeddings, but the full
2560-dim combined vector (512-dim CLS + 2048-dim Morgan fingerprint) that
UniMolLoRAClassifier.get_embeddings() returns, not just the 512-dim CLS slice --
matches DILI's pca_model.pkl, mean_.shape == (2560,).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import faiss
import joblib
import numpy as np
import pandas as pd

from ..vendor.pc_only_modelling import extract_pc_features
from ..vendor.rag_feature_extraction import compute_rag_features
from ..vendor.safe_model_loading import load_tokenizer_safe
from . import unimol_e2e
from ._molformer_compat import ensure_transformers_onnx_shim
from .pc_only import _sanitize_single_row
from .unimol_e2e import DEVICE

MOLFORMER_MODEL_ID = "ibm/MoLFormer-XL-both-10pct"


class RafeBundle:
    def __init__(self, xgboost_model, index, meta: dict, pca, backbone, backbone_type: str, tokenizer=None):
        self.xgboost_model = xgboost_model
        self.index = index
        self.meta = meta
        self.pca = pca
        self.backbone = backbone
        self.backbone_type = backbone_type
        self.tokenizer = tokenizer


def _find_index_file(faiss_dir: Path) -> Path:
    matches = list(faiss_dir.glob("*.index"))
    if not matches:
        raise FileNotFoundError(f"No .index file found in {faiss_dir}")
    return matches[0]


def load_model(save_dir: Path, backbone_dir: Path, backbone_type: Optional[str]) -> RafeBundle:
    save_dir = Path(save_dir)
    backbone_dir = Path(backbone_dir)

    xgboost_model = joblib.load(str(save_dir / "XGBoost.pkl"))

    with open(save_dir / "pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(save_dir / "zinc250k_metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    index = faiss.read_index(str(_find_index_file(save_dir / "faiss_index")))

    if backbone_type == "molformer":
        from transformers import AutoModel

        # revision="main" overrides safe_model_loading.py's pinned revision, which
        # is broken against this webapp's newer transformers -- see ARCHITECTURE.md.
        tokenizer = load_tokenizer_safe(MOLFORMER_MODEL_ID, trust_remote_code=True, revision="main")

        # The revision fix above only affects the tokenizer/config fetched fresh
        # from the Hub. This checkpoint's own bundled configuration_molformer.py
        # (baked in at export time) has the same transformers.onnx import problem
        # regardless -- see _molformer_compat.py.
        ensure_transformers_onnx_shim()
        model = AutoModel.from_pretrained(str(backbone_dir), trust_remote_code=True)
        model.to(DEVICE)
        model.eval()
        return RafeBundle(xgboost_model, index, meta, pca, model, backbone_type, tokenizer)

    if backbone_type == "unimol":
        # Reuses unimol_e2e's own loader (handles the safetensors format our HF
        # upload ships -- see unimol_e2e.py). Only .get_embeddings() is used here,
        # not the classification head.
        model = unimol_e2e.load_model(backbone_dir)
        return RafeBundle(xgboost_model, index, meta, pca, model, backbone_type)

    raise ValueError(f"Unknown RAFE backbone type: {backbone_type!r}")


def _query_embedding(bundle: RafeBundle, smiles: str) -> np.ndarray:
    if bundle.backbone_type == "molformer":
        from ..vendor.finetuned_model_embeddings import get_embeddings

        return get_embeddings([smiles], bundle.tokenizer, bundle.backbone).astype(np.float32)

    # unimol: full 2560-dim combined embedding, unsliced -- see module docstring.
    return bundle.backbone.get_embeddings([smiles]).astype(np.float32)


def predict(bundle: RafeBundle, smiles: str) -> dict:
    raw_embedding = _query_embedding(bundle, smiles)

    # A normalized copy for the FAISS search -- faiss.normalize_L2() mutates in
    # place, and training concatenated the raw (unnormalized) embedding into the
    # feature vector below, so raw_embedding itself must stay untouched.
    search_embs = raw_embedding.copy()
    faiss.normalize_L2(search_embs)

    rag_features_df, _ = compute_rag_features(
        query_embs=search_embs,
        query_smiles=[smiles],
        index=bundle.index,
        meta=bundle.meta,
        pca=bundle.pca,
        batch_size=1,
    )

    pc_features_df = _sanitize_single_row(extract_pc_features(pd.Series([smiles])))

    # Order matters -- must match training's np.hstack([emb, pc_feats, rag_feats])
    # exactly; XGBoost has no column names here, only position.
    features = np.hstack(
        [raw_embedding, pc_features_df.to_numpy(dtype=np.float32), rag_features_df.to_numpy(dtype=np.float32)]
    )

    proba = bundle.xgboost_model.predict_proba(features)[0]
    predicted_class = int(np.argmax(proba))
    return {
        "predicted_class": predicted_class,
        "confidence": float(proba[predicted_class]),
    }
