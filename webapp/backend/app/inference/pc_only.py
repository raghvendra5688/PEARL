"""Inference wrapper for Caco-2's rank1 model: 473-dim physicochemical descriptor
vector + CatBoost. See ARCHITECTURE.md #5 for why extract_pc_features() is reused
unmodified here, and why sanitize_features() is adapted for single-row inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from ..vendor.pc_only_modelling import extract_pc_features, invert_target_transform


def load_model(save_dir: Path):
    """CatBoost/XGBoost/LightGBM models are all saved with plain joblib.dump and load
    identically -- see ARCHITECTURE.md #8. No per-library branch needed."""
    return joblib.load(str(Path(save_dir) / "CatBoost.pkl"))


def _sanitize_single_row(features: pd.DataFrame) -> pd.DataFrame:
    """Single-row variant of pc_only_modelling.sanitize_features(). The original
    fills NaNs with the batch's own median, which is meaningless for one row --
    fills with 0 instead."""
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    features = features.clip(lower=-1e6, upper=1e6)
    return features


def predict(model, smiles: str, task: str, target_transform: Optional[str]) -> dict:
    features = extract_pc_features(pd.Series([smiles]))
    features = _sanitize_single_row(features)

    if task == "classification":
        proba = model.predict_proba(features)[0]
        predicted_class = int(np.argmax(proba))
        return {
            "predicted_class": predicted_class,
            "confidence": float(proba[predicted_class]),
        }

    raw = float(model.predict(features)[0])
    value = float(invert_target_transform(np.array([raw]), target_transform)[0])
    return {"value": value}
