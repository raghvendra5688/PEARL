"""Nearest-neighbour search against ZINC-250k for the Step 2 "chemical context" panel.

A local Tanimoto scan over pre-computed Morgan fingerprints, not a hosted vector DB --
see ARCHITECTURE.md #7 for why this is the zero-cost choice for a free-tier Space.
"""

from __future__ import annotations

import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from .formula import molecular_formula

logger = logging.getLogger("pearl.webapp")

# A copy, not a reference into the main repo -- so webapp/ deploys standalone.
# See ARCHITECTURE.md.
BACKEND_ROOT = Path(__file__).resolve().parents[2]
ZINC_CSV = BACKEND_ROOT / "data" / "zinc250k_cleaned.csv"

# Disk cache for the precomputed fingerprint index -- rebuilding it from the raw
# CSV took ~213s, the largest single cost of every server restart. Invalidated by
# comparing mtimes against the source CSV.
CACHE_DIR = BACKEND_ROOT / "models_cache"
FINGERPRINT_CACHE = CACHE_DIR / "zinc_fingerprint_cache.pkl"

_lock = threading.Lock()
_state: dict = {}
# Same generator instance for the reference set and every query fingerprint --
# radius/bit-length must match for Tanimoto similarity to be meaningful. Also
# avoids the deprecated AllChem.GetMorganFingerprintAsBitVect, which printed one
# warning per molecule (249,455 of them) on every cold load.
_MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)


def _build_from_csv() -> dict:
    t0 = time.monotonic()
    df = pd.read_csv(ZINC_CSV)
    fps = []
    keep_idx = []
    for i, smi in enumerate(df["smiles"].astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(_MORGAN_GEN.GetFingerprint(mol))
        keep_idx.append(i)

    state = {"df": df.iloc[keep_idx].reset_index(drop=True), "fps": fps}
    logger.info("ZINC-250k fingerprint index built from CSV in %.1fs (%d molecules)", time.monotonic() - t0, len(fps))
    return state


def _load() -> dict:
    """Loads ZINC-250k once and precomputes a Morgan fingerprint per row -- from a
    disk cache when one exists and is newer than the source CSV, otherwise from
    scratch (and then writes the cache for next time). Cached in module state for
    the lifetime of the process -- see ARCHITECTURE.md #4."""
    if _state:
        return _state
    with _lock:
        if _state:
            return _state

        if FINGERPRINT_CACHE.exists() and FINGERPRINT_CACHE.stat().st_mtime >= ZINC_CSV.stat().st_mtime:
            t0 = time.monotonic()
            try:
                with open(FINGERPRINT_CACHE, "rb") as f:
                    state = pickle.load(f)
                logger.info("ZINC-250k fingerprint index loaded from disk cache in %.1fs", time.monotonic() - t0)
                _state.update(state)
                return _state
            except Exception:
                logger.exception("Fingerprint cache exists but failed to load -- rebuilding from CSV")

        state = _build_from_csv()
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(FINGERPRINT_CACHE, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # Best-effort -- a failed cache write just means the next restart
            # rebuilds from CSV again, not a request-serving failure now.
            logger.exception("Failed to write fingerprint cache -- will rebuild next time too")

        _state.update(state)
        return _state


def is_warm() -> bool:
    """Non-blocking check -- never triggers the load itself, just reports whether
    it's already finished. Used by /api/health."""
    return bool(_state)


def warm() -> None:
    """Triggers the one-time ZINC-250k load eagerly at process startup, instead of
    on a visitor's first /api/profile request -- that first load could take long
    enough for a dev proxy or load balancer to reset the connection. Same
    reasoning as model_registry.warm_all_models() -- see ARCHITECTURE.md #4/#7.
    """
    _load()


def nearest_neighbours(smiles: str, k: int = 5) -> Optional[list[dict]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    state = _load()
    query_fp = _MORGAN_GEN.GetFingerprint(mol)
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, state["fps"])

    top_k_idx = np.argsort(similarities)[::-1][:k]
    df = state["df"]
    # Formula is computed only for the returned top-k, not all 250K rows -- cheap
    # here, would be wasteful precomputed for every row at startup.
    return [
        {
            "smiles": df.iloc[int(i)]["smiles"],
            "formula": molecular_formula(df.iloc[int(i)]["smiles"]),
            "similarity": round(float(similarities[int(i)]), 4),
            "logp": round(float(df.iloc[int(i)]["logP"]), 3),
            "qed": round(float(df.iloc[int(i)]["qed"]), 3),
            "sas": round(float(df.iloc[int(i)]["SAS"]), 3),
        }
        for i in top_k_idx
    ]
