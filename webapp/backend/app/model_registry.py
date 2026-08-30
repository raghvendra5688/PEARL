"""Server-side registry of runnable rank1 models. Internal HF Hub paths never leave
this module as JSON -- only `public_dataset_info()` may cross the API boundary. See
ARCHITECTURE.md #4 (lazy loading) and #9 (non-disclosure).

Credibility numbers are copied verbatim from
results/summary/{classification,regression}_mean_se_by_method.csv. For BACE/HERG,
the classification CSV has only one Uni-Mol E2E LoRA row per dataset (the
weighted-loss variant), so that's the one used here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pearl.webapp")

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "raghvendramall/pearl-benchmark-models")
MODELS_CACHE_DIR = Path(
    os.environ.get("PEARL_MODELS_CACHE", Path(__file__).resolve().parents[1] / "models_cache")
)


@dataclass(frozen=True)
class Credibility:
    metric: str
    mean: float
    se: float


@dataclass(frozen=True)
class DatasetEntry:
    key: str
    display_name: str                # PLAN.md §3 -- safe to show visitors
    task: str                        # "classification" | "regression"
    method_family: str                # public-safe label, no internal slug/rank naming
    hf_subfolder: str                 # internal HF Hub path -- never serialize this
    inference_module: str             # "unimol_e2e" | "pc_only" | "rafe" | "ft_embed" | "chemprop"
    credibility: tuple[Credibility, ...]
    target_transform: Optional[str] = None   # None | "log1p" -- see invert_target_transform()
    unit_label: Optional[str] = None          # regression only
    # RAFE only: the embedding backbone lives in a *separate* HF folder from the
    # retrieval artifacts (rank1_rafe/), unlike Half-Life's FT-Embed backbone,
    # which is nested inside its own rank1 folder -- see ARCHITECTURE.md.
    extra_hf_subfolders: tuple[str, ...] = ()
    rafe_backbone: Optional[str] = None       # "molformer" | "unimol" -- RAFE only
    # Multiclass only (Flavor) -- ordered class names for predicted_class indices.
    class_labels: Optional[tuple[str, ...]] = None


REGISTRY: dict[str, DatasetEntry] = {
    "BACE": DatasetEntry(
        key="BACE",
        display_name="BACE1 Enzyme Inhibition",
        task="classification",
        method_family="Uni-Mol + LoRA (end-to-end)",
        hf_subfolder="BACE/rank1_e2e-lora-uni-mol",
        inference_module="unimol_e2e",
        credibility=(
            Credibility("MCC", 0.6234, 0.0651),
            Credibility("AUC", 0.8909, 0.0275),
        ),
    ),
    "HERG": DatasetEntry(
        key="HERG",
        display_name="Cardiotoxicity (hERG Channel Inhibition)",
        task="classification",
        method_family="Uni-Mol + LoRA (end-to-end)",
        hf_subfolder="HERG/rank1_e2e-lora-uni-mol",
        inference_module="unimol_e2e",
        credibility=(
            Credibility("MCC", 0.6093, 0.0154),
            Credibility("AUC", 0.8809, 0.0065),
        ),
    ),
    "CACO2": DatasetEntry(
        key="CACO2",
        display_name="Intestinal Permeability",
        task="regression",
        method_family="Physicochemical descriptors (CatBoost)",
        hf_subfolder="CACO2/rank1_pc-only",
        inference_module="pc_only",
        credibility=(
            Credibility("R2", 0.7444, 0.0358),
            Credibility("Spearman", 0.8534, 0.0211),
        ),
        target_transform=None,
        unit_label="log Papp (cm/s)",
    ),
    "BBBP": DatasetEntry(
        key="BBBP",
        display_name="Blood-Brain Barrier Permeability",
        task="classification",
        method_family="Retrieval-augmented features (MolFormer + XGBoost)",
        hf_subfolder="BBBP/rank1_rafe",
        extra_hf_subfolders=("BBBP/rank5_e2e-lora-hf",),
        rafe_backbone="molformer",
        inference_module="rafe",
        credibility=(
            Credibility("MCC", 0.7980, 0.0281),
            Credibility("AUC", 0.9400, 0.0166),
        ),
    ),
    "DILI": DatasetEntry(
        key="DILI",
        display_name="Drug-Induced Liver Injury Risk",
        task="classification",
        method_family="Retrieval-augmented features (Uni-Mol + XGBoost)",
        hf_subfolder="DILI/rank1_rafe",
        extra_hf_subfolders=("DILI/rank5_e2e-lora-uni-mol",),
        rafe_backbone="unimol",
        inference_module="rafe",
        credibility=(
            Credibility("MCC", 0.7897, 0.0515),
            Credibility("AUC", 0.8748, 0.0390),
        ),
    ),
    "HALF_LIFE": DatasetEntry(
        key="HALF_LIFE",
        display_name="Pharmacokinetic Half-Life",
        task="regression",
        method_family="Finetuned embeddings (MolFormer + CatBoost)",
        hf_subfolder="HALF_LIFE/rank1_ft-embed",
        inference_module="ft_embed",
        credibility=(
            Credibility("R2", 0.0832, 0.0522),
            Credibility("Spearman", 0.5995, 0.0562),
        ),
        target_transform="log1p",
        unit_label="hours",
    ),
    # class_labels: alphabetical LabelEncoder order of the training data's taste
    # column (0=bitter, 1=sour, 2=sweet, 3=umami, 4=undefined) -- confirmed
    # against the original training CSV.
    "FLAVOR": DatasetEntry(
        key="FLAVOR",
        display_name="Taste Profile",
        task="classification",
        method_family="Graph neural network (Chemprop)",
        hf_subfolder="FLAVOR/rank1_chemprop",
        inference_module="chemprop",
        credibility=(Credibility("MCC", 0.8086, 0.0115),),
        class_labels=("bitter", "sour", "sweet", "umami", "undefined"),
    ),
}

# Unused by any router (the frontend keeps its own catalog in lib/datasets.ts) --
# kept as documentation of registry state. BBBP is registered (the backend can
# serve it directly) but stays out of the frontend's enabled list -- its RAFE
# feature vector is 3 columns short of what its model expects, failing every
# call; see ARCHITECTURE.md. ClinTox is excluded entirely -- its model was never
# uploaded to the HF repo.
CATALOG_DISPLAY_ORDER: tuple[str, ...] = (
    "BACE", "HERG", "CACO2", "BBBP", "DILI", "HALF_LIFE", "FLAVOR",
)
NOT_YET_AVAILABLE: dict[str, str] = {}

_lock = threading.Lock()
_loaded: dict[str, object] = {}


def is_loaded(key: str) -> bool:
    return key in _loaded


def get_dataset_entry(key: str) -> DatasetEntry:
    entry = REGISTRY.get(key)
    if entry is None:
        raise KeyError(f"Unknown or not-yet-available dataset: {key}")
    return entry


def public_dataset_info(key: str) -> dict:
    """The only sanctioned way to expose a registry entry over the API. Deliberately
    whitelists fields rather than serializing the dataclass -- see ARCHITECTURE.md #9."""
    entry = get_dataset_entry(key)
    return {
        "key": entry.key,
        "display_name": entry.display_name,
        "task": entry.task,
        "method_family": entry.method_family,
        "unit_label": entry.unit_label,
        "class_labels": list(entry.class_labels) if entry.class_labels else None,
        "credibility": [
            {"metric": c.metric, "mean": c.mean, "se": c.se} for c in entry.credibility
        ],
    }


def _download_snapshot(entry: DatasetEntry) -> tuple[Path, tuple[Path, ...]]:
    """Downloads the primary rank1 folder plus any extra subfolders (RAFE's
    separate embedding backbone folder) in one call. Returns (primary_path,
    extra_paths) -- extra_paths is empty for datasets that don't need one."""
    from huggingface_hub import snapshot_download

    all_subfolders = (entry.hf_subfolder, *entry.extra_hf_subfolders)
    local_root = snapshot_download(
        repo_id=HF_MODEL_REPO,
        allow_patterns=[f"{sf}/*" for sf in all_subfolders],
        cache_dir=str(MODELS_CACHE_DIR),
        token=os.environ.get("HF_TOKEN") or None,
    )
    primary_path = Path(local_root) / entry.hf_subfolder
    extra_paths = tuple(Path(local_root) / sf for sf in entry.extra_hf_subfolders)
    return primary_path, extra_paths


def get_model(key: str):
    """Lazily download (first call) and load a dataset's rank1 model, then serve warm
    from an in-process cache. See ARCHITECTURE.md #4 for why this is lazy, not eager."""
    if key in _loaded:
        return _loaded[key]
    with _lock:
        if key in _loaded:
            return _loaded[key]
        entry = get_dataset_entry(key)
        local_path, extra_paths = _download_snapshot(entry)

        if entry.inference_module == "unimol_e2e":
            from .inference.unimol_e2e import load_model

            model = load_model(local_path)
        elif entry.inference_module == "pc_only":
            from .inference.pc_only import load_model

            model = load_model(local_path)
        elif entry.inference_module == "rafe":
            from .inference.rafe import load_model

            model = load_model(local_path, extra_paths[0], entry.rafe_backbone)
        elif entry.inference_module == "ft_embed":
            from .inference.ft_embed import load_model

            model = load_model(local_path)
        elif entry.inference_module == "chemprop":
            from .inference.chemprop_flavor import load_model

            model = load_model(local_path)
        else:
            raise ValueError(f"Unknown inference module: {entry.inference_module}")

        _loaded[key] = model
        return model


async def _warm_one(key: str) -> None:
    try:
        await asyncio.to_thread(get_model, key)
        logger.info("Warmed model: %s", key)
    except Exception:
        logger.exception("Failed to warm model: %s -- will retry lazily on first request", key)


async def warm_all_models() -> None:
    """Eagerly downloads and loads every registry model at startup instead of on
    the first request -- a cold first request could take minutes, long enough for
    a dev proxy or load balancer to reset the connection. See ARCHITECTURE.md #4.

    BACE warms first, alone, since it triggers unimol_tools' shared base-checkpoint
    download on a true first run, and two threads writing that file at once is a
    real risk. Every other dataset warms concurrently afterward, once that file is
    guaranteed to exist.
    """
    if "BACE" in REGISTRY:
        await _warm_one("BACE")
    await asyncio.gather(*(_warm_one(key) for key in REGISTRY if key != "BACE"))
