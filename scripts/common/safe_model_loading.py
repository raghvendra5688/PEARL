"""
Workaround for a transformers/torch version incompatibility discovered while
building the Phase 5 regression LoRA trainer: transformers>=4.something added
a hard check (check_torch_load_is_safe, see CVE-2025-32434) that refuses to
`torch.load` a non-safetensors checkpoint unless torch>=2.6 is installed.
The pinned environment here (pip-constraints-effichem.txt) has
transformers==4.57.3 + torch==2.5.0 -- deliberately pinned together to avoid
a *different* cross-package regression discovered earlier this session -- so
upgrading torch is not a safe first move.

DeepChem/ChemBERTa-77M-MLM and DeepChem/ChemBERTa-77M-MTR only ship
pytorch_model.bin (no .safetensors) on the HF Hub, so
AutoModelForSequenceClassification.from_pretrained(...) fails outright for
those two models under the current pinned environment. ibm/MoLFormer-XL-both-10pct
ships safetensors and is unaffected by *this* problem.

This affects every existing HF-based LoRA finetuning script
(scripts/smiles/finetuning/finetune_*.py) equally, not just the new
regression ones -- it was simply never hit before because no finetuning run
had actually been executed against this pinned environment yet.

Fix: convert the checkpoint to safetensors once into a local cache dir (no
env/package changes) and load from there instead of the bare model id.
lm_head.* tensors are dropped before conversion -- AutoModelForSequenceClassification
never uses them (it builds its own classification head), and they're tied to
the embedding weights, which safetensors refuses to serialize twice.

SECOND, UNRELATED bug found the same way: ibm/MoLFormer-XL-both-10pct's
`trust_remote_code=True` modeling file is fetched at HEAD by default, and its
latest revision ("Fix deprecated code (#7)", 2026-07-02) imports
`create_bidirectional_mask` from `transformers.masking_utils` -- a symbol that
does not exist in the installed transformers==4.57.3 (it's from a newer
transformers API). This also affects every existing script that loads
MolFormer, for the same "never actually executed yet" reason. Fixed by
pinning both the tokenizer and model to the last revision before that commit
(7b12d946c181a37f6012b9dc3b002275de070314, 2024-03-31, "Adding safetensors
variant"), which predates the incompatible remote-code update.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "pearl_safetensors_conversions"

# model_name -> HF Hub revision (commit sha) known to work under the pinned
# transformers==4.57.3. Only override models with a known incompatibility;
# everything else loads at HEAD as usual.
KNOWN_GOOD_REVISIONS: Dict[str, str] = {
    "ibm/MoLFormer-XL-both-10pct": "7b12d946c181a37f6012b9dc3b002275de070314",
}


def _convert_to_local_safetensors(model_name: str, cache_root: Path) -> Path:
    local_dir = cache_root / model_name.replace("/", "__")
    marker = local_dir / "model.safetensors"
    if marker.exists():
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Converting {model_name} to a local safetensors checkpoint at {local_dir} (one-time)")

    files = list_repo_files(model_name)
    for fname in files:
        if fname in ("pytorch_model.bin", "training_args.bin") or fname.endswith(".bin"):
            continue
        local_path = hf_hub_download(model_name, fname)
        shutil.copy(local_path, local_dir / fname)

    bin_path = hf_hub_download(model_name, "pytorch_model.bin")
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    state_dict = {k: v.contiguous() for k, v in state_dict.items() if not k.startswith("lm_head")}
    save_file(state_dict, str(local_dir / "model.safetensors"))

    return local_dir


def load_seq_cls_model_safe(
    model_name: str,
    cache_root: Optional[Path] = None,
    **from_pretrained_kwargs: Any,
) -> "AutoModelForSequenceClassification":
    """
    Drop-in replacement for AutoModelForSequenceClassification.from_pretrained(model_name, ...)
    that transparently:
      1. Pins a known-good revision for models whose HEAD remote code is
         incompatible with the pinned transformers version (see module docstring).
      2. Falls back to a local safetensors conversion if the Hub checkpoint is
         .bin-only and the installed transformers/torch combination refuses to
         load it.
    """
    cache_root = cache_root or DEFAULT_CACHE_ROOT
    revision = KNOWN_GOOD_REVISIONS.get(model_name)
    if revision is not None:
        from_pretrained_kwargs.setdefault("revision", revision)

    try:
        return AutoModelForSequenceClassification.from_pretrained(model_name, **from_pretrained_kwargs)
    except ValueError as e:
        if "torch.load" not in str(e) and "CVE-2025-32434" not in str(e):
            raise
        logging.warning(
            f"{model_name}: checkpoint is .bin-only and transformers refused to load it "
            f"under the current torch version. Falling back to a local safetensors conversion."
        )
        local_dir = _convert_to_local_safetensors(model_name, cache_root)
        from_pretrained_kwargs.pop("revision", None)  # local dir has no revisions
        return AutoModelForSequenceClassification.from_pretrained(str(local_dir), **from_pretrained_kwargs)


def load_tokenizer_safe(model_name: str, **from_pretrained_kwargs: Any) -> "AutoTokenizer":
    """Drop-in replacement for AutoTokenizer.from_pretrained(model_name, ...) that
    pins the same known-good revision as load_seq_cls_model_safe, so the
    tokenizer and model always come from the same checkpoint revision."""
    revision = KNOWN_GOOD_REVISIONS.get(model_name)
    if revision is not None:
        from_pretrained_kwargs.setdefault("revision", revision)
    return AutoTokenizer.from_pretrained(model_name, **from_pretrained_kwargs)
