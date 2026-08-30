"""Inference wrapper for BACE/HERG's rank1 model: Uni-Mol encoder (LoRA merged) +
Morgan fingerprint, end-to-end finetuned classification head.

Reuses UniMolLoRAClassifier and its forward()/preprocessing helpers from
vendor/unimol_lora_trainer.py unmodified -- forward() returns (logits, embedding),
which is the prediction path used here (.get_embeddings() is for RAFE, see rafe.py).

load_model() below does not call that module's own load_finetuned_unimol(): the
uploaded HF checkpoint ships .safetensors files, not the .pt files that function
expects. Loading via safetensors.torch.load_file() instead of torch.load() also
avoids CVE-2025-32434 (arbitrary code execution from an untrusted pickle checkpoint).
Model construction and state_dict loading otherwise mirror load_finetuned_unimol().
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

from ..vendor.unimol_lora_trainer import (
    DEVICE,
    UniMolLoRAClassifier,
    _collate_unimol_inputs,
    batch_morgan_fps,
    preprocess_smiles_for_unimol,
)


def load_model(save_dir: Path) -> UniMolLoRAClassifier:
    save_dir = Path(save_dir)
    with open(save_dir / "config.json") as f:
        cfg = json.load(f)

    model = UniMolLoRAClassifier(
        num_classes=cfg["num_classes"],
        task_type=cfg.get("task_type", "classification"),
    )

    head_state = load_safetensors(str(save_dir / "pytorch_model.safetensors"))
    model.load_state_dict(head_state, strict=False)

    encoder_path = save_dir / "unimol_encoder.safetensors"
    if encoder_path.exists():
        encoder_state = load_safetensors(str(encoder_path))
        model._repr.model.load_state_dict(encoder_state)

    model.to(DEVICE)
    model.eval()
    model._repr.model.eval()
    return model


@torch.no_grad()
def predict(model: UniMolLoRAClassifier, smiles: str) -> dict:
    """Runs the end-to-end classifier on a single canonical SMILES string."""
    unimol_inputs = preprocess_smiles_for_unimol([smiles], model._repr)
    padding_idx = model._repr.model.padding_idx
    unimol_batch = _collate_unimol_inputs(unimol_inputs, padding_idx)
    morgan_fp = torch.tensor(batch_morgan_fps([smiles]), dtype=torch.float32)

    logits, _ = model.forward(unimol_batch, morgan_fp)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    predicted_class = int(np.argmax(probs))
    return {
        "predicted_class": predicted_class,
        "confidence": float(probs[predicted_class]),
    }
