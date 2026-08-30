"""Inference wrapper for Flavor's rank1 model: a Chemprop (D-MPNN graph neural
network) multiclass classifier predicting a taste category.

New code, not adapted from an existing script -- scripts/gnn/chemprop_baseline.py
never runs inference in-process, it shells out to the `chemprop_train` CLI. The
calling pattern (load_checkpoint -> get_data_from_smiles -> MoleculeDataLoader ->
predict) needed three environment fixes, found by running it against the real
checkpoint:

1. chemprop==1.6.1 references `numpy.VisibleDeprecationWarning`, removed in
   newer numpy -- shimmed by aliasing it to `DeprecationWarning` before import.
2. `load_checkpoint()` calls `torch.load()` without `weights_only=False`; newer
   PyTorch's default refuses to unpickle the `argparse.Namespace` chemprop bundles
   its training args in -- fixed via `torch.serialization.add_safe_globals(...)`.
3. `MoleculeDataLoader`'s default multiprocessing workers don't inherit the
   in-process numpy shim above and crash -- `num_workers=0` avoids it.

Class order (bitter, sour, sweet, umami, undefined): alphabetical LabelEncoder
order of the training data's taste column, confirmed against the original
training CSV -- see model_registry.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Must happen before chemprop is imported anywhere -- see module docstring #1.
np.VisibleDeprecationWarning = DeprecationWarning  # type: ignore[attr-defined]

import torch  # noqa: E402

# Must happen before load_checkpoint() is called -- see module docstring #2.
torch.serialization.add_safe_globals([argparse.Namespace])

from chemprop.data import get_data_from_smiles  # noqa: E402
from chemprop.data.data import MoleculeDataLoader  # noqa: E402
from chemprop.train import predict as chemprop_predict  # noqa: E402
from chemprop.utils import load_checkpoint  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(save_dir: Path):
    model = load_checkpoint(str(Path(save_dir) / "model.pt"), device=DEVICE)
    model.eval()
    return model


def predict(model, smiles: str) -> dict:
    data = get_data_from_smiles(smiles=[[smiles]], skip_invalid_smiles=True)
    if len(data) == 0:
        raise ValueError(f"Chemprop's featurizer rejected this SMILES: {smiles!r}")

    loader = MoleculeDataLoader(dataset=data, batch_size=1, num_workers=0)
    raw = chemprop_predict(model=model, data_loader=loader)
    probabilities = raw[0][0]  # [[[c0, c1, ..., c4]]] -> one molecule, one task

    predicted_class = int(np.argmax(probabilities))
    return {
        "predicted_class": predicted_class,
        "confidence": float(probabilities[predicted_class]),
        "probabilities": [float(p) for p in probabilities],
    }
