"""Compatibility shims shared by rafe.py and ft_embed.py. A MolFormer checkpoint's
bundled `configuration_molformer.py`/`modeling_molformer.py` were exported against
an older `transformers` than this webapp pins (5.14.1) and reference three symbols
later versions removed. This is separate from the KNOWN_GOOD_REVISIONS pin problem
(see ARCHITECTURE.md): it's the checkpoint's own locally-bundled copy of these
files, which a `revision=` argument on a fresh Hub fetch can't affect.

1. `transformers.onnx.OnnxConfig` -- the whole submodule was removed. Only used as
   an unused base class for ONNX-export tooling, never instantiated on the normal
   forward path, so an empty stand-in class is safe.
2. `transformers.pytorch_utils.find_pruneable_heads_and_indices` -- removed. Only
   called from `MolformerAttention.prune_heads()`, an opt-in method this webapp
   never calls, so the stub raises rather than guessing at a reimplementation.
3. `PreTrainedModel.get_head_mask` -- removed, but genuinely on the forward path
   (every embedding call hits it). Reimplemented for the one case this webapp
   actually needs: `head_mask=None`, since `get_embeddings()` never passes one.
"""

from __future__ import annotations

import sys
import types


def ensure_transformers_onnx_shim() -> None:
    if "transformers.onnx" not in sys.modules:
        shim = types.ModuleType("transformers.onnx")

        class OnnxConfig:  # intentionally empty, see module docstring
            pass

        shim.OnnxConfig = OnnxConfig  # type: ignore[attr-defined]
        sys.modules["transformers.onnx"] = shim

    import transformers.pytorch_utils as pytorch_utils

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(*args, **kwargs):
            raise NotImplementedError(
                "find_pruneable_heads_and_indices was removed from this transformers "
                "version; this webapp never calls prune_heads(), see _molformer_compat.py."
            )

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "get_head_mask"):

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                raise NotImplementedError(
                    "get_head_mask was removed from this transformers version; this "
                    "shim only reimplements the head_mask=None path, see _molformer_compat.py."
                )
            return [None] * num_hidden_layers

        PreTrainedModel.get_head_mask = get_head_mask
