"""GET /api/health -- which rank1 models and the ZINC neighbour index are currently
warm. Booleans only -- no file paths, memory figures, or HF repo internals. See
ARCHITECTURE.md #9.
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import model_registry
from ..chem import zinc_neighbours

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "models": {key: model_registry.is_loaded(key) for key in model_registry.REGISTRY},
        "zinc_index": zinc_neighbours.is_warm(),
    }
