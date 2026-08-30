"""POST /api/predict -- SSE stream, one event per selected dataset's rank1 model
result. Per-dataset isolation: one failure emits a generic error event and the
stream continues -- see ARCHITECTURE.md #3 (why SSE) and #9 (why errors stay generic).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .. import model_registry
from ..inference import chemprop_flavor, ft_embed, pc_only, rafe, unimol_e2e

logger = logging.getLogger("pearl.webapp")
router = APIRouter()

# BBBP is registered here (the backend can serve it directly, e.g. for testing)
# even though the frontend's dataset catalog doesn't expose it yet -- see
# model_registry.py and ARCHITECTURE.md for why.
DatasetKey = Literal["BACE", "HERG", "CACO2", "BBBP", "DILI", "HALF_LIFE", "FLAVOR"]


class PredictRequest(BaseModel):
    smiles: str
    datasets: list[DatasetKey]


def _run_prediction(key: str, smiles: str) -> dict:
    """Runs one dataset's rank1 model synchronously (executed in a worker thread --
    see predict() below). Returns a client-safe payload built entirely from
    model_registry.public_dataset_info(); raw registry internals never appear here."""
    entry = model_registry.get_dataset_entry(key)
    info = model_registry.public_dataset_info(key)
    model = model_registry.get_model(key)

    if entry.inference_module == "unimol_e2e":
        result = unimol_e2e.predict(model, smiles)
    elif entry.inference_module == "pc_only":
        result = pc_only.predict(model, smiles, entry.task, entry.target_transform)
    elif entry.inference_module == "rafe":
        result = rafe.predict(model, smiles)
    elif entry.inference_module == "ft_embed":
        result = ft_embed.predict(model, smiles)
    elif entry.inference_module == "chemprop":
        result = chemprop_flavor.predict(model, smiles)
    else:
        raise ValueError(f"Unknown inference module: {entry.inference_module}")

    return {**info, "dataset": key, **result}


@router.post("/predict")
async def predict(req: PredictRequest) -> EventSourceResponse:
    async def event_stream():
        tasks = {
            key: asyncio.create_task(asyncio.to_thread(_run_prediction, key, req.smiles))
            for key in req.datasets
        }
        pending = set(tasks.values())
        task_to_key = {task: key for key, task in tasks.items()}

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                key = task_to_key[task]
                try:
                    payload = task.result()
                    yield {"event": "result", "data": json.dumps(payload)}
                except Exception:
                    logger.exception("Prediction failed for dataset=%s", key)
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "dataset": key,
                            "error": "Prediction unavailable for this property right now.",
                        }),
                    }
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(event_stream())
