"""App factory. Serves the Next.js static export at "/" and FastAPI at "/api/*" from
one process -- see ARCHITECTURE.md #1/#2 for why one process, one origin.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv

# Must run before any sibling module reads an env var at import time
# (model_registry.py reads HF_MODEL_REPO/PEARL_MODELS_CACHE at module load).
load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from . import model_registry  # noqa: E402
from .chem import zinc_neighbours  # noqa: E402
from .errors import register_error_handlers  # noqa: E402
from .routers import health, predict, profile, resolve  # noqa: E402

logger = logging.getLogger("pearl.webapp")

IS_PRODUCTION = os.environ.get("ENV", "development") == "production"
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "out"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Backgrounded, not awaited: the server accepts connections immediately instead
    # of blocking on model downloads and the ZINC index build. See ARCHITECTURE.md
    # #4/#7. The two warm-ups run concurrently -- they share no file, so no race.
    warm_models_task = asyncio.create_task(model_registry.warm_all_models())
    warm_zinc_task = asyncio.create_task(asyncio.to_thread(zinc_neighbours.warm))
    yield
    warm_models_task.cancel()
    warm_zinc_task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(
        title="PEARL Experience API",
        lifespan=lifespan,
        # /docs and /redoc stay off in production -- an API surface map is not
        # something a public visitor needs. See ARCHITECTURE.md #9.
        docs_url=None if IS_PRODUCTION else "/docs",
        redoc_url=None if IS_PRODUCTION else "/redoc",
    )

    register_error_handlers(app)

    # Dev only: `next dev` runs on a different port and calls the backend directly,
    # which needs CORS. Production serves both from one origin, so none is needed.
    # See ARCHITECTURE.md #9/#14.
    if not IS_PRODUCTION:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    app.include_router(resolve.router, prefix="/api")
    app.include_router(profile.router, prefix="/api")
    app.include_router(predict.router, prefix="/api")
    app.include_router(health.router, prefix="/api")

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

    return app


app = create_app()
