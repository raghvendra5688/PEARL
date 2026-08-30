# Build context is this webapp/ folder itself -- everything the image needs
# (backend, its vendored copies of the PEARL functions it reuses, the ZINC-250k
# data, and the frontend) lives inside webapp/ now. Deliberately self-contained:
# deploying doesn't require or expose the rest of the research repo. See
# ARCHITECTURE.md and DEPLOY.md.
#
# Local build:  docker build -f webapp/Dockerfile -t pearl-experience webapp
# Local run:    docker run -p 7860:7860 pearl-experience

# ---- Frontend build stage --------------------------------------------------
FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN BUILD_STATIC=true npm run build

# ---- Backend runtime stage --------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    ENV=production \
    PEARL_MODELS_CACHE=/app/backend/models_cache

WORKDIR /app

# build-essential/git: some scientific packages (e.g. unimol_tools) may need to
# build from sdist if no prebuilt wheel matches this exact platform/Python combo.
# libgomp1: OpenMP runtime XGBoost/LightGBM/FAISS need at import time, not just
# build time -- kept in the final image, not just the build layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-web.txt ./backend/requirements-web.txt
RUN pip install --no-cache-dir -r backend/requirements-web.txt

COPY backend/ ./backend/
COPY --from=frontend-build /frontend/out ./frontend/out

RUN mkdir -p ${PEARL_MODELS_CACHE}

WORKDIR /app/backend
# HF Spaces' Docker SDK expects the app to listen on 7860 by default.
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
