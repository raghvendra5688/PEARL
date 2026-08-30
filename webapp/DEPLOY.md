# Deploying to Hugging Face Spaces

`webapp/` is a self-contained folder — the backend doesn't import anything from
outside it (see `ARCHITECTURE.md` for what's vendored and why). That means the
Space deployment can be just `webapp/`'s contents, as their own standalone repo,
with no shared history and nothing else from this research repo included.

## 1. Create a standalone copy of webapp/

Simplest and safest — a fresh repo containing only what's currently in `webapp/`,
no history from this repo at all:

```bash
# from the PEARL-1 repo root
rm -rf /tmp/pearl-experience-deploy
cp -r webapp /tmp/pearl-experience-deploy
cd /tmp/pearl-experience-deploy
rm -rf backend/models_cache frontend/node_modules frontend/.next frontend/out
git init
git add .
git commit -m "Deploy PEARL Experience"
```

(If you'd rather keep this folder's git history intact instead of starting fresh,
`git subtree split --prefix=webapp -b webapp-only` from the main repo produces a
branch with full history for just this folder — push that branch instead of doing
the copy above. Either way, nothing outside `webapp/` is included.)

## 2. Create the Space and push

```bash
huggingface-cli repo create pearl-experience --type space --space_sdk docker
cd /tmp/pearl-experience-deploy
git remote add space https://huggingface.co/spaces/<your-username>/pearl-experience
git push space main
```

## 3. Add the Space config

Add a `README.md` at the root of that pushed repo (it's now the Space's own root,
so this doesn't touch anything from the main research repo):

```yaml
---
title: PEARL Experience
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
```

No `dockerfile:` field needed — `Dockerfile` is already at the root of what got
pushed, which is where HF Spaces looks by default.

## Secrets

Set `HF_TOKEN` in the Space's Settings → Repository secrets **only if** a model
artifact ever needs authenticated download (public repos don't need it). Never
commit a real token to `.env` — see `ARCHITECTURE.md` and `.env.example`.

## Startup time

Every model and the ZINC-250k similarity index load in the background right
after the container starts (not lazily on first request — see `ARCHITECTURE.md`).
Expect the Space to take a few minutes after it comes up before predictions are
fast; `/api/health` shows what's ready.
