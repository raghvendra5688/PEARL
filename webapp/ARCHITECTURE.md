# Architecture

This doc explains how the PEARL Experience webapp is put together, where things
live, and how to run it. For the original feature spec see `PLAN.md`; for
deployment to Hugging Face Spaces see `DEPLOY.md`.

## What this is

A FastAPI backend + Next.js (React) frontend that lets a visitor enter a molecule
(SMILES or a chemical formula), see an instant profile (3D structure,
physicochemical descriptors, nearest known compounds), and predict properties
(BACE1 inhibition, hERG cardiotoxicity, Caco-2 permeability, liver injury risk,
pharmacokinetic half-life, taste profile) using PEARL's benchmarked models. A
seventh property (blood-brain barrier permeability) is fully wired up on the
backend but not yet shown in the UI — see "Why RAFE's feature vector is
[embedding, PC features, retrieval features]" below.

---

## Project structure

### Backend — `webapp/backend/`

| File | What it does |
|---|---|
| `app/main.py` | App factory. Wires up routers, starts model/ZINC-index warm-up on startup, mounts the built frontend, adds CORS (dev only). |
| `app/model_registry.py` | The catalog of runnable models (BACE, HERG, CACO2, BBBP, DILI, Half-Life, Flavor) — where each one's files live on Hugging Face, its credibility numbers, and functions to download/load/cache them. |
| `app/errors.py` | One global error handler. Client always gets a short, generic message; the real error is logged server-side only. |
| `app/routers/resolve.py` | `POST /api/resolve` — SMILES or formula → canonical SMILES (or a disambiguation list if a formula matches multiple compounds). |
| `app/routers/profile.py` | `GET /api/profile` — 3D structure, physicochemical summary, nearest ZINC neighbours, formula + name. |
| `app/routers/predict.py` | `POST /api/predict` — runs the selected models and streams results back as they finish (Server-Sent Events). |
| `app/routers/health.py` | `GET /api/health` — which models/indexes are currently warm. |
| `app/resolve/pubchem.py` | Talks to PubChem: resolves a formula to candidate compounds, and looks up a common name for display. |
| `app/chem/conformer.py` | Generates a 3D conformer (RDKit) and returns it as a MOL block for the viewer. |
| `app/chem/pc_profile.py` | Computes the human-readable physicochemical summary (molecular weight, LogP, TPSA, etc). |
| `app/chem/formula.py` | Computes a molecular formula from SMILES. |
| `app/chem/zinc_neighbours.py` | Finds the most similar known compounds in ZINC-250k (local similarity search). |
| `app/inference/pc_only.py` | Runs Caco-2's model (physicochemical features → CatBoost). |
| `app/inference/unimol_e2e.py` | Runs BACE/hERG's model (Uni-Mol + LoRA, end-to-end finetuned). |
| `app/inference/rafe.py` | Runs BBBP/DILI's models (retrieval-augmented features: nearest ZINC-250k neighbours by embedding similarity → XGBoost). |
| `app/inference/ft_embed.py` | Runs Half-Life's model (finetuned MolFormer embedding, no other features → CatBoost). |
| `app/inference/chemprop_flavor.py` | Runs Flavor's model (Chemprop graph neural network, multiclass). |
| `app/vendor/` | Vendored (copied, not imported) copies of the PEARL research-repo files the backend actually reuses — see "Why the backend vendors its own copy of a few files" below. |
| `data/zinc250k_cleaned.csv` | A copy of the ZINC-250k dataset used for the nearest-neighbour search — same reason as `vendor/`. |
| `requirements-web.txt` | Backend dependencies. |

### Frontend — `webapp/frontend/`

| File | What it does |
|---|---|
| `app/layout.tsx` | Root layout — fonts, page background, decorative molecule shapes. |
| `app/page.tsx` | The whole page flow: input → profile → property selection → results. Holds all the top-level state. |
| `app/globals.css` | Global styles and a couple of small animations. |
| `components/MoleculeInput.tsx` | The SMILES/formula input box and toggle. |
| `components/DisambiguationPicker.tsx` | Shown when a formula matches multiple compounds. |
| `components/MoleculeViewer3D.tsx` | The 3D structure viewer (3Dmol.js). |
| `components/PCProfileCard.tsx` | The physicochemical stat tiles. |
| `components/NeighbourPanel.tsx` | The "nearest known compounds" card grid. |
| `components/PropertySelector.tsx` | The property picker + "Predict" button. |
| `components/DatasetResultCard.tsx` | One prediction result — a binary classification badge/meter, a multiclass ranked breakdown, or a regression value. |
| `components/ChemFormula.tsx` | Renders a formula like `C9H8O4` with real subscripts. |
| `components/icons.tsx` | The small hand-drawn icon set used throughout. |
| `components/MoleculeBackdrop.tsx` | The decorative rotating background shapes. |
| `lib/api.ts` | All calls to the backend, including the SSE stream parser for predictions. |
| `lib/types.ts` | TypeScript types matching the backend's response shapes. |
| `lib/datasets.ts` | The property catalog — display names, descriptions, which are enabled, classification labels. |
| `next.config.js` | Build config (static export in production). |
| `tailwind.config.ts` | Theme tokens — colors, shadows, background grid, animations. |
| `.env.development.local` | Dev-only: points the frontend at the local backend (gitignored). Must use this name, not the more common `.env.local` — see `lib/api.ts`. |

### Root

| File | What it does |
|---|---|
| `Dockerfile` | Builds the production image (frontend static export + backend, one container). |
| `DEPLOY.md` | How to deploy to a Hugging Face Space. |
| `PLAN.md` | The original build spec this was built from. |

---

## How to run it

### Backend

```powershell
cd webapp\backend
pip install -r requirements-web.txt
uvicorn app.main:app --reload --port 8000
```

First startup downloads and loads every model plus builds the ZINC-250k
similarity index in the background — a few minutes on a cold machine, much
faster once the ZINC index is cached to disk (see "Speeding up startup and
`--reload` cycles" below). Check readiness with:

```powershell
curl http://localhost:8000/api/health
```

### Frontend

```powershell
cd webapp\frontend
npm install
npm run dev
```

Open `http://localhost:3000`. This talks directly to the backend on port 8000 (see
`.env.development.local`) — it does not proxy through Next.js.

### Testing it exactly like production (one origin, no dev servers)

```powershell
cd webapp\frontend
$env:BUILD_STATIC = "true"
npm run build
cd ..\backend
uvicorn app.main:app --port 8000
```

Then visit `http://localhost:8000` directly — the backend serves the built
frontend itself.

### Docker

Build with `webapp/` itself as the context — everything the image needs lives
inside it:

```powershell
docker build -f webapp\Dockerfile -t pearl-experience webapp
docker run -p 7860:7860 pearl-experience
```

---

## How it works

```
Browser
  │
  ├─ dev:  talks straight to the backend (localhost:8000)
  └─ prod: same origin, backend serves the built frontend too
              │
              ▼
        FastAPI (webapp/backend)
              │
    ┌─────────┼─────────────┬──────────────┐
    ▼         ▼              ▼              ▼
 resolve   profile        predict        health
 (PubChem) (RDKit +       (loads a       (are models/
            ZINC scan)     model, runs    index warm?)
                            it, streams
                            result back)
```

- **Resolve and profile** are instant — pure RDKit plus (for profile) a
  best-effort PubChem name lookup that never blocks the response.
- **Predict** streams results one at a time as each model finishes, instead of
  waiting for all of them.
- **Every model and the ZINC similarity index load at server startup**, in the
  background, so a visitor's first request is never the one paying that cost.

---

## Key design decisions

- **One process, one origin in production.** The frontend is built as static
  files and served directly by FastAPI — no separate Node server, no proxy.
- **Predictions stream over Server-Sent Events**, not WebSockets — it's a
  one-directional feed, and SSE works over plain HTTP.
- **Models and the ZINC index warm up eagerly at startup**, not on first
  request. A cold first prediction could otherwise take minutes (downloads
  plus a one-time index build) — long enough for anything sitting in front of
  the connection to give up before the answer arrived.
- **In development, the frontend calls the backend directly**, not through
  Next.js's built-in proxy. Next's proxy doesn't reliably carry a streaming
  SSE response through, so it's removed from that path entirely.
- **The 473-dim feature vector used for Caco-2's model** and **the
  human-readable physicochemical summary shown to visitors** are two separate
  computations, not one — several of the display fields are filtered out of
  the model's feature set for statistical reasons and aren't reliably present
  there.
- **3D structure needs no model** — it's pure RDKit, which is why the "instant
  profile" is genuinely instant.
- **Nearest-neighbour search is a local similarity scan**, not a hosted vector
  database — cheap, needs no extra infrastructure, and is precomputed once at
  startup rather than on every request.
- **Every tree model (CatBoost/XGBoost/LightGBM) loads through the same code
  path** — they're all saved the same way, so there's no per-library branching.
- **Nothing internal reaches the client.** Errors are always a short, generic
  message (the real one is logged server-side only); internal file paths, model
  storage locations, and debug endpoints are never exposed.
- **A formula can match more than one compound** — `/api/resolve` returns a
  picker in that case instead of guessing.
- **Input is SMILES or formula only** — no free-text name lookup. Simpler UI,
  one fewer external dependency to fail.
- **The molecule display always shows a formula** (computed locally, instant)
  and **a common name when one can be found** (PubChem, cached, best-effort —
  never blocks the page if it's slow or the name doesn't exist).
- **`webapp/` is a self-contained folder** — it doesn't read or import anything
  from elsewhere in the research repo. The handful of files and the ZINC dataset
  it needs are copied into `webapp/` itself (see below) so the whole app can be
  deployed on its own, without shipping the rest of the repo.
- **BBBP is registered on the backend but not enabled in the UI** — its
  feature vector is short by 3 columns against what its model expects. See
  "Why RAFE's feature vector is [embedding, PC features, retrieval features]"
  below.
- **ClinTox is excluded entirely**, not shown as "coming soon" — no model for it
  was ever exported.

---

## Why the backend vendors its own copy of a few files

The backend needs a handful of functions from the wider PEARL research repo: 3D
conformer generation, physicochemical feature extraction, the Uni-Mol + LoRA
model class, RAFE's retrieval-feature computation, MolFormer embedding
extraction, and a tokenizer-loading safety wrapper. Rather than importing them
from `scripts/` (which would mean deploying `webapp/` also has to ship or
reference the whole research repo — manuscript, results, unrelated training
scripts and all), those six files are copied verbatim into `app/vendor/`, and
the ZINC-250k CSV is copied into `data/`. The functions themselves are
unmodified, so behavior is identical to importing them.

The cost of this is that the vendored copies can drift from the originals — a
fix made in the research repo's `scripts/` won't automatically reach the
webapp. That's a deliberate tradeoff: each vendored file has a header comment
pointing back to its original location for whenever it needs re-syncing.

---

## Why RAFE needs two Hugging Face folders per dataset, not one

BBBP and DILI's models (`inference/rafe.py`) don't just run a checkpoint on the
query molecule — they embed it, search a FAISS index of ZINC-250k for its
nearest neighbours, and build features from what they find. That means two
separate pieces have to be downloaded: the retrieval artifacts (`rank1_rafe/` —
the XGBoost model, the FAISS index, a metadata file describing every ZINC-250k
row, and a fitted PCA that compresses neighbour embeddings into what XGBoost
was actually trained on) and, separately, the embedding backbone that turns a
new query SMILES into a vector in the first place — which lives in a
*different* ranked folder entirely (BBBP's is at `rank5_e2e-lora-hf/`, DILI's
at `rank5_e2e-lora-uni-mol/`), because that backbone was ranked and uploaded on
its own merits as an end-to-end model, not specifically as RAFE's embedder.
`model_registry.py`'s `extra_hf_subfolders` field exists for exactly this case;
every other dataset needs only one folder.

The two datasets also embed differently — BBBP uses MolFormer (a 768-dim
pooled vector), DILI uses Uni-Mol (a 2560-dim vector: a 512-dim CLS embedding
concatenated with a 2048-dim Morgan fingerprint, the same combined
representation `unimol_e2e.py` already produces for BACE/hERG). Which one
applies to which dataset was confirmed by downloading and inspecting both
fitted PCA objects directly (`pca.mean_.shape`), not guessed from a docstring
— that's what revealed DILI needs the full 2560-dim vector unsliced, not just
the 512-dim CLS portion its embedding method's docstring calls out for other
uses.

## Why RAFE's feature vector is [embedding, PC features, retrieval features]

`compute_rag_features()`'s own docstring only describes its own ~60-column
output, which is easy to mistake for the entire feature vector XGBoost
expects. It isn't. PLAN.md's own description says so correctly ("FT Embed+PC
further augmented with ZINC-250k chemical-neighbourhood features") — that's
about what gets concatenated *around* `compute_rag_features()`, not about it
alone. The real vector, confirmed against the training script
(`scripts/unimol/ml/rag_modelling_unimol_new_datasets.py`,
`build_feature_matrix()`), is `np.hstack([embedding, pc_features,
rag_features])`, in that order: 2560 (Uni-Mol) + 473 (PC, the same
`extract_pc_features()` already reused for Caco-2) + 60 (retrieval) = 3093.
`rafe.py` builds all three and concatenates them in that order for both BBBP
and DILI.

One more subtlety: the embedding used for the *search* step has to be
L2-normalized (`compute_rag_features()` expects it), but the embedding
concatenated into the final feature vector does not — training's feature
matrix script never normalizes it. `faiss.normalize_L2()` mutates its argument
in place, so `rafe.py` normalizes a **copy** for the search call and keeps the
raw embedding for concatenation.

**DILI works end-to-end**, confirmed with a real prediction (aspirin: low
liver-injury risk, 92% confidence). **BBBP does not** — its feature count is
short by exactly 3 (1301 built vs 1304 the model expects), even though DILI's
structurally identical pipeline matches exactly. `extract_pc_features()` is
fully deterministic (fixed columns regardless of molecule), so this isn't a
per-molecule gap. BBBP's original training script builds its PC columns from
an external file (`PC_FT_All_Embeddings/BBBP_Embeddings/bbbp_{split}_features.csv`)
generated by a different script
(`scripts/smiles/ml/pc_feature_extraction_ft_model_refactored.py`) on
infrastructure outside this repo. That script's RDKit-descriptor exclusion
list is byte-identical to the one vendored here, and testing both the
currently-installed RDKit (2026.3.5) and the repo's own pinned constraint
(2025.9.4) produces the same 473-column count either way — ruling out RDKit
version drift as the cause. The 3 extra columns genuinely only exist in that
external file, which was never uploaded anywhere accessible from this repo.
`lib/datasets.ts` keeps BBBP's card disabled until that file (or the missing
column names) can be obtained from the original author, rather than shipping
a result that would quietly be wrong.

## Why concurrent predictions across datasets can crash the whole process

Reproduced once: requesting DILI and BBBP in the same `/api/predict` call
(each dispatched to its own thread via `asyncio.to_thread`, same pattern used
for every other dataset) crashed the entire Python process with no traceback
logged at all — not a caught exception, not a per-dataset error event.
Requesting BBBP alone afterward failed normally instead (the feature-count bug
above), so the crash was specific to running two different native ML code
paths (Uni-Mol/PyTorch for DILI, MolFormer/PyTorch for BBBP) in separate OS
threads at the same moment. This has the signature of a native (non-Python)
crash — a thread-safety issue inside a C/C++ extension (PyTorch, FAISS, or a
shared BLAS/MKL library) rather than anything `asyncio.to_thread` itself is
responsible for.

Not yet fixed. It's a real risk for any two datasets that both load a heavy
native model concurrently, not just this pair. A later test (see below) found
that concurrent model *loading* is safe — only concurrent *inference* has been
shown to crash — but that's a hint, not a full fix; the likely fix is
serializing native-heavy inference behind a lock rather than trusting
`asyncio.to_thread` to isolate it.

## Speeding up startup and `--reload` cycles

Two real, measured costs in how long the server takes to become ready:

1. **The ZINC-250k fingerprint index (~213s) was rebuilt from scratch on every
   process start** — it only ever lived in memory. That's pure computation
   over a static CSV that never changes, so it's now cached to disk
   (`models_cache/zinc_fingerprint_cache.pkl`, gitignored like every other
   downloaded/derived artifact here), invalidated by comparing mtimes against
   the source CSV. This matters most for `--reload` during development, which
   re-runs the entire startup sequence on every file save.
2. **Models warmed one at a time.** They only need to be sequential for one
   reason: BACE and HERG both trigger `unimol_tools`' shared base-checkpoint
   download the first time either runs, and two threads racing to write that
   file is a real risk. That risk disappears once the file already exists —
   which it does on every run after the first. So BACE now warms alone first
   (guaranteeing that file exists), then every other dataset warms
   concurrently via `asyncio.gather`.

Measured back to back on the same machine: a cold run (building the ZINC
cache from scratch) took 250s to fully warm; the next restart, with the cache
already on disk, took 108s — over 2x faster. Predictions were re-verified
correct afterward.

This also produced a useful data point about the concurrency crash above:
parallel model *warm-up* (loading several native models at once) completed
cleanly with no crash, twice in a row. That doesn't contradict the earlier
finding — the crash happened during concurrent *inference* (an actual forward
pass), and loading weights from disk is a much lighter operation than running
one. It's a hint that the crash risk is specific to concurrent native
compute, not confirmation — the inference-time risk is still open.

## Why Half-Life's model takes only an embedding, no other features

Half-Life's rank1 method is "FT Embed" — its CatBoost model was trained on the
raw pooled MolFormer embedding vector alone, with no physicochemical features
concatenated ("FT Embed+PC" is a different, lower-ranked model that does
concatenate PC features; feeding rank1's model the wrong shape would be a real
bug). Its embedding backbone is also structured differently from RAFE's: it's
bundled *inside* Half-Life's own `rank1_ft-embed/encoder_checkpoint/`
subfolder rather than living in a separate ranked folder, so
`inference/ft_embed.py` needs no `extra_hf_subfolders` at all.

## Why Flavor took extra verification before enabling it

Flavor's checkpoint (a Chemprop graph neural network predicting a 5-way taste
category) needed real, hands-on verification for two independent reasons:

**No prior art in this repo.** `scripts/gnn/chemprop_baseline.py` never runs
inference in-process — training and evaluation both go through the
`chemprop_train` CLI, with predictions read back from a CSV file the CLI
writes itself. So `inference/chemprop_flavor.py` is new code, proven against
the real downloaded checkpoint. That surfaced two environment
incompatibilities, both fixed in that file: `chemprop==1.6.1` references a
numpy attribute (`VisibleDeprecationWarning`) removed in this repo's newer
numpy, and its own `load_checkpoint()` calls `torch.load()` in a way newer
PyTorch's default security setting refuses (fixed by allowlisting the one
specific class it needs, not by disabling that protection outright).

**The taste-category order wasn't saved anywhere in this repo.** Nothing
recorded which output index means which taste. The order used (`bitter, sour,
sweet, umami, undefined`) was re-derived by alphabetically sorting the
training data's own taste labels — the same order `sklearn.LabelEncoder`
produces. Flavor was kept disabled until this could be confirmed.

**Now confirmed.** Independently re-running `LabelEncoder` against the
original training CSV reproduced the exact same order — matching what was
already wired into `model_registry.py`. Flavor is now enabled in
`lib/datasets.ts`.

---

## Things worth knowing before you debug something similar

- **Next.js loads `.env.local` for every build, not just `next dev`.** A static
  `next build` (even with `BUILD_STATIC=true`) sets `NODE_ENV=production`
  internally, and `.env.local` is loaded regardless of `NODE_ENV` — so a
  dev-only value like `NEXT_PUBLIC_API_BASE` would get baked into the
  production bundle, pointing it at `127.0.0.1:8000` even when the page itself
  loads from `localhost:8000` (a different origin to the browser), silently
  turning every API call cross-origin and failing the CORS preflight.
  `.env.development.local` is the fix — it only loads under `next dev`.
- **PubChem's property endpoint is inconsistent with its own request.** Asking
  for `IsomericSMILES` returns it under the key `SMILES` in the response, not
  `IsomericSMILES`. Not documented anywhere.
- **`unimol_tools` downloads its own ~190MB base checkpoint** from a completely
  different Hugging Face repo the first time it's used, separate from our own
  model download. This is part of why the startup warm-up matters.
- **A hand-rolled SSE parser must handle `\r\n`, not just `\n`.** The server
  library used here sends CRLF line endings; a parser that only splits on `\n\n`
  will silently parse zero events no matter how correct the server's response is.
- **A bright/light status color can fail contrast as plain text on a light
  background** even though it works fine on dark. The classification badges use
  a solid color fill with a chosen text color instead of colored text, so it
  works either way.
- **A checkpoint that "embeds its own args" isn't automatically loadable in a
  newer environment.** Flavor's Chemprop checkpoint bundles its training args
  directly inside `model.pt` — no separate `args.json` needed. But the library
  version that produced it (`chemprop==1.6.1`) still references a numpy
  attribute removed in this repo's newer numpy, and its own loader calls
  `torch.load()` in a way current PyTorch's default security setting refuses.
- **A `DataLoader` with `num_workers > 0` spawns fresh subprocesses that don't
  inherit an in-process patch.** Chemprop's `MoleculeDataLoader` defaults to
  multiple worker processes; a monkeypatch applied in the main process (the
  numpy shim above) isn't visible inside them, so they crash immediately on
  import. `num_workers=0` sidesteps it — also correct anyway for predicting
  one molecule at a time.
- **A revision pin chosen to fix one environment can break a different one.**
  `safe_model_loading.py`'s `KNOWN_GOOD_REVISIONS` pins MolFormer to a specific
  commit to avoid a problem in the *research repo's* pinned `transformers==4.57.3`.
  Reused as-is against this webapp's newer `transformers==5.14.1`, it fails
  outright — that pinned revision's bundled code imports `transformers.onnx`, a
  module later versions of `transformers` removed. HEAD doesn't have this
  problem, so `rafe.py`/`ft_embed.py` explicitly override the pin with
  `revision="main"`. That fix only covers the tokenizer/config resolved fresh
  from the Hub, though — the checkpoint's own locally bundled
  `configuration_molformer.py` (baked into the downloaded folder at export
  time) has the identical `transformers.onnx` import, and no `revision=`
  argument can touch a file already sitting on disk. `_molformer_compat.py`
  shims a dummy `transformers.onnx.OnnxConfig` into `sys.modules` instead —
  the real one is only ever used as an unused base class for ONNX-export
  tooling, never instantiated during a normal forward pass, so a stand-in is
  safe.
