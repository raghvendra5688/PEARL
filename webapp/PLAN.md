# PEARL Experience Website — Build Spec

> Paste this whole file into a Claude Code (VS Code) chat with this repo open, or say
> "read webapp/PLAN.md and start implementing" — it's self-contained and doesn't assume
> any prior conversation.

## 1. What this is

PEARL (this repo) benchmarks molecular-property prediction across 8 datasets using 7
method families (PC-only, Chemprop, GIN, E2E LoRA, FT Embed, FT Embed+PC, RAFE). The
trained models are public on Hugging Face:
**https://huggingface.co/raghvendramall/pearl-benchmark-models** — one folder per
dataset, each containing its top-5 ranked methods as `{DATASET}/rank{1-5}_{method}/`.

We're building a public website: a visitor enters a molecule (SMILES / name / chemical
formula), immediately sees an instant profile of it (3D structure, physicochemical
summary, optionally nearest ZINC-250k neighbours), then picks which properties they
want predicted. **There is no method-selection UI** — each property always uses that
dataset's single best (`rank1`) model. No live comparison across methods.

Hosting must be free (self-funded project): **Hugging Face Spaces** (Docker SDK, free
CPU tier) for the app, **Hugging Face Hub** for all models/data — both genuinely $0,
no billing account.

---

## 2. UX flow

**Step 1 — Resolve input.** Accept SMILES, a name, or a formula.
- SMILES → validate/canonicalize with RDKit (`Chem.MolFromSmiles`).
- Name → PubChem PUG REST: `GET /rest/pug/compound/name/{name}/property/IsomericSMILES,IUPACName/JSON` (free, no key).
- Formula (e.g. `C9H8O4`) → PubChem `fastformula`: `GET /rest/pug/compound/fastformula/{formula}/cids/JSON?MaxRecords=10`. A formula can match many compounds:
  - 0 matches → clear error.
  - 1 match → resolve directly.
  - 2+ matches → return a short disambiguation list (name + SMILES for each candidate CID) for the frontend to show as a picker.

**Step 2 — Instant profile** (no property chosen yet):
- **3D structure**: one RDKit ETKDGv3 + MMFF94 conformer (see `scripts/unimol/finetuning/smiles_to_conformers.py::smiles_to_unimol_input()` for the exact recipe — fallback chain ETKDGv3 → ETKDGv2 → plain distance geometry, then MMFF94 or UFF optimization), emitted as a MOL block, rendered client-side with **3Dmol.js** (free, WebGL, no server GPU).
- **Physicochemical profile**: call `scripts/smiles/ml/pc_only_modelling.py::extract_pc_features()` (pure RDKit + NetworkX + Morgan/MACCS — no model needed) and surface a curated human-readable subset (MolWt, LogP, TPSA, HBD/HBA, rotatable bonds, aromatic rings). This is identical for every downstream method — confirmed from `pc_feature_merge_new_datasets.py` reusing the same function for both HF and Uni-Mol modalities — so there's exactly one profile to compute, not per-backbone variants.
- **Nearest ZINC-250k neighbours** (optional, decide before building — see open item in §6): a general "chemical context" feature, decoupled from any prediction. Since no universal embedding index exists yet, the pragmatic zero-upload option is: load `data/zinc250k/zinc250k_cleaned.csv` (already in this repo, not gitignored) at startup, precompute Morgan fingerprints once, and do a `DataStructs.BulkTanimotoSimilarity` search per query — needs no HF upload, no embedding model, no FAISS.

**Step 3 — Property selection.** Show datasets as plain-language cards, multi-select, "select all" convenience:

| Dataset | Shown as | Status (see §4) |
|---|---|---|
| BACE | BACE1 Enzyme Inhibition | Ready |
| hERG | Cardiotoxicity (hERG Channel Inhibition) | Ready |
| Caco-2 | Intestinal Permeability | Ready |
| BBBP | Blood-Brain Barrier Permeability | Needs 2 artifacts |
| DILI | Drug-Induced Liver Injury Risk | Needs 2 artifacts |
| Half-Life | Pharmacokinetic Half-Life | Needs 1 artifact |
| Flavor | Taste Profile (bitter/sweet/sour/umami) | Needs verification |
| ClinTox | Clinical Toxicity Risk | Not uploaded — omit for v1 |

Only load/run models for datasets actually selected (lazy loading — keeps memory bounded
on a free Space).

**Step 4 — Predictions.** For each selected dataset, load its `rank1` model, run it,
stream the result back over SSE as soon as it's ready (fast methods ~1s; Uni-Mol-based
ones slower — show a "still computing" state, don't block other cards). Each result:
- Classification: predicted class + probability as confidence (`predict_proba` / softmax).
- Regression (Caco-2, Half-Life): point estimate + units (Half-Life needs its log1p
  transform inverted — see `invert_target_transform()` in `pc_only_modelling.py` /
  `chemprop_baseline.py` / `gcn_baseline.py`) + the method's benchmark R²/Spearman.
- A static credibility line: that one model's Mean±SE from
  `results/summary/classification_mean_se_by_method.csv` /
  `regression_mean_se_by_method.csv`.
- **Recommended (optional) addition**: for RAFE-based results (BBBP, DILI), also surface
  the retrieval similarity score (`zinc_nearest_sim` / `k10_sim_mean`) and flag low
  values (e.g. below whatever threshold looks right once you inspect the real training
  distribution) with a caveat like "this molecule is structurally unlike anything in our
  reference library — treat this result with extra caution." RAFE's fixed top-k
  retrieval never filters by similarity, so a novel molecule can silently get neighbor
  features computed from poor matches; this surfaces that honestly instead of hiding it.
  Costs nothing extra since the similarity score is already produced during retrieval.

---

## 3. Architecture

- **Frontend**: React/Next.js (not Streamlit — need a real 3D viewer and independently
  streaming result cards, which don't fit Streamlit's rerun-the-whole-script model).
- **Backend**: one FastAPI service, self-hosting all inference in-process. Not HF
  Inference API/Endpoints — most of these methods (tree models, Chemprop, Uni-Mol,
  FAISS retrieval) aren't standard `transformers` pipelines anyway, so a custom handler
  is required either way; doing it in one process avoids paying for/operating several
  separate endpoints.
- **Hosting**: single Hugging Face Space (Docker SDK, free CPU tier) serving the built
  Next.js app at `/` and FastAPI at `/api/*` from one container, one URL.
- Models/artifacts pulled from HF Hub via `huggingface_hub.hf_hub_download` /
  `snapshot_download`, lazily on first request for that dataset, cached warm afterward.

```
Browser ──► HF Space (1 free Docker container)
              ├─ Next.js build at "/"
              └─ FastAPI at "/api/*"
                   ├─ POST /api/resolve   → SMILES/name/formula → canonical SMILES
                   │                        (+ disambiguation list if formula is ambiguous)
                   ├─ GET  /api/profile   → 3D MOL block, PC feature summary,
                   │                        (optional) nearest ZINC neighbours
                   ├─ POST /api/predict   → SSE stream; one event per selected dataset's
                   │                        rank1-model result; isolated per-dataset
                   │                        error handling (one failure doesn't kill the
                   │                        stream)
                   └─ GET  /api/health    → which dataset models are currently loaded
```

---

## 4. Model registry — verified against the actual HF repo

Fetched directly from `raghvendramall/pearl-benchmark-models` (not assumed). Use this
table as the literal source for `app/model_registry.py`.

| Dataset | rank1 method | HF folder | Files present | Runnable now? |
|---|---|---|---|---|
| BACE | E2E LoRA (Uni-Mol, FL) | `BACE/rank1_e2e-lora-uni-mol/` | `config.json`, `pytorch_model.safetensors`, `unimol_encoder.safetensors` | **Yes** — self-contained |
| hERG | E2E LoRA (Uni-Mol, FL) | `HERG/rank1_e2e-lora-uni-mol/` | same as BACE | **Yes** — self-contained |
| Caco-2 | PC-only | `CACO2/rank1_pc-only/` | `CatBoost.pkl` | **Yes** — no backbone needed at all |
| BBBP | RAFE (MolFormer, FL) | `BBBP/rank1_rafe/` | `XGBoost.pkl`, `faiss_index/MolFormer_Finetuned_FL.index` | **Partial** — see below |
| DILI | RAFE (Uni-Mol, FL) | `DILI/rank1_rafe/` | `XGBoost.pkl`, `faiss_index/UniMol_FL.index` | **Partial** — see below |
| Half-Life | FT Embed (MolFormer, Huber) | `HALF_LIFE/rank1_ft-embed/` | `CatBoost.pkl` | **No** — backbone missing entirely |
| Flavor | Chemprop | `FLAVOR/rank1_chemprop/` | `README.md`, `model.pt` (14MB) | **Unverified** — no `args.json` present; Chemprop checkpoints normally need one to reconstruct hyperparameters before loading weights. Check whether this `.pt` embeds its own args (some Chemprop versions do) before trusting it standalone. |
| ClinTox | — | not in repo | — | **No** — not uploaded at all; omit from v1 |

**Important discovery for BBBP and DILI — their embedding backbones already exist
elsewhere in the repo, just not inside the `rank1_rafe` folder itself:**
- BBBP needs "MolFormer_Finetuned_FL" embeddings. `BBBP/rank5_e2e-lora-hf/` is
  confirmed to be `ibm/MoLFormer-XL-both-10pct` + Focal Loss — the exact same backbone.
  Load it with `AutoModel` (not `AutoModelForSequenceClassification`) to get embeddings
  only, same pattern as `finetuned_model_embeddings.py::get_embeddings()` (mean-pool
  `last_hidden_state`, not `pooler_output`).
- DILI needs "UniMol_FL" embeddings. `DILI/rank5_e2e-lora-uni-mol/` is confirmed to be
  Uni-Mol + Focal Loss — the exact same backbone. Load via
  `unimol_lora_trainer.py::load_finetuned_unimol()` and use `.get_embeddings()` instead
  of the classification head.
- Half-Life needs "MolFormer_Finetuned_Huber" embeddings, but
  `HALF_LIFE/rank4_e2e-lora-hf/` turned out to be **ChemBERTa-77M-MTR + Huber** — a
  different backbone family, not a match. This one is genuinely still missing.

**What's still actually missing (only these, not full backbones anymore) for BBBP/DILI
RAFE to fully work:**
1. The ZINC-250k **metadata** (SMILES + logP + QED + SAS arrays, keyed by the same row
   order as the FAISS index) — needed to know *what* each retrieved neighbor is. Not
   found anywhere in the repo or its READMEs.
2. The fitted **PCA model** (32-component, fit once on training-set neighbor centroids)
   — needed to reproduce the exact 32 compressed-centroid features the tree model was
   trained on. Not found anywhere either. This can't be safely refit from scratch —
   reusing the exact saved PCA object is required, or the tree model receives inputs it
   was never trained to interpret and predictions silently degrade.

---

## 5. Reused existing PEARL code — import these, don't reimplement

- `scripts/unimol/finetuning/smiles_to_conformers.py::smiles_to_unimol_input()` — 3D conformer generation.
- `scripts/smiles/ml/pc_only_modelling.py::extract_pc_features()`, `sanitize_features()`, `invert_target_transform()`.
- `scripts/smiles/rag/rag_feature_extraction.py::compute_rag_features()` — exact RAFE feature formulas (57 features: physicochemical neighbor stats, density/similarity, PCA-32 centroid, Tanimoto/scaffold). Once metadata+PCA are available, call this unmodified.
- `scripts/smiles/finetuning/finetuned_model_embeddings.py::get_embeddings()` — pattern for extracting embeddings from a merged E2E-LoRA HF checkpoint (mean-pooled `last_hidden_state`).
- `scripts/unimol/finetuning/unimol_lora_trainer.py` — `load_finetuned_unimol()`, `UniMolLoRAClassifier`, `.get_embeddings()`.
- `scripts/gnn/chemprop_baseline.py`, `scripts/gnn/gcn_baseline.py` — Chemprop/GIN inference patterns (not currently needed for any `rank1` model, kept for reference / if a `rank1` changes).
- `scripts/common/safe_model_loading.py::load_seq_cls_model_safe()`, `load_tokenizer_safe()` — required for any ChemBERTa/MolFormer load; handles a known CVE-2025-32434 safetensors issue and a MolFormer `trust_remote_code` revision pin. (The uploaded HF checkpoints are already safetensors, which sidesteps the worst of the `.bin` issue, but the MolFormer revision pin is still relevant.)
- `results/summary/classification_mean_se_by_method.csv` / `regression_mean_se_by_method.csv` — the Mean±SE credibility numbers for each `rank1` model.

---

## 6. Build order

**Phase 1 — build and fully verify against the 3 ready datasets (BACE, hERG, Caco-2):**
1. Scaffold `webapp/backend/app/` (FastAPI) and `webapp/frontend/` (Next.js).
2. `/api/resolve` — SMILES/name/formula → canonical SMILES, with PubChem + RDKit.
3. `/api/profile` — 3D MOL block + PC feature summary. Decide now whether the ZINC
   neighbour panel is in v1 (zero-cost Tanimoto option described in §2) or deferred.
4. `model_registry.py` seeded with just BACE, hERG, Caco-2 (table in §4).
5. Three inference wrappers: `unimol_e2e.py` (BACE/hERG), `pc_only.py` (Caco-2).
6. `/api/predict` SSE endpoint, wired to those three.
7. Frontend: input → profile → property selector (only these 3 enabled) → streaming result cards.
8. Dockerfile, deploy to a free HF Space, verify the public URL end-to-end.

**Phase 2 — add BBBP/DILI RAFE once metadata+PCA are uploaded** (or sooner, using a
temporary local copy if you already have `meta.pkl`/`{backbone}_pca.pkl` from your own
training run — check your `PEARL_EXTRAS`/`PEARL_EXTRAS_V2` storage under
`rag_pca/{dataset}/` and wherever the ZINC metadata was saved):
9. `rafe.py` inference wrapper using `compute_rag_features()` + the embedding-extraction
   pattern from the `rank5_e2e-lora-hf` / `rank5_e2e-lora-uni-mol` checkpoints (§4).
10. Add the optional low-similarity confidence caveat (§2, Step 4).

**Phase 3 — add the rest as their artifacts land:**
11. Half-Life once its MolFormer-Huber backbone is uploaded (`ft_embed.py` wrapper).
12. Flavor once the Chemprop checkpoint's standalone-loadability is confirmed.
13. ClinTox once its top model is exported and uploaded from scratch.

---

## 7. Repo layout

```
webapp/
├── PLAN.md                  ← this file
├── Dockerfile
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── model_registry.py
│   │   ├── routers/{resolve,profile,predict,health}.py
│   │   ├── inference/{unimol_e2e,pc_only,rafe,ft_embed,chemprop}.py
│   │   └── resolve/pubchem.py
│   └── requirements-web.txt   # root requirements.txt + fastapi, uvicorn, sse-starlette,
│                                huggingface_hub, faiss-cpu
└── frontend/
    ├── app/page.tsx
    └── components/
        ├── MoleculeInput.tsx
        ├── MoleculeViewer3D.tsx      # 3Dmol.js
        ├── PCProfileCard.tsx
        ├── NeighbourPanel.tsx        # optional, see §2/§6
        ├── PropertySelector.tsx
        └── DatasetResultCard.tsx
```

---

## 8. Open items to resolve with the mentor (not blocking Phase 1)

1. BBBP + DILI: upload the ZINC-250k metadata (SMILES/logP/QED/SAS) and the two fitted PCA models.
2. Half-Life: upload the finetuned MolFormer (Huber) backbone.
3. Flavor: confirm whether `model.pt` needs a companion `args.json`.
4. ClinTox: export and upload its top model (some methods were near-tied on MCC — confirm the actual winner including tiebreak).
5. Decide whether the Step 2 "nearest neighbours" landing panel ships in v1 (no upload needed either way — it's a local Tanimoto search over the existing `data/zinc250k/zinc250k_cleaned.csv`) or is dropped for a leaner launch.
