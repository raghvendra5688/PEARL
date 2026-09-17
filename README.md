# PEARL: Parameter-Efficient Adaptation with Retrieval-Augmented Learning

Code for the manuscript:

> **PEARL: Parameter-Efficient Adaptation with Retrieval-Augmented Learning for Molecular Property Prediction**

PEARL benchmarks **three chemical language model (CLM) families** — ChemBERTa, MolFormer, and the
3D SE(3)-equivariant Uni-Mol — under a **four-mode adaptation ladder**, across **seven molecular
property datasets** spanning binary classification, multi-class classification, and regression.
Every CLM-based result is checked against three baselines trained **in-house, under the identical
scaffold split and evaluation protocol**: a PC-only (no-CLM) tree ensemble, Chemprop, and a
hand-rolled GCN — rather than against numbers cited from other papers.

| Mode | Description |
|------|-------------|
| **E2E LoRA** | End-to-end LoRA finetuning of a CLM backbone; task head trained jointly with the adapters |
| **FT Embed** | Optuna-tuned XGBoost/LightGBM/CatBoost models on embeddings extracted from the finetuned CLM |
| **FT Embed+PC** | Same as FT Embed, but the embedding is concatenated with 473-dim physicochemical (PC) features (148 RDKit descriptors, 30 graph features, 128-bit Morgan fingerprints, 167-bit MACCS keys) |
| **RAFE** | FT Embed+PC further augmented with ~69-dim ZINC-250k chemical-neighbourhood features via FAISS retrieval (Retrieval-Augmented Feature Enhancement) |

| Baseline *(in-house, outside the ladder)* | Description |
|---|---|
| **PC-only** | Same PC feature vector as FT Embed+PC, but with **no CLM embedding at all** — isolates how much the CLM contributes over classical descriptors |
| **Chemprop** | D-MPNN graph model, retrained in-house on PEARL's own scaffold splits (not cited from the original Chemprop paper) |
| **GCN** | Hand-rolled GINConv-based graph model, same in-house protocol as Chemprop |

The progression E2E LoRA → FT Embed → FT Embed+PC → RAFE forms an ablation ladder that isolates
each component's contribution: finetuning quality, physicochemical feature engineering, and
retrieval-based neighbourhood signal from the 249,455-molecule ZINC-250k knowledge base. PC-only,
Chemprop, and GCN are zero-CLM / non-CLM control points evaluated alongside this ladder on every
dataset, with bootstrapped confidence intervals used to decide whether a ladder "win" is
statistically real or just a point-estimate.

---

## Datasets

| Dataset | Task | Compounds | Split | Source |
|---|---|---|---|---|
| **BACE** | Binary classification (β-secretase-1 inhibition) | 1,513 | 80:10:10, scaffold | MoleculeNet |
| **BBBP** | Binary classification (blood-brain-barrier permeability) | >2,000 | 80:10:10, scaffold | MoleculeNet |
| **Flavor (FART)** | 5-class classification (bitter/sour/sweet/umami/undefined) | 15,025 | curated splits from FART paper | Zimmermann et al. |
| **hERG** | Binary classification (cardiac ion-channel inhibition) | 13,445 | 70:10:20, scaffold | TDC ADMET Group |
| **DILI** | Binary classification (drug-induced liver injury) | 475 | 70:10:20, scaffold | TDC ADMET Group |
| **Caco-2** | Regression (log apparent permeability) | 906 | 70:10:20, scaffold | TDC ADMET Group |
| **Half-Life** | Regression (plasma half-life, log1p-transformed) | 667 | 70:10:20, scaffold | TDC ADMET Group |

These are the seven datasets reported in the manuscript. The codebase also still contains scripts
and legacy results for an eighth, earlier dataset, **ClinTox** — it predates the switch to the TDC
ADMET Group datasets and is no longer part of the paper's headline benchmark, but the pipelines
still accept it as a `--dataset` option if you want to run it.

hERG, DILI, Caco-2, and Half-Life were pulled and curated via
`scripts/data_curation/tdc_data_curation.py`, which mirrors the standardization/deduplication
conventions of the original `data_cleaning.py` pipeline but sources from the Therapeutics Data
Commons (TDC) ADMET Benchmark Group and additionally records task type (classification/regression)
in `data/clean/tdc_dataset_manifest.json` so downstream scripts can branch automatically.

---

## Repository Structure

```
PEARL/
├── manuscript/          LaTeX source for the PEARL paper (+ HTML/PDF comparison exports)
├── data/
│   ├── raw/             Raw train/valid/test splits (all 8 datasets)
│   ├── clean/           Cleaned SMILES splits (+ tdc_dataset_manifest.json)
│   ├── tdc_cache/        Raw TDC pulls (hERG, DILI, Caco-2, Half-Life)
│   ├── zinc250k/        ZINC-250k reference library (SMILES + logP/QED/SAS)
│   ├── pc_only_features/  Cached 473-dim PC feature vectors (PC-only baseline)
│   ├── finetuned_embeddings/     LoRA model embeddings — SMILES track
│   ├── finetuned_pc_embeddings/  Embeddings merged with PC features
│   ├── conformers/      3D conformers for Uni-Mol
│   ├── rag_features/    RAFE features from ZINC-250k retrieval (SMILES track)
│   └── rag_features_unimol/  RAFE features (Uni-Mol track)
├── scripts/
│   ├── data_curation/   Data cleaning (MoleculeNet-era) + TDC ADMET curation
│   ├── smiles/
│   │   ├── finetuning/  LoRA finetuning + embedding extraction (ChemBERTa, MolFormer; incl. Huber-loss regression variants)
│   │   ├── ml/          FT Embed / FT Embed+PC modelling, PC-only baseline, feature prep
│   │   └── rag/         RAFE pipeline (ZINC-250k indexing, features, modelling)
│   ├── unimol/
│   │   ├── finetuning/  Uni-Mol LoRA finetuning + conformers + embeddings
│   │   ├── ml/          FT Embed modelling on Uni-Mol embeddings
│   │   └── rag/         Uni-Mol RAFE pipeline
│   ├── gnn/             In-house Chemprop and GCN baselines (all 8 datasets)
│   ├── eval/            Bootstrapped-CI evaluation + ROC/PR plotting
│   └── common/          Bootstrap CI utilities, safe model loading, Hugging Face upload scripts
├── bash/                SLURM job scripts (smiles/, unimol/, gnn/, tdc/, rag_new_datasets/, ft_modelling/)
└── results/
    ├── pc_only/         PC-only baseline results (all 8 datasets) + pc_only_summary.csv
    ├── gnn/             Chemprop and GCN in-house baseline results
    ├── lora_finetuning/ E2E LoRA (SMILES track) test-set results, per dataset/loss
    ├── unimol_finetuning/  E2E LoRA (Uni-Mol track) evaluation metrics
    ├── ft_embeddings/   FT Embed / FT Embed+PC results (XGB/LGBM/CatBoost)
    ├── rag/             SMILES RAFE results
    ├── rag_unimol/       Uni-Mol RAFE results
    ├── summary/         Cross-dataset, cross-method mean±SE summary tables
    └── figures/         Generated ROC/PR/summary plots used in the manuscript
```

---

## Scripts

### Data Curation (`scripts/data_curation/`)

| Script | Purpose |
|--------|---------|
| `data_cleaning.py` | Clean raw SMILES for BBBP, ClinTox, Flavor |
| `bace_data_cleaning.py` | BACE-specific SMILES cleaning |
| `tdc_data_curation.py` | Pull + curate hERG, DILI, Caco-2, Half-Life from the TDC ADMET Group; writes `tdc_dataset_manifest.json` |

---

### E2E LoRA Finetuning

#### SMILES Track (`scripts/smiles/finetuning/`)

| Script | Purpose |
|--------|---------|
| `finetune_{bace\|bbbp\|clintox\|flavor\|herg\|dili}_{fl\|wl}.py` | LoRA finetuning with focal loss / weighted loss (classification, 12 scripts) |
| `finetune_{caco2\|half_life}_huber.py` | LoRA finetuning with Huber loss (regression, 2 scripts) |
| `finetuned_model_embeddings.py` | Extract embeddings from finetuned CLMs (original 4 datasets) |
| `finetuned_model_embeddings_new_datasets.py` | Same, for hERG/DILI/Caco-2/Half-Life |

#### Uni-Mol Track (`scripts/unimol/finetuning/`)

| Script | Purpose |
|--------|---------|
| `smiles_to_conformers.py` | Generate ETKDGv3 + MMFF94 3D conformers |
| `unimol_lora_trainer.py` | Shared LoRA trainer utilities |
| `finetune_unimol_{dataset}_{fl\|wl}.py` | Uni-Mol LoRA finetuning, classification (12 scripts) |
| `finetune_unimol_{caco2\|half_life}_huber.py` | Uni-Mol LoRA finetuning, regression (2 scripts) |
| `unimol_embeddings.py` / `unimol_embeddings_new_datasets.py` | Extract 2560-dim Uni-Mol embeddings |

---

### FT Embed and FT Embed+PC (ML on Finetuned Embeddings)

#### Feature Preparation and Modelling (`scripts/smiles/ml/`)

| Script | Purpose |
|--------|---------|
| `bace_feature_extraction_ft_model.py` / `pc_feature_extraction_ft_model_refactored.py` | Merge finetuned embeddings + 473-dim PC features (original 4 datasets) |
| `pc_feature_merge_new_datasets.py` | Same, for hERG/DILI/Caco-2/Half-Life |
| `{bace\|bbbp\|clintox\|flavor}_modelling_refactored.py` / `_pc_modelling_refactored.py` | Optuna-tuned XGB/LGBM/CatBoost on embeddings / embeddings+PC (original 4 datasets) |
| `ft_modelling_new_datasets.py` | Same, for hERG/DILI/Caco-2/Half-Life (classification + regression) |
| `pc_only_modelling.py` | PC-only baseline, all 8 datasets (see below) |
| `rag_modelling_new_datasets.py` | RAFE modelling for hERG/DILI/Caco-2/Half-Life (SMILES track) |
| `aggregate_metrics.py` / `aggregate_all_metrics.py` | Aggregate FT Embed metrics per-dataset / across all datasets |

#### Uni-Mol FT Embed Modelling (`scripts/unimol/ml/`)

| Script | Purpose |
|--------|---------|
| `unimol_modelling.py` | XGB/LGBM/CatBoost on Uni-Mol LoRA embeddings (original 4 datasets) |
| `rag_modelling_unimol_new_datasets.py` | RAFE modelling for hERG/DILI/Caco-2/Half-Life (Uni-Mol track) |

---

### PC-only Baseline — No CLM

Trains Optuna-tuned XGBoost / LightGBM / CatBoost on the 473-dim engineered physicochemical (PC)
feature vector alone (RDKit descriptors, graph features, Morgan fingerprints, MACCS keys) —
**no CLM embedding of any kind**. Computed directly from `data/clean/`, so it requires no
LoRA-finetuned checkpoint or externally-hosted artifacts. Now covers all seven benchmark datasets
(plus ClinTox).

```bash
python scripts/smiles/ml/pc_only_modelling.py --dataset {bace|bbbp|flavor|herg|dili|caco2|half_life|all}
```

Output: `results/pc_only/{DATASET}_PC_Only_Results/metrics/{model}_metrics.json`,
consolidated summary at `results/pc_only/pc_only_summary.csv`.

---

### Chemprop and GCN — In-house Graph Baselines (`scripts/gnn/`)

Retrained in-house on PEARL's own scaffold splits and Optuna search budget, so the
"does a graph model beat/match the CLM ladder" comparison is a controlled experiment rather than a
cross-paper citation, across all seven benchmark datasets (plus ClinTox).

```bash
python scripts/gnn/chemprop_baseline.py --dataset {bace|bbbp|flavor|herg|dili|caco2|half_life|all}
python scripts/gnn/gcn_baseline.py --dataset {bace|bbbp|flavor|herg|dili|caco2|half_life|all}
# or via SLURM:
sbatch bash/gnn/run_{dataset}_chemprop.sh
sbatch bash/gnn/run_{dataset}_gcn.sh
```

Output: `results/gnn/{chemprop|gcn}/{DATASET}_{Chemprop|GCN}_Results/`.

---

### RAFE (Retrieval-Augmented Feature Enhancement)

#### SMILES RAFE Pipeline (`scripts/smiles/rag/`)

| Script | Purpose |
|--------|---------|
| `embed_zinc250k.py` / `embed_zinc250k_new_datasets.py` | Embed all 249,455 ZINC-250k molecules with finetuned CLMs |
| `build_zinc_index.py` / `build_zinc_index_new_datasets.py` | Build GPU FAISS FlatIP indices from ZINC-250k embeddings |
| `rag_feature_extraction.py` / `rag_feature_extraction_new_datasets.py` | Query FAISS indices; compute RAFE features per molecule |
| `rag_modelling_{bace\|bbbp\|clintox\|flavor}.py` | Train Optuna-tuned XGB/LGBM/CatBoost (original 4 datasets) |
| `../ml/rag_modelling_new_datasets.py` | Same, for hERG/DILI/Caco-2/Half-Life |
| `aggregate_rag_results.py` / `aggregate_rag_results_new_datasets.py` | Consolidate metrics across models and datasets |

#### Uni-Mol RAFE Pipeline (`scripts/unimol/rag/`)

| Script | Purpose |
|--------|---------|
| `embed_zinc250k_unimol.py` / `embed_zinc250k_unimol_new_datasets.py` | Embed ZINC-250k with finetuned Uni-Mol models |
| `build_zinc_index_unimol.py` / `build_zinc_index_unimol_new_datasets.py` | Build Uni-Mol FAISS indices |
| `rag_feature_extraction_unimol.py` / `rag_feature_extraction_unimol_new_datasets.py` | Extract Uni-Mol RAFE features |
| `rag_modelling_{bace\|bbbp\|clintox\|flavor}_unimol.py` | Uni-Mol RAFE classifiers (original 4 datasets) |
| `../ml/rag_modelling_unimol_new_datasets.py` | Same, for hERG/DILI/Caco-2/Half-Life |

---

### Evaluation and Statistics (`scripts/eval/`, `scripts/common/`)

Every reported comparison in the paper is backed by bootstrapped confidence intervals, not raw
point estimates.

| Script | Purpose |
|--------|---------|
| `common/bootstrap_ci.py` | Canonical bootstrapped-CI implementation (10,000 stratified resamples for classification, normal-approximation for regression), matching the manuscript's stated protocol |
| `eval/bootstrap_ci.py` | Standalone CLI / importable variant for ad-hoc prediction CSVs |
| `common/backfill_ci.py` | Backfill CIs onto already-trained PC-only/Chemprop/GCN results without retraining |
| `eval/plot_bace_roc_pr.py` | ROC/PR curves for BACE (Figure in main text) |
| `eval/plot_roc_pr_4datasets.py` | ROC/PR curves across the four binary classification datasets |
| `eval/eval_rafe_flavor.py` | RAFE-specific evaluation utilities for Flavor |

---

### Publishing Models to Hugging Face (`scripts/common/`)

| Script | Purpose |
|--------|---------|
| `upload_top5_to_hf.py` | Ranks every method (by MCC/Spearman) per dataset, resolves each top-5 entry's saved artifact, and uploads all seven datasets' top-5 models into one consolidated Hugging Face model repo |
| `upload_rafe_faiss_indices.py` | Uploads the ZINC-250k FAISS index for whichever RAFE configuration made a dataset's top-5, alongside its model folder |

```bash
python scripts/common/upload_top5_to_hf.py --hf-user <your-hf-username> --do-upload
python scripts/common/upload_rafe_faiss_indices.py --hf-user <your-hf-username> --do-upload
```

Published models: <https://huggingface.co/raghvendramall/pearl-benchmark-models>

---

## Large Files (not in repo)

Finetuned model checkpoints, ZINC embeddings, and FAISS indices are stored externally, split
across two locations depending on which datasets they belong to. These are not part of the git
repository — download/copy them yourself into two directories (named however you like; we suggest
`PEARL_Extras` and `PEARL_Extras_v2`) and point these two variables at wherever you placed them:

```bash
export PEARL_EXTRAS="/path/to/PEARL_Extras"        # BACE, BBBP, ClinTox, Flavor
export PEARL_EXTRAS_V2="/path/to/PEARL_Extras_v2"  # hERG, DILI, Caco-2, Half-Life
```

## Requirements

```bash
pip install -r requirements.txt
```

> `faiss-gpu` requires a CUDA GPU. On CPU-only machines use `faiss-cpu` instead.
> `unimol-tools` is only needed for the Uni-Mol track.
> `pip-constraints-effichem.txt` pins the exact dependency versions used to reproduce the paper's reported numbers.

---

## End-to-End Workflow

```
Step 1   Data curation ─────────────┬── data_cleaning.py / bace_data_cleaning.py   (BACE, BBBP, ClinTox, Flavor)
                                    └── tdc_data_curation.py                       (hERG, DILI, Caco-2, Half-Life)
Step 2   LoRA finetuning ──────────┬── SMILES track (ChemBERTa, MolFormer)         [E2E LoRA]
                                    └── Uni-Mol track (conformers → finetune)      (FL/WL for classification, Huber for regression)
Step 3   Extract finetuned embeddings
Step 4   Prepare ML input + model  ─┬── FT Embed: embeddings only                  [FT Embed]
                                    ├── FT Embed: embeddings + PC features          [FT Embed+PC]
                                    ├── PC-only: PC features alone, no CLM          [PC-only baseline]
                                    └── Chemprop / GCN: trained directly on graphs  [in-house GNN baselines]
Step 5   RAFE pipeline ────────────┬── 5a  Embed ZINC-250k                        [RAFE]
                                    ├── 5b  Build FAISS indices
                                    ├── 5c  Extract RAFE features
                                    └── 5d  RAFE modelling
Step 6   Evaluation                 ── bootstrapped CIs + ROC/PR/summary figures
Step 7   Publish (optional)         ── upload top-5 models + RAFE FAISS indices to Hugging Face
```

For the four TDC ADMET datasets (hERG, DILI, Caco-2, Half-Life), Steps 2–5 use the
`*_new_datasets.py` script variants noted in the tables above; the underlying pipeline logic
(finetune → embed → RAFE-index → RAFE-features → model) is otherwise identical to the original
four datasets.

---

### Step 1 — Data Curation

```bash
python scripts/data_curation/data_cleaning.py          # BBBP, ClinTox, Flavor
python scripts/data_curation/bace_data_cleaning.py     # BACE
python scripts/data_curation/tdc_data_curation.py      # hERG, DILI, Caco-2, Half-Life
```

Output: `data/clean/{dataset}_datasets/{split}_clean.csv` (+ `tdc_dataset_manifest.json`)

---

### Step 2 — LoRA Finetuning  [E2E LoRA]

**SMILES track** — requires a `.env` file at the repo root with `WANDB_API_KEY=<key>`.

```bash
# Classification: focal loss or weighted loss
sbatch bash/smiles/run_{dataset}_finetuning_fl.sh   # dataset in {bace,bbbp,clintox,flavor,herg,dili}
sbatch bash/smiles/run_{dataset}_finetuning_wl.sh

# Regression: Huber loss
sbatch bash/smiles/run_{caco2|half_life}_finetuning_huber.sh
```

**Uni-Mol track** — generate conformers first.

```bash
sbatch bash/unimol/run_conformer_gen.sh
# or: python scripts/unimol/finetuning/smiles_to_conformers.py --dataset {bace|bbbp|clintox|flavor|herg|dili|caco2|half_life|zinc250k}

sbatch bash/unimol/run_unimol_{dataset}_{fl|wl}.sh       # classification
python scripts/unimol/finetuning/finetune_unimol_{caco2|half_life}_huber.py  # regression
```

Output: `$PEARL_EXTRAS(_V2)/{focal_loss|weighted_loss|huber_loss}_{DATASET}/{model}_LoRA_Finetuned/`

---

### Step 3 — Extract Finetuned Embeddings

```bash
# SMILES track
python scripts/smiles/finetuning/finetuned_model_embeddings.py               # original 4 datasets
python scripts/smiles/finetuning/finetuned_model_embeddings_new_datasets.py  # hERG, DILI, Caco-2, Half-Life

# Uni-Mol track
python scripts/unimol/finetuning/unimol_embeddings.py
python scripts/unimol/finetuning/unimol_embeddings_new_datasets.py
```

---

### Step 4 — FT Embed / FT Embed+PC / PC-only / GNN Baselines

```bash
# Merge embeddings + PC features
python scripts/smiles/ml/bace_feature_extraction_ft_model.py
python scripts/smiles/ml/pc_feature_extraction_ft_model_refactored.py  # BBBP, ClinTox, Flavor
python scripts/smiles/ml/pc_feature_merge_new_datasets.py              # hERG, DILI, Caco-2, Half-Life

# FT Embed / FT Embed+PC modelling
sbatch bash/smiles/run_{dataset}_ft.sh        # embeddings only     (dataset in {bace,bbbp,clintox,flavor})
sbatch bash/smiles/run_{dataset}_pc_ft.sh     # embeddings + PC     (dataset in {bace,bbbp,clintox,flavor})
python scripts/smiles/ml/ft_modelling_new_datasets.py --dataset {herg|dili|caco2|half_life}  # both variants
sbatch bash/unimol/run_unimol_ft_modelling.sh # Uni-Mol FT Embed, both variants

# PC-only baseline (no CLM)
python scripts/smiles/ml/pc_only_modelling.py --dataset all

# In-house GNN baselines
python scripts/gnn/chemprop_baseline.py --dataset all
python scripts/gnn/gcn_baseline.py --dataset all
```

Output: `results/ft_embeddings/{Dataset}_{FT|PC_FT}_Results/`,
`results/pc_only/{DATASET}_PC_Only_Results/`, `results/gnn/{chemprop|gcn}/{DATASET}_*_Results/`

---

### Step 5 — RAFE Pipeline  [RAFE]

```bash
# 5a. Embed ZINC-250k
sbatch bash/smiles/run_embed_zinc_{dataset}.sh                    # original 4 datasets
sbatch bash/rag_new_datasets/run_embed_zinc_hf.sh                 # hERG, DILI, Caco-2, Half-Life
sbatch bash/unimol/run_embed_zinc_unimol_{dataset}.sh
sbatch bash/rag_new_datasets/run_embed_zinc_unimol.sh

# 5b. Build FAISS indices
sbatch bash/smiles/run_build_zinc_index.sh
sbatch bash/rag_new_datasets/run_build_zinc_index_hf.sh
sbatch bash/unimol/run_build_zinc_index_unimol.sh
sbatch bash/rag_new_datasets/run_build_zinc_index_unimol.sh

# 5c. Extract RAFE features
sbatch bash/smiles/run_rag_features.sh
sbatch bash/rag_new_datasets/run_rag_features_hf.sh
sbatch bash/unimol/run_rag_features_unimol.sh
sbatch bash/rag_new_datasets/run_rag_features_unimol.sh

# 5d. RAFE modelling
sbatch bash/smiles/run_rag_modelling_{bace|bbbp|clintox|flavor}.sh
sbatch bash/rag_new_datasets/run_rag_modelling_hf.sh              # hERG, DILI, Caco-2, Half-Life
sbatch bash/unimol/run_rag_modelling_unimol_{bace|bbbp|clintox|flavor}.sh
sbatch bash/rag_new_datasets/run_rag_modelling_unimol.sh
```

Output: `data/rag_features{_unimol}/{dataset}/{col_name}_{split}_rag.csv`,
`results/rag{_unimol}/{dataset}/{col_name}/metrics.json`

---

### Step 6 — Evaluation

```bash
python scripts/common/bootstrap_ci.py --dataset all        # or scripts/eval/bootstrap_ci.py for ad-hoc CSVs
python scripts/common/backfill_ci.py                        # backfill CIs onto already-trained results
python scripts/eval/plot_bace_roc_pr.py
python scripts/eval/plot_roc_pr_4datasets.py
```

Output: `ci_metrics.json` alongside each `metrics.json`; summary tables in `results/summary/`;
figures in `results/figures/`.

---

## Environment Variables

| Variable          | Default                                              | Purpose                                              |
|--------------------|-------------------------------------------------------|-------------------------------------------------------|
| `PEARL_EXTRAS`     | *(none — set to your own `PEARL_Extras` path)*        | Root for large external files — BACE, BBBP, ClinTox, Flavor |
| `PEARL_EXTRAS_V2`  | *(none — set to your own `PEARL_Extras_v2` path)*     | Root for large external files — hERG, DILI, Caco-2, Half-Life |
| `WANDB_API_KEY`    | *(none — set in `.env`)*                              | WandB sweep key for LoRA hyperparameter tuning        |
| `RANDOM_SEED`      | `42`                                                   | Global random seed                                    |
| `N_JOBS`           | half of available CPUs                                 | Parallel jobs for tree-based classifiers              |
| `OPTUNA_TRIALS`    | `20`                                                   | Optuna hyperparameter trials per model                |

> Several scripts fall back to a hardcoded path if `PEARL_EXTRAS`/`PEARL_EXTRAS_V2` is unset — that
> fallback is specific to the original development machine and will not exist elsewhere, so always
> set both variables explicitly rather than relying on the default.

## Citation

If you use this code or data, please cite the PEARL manuscript.

---

&copy; Raghvendra Mall. All rights reserved.
