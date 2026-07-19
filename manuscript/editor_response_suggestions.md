# Addressing the Editor's Comments on PEARL — Suggested Revisions

**Context.** The editor raised two rejection-driving issues for *Journal of Cheminformatics*:

1. PEARL benchmarks on "old" datasets (BACE, BBBP, ClinTox from MoleculeNet, plus FART/Flavor) when much higher-quality benchmark resources now exist through the **OpenADMET** initiative.
2. PEARL only positions CLMs (ChemBERTa, MolFormer, Uni-Mol) against each other and cites GNN results *from other papers* rather than running graph-based models (GCN/GIN/D-MPNN/etc.) in-house under the same protocol — so the claim "CLMs are competitive/best" is not actually substantiated by a controlled benchmark.

This document lays out concrete, scoped changes to the repo and manuscript that address both points, plus novel extensions that turn the revision into a stronger contribution rather than a defensive patch.

---

## 1. The benchmark datasets are dated — bring in OpenADMET / TDC

### What's true about the current setup
Checked against `manuscript/PEARL_paper.tex` (§Materials and Methods) and `README.md`: PEARL trains on **BACE, BBBP, ClinTox** (MoleculeNet, scaffold split, DeepChem reference splits from 2018) and **Flavor/FART** (2024, but small and task-idiosyncratic). All four are classification tasks, three binary, one 5-class. There are no regression endpoints and no exposure/PK-relevant ADMET endpoints (clearance, half-life, CYP inhibition, hERG, solubility, etc.).

### Why the editor is right to push back
- MoleculeNet scaffold splits are known to contain label noise, near-duplicate leakage across splits, and are heavily saturated (GNN/CLM papers have reported AUC > 0.93 on BBBP since ~2020), so a new method "matching" these numbers is weak evidence of real advance.
- **OpenADMET** (UCSF / Octant / Open Molecular Software Foundation, ARPA-H + Gates Foundation funded) exists specifically to fix this: it produces open, assay-consistent, pre-competitive ADMET data and runs **blind prospective challenges** (e.g., the ExpansionRx–OpenADMET challenge, ~7,000 molecules from a real lead-optimization campaign against RNA-mediated disease targets) rather than static, possibly-leaky retrospective splits.
- The **TDC ADMET Benchmark Group** (Therapeutics Data Commons) is the community-standard modern alternative even short of a blind challenge: 22 curated endpoints (475–13,130 molecules each), standardized scaffold splits, and a public leaderboard, spanning absorption, distribution, metabolism, excretion, and toxicity — i.e., exactly the same property classes PEARL already targets (BBB permeability, toxicity) but with more endpoints and more rigorous curation.

### Concrete actions
1. **Add 3–5 TDC ADMET Group endpoints** that are direct analogues of what PEARL already models, so existing infrastructure (LoRA finetuning → FT Embed → FT Embed+PC → RAFE) transfers with minimal new code:
   - `BBB_Martins` (already conceptually covered — cross-check against current BBBP for consistency/leakage),
   - `hERG` or `hERG_Karim` (binding-affinity/binding-pocket task, directly comparable in spirit to BACE — a good second geometry-sensitive target for the Uni-Mol 3D story),
   - `AMES` or `DILI` (toxicity, analogous to ClinTox but larger and better curated),
   - `Caco2_Wang`, `Solubility_AqSolDB`, or `Half_Life_Obach` (regression — see point 2 below).
   TDC datasets are pip-installable (`pip install PyTDC`) and ship canonical scaffold splits, so `scripts/data_curation/` only needs a thin new loader, not a rewrite.
2. **Add at least one regression endpoint.** The manuscript itself lists "extending PEARL to regression tasks... with calibrated uncertainty" as unaddressed future work (Discussion). Doing this now, on a TDC regression endpoint, kills two birds: it answers the editor's benchmark-quality concern *and* pre-empts a "why classification only" review comment. RAFE's neighborhood features (logP/QED/SAS distributions, similarity-weighted means) are already real-valued and require no redesign for a regression head — only the tree-classifier objective needs to switch to Spearman/MAE-optimized regressors.
3. **Report a held-out, temporally- or scaffold-disjoint generalization check** using an OpenADMET blind-challenge-style split (even a self-constructed time/scaffold split on a TDC set works if the live challenge window has closed) to directly counter "these are old, saturated splits" — show PEARL/RAFE performance doesn't collapse under a harder, leakage-resistant split.
4. **Explicitly retain BACE/BBBP/ClinTox/Flavor** as a legacy comparison layer (for continuity with EffiChem and published GNN numbers) but reframe them in the manuscript as a validation set, with the new OpenADMET/TDC endpoints as the primary evidence base. Journal of Cheminformatics reviewers are likely to accept "old + new" more readily than a wholesale swap that discards prior work.

---

## 2. CLMs vs. everything else — the GNN comparison is not a real benchmark

### What's actually in the repo
`grep` across `scripts/` and `bash/` finds **no GCN/GNN/Chemprop/D-MPNN implementation anywhere in PEARL**. The "GNN baselines" in the manuscript (Table `tab:external_main`, Supplementary Table S5: Chemprop, GROVER-Large, AttentiveFP, Mole-BERT) are **numbers copied from the original publications**, not runs performed on the same data splits, seeds, hardware, or evaluation code. The manuscript is transparent about this (it flags a "split-difference caveat" for ClinTox in Supplementary §S5), which is honest — but it also means every "PEARL matches/beats GROVER" claim in the paper is a cross-paper comparison, not a controlled experiment. This is precisely the gap the editor is naming, and it's the harder of the two issues to fix, but also the highest-leverage one for the resubmission.

### Concrete actions
1. **Implement and run at least two GNN baselines in-house**, on the identical scaffold splits / train-val-test files already produced by `scripts/data_curation/`:
   - **Chemprop (D-MPNN)** — the de facto standard baseline, has a stable, well-documented Python package; minimal integration effort.
   - **GIN or GCN** (e.g., via PyTorch Geometric / DGL-LifeSci) — a plain, non-pretrained graph baseline. Include this *specifically* because reviewers increasingly ask "does a simple GCN close most of the gap?" — if it does, that's an important, publishable finding in itself, not a threat to the paper.
   Both should go through the *same* Optuna-tuned hyperparameter search budget, the *same* bootstrapped-CI evaluation protocol, and the *same* WL/FL class-imbalance handling already used for the tree classifiers, so the comparison is apples-to-apples rather than borrowed-numbers-to-apples.
2. **Add a "cheap baseline" tier that should have existed from the start.** PEARL already computes the 473-dim PC feature vector (RDKit descriptors + graph features + Morgan + MACCS) for every molecule (`scripts/smiles/ml/pc_feature_extraction_ft_model_refactored.py`). Training the existing Optuna-tuned XGBoost/LightGBM/CatBoost stage on **PC features alone, with no CLM embedding at all**, is a near-zero-cost ablation using code that already exists — it directly tests whether the CLM embedding is earning its keep at all versus classical descriptors + tree ensemble. If PC-only tree models are competitive with CLM-based ones on some datasets (plausible, given the manuscript's own finding that "adding PC features to finetuned embeddings provides minimal improvement"), that is an important and easy-to-report result.
3. **Report training/inference cost, not just parameter count.** PEARL's core efficiency argument is % trainable parameters reduced via LoRA. GNN baselines (especially plain GCN/GIN) are typically 10–100x cheaper to train from scratch than any LoRA-adapted 46M–100M-parameter CLM. A reviewer sympathetic to GNNs will ask "efficient relative to what — a needlessly large model class?" Reporting GPU-hours and wall-clock alongside parameter-percentage pre-empts that and lets PEARL make an honest "efficiency vs. accuracy" Pareto argument instead of a parameter-count argument alone.
4. **Reframe the contribution claim.** Instead of "CLMs (with LoRA) are the best models," which the current evidence doesn't support, position PEARL as **"a controlled framework for comparing representation classes (1D sequence, 2D graph, 3D geometry) under matched parameter-efficient adaptation and matched retrieval augmentation"** — with an honest result that whichever family wins is task-dependent (already true for CLM vs. Uni-Mol in the current results) and now extended to include graphs. This reframing is defensible with the data the authors will actually have after adding GNN baselines, whereas the current framing is not.

---

## 3. Novelty to add during revision (turns a defensive fix into a stronger paper)

These go beyond "satisfy the editor" and would make the resubmission read as an advance rather than a patch:

1. **A fourth representation track, not just a fourth baseline table.** Slot a graph model (Chemprop D-MPNN or GIN) into PEARL's existing four-mode ladder (E2E → FT Embed → FT Embed+PC → RAFE) exactly like Uni-Mol was. This requires: (a) LoRA-style parameter-efficient adaptation for the GNN (e.g., low-rank adapters on message-passing weight matrices, or adapter layers between MPNN blocks — there is emerging literature on PEFT for GNNs to draw on), and (b) RAFE feature extraction from GNN embeddings via the same FAISS/ZINC-250k pipeline. This directly extends the paper's central "parameter-efficient adaptation across representation classes" narrative from {1D, 3D} to {1D, 2D, 3D} and would make PEARL, to our knowledge, the first framework to apply matched PEFT + retrieval augmentation across all three molecular representation paradigms.
2. **RAFE as an applicability-domain / uncertainty signal.** RAFE already computes `zinc_nearest_sim` and neighborhood-density features as a byproduct of retrieval. Repurpose these explicitly as a confidence/OOD flag and validate them against the OpenADMET blind-challenge-style split from Section 1 — i.e., show that low ZINC-neighborhood similarity predicts higher prediction error. This is a natural, low-cost extension of existing RAFE features into an applicability-domain contribution, which is exactly the kind of practical, deployment-relevant result Journal of Cheminformatics favors.
3. **Extend RAFE to regression** (paired with the new TDC regression endpoint) by replacing the mean/std neighborhood statistics with the regression-target analogue (e.g., neighborhood mean/variance of the predicted property itself, computed leakage-free from ZINC-250k's own logP/QED/SAS, or from a second held-out property-labeled corpus). This operationalizes the "future work" the Discussion already promises, inside the same revision.
4. **Cross-representation retrieval.** Currently each CLM family gets its own FAISS index (ChemBERTa-indexed ZINC, MolFormer-indexed ZINC, Uni-Mol-indexed ZINC). An interesting ablation: does retrieving neighbors using a *different* model's embedding space (e.g., query with MolFormer, retrieve via the Uni-Mol 3D index) add complementary signal beyond same-space retrieval? This tests whether RAFE's value is representation-specific or general, and is a cheap experiment given the indices already exist for all three families.

---

## 4. Suggested order of operations

1. Add the PC-only tree-classifier baseline (§2.2) — trivial, uses existing code, immediately clarifies how much the CLM embedding is contributing.
2. Stand up Chemprop and one plain GNN (GIN/GCN) on the *existing* BACE/BBBP/ClinTox/Flavor splits first, to get an in-house-controlled version of Table `tab:external_main`/S5 (§2.1) — this is the single highest-priority fix since it's the crux of the editor's second objection.
3. Add 2–3 TDC ADMET Group endpoints (start with `hERG` and one toxicity + one regression endpoint) and rerun the full four-mode ladder plus the new GNN baselines on them (§1).
4. Only after 1–3 are in hand, decide whether to attempt the GNN-into-the-ladder extension (§3.1) and the OOD/applicability-domain analysis (§3.2) — these are higher-effort and are what would elevate the resubmission beyond "responded to reviewers" into a stronger paper, but 1–3 are what make the paper acceptable in the first place.

---

## References for the above (for the response letter / methods section)

- OpenADMET consortium (UCSF, Octant, Open Molecular Software Foundation; ARPA-H / Gates Foundation funded); ExpansionRx–OpenADMET blind challenge (Oct 2025), ~7,000-molecule real lead-optimization dataset across 9 ADMET endpoints. See `openadmet.org` and the associated Nature Communications paper on systematic open-science ADMET prediction ("Mapping the avoid-ome").
- Therapeutics Data Commons ADMET Benchmark Group: 22 endpoints, standardized scaffold splits, public leaderboard (`tdcommons.ai/benchmark/overview`).
- Chemprop / D-MPNN (Yang et al. 2019) — already cited in PEARL as a literature number; needs an in-house run.
- GIN (Xu et al. 2019) / GCN (Kipf & Welling 2017) as minimal, non-pretrained graph baselines.
- PEFT-for-GNN literature (adapters/low-rank updates on message-passing networks) as a starting point for the fourth-track extension in §3.1.
