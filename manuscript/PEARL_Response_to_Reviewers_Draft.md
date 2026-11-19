# Response to Reviewers

**Manuscript:** PEARL: Parameter-Efficient Adaptation with Retrieval-Augmented Learning for Molecular Property Prediction (Journal of Cheminformatics)

*Written for: the handling editor and Reviewers 1, 2 and 3. Status markers: **[PENDING: ...]** marks a result that is still being computed and will be filled in before submission; **[v3]** marks a change that will appear in the revised manuscript or supplement.*

Dear Editor and Reviewers,

We are grateful for the time and care that went into these reports. They pointed us to the right questions: whether the gains of RAFE can be separated from the choice of tree model, whether simple baselines explain the results, and whether our statistics support the claims. We answer each point below with new experiments run under one protocol: identical scaffold splits, the same tuning budget (20 Optuna trials with 5-fold cross-validation on the training split only), three tree models, and five seeds. Quoted reviewer text is abridged.

## Overview of our answers

1. **What the evidence supports.** Along the PEARL ladder (fine-tuned embedding, plus descriptors, plus RAFE), the full pipeline improves significantly on the embedding alone on 4 of 7 datasets (BBBP, hERG, DILI, Caco-2; matched-configuration analysis, Figure 1). In controlled comparisons, RAFE is a safe final stage: in 24 paired comparisons against FT Embed+PC (3 trees × 8 dataset configurations) none is significantly worse and none is significantly better, and the stage costs about 0.13 s per molecule at prediction time and its retrieval settings do not need tuning (Figure 5).
2. **Benchmark view.** As a benchmark of which embedding, loss, tree and stage works best for each dataset, RAFE's best configuration ranks in the top three on 6 of 7 datasets and first on BBBP and DILI (Figure 2, Table 1). We now back every best-configuration value with seed-averaged results.
3. **Where we have revised our claims.** RAFE's advantage over FT Embed+PC is small and dataset-specific, and physicochemical descriptors alone (PC-only) are a strong baseline that RAFE does not significantly exceed on any dataset. We state this directly and explain how the fine-tuned embedding and RAFE contribute where they do (Figure 8).
4. **Statistics and reproducibility.** Paired bootstrap confidence intervals on identical test molecules replace the ±1 SE overlap; the training and tuning protocol is stated explicitly. [v3]

---

## Reviewer 1

**R1.1 Limited methodological novelty** *("integrates existing techniques ... RAFE constructs statistical features from retrieved molecules without a new representation-learning mechanism")*.
We respectfully agree that PEARL assembles known components, and we have revised the manuscript so that it no longer presents RAFE as a new learning method. We would, however, argue that the contribution is not a new learner but a systematic, reproducible study that had not been done: (a) a ladder from end-to-end LoRA through frozen embeddings, descriptors and retrieval features, evaluated under one scaffold-split protocol against in-house PC-only, Chemprop and GCN baselines on seven datasets; (b) retrieval from an external molecular library that uses structures only, so no label information enters the features and no test leakage is possible; and (c) LoRA adaptation of Uni-Mol through its fused attention projection, which the standard adapter recipes do not cover. Relative to EffiChem, the new elements are the Uni-Mol branch, the descriptor and retrieval stages, and the controlled comparisons reported in this response. We have added a paragraph on exactly what RAFE adds. [v3: Introduction]

**R1.2 Inconsistent improvements from retrieval augmentation** *("small or negative improvement on several datasets; PC-only outperforms RAFE on Caco-2")*.
We agree that RAFE does not improve every dataset, and we think this is what a benchmark of this kind should show (no single configuration is best everywhere). Three observations place the results in context.

First, the full pipeline's gain over the embedding alone is clear. Across matched embedding-loss-tree configurations, RAFE exceeds FT Embed significantly on 4 of 7 datasets (BBBP: +0.014, p = 0.025; hERG: +0.006, p = 0.015; DILI: +0.052, p = 0.012; Caco-2: +0.085, p < 0.001; Figure 1). Second, as a ranking of best configurations, RAFE is first on BBBP and DILI and in the top three on 6 of 7 datasets (Table 1, Figure 2). Third, RAFE's contribution beyond the descriptors is modest: against FT Embed+PC the matched-configuration difference is +0.034 on Flavor (Wilcoxon p = 0.001) and +0.013 on BACE (p = 0.012), and within about ±0.01 elsewhere. The cost-performance view (Figure 3) shows on which datasets the extra stage reaches the frontier.

We add a caution that we think the reviewer would also want: best-configuration values are single runs and are optimistic. When the same configurations are rerun with five seeds, the two largest values move (DILI: 0.790 in the table, 0.697 ± 0.022 over five seeds with the stored RAFE features; BBBP: 0.798 versus 0.772 to 0.779), and we will recompute the rankings on seed-averaged values. We will report these next to the best-run values and base the rankings on them. **[PENDING: seed-averaged, CV-selected ranking, replaces Table 1; see Table 10]**

**Table 1.** Rank of RAFE's best configuration among seven methods (single-run values from the supplementary tables; Chemprop, GCN and PC-only from supplementary Table S11). Mean rank 2.29; over the medians of each method's configurations, RAFE is in the top three on 2 datasets (mean rank 4.14; PC-only 1.43).

| Dataset | RAFE best configuration | RAFE rank (of 7) | Best method |
|---|---|---|---|
| BACE | 0.561 | 2 | E2E LoRA (0.623) |
| BBBP | 0.798 | 1 | RAFE (0.798) |
| Flavor | 0.804 | 3 | Chemprop (0.809) |
| hERG | 0.589 | 4 | E2E LoRA (0.609) |
| DILI | 0.790 | 1 | RAFE (0.790) |
| Caco-2 | 0.835 | 2 | PC-only (0.853) |
| Half-Life | 0.568 | 3 | FT Embed (0.623) |

![**Figure 1.** Difference between pipeline stages over matched embedding-loss-tree configurations (supplementary tables; intervals are bootstrap over configurations and are descriptive because configurations share a test set).](../results/revision/figures_response/figA_ladder_gain.pdf)

![**Figure 2.** Best configuration of each method per dataset (red: RAFE).](../results/revision/figures_response/figB_best_config_ranking.pdf)

![**Figure 3.** Cost-performance frontier of the best configuration of each stage (median view over matched configurations: `results/revision/e12_pareto/pareto_frontier_median.pdf`).](../results/revision/e12_pareto/pareto_frontier.pdf)

**R1.3 Insufficient ablations** *(number of neighbours, similarity metric, feature groups, mixed tree models)*.
We have added each of these controls under the identical protocol.

*Feature groups.* Adding or removing any of the four RAFE groups changes the score by no more than 0.02 in 85% of the 72 dataset-by-arm cells (Table 2, Figure 4); the 11 larger shifts are all decreases and are confined to BACE (test n = 152), DILI (test n = 96), Half-Life (test n = 133), which have the smallest test sets in the study (n = 96, 133, 152); individual values are listed in Table 2. No single group is responsible for the effect of RAFE, which is consistent with the groups carrying overlapping information (see R2.3).

*Retrieval metric and neighbour count.* We compared CLM-cosine retrieval with Morgan-Tanimoto retrieval and with random retrieval (five independent random draws), and swept k from 1 to 100. Of 44 paired comparisons, 2 has an interval that excludes zero (Figure 5). Practically, RAFE does not need its retrieval settings tuned. We read the equal performance of random neighbours as showing that RAFE works as a source of chemical-space context rather than as a nearest-analogue lookup, and we have changed the manuscript's description accordingly. **[PENDING: pooled paired intervals for the random-retrieval control (5 draws × 5 seeds) from the final cluster cells]**

*Mixed tree models.* The reviewer is right that the main-text tables could pair different trees. The controlled matrix (Table 5, R2.1) runs every representation with XGBoost, LightGBM and CatBoost.

**Table 2.** RAFE feature-group ablation (5-seed means, each dataset at its own tree; "+X" adds one group to FT Embed+PC, "RAFE-X" removes it from full RAFE; A physchem neighbourhood, B density, C centroid, D Tanimoto).

| Dataset (tree) | FT+PC | +A | +B | +C | +D | RAFE | RAFE-A | RAFE-B | RAFE-C | RAFE-D |
|---|---|---|---|---|---|---|---|---|---|---|
| BBBP (MolFormer-FL) (XGBoost) | 0.785 | 0.790 | 0.781 | 0.790 | 0.780 | 0.779 | 0.783 | 0.779 | 0.789 | 0.769 |
| DILI (Uni-Mol-FL) (XGBoost) | 0.693 | 0.680 | 0.686 | 0.673 | 0.704 | 0.697 | 0.671 | 0.693 | 0.705 | 0.651 |
| Caco-2 (ChemBERTa-MTR) (CatBoost) | 0.833 | 0.826 | 0.832 | 0.834 | 0.835 | 0.833 | 0.830 | 0.835 | 0.836 | 0.826 |
| BACE (MolFormer-WL) (LightGBM) | 0.549 | 0.554 | 0.553 | 0.558 | 0.550 | 0.553 | 0.550 | 0.544 | 0.548 | 0.558 |
| BACE (Uni-Mol-WL) (XGBoost) | 0.440 | 0.455 | 0.418 | 0.444 | 0.433 | 0.419 | 0.421 | 0.433 | 0.420 | 0.445 |
| hERG (MolFormer-FL) (LightGBM) | 0.545 | 0.550 | 0.544 | 0.542 | 0.540 | 0.544 | 0.552 | 0.542 | 0.545 | 0.544 |
| Flavor (MolFormer-WL) (LightGBM) | 0.783 | 0.786 | 0.785 | 0.782 | 0.781 | 0.785 | 0.783 | 0.782 | 0.783 | 0.777 |
| Half-Life (MolFormer) (XGBoost) | 0.595 | 0.571 | 0.591 | 0.581 | 0.593 | 0.559 | 0.580 | 0.565 | 0.572 | 0.551 |

![**Figure 4.** Change relative to FT Embed+PC for each feature-group arm (RAFE-X: RAFE without group X).](../results/revision/figures_response/figC_ablation_heatmap.pdf)

![**Figure 5.** Left: effect of the number of retrieved neighbours (difference to k = 10). Right: effect of the retrieval strategy (difference to CLM-cosine retrieval; error bars: spread over the five random draws).](../results/revision/figures_response/figD_retrieval_robustness.pdf)

**R1.4 Adequacy of the ZINC-250k retrieval database.**
We measured, for every dataset molecule, the largest Morgan-Tanimoto similarity to ZINC-250k (Table 3, Figure 6). The median is 0.40 over all 34,018 molecules, and about 5% have a neighbour at 0.6 or above; BACE is the least covered (median 0.338). ZINC-250k therefore provides background chemical context rather than close analogues. This is compatible with the results above: the per-molecule gain from RAFE is not related to coverage (R2.3), and random neighbours perform like similar ones (R1.3). In other words, the module does not depend on the library covering the task chemistry closely, which makes a generic library workable; whether a task-matched library would do better is untested and is stated as a limitation. [v3]

**Table 3.** Coverage of the dataset molecules by ZINC-250k.

| Dataset | Molecules | Median max Tanimoto to ZINC | Fraction >=0.4 | Fraction >=0.6 | Fraction >=0.8 |
|---|---|---|---|---|---|
| BACE | 1513 | 0.338 | 0.151 | 0.002 | 0.000 |
| BBBP | 1996 | 0.418 | 0.581 | 0.128 | 0.021 |
| Caco-2 | 910 | 0.404 | 0.526 | 0.080 | 0.008 |
| DILI | 475 | 0.425 | 0.587 | 0.135 | 0.013 |
| Flavor | 15025 | 0.431 | 0.624 | 0.070 | 0.006 |
| Half-Life | 654 | 0.407 | 0.526 | 0.104 | 0.015 |
| hERG | 13445 | 0.375 | 0.382 | 0.016 | 0.001 |

![**Figure 6.** Joint UMAP of the dataset molecules and ZINC-250k.](../results/revision/e07_coverage/umap_coverage.pdf)

**R1.5 End-to-end computational cost.**
Table 4 and Figure 7 give the cost of each stage, measured or taken from the logs of the original runs (no cross-method comparison, as discussed). One-off costs are small for the SMILES models (LoRA fine-tuning in seconds to minutes; embedding the 249k-molecule library in about 4.3 minutes) and large for Uni-Mol (hours, dominated by conformer generation). At prediction time RAFE adds about 0.13 s per molecule on a shared GPU, using exact search; tree inference takes under a millisecond. The figures come from different hardware and are indicative. [v3: new subsection "Computational cost"]

**Table 4.** Runtime of the pipeline stages.

| Stage | Cost | Source |
|---|---|---|
| LoRA fine-tuning, one run | median 13 s, IQR 6-43 s, max 540 s | training logs; includes small test runs, so a lower bound per dataset |
| Library embedding, SMILES models (6 variants) | 4.4 min (hERG), 4.2 min (DILI) | original run logs |
| Library embedding, Uni-Mol | herg: 8.4 h, dili: 9.1 h, caco2: 1.7 h (hERG and DILI: 2 variants each; Caco-2 rerun: 1 variant) | original run logs |
| RAFE at inference, one molecule | retrieval 125.8 ms, features 3.8 ms, total 129.6 ms (129.63 ms per molecule) | measured, shared P100, exact search over 249k molecules |
| RAFE at inference, 192 molecules | retrieval 139.8 ms, features 456.2 ms, total 596.0 ms (3.10 ms per molecule) | measured |
| XGBoost inference, one molecule | 0.5 ms | measured |
| Tree stage per cell (20 Optuna trials × 5-fold CV + fit) | XGBoost median 639 s, IQR 398-888 s; LightGBM median 213 s, IQR 33-778 s; CatBoost median 1560 s, IQR 833-4380 s | E1 and E4 cells on a loaded shared server; an upper bound |

![**Figure 7.** Cost of the pipeline stages (blue: one-off; red: per query molecule).](../results/revision/figures_response/figE_runtime.pdf)

---

## Reviewer 2

**R2.1 Lack of controlled comparisons between representations and algorithms** *(primary concern)*.
We thank the reviewer for pressing on this. The main-text tables did report the best row for each representation, and the supplementary tables already contain every representation-by-tree combination, but we agree that this does not constitute a controlled comparison. We have now run PC-only, FT Embed, FT Embed+PC and RAFE through the same code, on the same scaffold splits and test molecules, with the same tuning budget (20 Optuna trials, 5-fold cross-validation on the training split only), the same three tree models and the same five seeds (Table 5, Figure 8). All hyper-parameters are optimised on the training split alone, and the test split is evaluated once per fitted model. [v3: Methods, and a main-text table with the full matrix in the supplement]

The controlled comparison gives a clear and useful picture. RAFE is never significantly different from FT Embed+PC (24 paired comparisons: 0 better, 0 worse), so we found no detectable cost to adding the final stage. RAFE is significantly better than the fine-tuned embedding alone in 5 of 24 comparisons (Caco-2, DILI, hERG) and never worse. Which stage is best depends on the dataset: the fine-tuned embedding helps over descriptors alone on BACE (MolFormer), BBBP and Half-Life, whereas PC-only is the strongest or tied on Caco-2, hERG and Flavor, and RAFE is significantly below PC-only in 4 of 24 comparisons (Flavor, hERG) and significantly above it in 0. We state this plainly, and note that identifying such dataset-specific winners is the purpose of the benchmark. 

**Table 5.** Controlled comparison (mean over seeds; MCC for classification, Spearman for regression; one embedding configuration per dataset).

| Dataset (embedding) | Metric | Tree | PC-only | FT Embed | FT Embed+PC | RAFE |
|---|---|---|---|---|---|---|
| BACE (MolFormer-WL) | MCC | CatBoost | 0.465 | 0.546 | 0.528 | 0.518 |
| BACE (MolFormer-WL) | MCC | LightGBM | 0.485 | 0.539 | 0.549 | 0.553 |
| BACE (MolFormer-WL) | MCC | XGBoost | 0.477 | 0.542 | 0.538 | 0.527 |
| BACE (Uni-Mol-WL) | MCC | CatBoost | 0.465 | 0.390 | 0.440 | 0.425 |
| BACE (Uni-Mol-WL) | MCC | LightGBM | 0.485 | 0.421 | 0.439 | 0.424 |
| BACE (Uni-Mol-WL) | MCC | XGBoost | 0.477 | 0.429 | 0.440 | 0.419 |
| BBBP (MolFormer-FL) | MCC | CatBoost | 0.726 | 0.762 | 0.768 | 0.799 |
| BBBP (MolFormer-FL) | MCC | LightGBM | 0.733 | 0.768 | 0.755 | 0.767 |
| BBBP (MolFormer-FL) | MCC | XGBoost | 0.731 | 0.783 | 0.785 | 0.779 |
| Caco-2 (ChemBERTa-MTR) | Spearman | CatBoost | 0.840 | 0.809 | 0.833 | 0.833 |
| Caco-2 (ChemBERTa-MTR) | Spearman | LightGBM | 0.834 | 0.789 | 0.819 | 0.816 |
| Caco-2 (ChemBERTa-MTR) | Spearman | XGBoost | 0.833 | 0.788 | 0.813 | 0.817 |
| DILI (Uni-Mol-FL) | MCC | CatBoost | 0.738 | 0.595 | 0.683 | 0.669 |
| DILI (Uni-Mol-FL) | MCC | LightGBM | 0.654 | 0.574 | 0.690 | 0.658 |
| DILI (Uni-Mol-FL) | MCC | XGBoost | 0.672 | 0.520 | 0.693 | 0.697 |
| Flavor (MolFormer-WL) | MCC | CatBoost | 0.798 | 0.789 | 0.789 | 0.786 |
| Flavor (MolFormer-WL) | MCC | LightGBM | 0.798 | 0.779 | 0.783 | 0.785 |
| Flavor (MolFormer-WL) | MCC | XGBoost | 0.809 | 0.782 | 0.780 | 0.782 |
| Half-Life (MolFormer) | Spearman | CatBoost | 0.530 | 0.604 | 0.596 | 0.577 |
| Half-Life (MolFormer) | Spearman | LightGBM | 0.477 | 0.550 | 0.558 | 0.512 |
| Half-Life (MolFormer) | Spearman | XGBoost | 0.499 | 0.592 | 0.595 | 0.559 |
| hERG (MolFormer-FL) | MCC | CatBoost | 0.593 | 0.536 | 0.543 | 0.537 |
| hERG (MolFormer-FL) | MCC | LightGBM | 0.577 | 0.532 | 0.545 | 0.544 |
| hERG (MolFormer-FL) | MCC | XGBoost | 0.594 | 0.545 | 0.546 | 0.545 |

![**Figure 8.** The controlled comparison of Table 5 (thick line: mean over trees; markers: individual trees).](../results/revision/figures_response/figF_controlled_matrix.pdf)

**R2.2 Insufficient comparison with conventional fingerprints.**
We agree that these baselines belong in the main benchmark. Table 6 evaluates ECFP4, ECFP6, MACCS, ECFP4+MACCS, RDKit descriptors and the 473 physicochemical descriptors with the same trees and tuning (best tree shown; 3 to 3 seeds on the smaller datasets and 1 on hERG and Flavor because of cost). Fingerprints alone are clearly weaker than descriptors on Caco-2 (0.607 for ECFP4 versus 0.838), DILI and Half-Life, and comparable on BACE, hERG and Flavor. RAFE is clearly better than ECFP4 alone on DILI (+0.26) and Caco-2 (+0.22); against PC-only the intervals include zero on all three datasets compared (BBBP, DILI, Caco-2). [v3]

**Table 6.** Fingerprint and descriptor baselines (best tree per cell; MCC for classification, Spearman for regression).

| Dataset | ECFP4 | ECFP6 | MACCS | ECFP4+MACCS | RDKit descriptors | 473 physchem (PC-only) |
|---|---|---|---|---|---|---|
| BACE | 0.515 | 0.473 | 0.505 | 0.516 | 0.426 | 0.498 |
| BBBP | 0.688 | 0.673 | 0.700 | 0.700 | 0.729 | 0.748 |
| Flavor | 0.799 | 0.797 | 0.804 | 0.815 | 0.800 | 0.806 |
| hERG | 0.584 | 0.560 | 0.539 | 0.587 | 0.593 | 0.601 |
| DILI | 0.546 | 0.486 | 0.612 | 0.607 | 0.731 | 0.732 |
| Caco-2 | 0.607 | 0.517 | 0.800 | 0.791 | 0.842 | 0.838 |
| Half-Life | 0.444 | 0.414 | 0.526 | 0.524 | 0.516 | 0.526 |

**R2.3 The independent contribution of RAFE is not convincingly established** *(complementarity with the language-model representations)*.
We have examined this from three directions. *Where RAFE helps:* the per-molecule gain over FT Embed+PC is unrelated to ZINC coverage (Spearman between -0.02 and +0.14; all intervals include zero; Figure 9, left). *What it adds:* a ridge model recovers 36 to 52% of the variance of the physicochemical-neighbourhood group, 47 to 76% of the density group and 32 to 74% of the centroid group from the embedding and descriptors on held-out molecules (a linear estimate), so these groups are partly complementary and partly redundant; the Tanimoto group is not predictable from them (Figure 9, right). *How much predictions change:* RAFE corrects or breaks between 0 and 4 test molecules on the classification sets. We read this as RAFE being a conservative refinement. To test whether its redundant part can be removed, we are evaluating five variants that use no labels in retrieval or feature construction (groups A and B only, residualised features, additional ZINC properties, compressed features, hybrid retrieval) (Table 7). **[PENDING: RAFE variants, Table 7, from `results/revision/e11/`]**

![**Figure 9.** Left: per-molecule gain of RAFE over FT Embed+PC by ZINC-coverage tercile. Right: held-out R² of each RAFE group predicted from the embedding and descriptors.](../results/revision/e10/e10_summary.pdf)

**Table 7 (placeholder).** Label-free RAFE variants at each dataset's best configuration (5 seeds; variant chosen by training cross-validation).

| Dataset | FT+PC | RAFE | V1 groups A+B | V2 residual | V3 wide | V4 compressed | V5 hybrid | Chosen by CV |
|---|---|---|---|---|---|---|---|---|
| BACE | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| BBBP | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| Flavor | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| hERG | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| DILI | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| Caco-2 | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| Half-Life | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |

**R2.4 Insufficient isolation of the contribution of 3D information.**
We agree and have separated the three components (Table 8, Figure 10). The answer depends on the endpoint. On BACE the ECFP4 part alone scores higher than the 3D vector (0.495 versus 0.394 with XGBoost; the difference is significant for LightGBM and CatBoost). On Caco-2 the concatenation is clearly better than either part (+0.13 to +0.14 over the 3D vector for XGBoost and CatBoost), so the two views are complementary there. On DILI the 3D vector outperforms ECFP4 for CatBoost. We have corrected the wording accordingly: the headline BACE result (MCC 0.623) is the end-to-end LoRA Uni-Mol model with its own head, which does not use ECFP4, so the ECFP4 confound the reviewer identified applies only to the frozen-embedding rows. [v3]

**Table 8.** 3D versus ECFP4 within the Uni-Mol vector (XGBoost, 5 seeds; MCC, and Spearman for Caco-2).

| Dataset | 3D vector (512-d) | ECFP4 (2048-bit) | Both (2560-d) |
|---|---|---|---|
| BACE | 0.394 | 0.495 | 0.429 |
| BBBP | 0.690 | 0.694 | 0.721 |
| Caco-2 | 0.522 | 0.518 | 0.653 |
| DILI | 0.521 | 0.410 | 0.520 |

We also examined the influence of the conformer (relevant to R3.2). Re-embedding BACE with five RDKit ETKDG seeds (the original seed reproduces the stored embeddings exactly) shows that the 3D vector changes noticeably between conformers (median cosine similarity to the original 0.91 to 0.92). The resulting spread in test MCC is 4 to 7 times the tuning noise for the 3D vector alone and 2 to 3 times with ECFP4 added (Table 9, Figure 10), so a single conformer should be reported with this caveat. Pipelines built on SMILES-based models, including the RAFE configurations with MolFormer and ChemBERTa, do not involve conformers at all.

**Table 9.** Conformer-seed sensitivity on BACE (5 conformer seeds × 3 tree seeds).

| Input | Tree | Mean MCC | SD across 5 conformer seeds | SD across 3 tree seeds | Range over conformers |
|---|---|---|---|---|---|
| 3D vector (512-d) | LightGBM | 0.363 | 0.042 | 0.011 | 0.096 |
| 3D vector (512-d) | XGBoost | 0.349 | 0.045 | 0.006 | 0.099 |
| 3D + ECFP4 (2560-d) | LightGBM | 0.428 | 0.024 | 0.011 | 0.063 |
| 3D + ECFP4 (2560-d) | XGBoost | 0.411 | 0.022 | 0.007 | 0.049 |

![**Figure 10.** Left: 3D versus ECFP4 versus both (XGBoost). Right: test MCC for each conformer seed (42 = original), both inputs, XGBoost and LightGBM.](../results/revision/figures_response/figG_3d_vs_ecfp4_conformers.pdf)

**R2.5 Statistical evaluation requires greater rigour** *(paired bootstrap; uncertainty from selecting the best configuration)*.
We agree and have replaced the ±1 SE overlap with paired bootstrap confidence intervals on identical test molecules (differences averaged over seeds within each resample). Every arm is run with five seeds, and the seed controls the Optuna sampler, the cross-validation shuffle and the model initialisation. On the second point, the reviewer is right that the best of many configurations is optimistic when taken from a single run. To make the selection transparent we (i) report seed-averaged values next to every best-run value (DILI as an example: 0.697 ± 0.022 over five seeds with the stored RAFE features versus 0.790 in the table, with FT Embed+PC at 0.693 ± 0.019); (ii) choose configurations by the training-only cross-validation score when quoting a selected result; and (iii) give the paired difference between the best and runner-up configurations (Table 10). The full configuration grid (all embedding models, losses and trees) is complete for XGBoost and LightGBM on five datasets. **[PENDING: CatBoost, hERG and Flavor of the full grid, then Table 10]**

**Table 10 (placeholder).** Seed-averaged best configurations.

| Dataset | Best by training CV (embedding, loss, tree, stage) | Test score (seed mean ± SD) | Best by test score | Test score (seed mean ± SD) | Optimism gap | Runner-up difference (paired 95% CI) |
|---|---|---|---|---|---|---|
| BACE | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| BBBP | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| Flavor | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| hERG | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| DILI | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| Caco-2 | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |
| Half-Life | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** | **[PENDING: ]** |

**R2.6 Limited coherence of the central scientific narrative.**
We agree and have reorganised the paper around three hypotheses: H1, LoRA adaptation is parameter-efficient without loss of accuracy; H2, retrieval features add information beyond the language-model embedding and descriptors; H3, 3D geometry helps structure-dependent endpoints. H1 is supported by the existing results. For H2, the evidence is that RAFE is a safe addition that improves on the embedding alone on several datasets, with a small and dataset-specific advantage over embedding plus descriptors; for H3, see R2.4. The strength of the descriptor baseline is now treated as a result in its own right (R2.1, R2.2). [v3]

**R2.7 Organisation and repetition.**
We merged "Results and Discussion" and restricted the discussion to interpretation, limitations and implications. [v3]

**R2.8 References and the LoRA initialisation.**
We reformatted the references with DOIs throughout. **[PENDING: final DOI check of every reference]** The LoRA description now matches the implementation: ΔW = (α/r)·B·A, with A initialised by Kaiming-uniform and B initialised to zero (the library default); Figure 2 of the paper was redrawn, and the Methods and supplement use the same notation. [v3]

---

## Reviewer 3

**R3.1 Feature-group ablation; random and Morgan-similarity retrieval.** These controls are in Table 2 and Figures 4 and 5 (R1.3), with the interpretation in R2.3. We also discuss the conditions under which RAFE helps (R1.2, R2.3).

**R3.2 3D versus ECFP4; conformer selection.** See Tables 8 and 9 and Figure 10 (R2.4). We did not repeat the conformer analysis for hERG and Flavor because of cost, and we say so in the limitations. **[PENDING: decision whether to add conformer runs for a second dataset]**

**R3.3 Statistical significance, multiple runs and test-set use.** The paired bootstrap and the five-seed protocol are described in R2.1 and R2.5. On model selection: the tree models are tuned by 5-fold cross-validation on the training split only (MCC for classification, negative RMSE for regression), and the test split is evaluated once per fitted model and never used for tuning. LoRA fine-tuning uses the training split, with the validation split for monitoring and checkpoint selection. As a limitation, seed variability was measured for the downstream stage; fine-tuning and library re-embedding were not repeated with independent seeds because of cost, so the intervals describe the downstream stage and are lower bounds on total variability. [v3: Methods and Limitations]

**R3.4 Interpretation of performance claims.** We revised each of the examples. (a) *BBBP:* the RAFE effect over FT Embed+PC depends on the tree (+0.031 with CatBoost, +0.012 with LightGBM, -0.006 with XGBoost; Table 5), and we now describe it as small and tree-dependent. (b) *DILI metrics:* the best configuration depends on the metric: for RAFE, MCC, accuracy and F1 select Uni-Mol FL with XGBoost (MCC 0.790) while AUC selects ChemBERTa-MLM FL with CatBoost (0.907); the rank correlation between MCC and AUC across configurations is 0.65 to 0.79. With 96 test molecules, MCC at a fixed threshold is noisier than AUC, so we state the ranking metric and report AUC alongside. (c) *Half-Life:* Spearman and R² behave differently because R² is computed in hours after inverting the log1p transform, where 57% of the test variance sits in three molecules; for the E4 baselines R² rises from 0.043 to 0.169 on the log scale (Figure 11). The rank correlation between Spearman and R² across the supplementary configurations is 0.40 (0.25 within RAFE). We report both and conclude that the models rank half-lives moderately well (Spearman 0.54 to 0.62 for the best configuration of each stage) but do not predict their magnitude. (d) *BACE and protein-ligand interactions:* we removed the statement that Uni-Mol captures protein-ligand interactions; the data show that a 3D-aware model performs best on BACE, which does not by itself establish interaction modelling. [v3]

![**Figure 11.** Half-Life. Left: Spearman versus R² (hours) over the supplementary configurations. Right: R² in hours versus on the log1p scale for the descriptor and fingerprint baselines.](../results/revision/figures_response/figH_halflife_metrics.pdf)

**R3.5 Novelty relative to EffiChem; inconsistencies.** The novelty statement is under R1.1. We audited the manuscript against the supplement and the code and corrected: the number of RAFE features (60 in the implementation, 18 + 5 + 34 + 3, not 69); the embedding dimension of MolFormer-XL; the LoRA initialisation and formula; the loss-function definitions; and the label of the best BACE configuration (the stored retrieval index is MolFormer FL, whereas the text said WL). A table of all corrections is in the supplement. **[PENDING: final consistency check of the revised tables]** [v3]

**R3.6 Additional related work.** We added a discussion of Fang et al. (Nature Machine Intelligence, 2022), Li et al. (Nature Communications, 2023), Su et al. (ChemRxiv, 2026) and Li et al. (Journal of Chemical Theory and Computation, 2026) to the Introduction and Discussion, in the context of geometric, knowledge-based and multimodal molecular representations. **[PENDING: verification of the four DOIs against the sources]** [v3]

---

## Limitations now stated in the manuscript
- Uni-Mol results depend on the conformer; only BACE was examined.
- Seed variability covers the downstream stage; fine-tuning and library embedding were not repeated with independent seeds.
- E2E LoRA, Chemprop and GCN are single runs.
- ZINC-250k covers the task chemistry only moderately; other libraries were not tested.
- The new experiments ran XGBoost on GPU, so absolute values differ slightly from the original CPU runs; comparisons are made within experiments.
- A training molecule's embedding comes from an encoder fine-tuned on that molecule, which can favour overfitting in frozen-embedding models.

We thank the reviewers again; the revised analyses have made the paper clearer and its claims better supported.
