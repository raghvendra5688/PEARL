# Change log: academic-humanizer pass on PEARL_paper_v3.tex

Tool: AIScientists-Dev/academic-humanizer v0.3.3 (rules applied by hand, audit script `scripts/revision/humanizer_audit.py`). Template: `PEARL_paper_MIR.tex`. Venue register: Journal of Cheminformatics (expository, results-forward). Pre-humanizer draft kept as `PEARL_paper_v3_prehumanized.tex`.

## Patterns removed
- **Over-claiming / significance hype:** "achieves the best overall result" (BBBP, DILI, hERG), "statistically confirmed win", "clearest confirmed win for the CLM ladder", "orthogonal signal", "establishes the ceiling" rewritten as single-run values with five-seed means and paired intervals beside them.
- **Em-dashes and spaced dashes** used as punctuation (Methods, problem formulation, Discussion, author roles): recast with commas, colons, parentheses or separate sentences. Remaining audit hits are minus signs in equations and numeric ranges.
- **AI vocabulary and filler:** "additionally" (6), "genuinely", "candid", "considerably", "perhaps most avoidably" reduced or removed; "to our knowledge, no prior work" replaced by a limited statement.
- **Novelty padding:** "novel RAFE pipeline" removed from the contributions and abstract.
- **Vague hedges and magnitudes:** replaced by numbers or intervals from the result files (macros in `manuscript/tables/numbers.tex`).
- **Long sentences:** the abstract Methods sentence, the Methods dataset paragraph, the problem formulation and the evaluation protocol were split.

## Claims softened or given evidence pointers
- "RAFE is the point-estimate winner on BBBP/DILI" now states the five-seed means (0.779 and 0.697) and the paired comparisons.
- "PC-only is statistically competitive" now cites the paired result (RAFE significantly below PC-only on hERG, all trees, and on Flavor, XGBoost).
- "3D geometry matters for binding affinity" now says the data are consistent with it, notes the ECFP4 block in the Uni-Mol vector and the conformer sensitivity, and withdraws the protein-ligand interaction statement.
- "PC features give minimal improvement because LoRA internalizes properties" and "regression gains are smaller" removed (not supported by the new grid).
- Loss preference statements ("weighted loss for imbalanced datasets") replaced by the observed best losses (focal on BBBP and DILI, weighted on BACE).
- The k and metric "not swept" statements removed; both were swept (E2, E3).

## Preserved (Layer 3)
Calibrated hedging ("suggests", "is consistent with"), "we", passive voice for procedures, equations, symbols, method names and all citation keys.

## Voice notes
Kept the author's habits from the template: bold run-in lead-ins for discussion points, explicit "we report this" statements, semicolon lists. Section order follows the template.

## Integrity check (`humanizer_audit.py --integrity`)
Numbers, `\cite` keys, `\ref`/`\label` and equations were compared between the pre-humanizer draft and the final file. Two number tokens differ and both are parts of names, not values: the "2" of "Caco-2" and one "250" of "ZINC-250k" in sentences that were shortened. No value, equation or citation was altered.

## Open items (not humanizer issues)
- Red `\pending{}` markers: hERG and Flavor grid, CatBoost columns of the full tables, Flavor row of the variants table, BBBP neighbor-count sweep, AI-use statement against the journal policy, BACE configuration label check.
- `RAFE_Framework_v2.pdf` and `PEARL_Workflow_v2.pdf` may still show 69 features; the figures must be redrawn. `PEARL_HM_Workflow_v2.pdf`, referenced in the template, is not in `Figures_v2/`.
- LaTeX not compiled on the server (no TeX installation); label, brace and environment checks pass.

## Numerical verification pass (2026-11-20)

Script: `/tmp/.../verify.py` logic folded into the checks below; figures and tables regenerated from the result files.

**Checked automatically (58 checks, all passing):** five-seed means quoted in prose against `e15_analysis/best_configs.csv`; PC-only five-seed values against the E1 cells; every paired difference and interval against `e06/paired_e1.csv` and `e15_analysis/full_tables/ci_*.csv`; the matched-pair ladder differences and Wilcoxon p-values against `e09_selection_audit/matched_pair_differences.csv`; the counts of significant comparisons (24 / 5 / 4 / 0); the mean FT+PC minus FT per dataset; all seven test-set sizes against `n_test` in the result cells.

**Checked by cross-reference:** every value quoted in a per-dataset paragraph now appears either in that paragraph's own table or among the verified paired quantities (7 of 7 paragraphs clean).

**Corrections made during this pass**
1. Half-Life: "R\textsuperscript{2} $< 0.11$ for every method" was false, since the GCN row of the same table reads 0.114. Changed to "$\le 0.114$".
2. BACE ROC/PR paragraph: "RAFE is the weakest of the four (AUC 0.858)" holds only for the plotted best-by-MCC configurations. RAFE's best-AUC configuration on BACE reaches 0.875 (Supplementary Table S4). The sentence now says which configurations are shown and gives 0.875.
3. External baselines: "RAFE exceeds GROVER-Large and Mole-BERT (both 0.940, tied)" was self-contradictory. Changed to "matches ... (all three at 0.940)".
4. Uni-Mol parameter efficiency: the "$<5\%$ trainable" claim had no table behind it. It is now derived in the text: the adapters add $2048r$ parameters per layer, at most 1.5M of about 47M at $r=32$ (verified: 3.1%).

**Rounding:** all 47 four-decimal values (the SE columns) were rounded to three decimals, so the whole manuscript now reports three decimals. The integrity check compares numbers at three decimals for this reason.

**Integrity check after the pass:** the only differences from the pre-humanizer draft are the four corrections above (`0.110` replaced by `0.114`, `0.875` added) and two name fragments ("Caco-2", "ZINC-250k") in rewritten sentences. No citation, equation or reference changed.

**Not verifiable from the shipped files:** the AUPR values quoted in the ROC/PR paragraph (0.862-0.872 on hERG, 0.940 on BBBP, 0.911 and 0.847 on DILI) come from the curve figures of the previous version, and no per-curve data file is kept in the repository. They should be re-read from the figure source before submission.

## Pass 3 (2026-11-21): complete-grid results folded in

Applied after every experiment finished, so the manuscript could state results rather than placeholders.

**Content changes (all driven by the completed grid, not by editing)**
- All nine result-dependent `\pending{}` markers replaced with real numbers: the abstract ranking, the five-seed values for Flavor and hERG, the complete-grid wording, the ranking paragraph, the full same-tree tables, the neighbour-count sweep, the variants table and the limitations. Only the AI-use policy wording remains, which is for the authors.
- The ranking is now over all seven datasets and all three tree models. RAFE's mean rank is \mrankRAFE{} of seven methods and it is top three on \ntop{} of them. The earlier 2.8 / five-of-five figure covered five datasets and two trees only; it is not comparable and has been removed rather than carried forward.
- Added the measurement caveat the ranking needs: our four stages are five-seed means, the external references are single runs, so the comparison is conservative with respect to RAFE.
- Added the pooled complete-grid comparison, which is the better-powered version of the controlled matrix: RAFE improves on the embedding it refines in 12 of 21 comparisons and loses none.

**Corrections**
- Removed a "correction" that was itself wrong. Table S16 claimed the stored BACE retrieval index is MolFormer FL and that the text should have said so. The stored files use `Molformer_Finetuned_WL` for both the embedding column and the ZINC index, so the original weighted-loss label was right. The row is gone and R3.5 now says the label was checked and needed no change.
- `sn-bibliography.bib`: 45 of 54 entries now carry a DOI, up from 6 (R2.8). Looked up against Crossref with a title-similarity threshold and a trusted-prefix filter so that no aggregator or mismatched DOI is written; a sample was verified to resolve to the right paper.

**Humanizer rules applied to the new prose**
- Split the three sentences this pass introduced that ran past 40 words; the count is back to the pre-pass baseline.
- Kept the register neutral and the claims tied to numbers. No em-dashes were introduced.

**Audit status.** The remaining flags are all cases Layer 3 says to preserve: the "em-dashes" at lines 242-246 are minus signs inside equations, and the "over-claiming verbs" are calibrated hedges ("not confirmed", "our data do not establish that mechanism") that are correct as written.

**Integrity check.** Against the pre-humanizer baseline the only differences are the intentional ones: `0.110` to `0.114` and `0.875` added (the earlier numeric-verification fixes), `2.800` removed (the superseded five-dataset mean rank), `24`, `12` and `10` added by the new pooled-comparison sentences, and two new section cross-references. No citation key or equation changed.

**Structural check.** Both files: braces balanced, environments matched, every `\ref` resolved against labels including those inside generated tables, every `\input` and `\includegraphics` target present.
