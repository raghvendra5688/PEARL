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
