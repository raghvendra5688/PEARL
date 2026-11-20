**Caco-2**: mean ± SD test Spearman over five seeds (42 to 46). Bold: best of the three stages within the row and tree.

| Embedding, loss | XGBoost: FT Embed | XGBoost: FT Embed+PC | XGBoost: RAFE | LightGBM: FT Embed | LightGBM: FT Embed+PC | LightGBM: RAFE |
|---|---|---|---|---|---|---|
| ChemBERTa-MLM Huber | 0.683 ± 0.010 | **0.750 ± 0.009** | 0.734 ± 0.005 | 0.671 ± 0.009 | **0.738 ± 0.012** | 0.711 ± 0.013 |
| ChemBERTa-MTR Huber | 0.788 ± 0.005 | 0.813 ± 0.007 | **0.817 ± 0.007** | 0.789 ± 0.003 | **0.819 ± 0.006** | 0.816 ± 0.005 |
| MolFormer Huber | 0.592 ± 0.028 | 0.712 ± 0.013 | **0.717 ± 0.016** | 0.655 ± 0.021 | 0.743 ± 0.014 | **0.748 ± 0.005** |
| Uni-Mol Huber | 0.653 ± 0.013 | 0.769 ± 0.009 | **0.774 ± 0.005** | 0.586 ± 0.017 | 0.758 ± 0.003 | **0.761 ± 0.011** |
| *PC-only (473 descriptors)* | 0.833 ± 0.004 |  |  | 0.834 ± 0.006 |  |  |
| *Mean difference over rows* | RAFE−FT: +0.082 | RAFE−FT+PC: -0.000 | 4 rows | RAFE−FT: +0.084 | RAFE−FT+PC: -0.005 | 4 rows |

Pending trees: CatBoost.
