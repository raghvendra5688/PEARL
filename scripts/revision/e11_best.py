"""E11 summary judged on test performance: per dataset (own tree) the variant with the highest 5-seed test mean among RAFE and V1-V5 ('best variant'),
with its paired difference to FT Embed+PC and to RAFE read from comparisons.csv (seed-pooled paired bootstrap). Output: results/revision/e11_analysis/best_variant.csv"""
import pandas as pd
from pathlib import Path
O = Path("results/revision/e11_analysis"); T = pd.read_csv(O / "variants_table.csv"); C = pd.read_csv(O / "comparisons.csv"); V = ["full_RAFE", "V1_AB", "V2_residual", "V3_wide", "V4_compressed", "V5_hybrid"]
rows = []
for _, r in T.iterrows():
    b = max(V, key=lambda a: r[a]); row = dict(Dataset=r.Dataset, Tree=r.Tree, best_variant=b, best_test=r[b], FT_PC=r["base_FT+PC"], RAFE=r["full_RAFE"], PC_only=r["PC-only"])
    for B in ("base_FT+PC", "full_RAFE"):
        x = C[(C.dataset == r.Dataset) & (C.tree == r.Tree) & (C.A == b) & (C.B == B)]
        if len(x): row[f"diff_vs_{B}"], row[f"lo_{B}"], row[f"hi_{B}"] = x.iloc[0]["diff"], x.iloc[0].ci_lo, x.iloc[0].ci_hi
    rows.append(row)
R = pd.DataFrame(rows); R.round(4).to_csv(O / "best_variant.csv", index=False); print(R.round(3).to_string(index=False))
