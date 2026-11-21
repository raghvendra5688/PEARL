"""Paired bootstrap CIs for the bottom row of the full same-tree tables. For one dataset and tree: the difference RAFE minus FT Embed (and minus FT Embed+PC)
averaged over all embedding x loss rows and the five seeds; molecules are resampled once per bootstrap draw (e06b_paired_all.cmp). Usage: e15_full_tables_ci.py <ds>"""
import sys, glob, json, pandas as pd; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import e06b_paired_all as pa
ds = sys.argv[1]; R = Path("results/revision"); rows = []
embs = sorted({p.split("/")[-1].split("@")[1] for p in glob.glob(str(R / f"e15/{ds}@*"))})
for clf in ("XGBoost", "LightGBM", "CatBoost"):
    for B, lab in (("FT_only", "FT Embed"), ("base_FT+PC", "FT Embed+PC")):
        pr = [(R / "e15" / f"{ds}@{e}" / f"full_RAFE_{clf}_s{s}", R / "e15" / f"{ds}@{e}" / f"{B}_{clf}_s{s}") for e in embs for s in range(42, 47)]
        pr = [p for p in pr if p[0].with_suffix(".npz").exists() and p[1].with_suffix(".npz").exists()]
        if len(pr) < 5 * len(embs): continue
        o, lo, hi, n = pa.cmp(ds, pr); rows.append(dict(dataset=ds, tree=clf, versus=lab, diff=o, ci_lo=lo, ci_hi=hi, n_pairs=n))
        print(ds, clf, lab, round(o, 4), round(lo, 4), round(hi, 4), flush=True)
pd.DataFrame(rows).to_csv(R / f"e15_analysis/full_tables/ci_{ds}.csv", index=False)
