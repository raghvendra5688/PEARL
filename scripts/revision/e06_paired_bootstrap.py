"""E6: paired bootstrap (10k) of metric differences on identical test molecules + multi-seed mean/SD.
usage: e06_paired_bootstrap.py  -> results/revision/e06/paired_*.csv  (reads preds.npz saved by run_cell)"""
import sys, json, glob, numpy as np, pandas as pd; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import *
OUT = RES / "e06"; OUT.mkdir(exist_ok=True)
def metric_fn(task):
    from sklearn.metrics import matthews_corrcoef
    from scipy.stats import spearmanr
    return (lambda y, p, pr: spearmanr(y, p).correlation) if task == "regression" else (lambda y, p, pr: matthews_corrcoef(y, p))
def load(prefix):
    z = np.load(prefix + ".npz"); return z["y"], z["pred"]
def paired(ds, pa, pb, B=10000, seed=0):
    f = metric_fn(DS[ds]["task"]); ya, a = load(pa); yb, b = load(pb); assert (ya == yb).all()
    rng = np.random.default_rng(seed); n = len(ya); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(ya[i])) < 2 and DS[ds]["task"] != "regression": continue
        try: d.append(f(ya[i], a[i], None) - f(ya[i], b[i], None))
        except Exception: pass
    d = np.array(d); obs = f(ya, a, None) - f(ya, b, None)
    return obs, np.percentile(d, 2.5), np.percentile(d, 97.5), float((d <= 0).mean() if obs > 0 else (d >= 0).mean())
def seed_table(exp, name, arm, clf):
    rows = [json.load(open(p)) for p in glob.glob(str(RES / exp / name / f"{arm}_{clf}_s*.json"))]
    m = primary(DS[rows[0]["dataset"]["task"] if False else rows[0]["dataset"]] ["task"]) if rows else None
    return rows, m
if __name__ == "__main__":
    out = []
    # E1: RAFE and each group vs FT+PC base, per tree seed-paired
    from rafe_lib import CFG
    for name in ("bbbp_mf", "dili_um", "caco2_mtr"):
        ds = CFG[name]["ds"]; clf = CFG[name]["clf"]; pm = primary(DS[ds]["task"])
        for arm in ["+A", "+B", "+C", "+D", "full_RAFE", "full-A", "full-B", "full-C", "full-D"]:
            diffs = []
            for sd in range(42, 47):
                pa = RES / "e1" / name / f"{arm}_{clf}_s{sd}"; pb = RES / "e1" / name / f"base_FT+PC_{clf}_s{sd}"
                if pa.with_suffix(".npz").exists() and pb.with_suffix(".npz").exists():
                    o, lo, hi, p = paired(ds, str(pa), str(pb), B=5000); diffs.append((sd, o, lo, hi, p))
            if diffs:
                od = np.array([d[1] for d in diffs])
                out.append(dict(exp="e1", cfg=name, A=arm, B="base_FT+PC", clf=clf, metric=pm, n_seeds=len(diffs), mean_diff=od.mean(), sd_diff=od.std(ddof=1) if len(od) > 1 else np.nan,
                                seed42_diff=diffs[0][1], seed42_ci_lo=diffs[0][2], seed42_ci_hi=diffs[0][3], seed42_p=diffs[0][4]))
    pd.DataFrame(out).round(4).to_csv(OUT / "e1_paired_vs_base.csv", index=False); print(pd.DataFrame(out).round(3).to_string())
