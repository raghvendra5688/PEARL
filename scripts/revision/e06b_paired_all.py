"""E6b-analysis (not the dropped E6b reseeding): seed-pooled paired bootstrap for E2, E3, E4-vs-RAFE, E5.
For each comparison A vs B on identical test molecules: per seed, predictions share the test set; each bootstrap draw resamples
molecules once and averages the metric difference over seeds. Output: results/revision/e06/paired_{e2,e3,e4,e5}.csv"""
import sys, glob, numpy as np, pandas as pd; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import *
from rafe_lib import CFG
OUT = RES / "e06"; OUT.mkdir(exist_ok=True); B = 3000
def metric(task):
    from sklearn.metrics import matthews_corrcoef; from scipy.stats import spearmanr
    return (lambda y, p: spearmanr(y, p).correlation) if task == "regression" else (lambda y, p: matthews_corrcoef(y, p))
def ld(p):
    z = np.load(str(p) + ".npz"); return z["y"], z["pred"]
def cmp(ds, pairs, seed=0):
    f = metric(DS[ds]["task"]); data = [(ld(a), ld(b)) for a, b in pairs]
    n = len(data[0][0][0]); rng = np.random.default_rng(seed); reg = DS[ds]["task"] == "regression"
    obs = np.mean([f(ya, a) - f(yb, b) for (ya, a), (yb, b) in data]); d = []
    for _ in range(B):
        i = rng.integers(0, n, n); y = data[0][0][0][i]
        if not reg and len(np.unique(y)) < 2: continue
        try: d.append(np.mean([f(ya[i], a[i]) - f(yb[i], b[i]) for (ya, a), (yb, b) in data]))
        except Exception: pass
    d = np.array(d); return obs, *np.percentile(d, [2.5, 97.5]), len(data)
def seeds_pairs(dirA, armA, dirB, armB, clf):
    out = []
    for sd in range(42, 47):
        a = dirA / f"{armA}_{clf}_s{sd}"; b = dirB / f"{armB}_{clf}_s{sd}"
        if a.with_suffix(".npz").exists() and b.with_suffix(".npz").exists(): out.append((a, b))
    return out
def run(rows, exp, ds, name, clf, A, B_, dA, dB, armA, armB):
    pr = seeds_pairs(dA, armA, dB, armB, clf)
    if pr:
        o, lo, hi, n = cmp(ds, pr); rows.append(dict(exp=exp, cfg=name, clf=clf, metric=primary(DS[ds]["task"]), A=A, B=B_, n_seeds=n, diff=o, ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0)))
if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2", "e3", "e4", "e5"]
    for exp in which:
        rows = []
        if exp == "e1":   # controlled representation matrix + ablation arms, per tree, pooled over seeds
            for name in ("bbbp_mf", "dili_um", "caco2_mtr", "bace_um", "bace_mf", "herg_mf", "flavor_mf", "half_life_mf"):
                if name not in CFG: continue
                ds = CFG[name]["ds"]
                for clf in ("XGBoost", "LightGBM", "CatBoost"):
                    d = RES / "e1" / name
                    for A, Bn in (("base_FT+PC", "PC_only"), ("base_FT+PC", "FT_only"), ("full_RAFE", "base_FT+PC"), ("full_RAFE", "PC_only"), ("full_RAFE", "FT_only"),
                                  ("+A", "base_FT+PC"), ("+B", "base_FT+PC"), ("+C", "base_FT+PC"), ("+D", "base_FT+PC"),
                                  ("full-A", "full_RAFE"), ("full-B", "full_RAFE"), ("full-C", "full_RAFE"), ("full-D", "full_RAFE")):
                        run(rows, exp, ds, name, clf, A, Bn, d, d, A, Bn)
        if exp in ("e2", "e3"):
            for name in ("bbbp_mf", "dili_um", "caco2_mtr", "bace_um", "bace_mf", "herg_mf", "flavor_mf"):
                ds = CFG[name]["ds"]; clf = CFG[name]["clf"]; d = RES / exp / name
                if exp == "e2":
                    run(rows, exp, ds, name, clf, "clm", "morgan", d, d, "clm", "morgan")
                    pr = [(d / f"clm_{clf}_s{sd}", d / f"random_r{r}_{clf}_s{sd}") for r in range(5) for sd in range(42, 47)
                          if (d / f"random_r{r}_{clf}_s{sd}.npz").exists() and (d / f"clm_{clf}_s{sd}.npz").exists()]
                    if pr:   # pooled over retrieval draws x model seeds (clm and random share the model seed)
                        o, lo, hi, n = cmp(ds, pr); rows.append(dict(exp=exp, cfg=name, clf=clf, metric=primary(DS[ds]["task"]), A="clm", B="random(5 draws x seeds)", n_seeds=n, diff=o, ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0)))
                else:
                    for k in (1, 5, 20, 50, 100): run(rows, exp, ds, name, clf, f"k{k}", "k10", d, d, f"k{k}", "k10")
        if exp == "e4":   # PC-only (pc473) and ECFP4 vs FT+PC base and vs full RAFE (same clf, same seeds)
            for name in ("bbbp_mf", "dili_um", "caco2_mtr", "bace_um", "bace_mf", "herg_mf", "flavor_mf"):
                ds = CFG[name]["ds"]; clf = CFG[name]["clf"]; d1 = RES / "e1" / name; d4 = RES / "e04_baselines" / ds
                run(rows, exp, ds, name, clf, "full_RAFE", "pc473_only", d1, d4, "full_RAFE", "pc473")
                run(rows, exp, ds, name, clf, "base_FT+PC", "pc473_only", d1, d4, "base_FT+PC", "pc473")
                run(rows, exp, ds, name, clf, "full_RAFE", "ecfp4", d1, d4, "full_RAFE", "ecfp4")
        if exp == "e5":
            for name in ("bace_um", "bbbp_um", "caco2_um", "dili_um"):
                ds = {"bace_um": "bace", "bbbp_um": "bbbp", "caco2_um": "caco2", "dili_um": "dili"}[name]; d = RES / "e05" / name
                for clf in ("XGBoost", "LightGBM", "CatBoost"):
                    for A, Bn in (("concat2560", "cls512"), ("concat2560", "ecfp2048"), ("cls512", "ecfp2048")):
                        run(rows, exp, ds, name, clf, A, Bn, d, d, A, Bn)
        df = pd.DataFrame(rows).round(4); df.to_csv(OUT / f"paired_{exp}.csv", index=False); print(exp); print(df.to_string())
