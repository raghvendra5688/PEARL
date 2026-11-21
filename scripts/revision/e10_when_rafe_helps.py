"""E10: when and why does RAFE help? (R1.2, R2.3, R3.1)
A. per-molecule gain of RAFE over FT+PC (same tree, same seeds, identical test molecules) vs ZINC coverage (max Tanimoto to ZINC-250k, E7).
B. complementarity: how much of each RAFE feature group is linearly predictable from [FT embedding | PC descriptors] (ridge, train -> test R^2).
C. prediction-level complementarity: molecules corrected vs broken by RAFE, error correlation between the two models.
Inputs: results/revision/e1/<cfg>/{base_FT+PC,full_RAFE}_<clf>_s<seed>.npz, e07 per_molecule_coverage.csv. Output: results/revision/e10/*.csv + e10_summary.pdf"""
import sys, glob, numpy as np, pandas as pd; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import *
from rafe_lib import CFG, get_base, build_rafe, group_of
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
OUT = RES / "e10"; OUT.mkdir(exist_ok=True)
cov = pd.read_csv(RES / "e07_coverage" / "per_molecule_coverage.csv"); cov = cov[cov.split == "test"]
CFGS = [(n, CFG[n]["ds"], CFG[n]["clf"]) for n in ("bbbp_mf", "dili_um", "caco2_mtr", "bace_um", "bace_mf", "flavor_mf", "herg_mf") if n in CFG]
rng = np.random.default_rng(0)
def seeds_avail(name, clf):
    return [s for s in range(42, 47) if all((RES / "e1" / name / f"{a}_{clf}_s{s}.npz").exists() for a in ("base_FT+PC", "full_RAFE"))]
def gain_vec(task, name, clf, seeds, y):
    g, ca, cb, errs = [], [], [], []
    for s in seeds:
        a = np.load(RES / "e1" / name / f"full_RAFE_{clf}_s{s}.npz"); b = np.load(RES / "e1" / name / f"base_FT+PC_{clf}_s{s}.npz")
        if task == "regression":
            ea, eb = np.abs(a["y"] - a["pred"]) / np.std(y), np.abs(b["y"] - b["pred"]) / np.std(y); g.append(eb - ea); errs.append((ea, eb))
        else:
            ok_a, ok_b = (a["pred"] == a["y"]).astype(float), (b["pred"] == b["y"]).astype(float); g.append(ok_a - ok_b); ca.append(ok_a); cb.append(ok_b)
    return np.mean(g, 0), (np.mean(ca, 0) if ca else None), (np.mean(cb, 0) if cb else None), errs
rows_bin, rows_sum, rows_comp, rows_cc = [], [], [], []
for name, ds, clf in CFGS:
    seeds = seeds_avail(name, clf)
    if len(seeds) < 2: print("skip", name, seeds); continue
    base = get_base(name); task = DS[ds]["task"]; sm = base["test"]["smiles"]; y = base["test"]["y"]
    c = cov[cov.dataset == ds].drop_duplicates("smiles").set_index("smiles").reindex(sm)["max_tanimoto"].values
    ok = ~np.isnan(c); g, ca, cb, errs = gain_vec(task, name, clf, seeds, np.asarray(y, float))
    rho, p = spearmanr(c[ok], g[ok]); bs = []
    for _ in range(2000):
        i = rng.integers(0, ok.sum(), ok.sum()); bs.append(spearmanr(c[ok][i], g[ok][i]).correlation)
    rows_sum.append(dict(cfg=name, ds=ds, clf=clf, n_seeds=len(seeds), n_test=int(ok.sum()), mean_gain=float(g[ok].mean()), spearman_gain_vs_coverage=rho, ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5)))
    q = np.quantile(c[ok], [1/3, 2/3]); bins = np.digitize(c, q)
    for b_, lab in enumerate(("low coverage", "mid coverage", "high coverage")):
        m = ok & (bins == b_); gg = g[m]; bm = [gg[rng.integers(0, len(gg), len(gg))].mean() for _ in range(2000)]
        rows_bin.append(dict(cfg=name, ds=ds, bin=lab, n=int(m.sum()), maxT_range=f"{c[m].min():.2f}-{c[m].max():.2f}", mean_gain=float(gg.mean()), ci_lo=np.percentile(bm, 2.5), ci_hi=np.percentile(bm, 97.5)))
    if task != "regression":
        rows_cc.append(dict(cfg=name, ds=ds, corrected_by_RAFE=int(((ca > .5) & (cb < .5)).sum()), broken_by_RAFE=int(((ca < .5) & (cb > .5)).sum()), both_wrong=int(((ca < .5) & (cb < .5)).sum()), both_right=int(((ca > .5) & (cb > .5)).sum())))
    else:
        ea = np.mean([e[0] for e in errs], 0); eb = np.mean([e[1] for e in errs], 0)
        rows_cc.append(dict(cfg=name, ds=ds, error_corr_RAFE_vs_base=float(np.corrcoef(ea, eb)[0, 1]), corrected=int((ea < eb - .05).sum()), broken=int((ea > eb + .05).sum())))
    # B. complementarity of RAFE groups with [emb | PC]
    from sklearn.linear_model import RidgeCV; from sklearn.preprocessing import StandardScaler; from sklearn.metrics import r2_score
    R = build_rafe(name, "clm", 0, "orig", 10); Xtr = np.hstack([base["train"]["emb"], base["train"]["pc"]]); Xte = np.hstack([base["test"]["emb"], base["test"]["pc"]])
    sc = StandardScaler().fit(Xtr); Xtr, Xte = np.nan_to_num(sc.transform(Xtr)), np.nan_to_num(sc.transform(Xte))
    r2 = {}
    for col in R["train"].columns:
        yt, ye = R["train"][col].values.astype(float), R["test"][col].values.astype(float)
        if np.std(yt) < 1e-9: continue
        m = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(Xtr, yt); r2[col] = r2_score(ye, m.predict(Xte))
    for grp in "ABCD":
        v = [r for col, r in r2.items() if group_of(col) == grp]
        if v: rows_comp.append(dict(cfg=name, ds=ds, group=grp, n_features=len(v), mean_R2_from_emb_PC=float(np.mean(v)), min_R2=float(np.min(v)), max_R2=float(np.max(v))))
    print("done", name, flush=True)
pd.DataFrame(rows_sum).round(4).to_csv(OUT / "gain_vs_coverage_summary.csv", index=False)
pd.DataFrame(rows_bin).round(4).to_csv(OUT / "gain_by_coverage_bin.csv", index=False)
pd.DataFrame(rows_comp).round(4).to_csv(OUT / "rafe_complementarity_R2.csv", index=False)
pd.DataFrame(rows_cc).round(4).to_csv(OUT / "prediction_level_complementarity.csv", index=False)
# figure (vector PDF)
b = pd.DataFrame(rows_bin); c_ = pd.DataFrame(rows_comp); cfgs = list(dict.fromkeys(b.cfg))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
w = .8 / max(len(cfgs), 1)
for k, n in enumerate(cfgs):
    s_ = b[b.cfg == n]; x = np.arange(3) + k * w
    ax[0].bar(x, s_.mean_gain, w * .9, yerr=[s_.mean_gain - s_.ci_lo, s_.ci_hi - s_.mean_gain], capsize=2, label=n)
ax[0].axhline(0, color="k", lw=.6); ax[0].set_xticks(np.arange(3) + .4 - w / 2); ax[0].set_xticklabels(["low", "mid", "high"]); ax[0].set_xlabel("ZINC coverage of test molecule (max Tanimoto tercile)")
ax[0].set_ylabel("per-molecule gain of RAFE over FT+PC\n(accuracy for classification; error/SD for regression)"); ax[0].legend(fontsize=7); ax[0].set_title("A. Does coverage predict where RAFE helps?", fontsize=9)
for k, n in enumerate(cfgs):
    s_ = c_[c_.cfg == n]; ax[1].bar(np.arange(len(s_)) + k * w, s_.mean_R2_from_emb_PC, w * .9, label=n)
ax[1].set_xticks(np.arange(4) + .4 - w / 2); ax[1].set_xticklabels(["A physchem-nbhd", "B density", "C centroid", "D Tanimoto"], fontsize=8)
ax[1].set_ylabel("held-out R$^2$ of RAFE features from [emb | PC] (ridge)"); ax[1].set_title("B. How much of RAFE is already in embedding + descriptors?", fontsize=9)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "e10_summary.pdf")
print(pd.DataFrame(rows_sum).round(3).to_string()); print(pd.DataFrame(rows_comp).round(3).to_string())
