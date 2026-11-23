"""E15/E14 analysis. Full configuration grid (embedding model x loss x tree x stage) with 5 seeds, for datasets whose grid is complete.
Outputs (results/revision/e15_analysis/): grid_means.csv, best_configs.csv (CV-selected vs test-oracle, optimism gap, runner-up paired CI), ranking.csv, e15_grid_heatmap.pdf
Trees: XGBoost, LightGBM (CatBoost cells for the small datasets are still on the cluster; the analysis states which trees it covers)."""
import sys, json, glob, re, numpy as np, pandas as pd; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import e06b_paired_all as pa
from rev_common import RES
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = RES / "e15_analysis"; OUT.mkdir(exist_ok=True); R = str(RES) + "/"
STAGES = {"FT_only": "FT Embed", "base_FT+PC": "FT Embed+PC", "full_RAFE": "RAFE"}; TREES = ["XGBoost", "LightGBM", "CatBoost"]
PC_CFG = {"bace": "bace_mf", "bbbp": "bbbp_mf", "dili": "dili_um", "caco2": "caco2_mtr", "half_life": "half_life_mf", "herg": "herg_mf", "flavor": "flavor_mf"}
NAME = {"bace": "BACE", "bbbp": "BBBP", "herg": "hERG", "dili": "DILI", "caco2": "Caco-2", "half_life": "Half-Life", "flavor": "Flavor"}
def pretty(e): m = re.match(r"(chemberta_77m_mlm|chemberta_77m_mtr|molformer_finetuned|unimol)_(\w+)", e); return {"chemberta_77m_mlm": "ChemBERTa-MLM", "chemberta_77m_mtr": "ChemBERTa-MTR", "molformer_finetuned": "MolFormer", "unimol": "Uni-Mol"}[m.group(1)] + " " + m.group(2).upper().replace("HUBER", "Huber")
rows = []
for f in glob.glob(R + "e15/*/*.json"):
    j = json.load(open(f)); dsemb = Path(f).parent.name; ds, emb = dsemb.split("@")
    rows.append(dict(ds=ds, emb=emb, arm=j["arm"], clf=j["clf"], seed=j["seed"], v=j.get("MCC", j.get("Spearman")), cv=j.get("cv_best", np.nan)))
d = pd.DataFrame(rows); d = d[d.clf.isin(TREES)]
g = d.groupby(["ds", "emb", "arm", "clf"]).agg(test=("v", "mean"), sd=("v", "std"), cv=("cv", "mean"), n=("v", "count")).reset_index()
exp_cfg = {ds: len(set(x.emb)) for ds, x in g.groupby("ds")}
complete = [ds for ds, x in g.groupby("ds") if (x.n == 5).all() and len(x) == exp_cfg[ds] * 3 * len(TREES)]
g = g[g.ds.isin(complete)]; g["config"] = g.emb.map(pretty) + " / " + g.clf; g.round(4).to_csv(OUT / "grid_means.csv", index=False)
print("complete datasets:", complete)
# PC-only reference from E1 (independent of the embedding; same trees)
def pc_only(ds):
    out = []
    for clf in TREES:
        v = [(json.load(open(f)).get("MCC", json.load(open(f)).get("Spearman")), json.load(open(f)).get("cv_best", np.nan)) for f in glob.glob(f"{R}e1/{PC_CFG[ds]}/PC_only_{clf}_s4?.json")]
        if v: out.append(dict(clf=clf, test=np.mean([a for a, _ in v]), sd=np.std([a for a, _ in v], ddof=1), cv=np.nanmean([b for _, b in v]), n=len(v)))
    return pd.DataFrame(out)
best_rows, rank_rows = [], []
S11 = pa  # placeholder to keep import used
import re as _re
tex = open("manuscript/PEARL_supplementary_v2.tex").read(); i = tex.index("label{tab:s11_inhouse}"); base = {}
for l in tex[i:i + 1800].split("\n"):
    m = _re.match(r"^(BACE|BBBP|Flavor|hERG|DILI|Caco-2|Half-Life)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)", l.strip())
    if m: base[m.group(1)] = tuple(float(m.group(k)) for k in (2, 3, 4))
sup = pd.read_csv(R + "e09_selection_audit/all_configs_parsed.csv")
for ds in complete:
    x = g[g.ds == ds]; per = {}
    for arm, lab in STAGES.items():
        s = x[x.arm == arm]; orc = s.loc[s.test.idxmax()]; sel = s.loc[s.cv.idxmax()]; s2 = s[s.config != orc.config].sort_values("test", ascending=False).iloc[0]
        pr = [(RES / "e15" / f"{ds}@{orc.emb}" / f"{arm}_{orc.clf}_s{sd}", RES / "e15" / f"{ds}@{s2.emb}" / f"{arm}_{s2.clf}_s{sd}") for sd in range(42, 47)]
        o, lo, hi, n = pa.cmp(ds, pr)
        mod = {"FT Embed": "FT Embed", "FT Embed+PC": "PC+FT", "RAFE": "RAFE"}[lab]; sr = sup[(sup.dataset == ds) & (sup.Modality == mod)].metric.max() if (sup.Modality == mod).any() else np.nan
        best_rows.append(dict(dataset=NAME[ds], stage=lab, selected_by_cv=sel.config, selected_test_mean=sel.test, selected_test_sd=sel.sd, oracle_config=orc.config, oracle_test_mean=orc.test, oracle_test_sd=orc.sd,
                              optimism_gap=orc.test - sel.test, single_run_original_tables=sr, runner_up_config=s2.config, runner_up_test_mean=s2.test, runner_up_diff=o, runner_up_ci_lo=lo, runner_up_ci_hi=hi))
        per[lab] = (orc.test, sel.test)
    pc = pc_only(ds); po = pc.loc[pc.test.idxmax()]; ps = pc.loc[pc.cv.idxmax()] if pc.cv.notna().any() else po; per["PC-only"] = (po.test, ps.test)   # PC-only cells without cv_best: both conventions use the test-best tree (2 candidates only)
    s11 = base[NAME[ds]]; per["Chemprop"] = (s11[1],) * 2; per["GCN"] = (s11[2],) * 2
    per["E2E LoRA"] = (sup[(sup.dataset == ds) & (sup.Modality == "E2E LoRA")].metric.max(),) * 2
    for k, nm in ((0, "best_of_grid"), (1, "cv_selected")):
        v = pd.Series({m: t[k] for m, t in per.items()}); rk = v.rank(ascending=False, method="min").astype(int)
        rank_rows.append(dict(dataset=NAME[ds], convention=nm, **{m: v[m] for m in v.index}, **{f"rank_{m}": rk[m] for m in rk.index}))
B = pd.DataFrame(best_rows); B.round(4).to_csv(OUT / "best_configs.csv", index=False); Rk = pd.DataFrame(rank_rows); Rk.round(4).to_csv(OUT / "ranking.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(B[["dataset", "stage", "selected_by_cv", "selected_test_mean", "oracle_config", "oracle_test_mean", "optimism_gap", "runner_up_diff", "runner_up_ci_lo", "runner_up_ci_hi"]].round(3).to_string(index=False))
for conv in ("best_of_grid", "cv_selected"):
    r = Rk[Rk.convention == conv]; print("\n", conv, "RAFE rank per dataset:", dict(zip(r.dataset, r.rank_RAFE)), "| mean rank RAFE", round(r.rank_RAFE.mean(), 2), "PC-only", round(r["rank_PC-only"].mean(), 2), "FT+PC", round(r["rank_FT Embed+PC"].mean(), 2), "FT", round(r["rank_FT Embed"].mean(), 2), "| RAFE top3:", int((r.rank_RAFE <= 3).sum()), "of", len(r))
# heatmap of the grid: fixed 3x3 layout (up to 9 datasets; empty panels switched off), vertical x-axis labels for legibility
nrows, ncols = 3, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 8.6 * nrows / 2), squeeze=False)
axs_flat = axs.ravel()
for k, ax in enumerate(axs_flat):
    if k >= len(complete): ax.axis("off"); continue
    ds = complete[k]
    x = g[g.ds == ds]; cols = [(a, c) for a in STAGES for c in TREES]; embs = sorted(set(x.emb), key=lambda e: pretty(e)); M = np.full((len(embs), len(cols)), np.nan)
    for i_, e in enumerate(embs):
        for j_, (a, c) in enumerate(cols):
            s = x[(x.emb == e) & (x.arm == a) & (x.clf == c)]
            if len(s): M[i_, j_] = s.test.iloc[0]
    im = ax.imshow(M, cmap="viridis", aspect="auto"); bi = np.unravel_index(np.nanargmax(M), M.shape); ax.add_patch(plt.Rectangle((bi[1] - .5, bi[0] - .5), 1, 1, fill=False, ec="red", lw=2.5))
    TABBR = {"XGBoost": "XGB", "LightGBM": "LGBM"}
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{STAGES[a].replace('FT Embed', 'FT')} {TABBR.get(c, c)}" for a, c in cols], fontsize=9, fontweight="bold", rotation=90, ha="center", va="top"); ax.set_yticks(range(len(embs))); ax.set_yticklabels([pretty(e) for e in embs], fontsize=10, fontweight="bold")
    ax.set_title(f"{NAME[ds]} (red box: best by test mean)", fontsize=11, fontweight="bold")
    for i_ in range(M.shape[0]):
        for j_ in range(M.shape[1]): ax.text(j_, i_, f"{M[i_, j_]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold", color="w")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.ax.tick_params(labelsize=8)
    for lbl in cbar.ax.get_yticklabels(): lbl.set_fontweight("bold")
fig.suptitle(f"Full configuration grid, mean test metric over 5 seeds (MCC; Spearman for Caco-2 and Half-Life) -- {len(complete)} of 7 datasets complete", fontsize=13, fontweight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / "e15_grid_heatmap.pdf"); print("saved figure")
