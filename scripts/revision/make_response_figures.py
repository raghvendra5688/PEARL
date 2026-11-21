"""Figures for the response to reviewers (vector PDF). Data: results/revision/*, supplement tables (all_configs_parsed.csv)."""
import json, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "results/revision/"; O = R + "figures_response/"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "axes.titlesize": 9})
RED, BLUE, GREY, GREEN, ORANGE = "#d62728", "#4c78a8", "#8c8c8c", "#2ca02c", "#ff7f0e"
NAME = {"bace": "BACE", "bbbp": "BBBP", "herg": "hERG", "dili": "DILI", "caco2": "Caco-2", "half_life": "Half-Life", "flavor": "Flavor"}
CFGDS = {"bace_um": "BACE", "bbbp_mf": "BBBP", "dili_um": "DILI", "caco2_mtr": "Caco-2", "bace_mf": "BACE", "herg_mf": "hERG", "flavor_mf": "Flavor", "half_life_mf": "Half-Life"}
CFGCLF = {"bbbp_mf": "XGBoost", "dili_um": "XGBoost", "caco2_mtr": "CatBoost", "bace_mf": "LightGBM", "herg_mf": "LightGBM", "flavor_mf": "LightGBM", "half_life_mf": "XGBoost", "bace_um": "XGBoost"}
def val(j): return j.get("MCC", j.get("Spearman"))
def cells(exp, cfg, arm, clf): return [val(json.load(open(f))) for f in sorted(glob.glob(f"{R}{exp}/{cfg}/{arm}_{clf}_s4?.json"))]
d = pd.read_csv(R + "e09_selection_audit/all_configs_parsed.csv"); rng = np.random.default_rng(0)

# F-A: ladder gains (matched configurations, supplement tables)
fig, ax = plt.subplots(figsize=(7.2, 3.6)); order = ["bace", "bbbp", "flavor", "herg", "dili", "caco2", "half_life"]
comps = [("PC+FT", "FT Embed", "FT Embed+PC vs FT Embed", BLUE), ("RAFE", "FT Embed", "RAFE vs FT Embed", RED), ("RAFE", "PC+FT", "RAFE vs FT Embed+PC", ORANGE)]
for i, ds in enumerate(order):
    s = d[d.dataset == ds]
    for k, (a, b, lab, c) in enumerate(comps):
        A = s[s.Modality == a].set_index(["CLM", "Loss", "ML"]).metric; B = s[s.Modality == b].set_index(["CLM", "Loss", "ML"]).metric
        diff = (A - B).dropna().values; m = diff.mean(); bs = [rng.choice(diff, len(diff)).mean() for _ in range(3000)]
        y = i + (k - 1) * 0.25; ax.errorbar(m, y, xerr=[[m - np.percentile(bs, 2.5)], [np.percentile(bs, 97.5) - m]], fmt="o", color=c, ms=4, capsize=2, label=lab if i == 0 else None)
ax.axvline(0, color="k", lw=.6); ax.set_yticks(range(7)); ax.set_yticklabels([NAME[x] for x in order]); ax.invert_yaxis(); ax.set_xlabel("mean difference in test metric over matched configurations (95% bootstrap CI over configurations)")
ax.legend(frameon=False, fontsize=7, loc="lower right"); ax.set_title("Gain along the PEARL ladder, matched embedding-loss-tree configurations (supplement tables)"); fig.tight_layout(); fig.savefig(O + "figA_ladder_gain.pdf"); plt.close()

# F-B: best-configuration scores of seven methods, RAFE highlighted
base = {"bace": (0.498, 0.528, 0.447), "bbbp": (0.759, 0.620, 0.694), "flavor": (0.808, 0.809, 0.798), "herg": (0.604, 0.558, 0.510), "dili": (0.765, 0.588, 0.729), "caco2": (0.853, 0.802, 0.733), "half_life": (0.502, 0.332, 0.418)}
meth = ["PC-only", "Chemprop", "GCN", "E2E LoRA", "FT Embed", "FT Embed+PC", "RAFE"]; fig, axs = plt.subplots(1, 7, figsize=(10.5, 2.8))
for ax, ds in zip(axs, order):
    s = d[d.dataset == ds]; v = list(base[ds]) + [s[s.Modality == m].metric.max() for m in ("E2E LoRA", "FT Embed", "PC+FT", "RAFE")]
    rk = pd.Series(v).rank(ascending=False, method="min").astype(int).values; cols = [GREY] * 6 + [RED]
    ax.hlines(range(7), min(v) - .03, v, color="#dddddd", lw=1); ax.scatter(v, range(7), c=cols, s=28, zorder=3); ax.set_yticks(range(7)); ax.set_yticklabels(meth if ds == "bace" else [""] * 7, fontsize=7); ax.invert_yaxis(); ax.set_xlim(min(v) - .03, max(v) + .03)
    ax.set_title(f"{NAME[ds]}\nRAFE rank {rk[-1]}", fontsize=8, color=RED if rk[-1] <= 3 else "k"); ax.tick_params(axis="x", labelsize=6)
fig.suptitle("Best configuration of each method per dataset (MCC; Spearman for Caco-2 and Half-Life)", fontsize=9); fig.tight_layout(); fig.savefig(O + "figB_best_config_ranking.pdf"); plt.close()

# F-C: ablation heatmap (delta vs FT+PC), each dataset at its own tree
arms = ["+A", "+B", "+C", "+D", "full_RAFE", "full-A", "full-B", "full-C", "full-D"]; lab = ["+A", "+B", "+C", "+D", "RAFE", "RAFE-A", "RAFE-B", "RAFE-C", "RAFE-D"]
cfgs = ["bbbp_mf", "dili_um", "caco2_mtr", "bace_mf", "bace_um", "herg_mf", "flavor_mf", "half_life_mf"]; M = np.full((len(cfgs), len(arms)), np.nan)
for i, c in enumerate(cfgs):
    clf = CFGCLF[c]; b = cells("e1", c, "base_FT+PC", clf)
    for j, a in enumerate(arms):
        x = cells("e1", c, a, clf)
        if x and b: M[i, j] = np.mean(x) - np.mean(b)
fig, ax = plt.subplots(figsize=(6.4, 3.4)); im = ax.imshow(M, cmap="RdBu_r", vmin=-.06, vmax=.06, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center", fontsize=6.5)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab); ax.set_yticks(range(len(cfgs))); ax.set_yticklabels([CFGDS[c] + (" (Uni-Mol)" if c == "bace_um" else "") for c in cfgs]); ax.axvline(3.5, color="k", lw=.8)
plt.colorbar(im, ax=ax, label="difference to FT Embed+PC (5-seed mean)"); ax.set_title("RAFE feature groups: effect on top of FT Embed+PC (RAFE-X = RAFE without group X)"); fig.tight_layout(); fig.savefig(O + "figC_ablation_heatmap.pdf"); plt.close()

# F-D: retrieval robustness: k sweep and retrieval strategy
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2)); ks = [1, 5, 10, 20, 50, 100]
for c in ["bbbp_mf", "dili_um", "caco2_mtr", "bace_mf", "herg_mf", "half_life_mf"]:
    clf = CFGCLF[c]; m = [np.mean(cells("e3", c, f"k{k}", clf)) if cells("e3", c, f"k{k}", clf) else np.nan for k in ks]
    if np.isfinite(m).all(): ax[0].plot(ks, np.array(m) - m[2], marker="o", ms=3, label=CFGDS[c])
ax[0].axhline(0, color="k", lw=.5); ax[0].set_xscale("log"); ax[0].set_xlabel("number of retrieved neighbours k"); ax[0].set_ylabel("difference to k = 10"); ax[0].legend(frameon=False, fontsize=6, ncol=2); ax[0].set_title("A. Number of neighbours")
xs = []; 
for i, c in enumerate(["bbbp_mf", "dili_um", "caco2_mtr", "bace_mf", "herg_mf", "half_life_mf"]):
    clf = CFGCLF[c]; ref = np.mean(cells("e2", c, "clm", clf)); mo = np.mean(cells("e2", c, "morgan", clf))
    rd = [np.mean(cells("e2", c, f"random_r{r}", clf)) for r in range(5) if cells("e2", c, f"random_r{r}", clf)]
    ax[1].bar(i - .27, mo - ref, .25, color=BLUE, label="Morgan-Tanimoto" if i == 0 else None); ax[1].bar(i, np.mean(rd) - ref if rd else 0, .25, color=GREY, yerr=np.std(rd) if len(rd) > 1 else 0, capsize=2, label="random neighbours" if i == 0 else None)
ax[1].axhline(0, color="k", lw=.5); ax[1].set_xticks(range(6)); ax[1].set_xticklabels(["BBBP", "DILI", "Caco-2", "BACE", "hERG", "Half-Life"], fontsize=7); ax[1].set_ylabel("difference to CLM-cosine retrieval"); ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("B. Retrieval strategy")
fig.tight_layout(); fig.savefig(O + "figD_retrieval_robustness.pdf"); plt.close()

# F-E: runtime
rt = pd.read_csv(R + "e08/runtime_summary.csv")
import re as _re
def _sec(txt, key=None):   # first number in the text field, converted to seconds (s, ms or h)
    m = _re.search(r"([\d.]+) (ms|s|h)", txt if key is None else txt.split(key)[1]); v = float(m.group(1)); return v / 1000 if m.group(2) == "ms" else v * 3600 if m.group(2) == "h" else v
def hrs(pattern, ds=None):
    x = rt[rt.stage.str.contains(pattern, regex=False)]; x = x[x.dataset == ds] if ds else x; return float(x.hours.iloc[0]) * 3600
lora = _sec(rt[rt.stage.str.contains("LoRA fine-tuning", regex=False)].file.iloc[0], "median")
smiles_zinc = np.mean([hrs("6 fine-tuned SMILES-CLM variants", "herg"), hrs("6 fine-tuned SMILES-CLM variants", "dili")])
unimol = np.mean([hrs("fine-tuned Uni-Mol", "herg"), hrs("fine-tuned Uni-Mol", "dili")])
tree = {c: _sec(rt[rt.stage.str.contains(f"({c})", regex=False)].file.iloc[0], "median") for c in ("LightGBM", "XGBoost")}
q1 = _sec(rt[rt.stage.str.contains("single molecule: exact retrieval", regex=False)].file.iloc[0], "total"); xg = _sec(rt[rt.stage.str.contains("XGBoost inference", regex=False) & rt.stage.str.contains("single molecule", regex=False)].file.iloc[0])
st = [("LoRA fine-tuning (1 run, median)", lora, "one-off"), ("ZINC embedding, SMILES models (6 variants)", smiles_zinc, "one-off"), ("ZINC embedding, Uni-Mol (2 variants)", unimol, "one-off"), ("Tree tuning + fit per cell, LightGBM", tree["LightGBM"], "one-off"), ("Tree tuning + fit per cell, XGBoost (GPU)", tree["XGBoost"], "one-off"),
      ("RAFE per query molecule (retrieval + features)", q1, "per query"), ("XGBoost inference per molecule", xg, "per query")]
fig, ax = plt.subplots(figsize=(6.6, 2.9)); ax.barh(range(len(st)), [s[1] for s in st], color=[BLUE if s[2] == "one-off" else RED for s in st]); ax.set_xscale("log"); ax.set_yticks(range(len(st))); ax.set_yticklabels([s[0] for s in st], fontsize=7); ax.invert_yaxis()
ax.set_xlabel("wall-clock seconds (log scale)"); [ax.text(s[1] * 1.15, i, (f"{s[1]/3600:.1f} h" if s[1] >= 3600 else f"{s[1]/60:.1f} min" if s[1] >= 120 else f"{s[1]:.4g} s"), va="center", fontsize=6.5) for i, s in enumerate(st)]; ax.set_xlim(1e-4, 5e5); ax.set_title("Cost of the pipeline stages (blue: one-off, red: per query molecule)"); fig.tight_layout(); fig.savefig(O + "figE_runtime.pdf"); plt.close()

# F-F: controlled matrix
mat = pd.read_csv(R + "e06/controlled_matrix_means.csv"); fig, axs = plt.subplots(2, 4, figsize=(10, 4.6)); cl = {"CatBoost": "^", "LightGBM": "s", "XGBoost": "o"}
for ax, c in zip(axs.ravel(), ["bace_mf", "bace_um", "bbbp_mf", "dili_um", "caco2_mtr", "herg_mf", "flavor_mf", "half_life_mf"]):
    s = mat[mat.cfg == c]; cols = ["PC_only", "FT_only", "base_FT+PC", "full_RAFE"]
    ax.hlines([s[x].mean() for x in cols], np.arange(4) - .3, np.arange(4) + .3, color=[GREY, "#bcbd22", BLUE, RED], lw=2.5)
    for _, r in s.iterrows(): ax.scatter(np.arange(4) + {"CatBoost": -.12, "LightGBM": 0, "XGBoost": .12}[r.clf], [r[x] for x in cols], marker=cl[r.clf], c="k", s=14, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["PC-only", "FT", "FT+PC", "RAFE"], fontsize=7); ax.set_title(CFGDS[c] + (" (Uni-Mol)" if c == "bace_um" else ""), fontsize=8)
    lo = min(s[x].min() for x in cols); hi = max(s[x].max() for x in cols); ax.set_ylim(lo - .06, hi + .03)
axs[0, 0].set_ylabel("MCC (Spearman: Caco-2, Half-Life)"); [axs[0, 0].scatter([], [], marker=m, c="k", s=14, label=k) for k, m in cl.items()]; axs[0, 0].legend(frameon=False, fontsize=6)
fig.suptitle("Controlled comparison: identical splits, tuning budget, three trees, five seeds (thick line: mean over trees; markers: individual trees)", fontsize=9); fig.tight_layout(); fig.savefig(O + "figF_controlled_matrix.pdf"); plt.close()

# F-G: 3D vs ECFP4 and conformer sensitivity
e5 = {}
for nm_, cf_ in (("BACE", "bace_um"), ("BBBP", "bbbp_um"), ("Caco-2", "caco2_um"), ("DILI", "dili_um")):
    e5[nm_] = tuple(np.mean([val(json.load(open(f))) for f in glob.glob(f"{R}e05/{cf_}/{a}_XGBoost_s4?.json")]) for a in ("cls512", "ecfp2048", "concat2560"))
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2)); w = .26
for k, (lab_, c) in enumerate([("3D vector (512-d)", BLUE), ("ECFP4 (2048-bit)", GREY), ("both (2560-d)", RED)]): ax[0].bar(np.arange(4) + (k - 1) * w, [v[k] for v in e5.values()], w, color=c, label=lab_)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(e5.keys()); ax[0].set_ylabel("XGBoost MCC (Caco-2: Spearman), 5 seeds"); ax[0].set_ylim(0.3, 0.8); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("A. 3D versus ECFP4 within the Uni-Mol vector")
rows = []
for f in glob.glob(R + "e05b/bace_um/*.json"):
    j = json.load(open(f)); rows.append(dict(arm=j["arm"], clf=j["clf"], cs=j["conformer_seed"], ts=j["seed"], v=j["MCC"]))
x = pd.DataFrame(rows)
for k, (arm, lab_, c) in enumerate([("cls512", "3D vector", BLUE), ("concat2560", "3D + ECFP4", RED)]):
    for s in range(42, 47):
        v = x[(x.arm == arm) & (x.cs == s)].v; ax[1].scatter(np.full(len(v), s + (k - .5) * .25), v, color=c, s=8, alpha=.6)
    ax[1].scatter([], [], color=c, s=14, label=lab_)
ax[1].set_xticks(range(42, 47)); ax[1].set_xlabel("conformer (ETKDG) seed; 42 = original"); ax[1].set_ylabel("test MCC (XGBoost and LightGBM, 3 tree seeds)"); ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("B. Conformer sensitivity on BACE")
fig.tight_layout(); fig.savefig(O + "figG_3d_vs_ecfp4_conformers.pdf"); plt.close()

# F-H: Half-Life metrics
h = d[d.dataset == "half_life"].rename(columns={"R": "R2"}); hh = pd.read_csv(R + "r34/half_life_raw_vs_log_R2.csv")
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2)); cm = {"E2E LoRA": GREEN, "FT Embed": GREY, "PC+FT": BLUE, "RAFE": RED}
for m_, g in h.groupby("Modality"): ax[0].scatter(g.Spearman, g.R2, s=14, color=cm[m_], label=m_)
ax[0].set_xlabel("Spearman (rank agreement)"); ax[0].set_ylabel("R$^2$ in hours"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("A. Ranking is moderate, magnitude is hard")
ax[1].scatter(hh.R2_raw_hours, hh.R2_log1p_space, s=12, color=BLUE); lim = [min(hh.R2_raw_hours.min(), hh.R2_log1p_space.min()) - .02, hh.R2_log1p_space.max() + .02]; ax[1].plot(lim, lim, color="k", lw=.6)
ax[1].set_xlabel("R$^2$ in hours"); ax[1].set_ylabel("R$^2$ on log1p scale"); ax[1].set_title("B. Same predictions, two scales (fingerprint and descriptor baselines)")
fig.tight_layout(); fig.savefig(O + "figH_halflife_metrics.pdf"); plt.close(); print("figures:", sorted(p.split("/")[-1] for p in glob.glob(O + "*.pdf")))
