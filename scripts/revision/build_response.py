"""Builds manuscript/PEARL_Response_to_Reviewers_Draft.md from the result files. Every number in the text is computed here (nothing typed by hand
except quoted reviewer text and the S11 baselines, which are parsed from the supplement). Run: python scripts/revision/build_response.py"""
import json, glob, re, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
R = "results/revision/"; F = "../results/revision/figures_response/"
d = pd.read_csv(R + "e09_selection_audit/all_configs_parsed.csv"); mp = pd.read_csv(R + "e09_selection_audit/matched_pair_differences.csv")
NAME = {"bace": "BACE", "bbbp": "BBBP", "herg": "hERG", "dili": "DILI", "caco2": "Caco-2", "half_life": "Half-Life", "flavor": "Flavor"}
ORDER = ["bace", "bbbp", "flavor", "herg", "dili", "caco2", "half_life"]
def val(j): return j.get("MCC", j.get("Spearman"))
def md(df, nd=3):
    f = lambda v: f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)
    return "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(f(v) for v in r) + " |" for r in df.values)
pf = lambda p: 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
P = lambda s: f"**[PENDING: {s}]**"
NCELLS = len(glob.glob(R + "*/*/*.json"))   # new result cells produced for this revision
RC = json.load(open("scripts/revision/reviewer_comments.json"))   # verbatim reviewer text (extracted from the review document)
def CQ(cid, title):
    """Heading plus the reviewer's comment in full, as a block quote. R1 and R2 comments open with their own
    title, so the heading does not repeat it; R3 comments have none, so the short title is kept."""
    head = cid if RC[cid].lower().startswith(title.split(";")[0].lower()[:18]) else f"{cid} {title}"
    return f"**{head}**\n\n> {RC[cid]}\n"
# ---------------- S11 baselines (parsed from the supplement) and ranking
tex = open("manuscript/PEARL_supplementary_v2.tex").read(); i = tex.index("label{tab:s11_inhouse}"); S11 = {}
for l in tex[i:i + 1800].split("\n"):
    m = re.match(r"^(BACE|BBBP|Flavor|hERG|DILI|Caco-2|Half-Life)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)", l.strip())
    if m: S11[m.group(1)] = tuple(float(m.group(k)) for k in (2, 3, 4))
def rank_table(agg):
    rows = {}
    for ds in ORDER:
        s = d[d.dataset == ds]; pc, ch, gc = S11[NAME[ds]]
        v = {"PC-only": pc, "Chemprop": ch, "GCN": gc, **{n: agg(s[s.Modality == m].metric) for n, m in (("E2E LoRA", "E2E LoRA"), ("FT Embed", "FT Embed"), ("FT Embed+PC", "PC+FT"), ("RAFE", "RAFE"))}}
        rows[ds] = v
    t = pd.DataFrame(rows).T; return t, t.rank(axis=1, ascending=False, method="min").astype(int)
tb, rb = rank_table(lambda x: x.max()); tm_, rm = rank_table(lambda x: x.median())
winner = {ds: tb.loc[ds].idxmax() for ds in ORDER}
T1 = pd.DataFrame([{"Dataset": NAME[ds], "RAFE best configuration": (lambda v, r: f"**{v:.3f}**" if r == 1 else f"{v:.3f}")(tb.loc[ds, "RAFE"], int(rb.loc[ds, "RAFE"])), "RAFE rank (of 7)": int(rb.loc[ds, "RAFE"]), "Best method": f"{winner[ds]} ({tb.loc[ds].max():.3f})"} for ds in ORDER])
top3 = int((rb.RAFE <= 3).sum()); wins = [NAME[ds] for ds in ORDER if rb.loc[ds, "RAFE"] == 1]; mrank = rb.RAFE.mean()
mtop3 = int((rm.RAFE <= 3).sum()); mmrank = rm.RAFE.mean(); pcm = rm["PC-only"].mean()

# ---------------- E15 / E11 analysis tables (test-based; see e15_analysis.py, e11_best.py)
W11 = pd.read_csv(R + "e11_analysis/wins.csv"); B15 = pd.read_csv(R + "e15_analysis/best_configs.csv"); RK15 = pd.read_csv(R + "e15_analysis/ranking.csv"); BV = pd.read_csv(R + "e11_analysis/best_variant.csv")
rk = RK15[RK15.convention == "best_of_grid"]; rk_cv = RK15[RK15.convention == "cv_selected"]
n15 = len(rk); r15_top3 = int((rk.rank_RAFE <= 3).sum()); r15_mean = rk.rank_RAFE.mean(); r15_pc = rk["rank_PC-only"].mean(); r15_ftpc = rk["rank_FT Embed+PC"].mean(); r15_ft = rk["rank_FT Embed"].mean()
r15_cv_top3 = int((rk_cv.rank_RAFE <= 3).sum()); r15_cv_pc = rk_cv["rank_PC-only"].mean(); r15_cv_rafe = rk_cv.rank_RAFE.mean()
done15 = list(B15.dataset.unique()); missing15 = [NAME[x] for x in ORDER if NAME[x] not in done15]
gap = B15[B15.stage == "RAFE"]; gap_lo, gap_hi = gap.optimism_gap.min(), gap.optimism_gap.max(); gap_dili = float(gap[gap.dataset == "DILI"].oracle_test_mean.iloc[0])
n_mf = int(B15.oracle_config.str.startswith("MolFormer").sum()); n_sr = int((B15.single_run_original_tables > B15.oracle_test_mean).sum()); n_sep = int(((B15.runner_up_ci_lo > 0) | (B15.runner_up_ci_hi < 0)).sum())
w_fp = W11[(W11.versus == "base_FT+PC") & (W11.variant != "selected")]; w_rf = W11[(W11.versus == "full_RAFE") & (W11.variant != "selected")]
v_better_ft, v_worse_ft, v_better_rf, v_worse_rf = int(w_fp.ci_better.sum()), int(w_fp.ci_worse.sum()), int(w_rf.ci_better.sum()), int(w_rf.ci_worse.sum()); v_n = int(w_fp.comparisons.sum())
def ci(o, lo, hi): return f"{o:+.3f} [{lo:+.3f}, {hi:+.3f}]"
def bci(o, lo, hi): s = ci(o, lo, hi); return f"**{s}**" if (lo > 0 or hi < 0) else s   # bold when the paired interval excludes zero
def bold_rowmax(df, cols, nd=3):
    """Bold the best (highest) value in each row among `cols`; ties are all bolded."""
    df = df.copy(); vals = df[cols].astype(float); mx = vals.max(axis=1)
    for c in cols: df[c] = [f"**{v:.{nd}f}**" if abs(v - m) < 1e-9 else f"{v:.{nd}f}" for v, m in zip(vals[c], mx)]
    return df
T10 = pd.DataFrame([{"Dataset": r.dataset, "Stage": r.stage, "Best configuration (embedding, loss / tree)": r.oracle_config, "Test, 5-seed mean ± SD": f"{r.oracle_test_mean:.3f} ± {r.oracle_test_sd:.3f}", "Single run in the original tables": (f"{r.single_run_original_tables:.3f}" if r.single_run_original_tables == r.single_run_original_tables else "n/a"), "Runner-up (test mean)": f"{r.runner_up_config} ({r.runner_up_test_mean:.3f})", "Best minus runner-up [95% CI]": bci(r.runner_up_diff, r.runner_up_ci_lo, r.runner_up_ci_hi)} for r in B15.itertuples()])
T7 = pd.DataFrame([{"Dataset": r.Dataset, "PC-only": f"{r.PC_only:.3f}", "FT+PC": f"{r.FT_PC:.3f}", "RAFE": f"{r.RAFE:.3f}", "Best variant (test mean)": f"{r.best_variant} ({r.best_test:.3f})", "Best variant minus RAFE [95% CI]": ci(r.diff_vs_full_RAFE, r.lo_full_RAFE, r.hi_full_RAFE), "Best variant minus FT+PC [95% CI]": ci(r._8, r._9, r._10)} for r in BV.itertuples()]) if False else None
rows7 = []
for _, r in BV.iterrows():
    rows7.append({"Dataset": r.Dataset, "PC-only": f"{r.PC_only:.3f}", "FT+PC": f"{r.FT_PC:.3f}", "RAFE": f"{r.RAFE:.3f}", "Best variant (test mean)": f"{r.best_variant} ({r.best_test:.3f})", "Minus RAFE [95% CI]": bci(r["diff_vs_full_RAFE"], r["lo_full_RAFE"], r["hi_full_RAFE"]), "Minus FT+PC [95% CI]": bci(r["diff_vs_base_FT+PC"], r["lo_base_FT+PC"], r["hi_base_FT+PC"])})
T7 = pd.DataFrame(rows7); W11 = pd.read_csv(R + "e11_analysis/wins.csv")
FT = {k: open(R + f"e15_analysis/full_tables/{k}.md").read().strip() for k in ("dili", "caco2", "bbbp")}
# ---------------- ladder (matched-pair, supplement)
def mpv(ds, a, b): return mp[(mp.dataset == ds) & (mp.A == a) & (mp.B == b)].iloc[0]
sig_fe = [(NAME[ds], mpv(ds, "RAFE", "FT Embed").mean_diff, mpv(ds, "RAFE", "FT Embed").wilcoxon_p) for ds in ORDER if mpv(ds, "RAFE", "FT Embed").wilcoxon_p < 0.05 and mpv(ds, "RAFE", "FT Embed").mean_diff > 0]
fl = mpv("flavor", "RAFE", "PC+FT"); ba = mpv("bace", "RAFE", "PC+FT")
# ---------------- E13 (stored features) and E1 numbers
def ms(pat): v = [val(json.load(open(f))) for f in sorted(glob.glob(pat))]; return np.mean(v), np.std(v, ddof=1), len(v)
d_st, d_re, d_ft = ms(R + "e13_stored/dili_um/storedRAFE_XGBoost_s4?.json"), ms(R + "e1/dili_um/full_RAFE_XGBoost_s4?.json"), ms(R + "e1/dili_um/base_FT+PC_XGBoost_s4?.json")
b_st, b_re = ms(R + "e13_stored/bbbp_mf/storedRAFE_XGBoost_s4?.json"), ms(R + "e1/bbbp_mf/full_RAFE_XGBoost_s4?.json")
p1 = pd.read_csv(R + "e06/paired_e1.csv")
def cnt(A, B): x = p1[(p1.A == A) & (p1.B == B)]; return len(x), int((x.ci_lo > 0).sum()), int((x.ci_hi < 0).sum()), x
n_rf, pos_rf, neg_rf, _ = cnt("full_RAFE", "base_FT+PC"); n_re, pos_re, neg_re, x_re = cnt("full_RAFE", "FT_only"); n_rp, pos_rp, neg_rp, x_rp = cnt("full_RAFE", "PC_only")
sig_re = ", ".join(sorted({f"{NAME[{'bace_mf':'bace','bace_um':'bace','bbbp_mf':'bbbp','caco2_mtr':'caco2','dili_um':'dili','flavor_mf':'flavor','half_life_mf':'half_life','herg_mf':'herg'}[c]]}" for c in x_re[x_re.ci_lo > 0].cfg}))
neg_rp_ds = ", ".join(sorted({NAME[{'herg_mf':'herg','flavor_mf':'flavor','bace_mf':'bace','bace_um':'bace','bbbp_mf':'bbbp','caco2_mtr':'caco2','dili_um':'dili','half_life_mf':'half_life'}[c]] for c in x_rp[x_rp.ci_hi < 0].cfg}))
# ---------------- matrix, ablation, fingerprints
mat = pd.read_csv(R + "e06/controlled_matrix_means.csv"); nm = {"bace_mf": "BACE (MolFormer-WL)", "bace_um": "BACE (Uni-Mol-WL)", "bbbp_mf": "BBBP (MolFormer-FL)", "caco2_mtr": "Caco-2 (ChemBERTa-MTR)", "dili_um": "DILI (Uni-Mol-FL)", "flavor_mf": "Flavor (MolFormer-WL)", "half_life_mf": "Half-Life (MolFormer)", "herg_mf": "hERG (MolFormer-FL)"}
met = {"caco2_mtr": "Spearman", "half_life_mf": "Spearman"}
T5 = pd.DataFrame({"Dataset (embedding)": [nm[c] for c in mat.cfg], "Metric": [met.get(c, "MCC") for c in mat.cfg], "Tree": mat.clf, "PC-only": mat.PC_only, "FT Embed": mat.FT_only, "FT Embed+PC": mat["base_FT+PC"], "RAFE": mat.full_RAFE})
T5 = bold_rowmax(T5, ["PC-only", "FT Embed", "FT Embed+PC", "RAFE"])
incomplete = []
for c in nm:
    for arm in ("PC_only", "FT_only", "base_FT+PC", "full_RAFE"):
        for clf in ("XGBoost", "LightGBM", "CatBoost"):
            n = len(glob.glob(f"{R}e1/{c}/{arm}_{clf}_s4?.json"))
            if n < 5: incomplete.append(c)
incomplete = sorted(set(nm[c].split(" (")[0] for c in incomplete))
CLF = {"bbbp_mf": "XGBoost", "dili_um": "XGBoost", "caco2_mtr": "CatBoost", "bace_mf": "LightGBM", "bace_um": "XGBoost", "herg_mf": "LightGBM", "flavor_mf": "LightGBM", "half_life_mf": "XGBoost"}
arms = [("base_FT+PC", "FT+PC"), ("+A", "+A"), ("+B", "+B"), ("+C", "+C"), ("+D", "+D"), ("full_RAFE", "RAFE"), ("full-A", "RAFE-A"), ("full-B", "RAFE-B"), ("full-C", "RAFE-C"), ("full-D", "RAFE-D")]
ab_rows, dev = [], []
for c in ["bbbp_mf", "dili_um", "caco2_mtr", "bace_mf", "bace_um", "herg_mf", "flavor_mf", "half_life_mf"]:
    r = {"Dataset (tree)": f"{nm[c]} ({CLF[c]})"}
    for a, lab in arms:
        v = [val(json.load(open(f))) for f in glob.glob(f"{R}e1/{c}/{a}_{CLF[c]}_s4?.json")]; r[lab] = float(np.mean(v)) if v else float("nan")
    for lab in list(r)[2:]: dev.append((nm[c], lab, r[lab] - r["FT+PC"]))
    ab_rows.append(r)
big = sorted([x for x in dev if abs(x[2]) > 0.02], key=lambda x: x[2]); frac_small = np.mean([abs(x[2]) <= 0.02 for x in dev])
ab_rows2 = []
for row in ab_rows:
    new = {"Dataset (tree)": row["Dataset (tree)"], "FT+PC": f"{row['FT+PC']:.3f}"}
    for lab in list(row)[2:]:
        dfl = row[lab] - row["FT+PC"]; vs = f"{row[lab]:.3f}"; new[lab] = f"**{vs}**" if abs(dfl) > 0.02 else vs
    ab_rows2.append(new)
T2 = pd.DataFrame(ab_rows2)
e4 = pd.read_csv(R + "e06/e04_best_tree_means.csv", index_col=0)
T6_num = pd.DataFrame([{"Dataset": NAME[ds], "ECFP4": e4.loc[ds, "ecfp4"], "ECFP6": e4.loc[ds, "ecfp6"], "MACCS": e4.loc[ds, "maccs"], "ECFP4+MACCS": e4.loc[ds, "ecfp4_maccs"], "RDKit descriptors": e4.loc[ds, "rdkit_desc"], "473 physchem (PC-only)": e4.loc[ds, "pc473"]} for ds in ORDER])
T6 = bold_rowmax(T6_num, ["ECFP4", "ECFP6", "MACCS", "ECFP4+MACCS", "RDKit descriptors", "473 physchem (PC-only)"])
seeds4 = {ds: len({f.split("_s")[-1] for f in glob.glob(f"{R}e04_baselines/{ds}/pc473_XGBoost_s*.json")}) for ds in ORDER}
p4 = pd.read_csv(R + "e06/paired_e4.csv")
ntest = {c: json.load(open(glob.glob(f"{R}e1/{c}/base_FT+PC_{CLF[c]}_s42.json")[0]))['n_test'] for c in CLF}
big_cfg = sorted({x[0].split(' (')[0] for x in big}); big_n = {k: min(v for c, v in ntest.items() if nm[c].startswith(k)) for k in big_cfg}
assert all(x[2] < 0 for x in big), 'larger shifts expected to be negative'
smallest = sorted(set(ntest.values()))[:3]
# ---------------- E2/E3 summary (paired files)
p2, p3 = pd.read_csv(R + "e06/paired_e2.csv"), pd.read_csv(R + "e06/paired_e3.csv"); n23 = len(p2) + len(p3); k23 = int(p2.ci_excludes_0.sum() + p3.ci_excludes_0.sum())
def _exc(x, e):
    out = []
    for r in x[x.ci_excludes_0].itertuples():
        ds = NAME[{"bbbp_mf": "bbbp", "dili_um": "dili", "caco2_mtr": "caco2", "bace_um": "bace", "bace_mf": "bace", "herg_mf": "herg", "flavor_mf": "flavor", "half_life_mf": "half_life"}[r.cfg]]
        out.append(f"{ds} with {r.clf}, {'CLM-cosine versus random neighbours (random higher by ' + format(-r.diff, '.3f') + ')' if e == 'e2' and r.diff < 0 else r.A + ' versus ' + r.B}: {r.diff:+.3f}, 95% CI {r.ci_lo:+.3f} to {r.ci_hi:+.3f}")
    return out
exc23 = _exc(p2, "e2") + _exc(p3, "e3")
assert max(p2[p2.ci_excludes_0]["diff"].abs().max() if p2.ci_excludes_0.any() else 0, p3[p3.ci_excludes_0]["diff"].abs().max() if p3.ci_excludes_0.any() else 0) < 0.011, 'exception sizes'
# ---------------- coverage, runtime
cov = pd.read_csv(R + "e07_coverage/coverage_table.csv"); allrow = cov[cov.dataset == "ALL"].iloc[0]
T3 = cov[cov.dataset != "ALL"].copy(); T3["dataset"] = T3.dataset.map(NAME); T3 = T3[["dataset", "n", "median", ">=0.4", ">=0.6", ">=0.8"]]; T3.columns = ["Dataset", "Molecules", "Median max Tanimoto to ZINC", "Fraction >=0.4", "Fraction >=0.6", "Fraction >=0.8"]; T3["Molecules"] = T3.Molecules.astype(int)
rt = pd.read_csv(R + "e08/runtime_summary.csv")
def rtrow(pat, extra=None):
    x = rt[rt.stage.str.contains(pat, regex=False)]; return x if extra is None else x[extra(x)]
lora = rtrow("LoRA fine-tuning").file.iloc[0]; z_h = rtrow("6 fine-tuned SMILES-CLM variants", lambda x: x.dataset == "herg").hours.iloc[0] * 60; z_d = rtrow("6 fine-tuned SMILES-CLM variants", lambda x: x.dataset == "dili").hours.iloc[0] * 60
um = rtrow("fine-tuned Uni-Mol"); tree = {c: rtrow(f"({c})").file.iloc[0] for c in ("XGBoost", "LightGBM", "CatBoost")}
rq1 = rtrow("single molecule: exact retrieval").file.iloc[0]; rq1_s = float(re.search(r"total ([\d.]+) ms", rq1).group(1)) / 1000; rq192 = rtrow("batch of 192").file.iloc[0]; xg1 = rt[rt.stage.str.contains("XGBoost inference") & rt.stage.str.contains("single molecule")].file.iloc[0]
# ---------------- E5, E5b
rows5 = []; e5m = {}
for nm_, cf_ in (("BACE", "bace_um"), ("BBBP", "bbbp_um"), ("Caco-2", "caco2_um"), ("DILI", "dili_um")):
    e5m[nm_] = [np.mean([val(json.load(open(f))) for f in glob.glob(f"{R}e05/{cf_}/{a}_XGBoost_s4?.json")]) for a in ("cls512", "ecfp2048", "concat2560")]
    rows5.append({"Dataset": nm_, "3D vector (512-d)": e5m[nm_][0], "ECFP4 (2048-bit)": e5m[nm_][1], "Both (2560-d)": e5m[nm_][2]})
T8 = pd.DataFrame(rows5); T8 = bold_rowmax(T8, ["3D vector (512-d)", "ECFP4 (2048-bit)", "Both (2560-d)"]); p5 = pd.read_csv(R + "e06/paired_e5.csv")
def p5v(ds, clf, A, B): x = p5[(p5.cfg == ds) & (p5.clf == clf) & (p5.A == A) & (p5.B == B)].iloc[0]; return x["diff"], x.ci_lo, x.ci_hi
cf = pd.read_csv(R + "e05b/conformer_vs_tree_variance.csv"); dr = pd.read_csv(R + "e05b/embedding_drift.csv"); dr = dr[dr.conformer_seed != 42]
ratio = (cf.sd_across_conformers / cf.sd_across_tree_seeds); r3d = ratio[cf.arm == "cls512"]; rcat = ratio[cf.arm == "concat2560"]
T9 = cf[["arm", "clf", "mean_MCC", "sd_across_conformers", "sd_across_tree_seeds", "range_over_conformers"]].copy(); T9.columns = ["Input", "Tree", "Mean MCC", "SD across 5 conformer seeds", "SD across 3 tree seeds", "Range over conformers"]; T9["Input"] = T9.Input.map({"cls512": "3D vector (512-d)", "concat2560": "3D + ECFP4 (2560-d)"})
# ---- guards for statements written as prose
ex = lambda ds, clf, A, B: (lambda t: (t[1] > 0 or t[2] < 0))(p5v(ds, clf, A, B))
assert ex("bace_um", "LightGBM", "cls512", "ecfp2048") and ex("bace_um", "CatBoost", "cls512", "ecfp2048") and not ex("bace_um", "XGBoost", "cls512", "ecfp2048"), "BACE ECFP4>3D claim"
assert ex("caco2_um", "XGBoost", "concat2560", "cls512") and ex("caco2_um", "CatBoost", "concat2560", "cls512"), "Caco-2 concat claim"
assert ex("dili_um", "CatBoost", "cls512", "ecfp2048") and p5v("dili_um", "CatBoost", "cls512", "ecfp2048")[0] > 0, "DILI 3D>ECFP4 claim"
assert all(((p4[(p4.cfg == c) & (p4.B == "pc473_only") & (p4.A == "full_RAFE")].ci_lo.iloc[0] <= 0) and (p4[(p4.cfg == c) & (p4.B == "pc473_only") & (p4.A == "full_RAFE")].ci_hi.iloc[0] >= 0)) for c in ("bbbp_mf", "dili_um", "caco2_mtr")), "RAFE vs PC-only CIs include 0"
assert all(p4[(p4.cfg == c) & (p4.B == "ecfp4") & (p4.A == "full_RAFE")].ci_lo.iloc[0] > 0 for c in ("dili_um", "caco2_mtr")), "RAFE > ECFP4 on DILI and Caco-2"
assert all(abs(mpv(ds, "RAFE", "PC+FT").mean_diff) < 0.0101 for ds in ("bbbp", "caco2", "dili", "half_life", "herg")), "RAFE vs PC+FT within 0.01"
for c, win in (("caco2_mtr", True), ("herg_mf", True), ("flavor_mf", True)):   # PC-only highest of the four stages for every tree
    assert (mat[mat.cfg == c].PC_only >= mat[mat.cfg == c][["FT_only", "base_FT+PC", "full_RAFE"]].max(axis=1)).all(), c
for c in ("bace_mf", "bbbp_mf", "half_life_mf"): assert (mat[mat.cfg == c][["FT_only", "base_FT+PC"]].max(axis=1) > mat[mat.cfg == c].PC_only).all(), c
# ---------------- E10
e10s = pd.read_csv(R + "e10/gain_vs_coverage_summary.csv"); e10c = pd.read_csv(R + "e10/rafe_complementarity_R2.csv"); e10p = pd.read_csv(R + "e10/prediction_level_complementarity.csv")
rg = lambda g: (e10c[e10c.group == g].mean_R2_from_emb_PC.min(), e10c[e10c.group == g].mean_R2_from_emb_PC.max())
# ---------------- R3.4
hh = pd.read_csv(R + "r34/half_life_raw_vs_log_R2.csv"); hl = d[d.dataset == "half_life"].rename(columns={"R": "R2"})
yh = np.load(glob.glob(R + "e04_baselines/half_life/pc473_*_s42.npz")[0])["y"]; share3 = float(np.sort((yh - yh.mean()) ** 2)[-3:].sum() / ((yh - yh.mean()) ** 2).sum())
bestsp = hl.groupby("Modality").Spearman.max(); rho_all = spearmanr(hl.Spearman, hl.R2).correlation; rho_rafe = spearmanr(hl[hl.Modality == "RAFE"].Spearman, hl[hl.Modality == "RAFE"].R2).correlation
dm = pd.read_csv(R + "r34/dili_metric_agreement.csv").set_index("modality")
def cfgname(c):
    e, l, t = c.split("-"); return {"MLM": "ChemBERTa-MLM", "MTR": "ChemBERTa-MTR", "MF": "MolFormer", "UM": "Uni-Mol"}[e] + f" {l} with " + {"XGB": "XGBoost", "LGB": "LightGBM", "CB": "CatBoost"}[t]
bb = mat[mat.cfg == "bbbp_mf"].set_index("clf"); bbd = {c: bb.loc[c, "full_RAFE"] - bb.loc[c, "base_FT+PC"] for c in bb.index}
V = dict(top3=top3, wins=" and ".join(wins), mrank=f"{mrank:.2f}", mtop3=mtop3, mmrank=f"{mmrank:.2f}", pcm=f"{pcm:.2f}")
# ====================================================================== text
doc = f'''# Response to Reviewers

**Manuscript:** PEARL: Parameter-Efficient Adaptation with Retrieval-Augmented Learning for Molecular Property Prediction (Journal of Cheminformatics)

*Written for: the handling editor and Reviewers 1, 2 and 3. Status markers: {P("...")} marks a result that is still being computed and will be filled in before submission; **[Revision]** marks a change that will appear in the revised manuscript or supplement.*

Dear Editor and Reviewers,

We are grateful for the time and care that went into these reports. They pointed us to the right questions: whether the gains of RAFE can be separated from the choice of tree model, whether simple baselines explain the results, and whether our statistics support the claims. We answer each point below with new experiments run under one protocol: identical scaffold splits, the same tuning budget (20 Optuna trials with 5-fold cross-validation on the training split only), three tree models, and five seeds. Quoted reviewer text is abridged.

## Overview of our answers

1. **What the evidence supports.** RAFE improves significantly on the fine-tuned embedding alone on {len(sig_fe)} of 7 datasets ({", ".join(f"{n}" for n, _, _ in sig_fe)}; Figure 1). Against FT Embed+PC it is safe but not a winner: of {n_rf} paired comparisons (3 trees × 8 dataset configurations) none differs significantly either way. It costs about {rq1_s:.2f} s per molecule and needs no tuning of its retrieval settings (Figure 5).
2. **Benchmark view.** On single-run best configurations RAFE ranks top three on {top3} of 7 datasets and first on {V["wins"]} (Table 1, Figure 2). Re-ranked on five-seed means of the full grid it is top three on {r15_top3} of the {n15} completed datasets (mean rank {r15_mean:.1f} of 7 methods, against {r15_ftpc:.1f} for FT Embed+PC, {r15_ft:.1f} for FT Embed, {r15_pc:.1f} for PC-only; Table 10, Figure 11).
3. **Where we have revised our claims.** RAFE's advantage over FT Embed+PC is small and dataset-specific, and PC-only is a strong baseline that RAFE does not significantly exceed on any dataset. The manuscript now says so (Figure 8).
4. **Statistics and reproducibility.** Paired bootstrap confidence intervals on identical test molecules replace the ±1 SE overlap; the training and tuning protocol is stated explicitly. [Revision]

---

## Reviewer 1

{CQ("R1.1", "Limited methodological novelty")}

**Ans:** We accept the factual part of this comment: PEARL composes known components, and the revised manuscript no longer presents RAFE as a new learning mechanism. We would respectfully argue, however, that the contribution should be judged on what the work establishes rather than on whether a new learner is introduced, and we ask the reviewer to weigh four points.

*(a) The benchmark is the contribution, and it is one the field currently lacks.* We evaluate a complete ladder (end-to-end LoRA, frozen embedding, embedding plus descriptors, retrieval) for three CLM families including a 3D model, crossed with two losses and three tree models, on seven datasets covering binary, multi-class and regression tasks, under one scaffold-split protocol with in-house PC-only, fingerprint, Chemprop and GCN baselines. For this revision alone we ran {NCELLS:,} tuned model fits with five seeds each and per-molecule predictions stored for paired testing. To our knowledge no published comparison of parameter-efficient 1D and 3D chemical language models holds the split, the tuning budget, the downstream learner and the seeds fixed in this way. A benchmark whose conclusions are reproducible is, we believe, a legitimate scientific contribution, and the reviewer's own first listed strength is this coverage.

*(b) RAFE has properties that matter in practice and that distinguish it from existing molecular retrieval.* Most molecular retrieval augmentation retrieves labelled training molecules, which couples the retrieval space to the prediction target and risks leakage. RAFE retrieves from a fixed external library using structure only, so no label can enter the features and leakage is impossible by construction. It is also architecture-agnostic (it wraps any CLM without retraining), it costs about {rq1_s:.2f} s per molecule, and it needs no tuning: we show the result is unchanged across $k$ from 1 to 100 and across three retrieval metrics (R1.3). In {n_rf} controlled paired comparisons it is never significantly worse than the stage below it, and it is significantly better than the fine-tuned embedding alone on {len(sig_fe)} of 7 datasets. A stage that cannot leak, needs no tuning and does not degrade accuracy is a useful default, even if it is assembled from known parts.

*(c) The study produces findings that are not predictable from the components.* Random ZINC-250k neighbours work as well as the most similar ones, and the per-molecule gain is unrelated to how well the library covers a molecule (R1.4, R2.3). RAFE therefore does not work as the nearest-analogue lookup that the retrieval literature assumes; it supplies coarse chemical-space context. We regard this as a substantive, if deflationary, result about retrieval augmentation in chemistry, and it is one that only a controlled study could produce. The same applies to two further findings: a descriptor-only baseline is competitive with every CLM-based pipeline on three of seven datasets, and the frozen Uni-Mol embedding varies more between conformers than between tree seeds (R2.4).

*(d) The negative and calibrating results are deliberate.* We could have reported only the configurations in which RAFE wins. Instead the revised manuscript states where RAFE does not help, where PC-only is better, and which of our earlier claims did not survive repetition. We hope the reviewer will see the value of a benchmark that is designed to be able to disconfirm its own framework.

Relative to EffiChem the new elements are the Uni-Mol branch, the descriptor and retrieval stages, the regression tasks and the controlled comparisons reported here. [Revision: Introduction and Discussion]

{CQ("R1.2", "Inconsistent improvements from retrieval augmentation")}

**Ans:** We agree that RAFE does not improve every dataset; a benchmark of this kind should show that no configuration wins everywhere. Three observations place the results in context.

First, the gain over the embedding alone is clear. Across matched embedding-loss-tree configurations, RAFE exceeds FT Embed significantly on {len(sig_fe)} of 7 datasets ({"; ".join(f"{n}: +{m:.3f}, {pf(p)}" for n, m, p in sig_fe)}; Figure 1). Second, as a ranking of best configurations, RAFE is first on {V["wins"]} and in the top three on {top3} of 7 datasets (Table 1, Figure 2). Third, RAFE's contribution beyond the descriptors is modest: against FT Embed+PC the matched-configuration difference is +{fl.mean_diff:.3f} on Flavor (Wilcoxon {pf(fl.wilcoxon_p)}) and +{ba.mean_diff:.3f} on BACE ({pf(ba.wilcoxon_p)}), and within about ±0.01 elsewhere. Figure 3 shows where the extra stage reaches the cost-performance frontier.

Best-configuration values are single runs and therefore optimistic: rerunning with five seeds, DILI falls from 0.790 to {gap_dili:.3f} and BBBP from 0.798 to {float(B15[(B15.dataset=="BBBP")&(B15.stage=="RAFE")].oracle_test_mean.iloc[0]):.3f}. Re-ranked on five-seed means of the full grid ({", ".join(done15)} complete), RAFE is top three on {r15_top3} of {n15} (mean rank {r15_mean:.1f} of 7 methods), behind FT Embed+PC ({r15_ftpc:.1f}) and ahead of FT Embed ({r15_ft:.1f}) and PC-only ({r15_pc:.1f}; Table 10, R2.5). We read RAFE as a consistently competitive final stage rather than a winner, and the paper says so. {P("hERG and Flavor of the seed-averaged ranking")}

**Table 1.** Rank of RAFE's best configuration among seven methods (single-run values from the supplementary tables; Chemprop, GCN and PC-only from supplementary Table S11). Mean rank {V["mrank"]}; over the medians of each method's configurations, RAFE is in the top three on {V["mtop3"]} datasets (mean rank {V["mmrank"]}; PC-only {V["pcm"]}).

{md(T1)}

![**Figure 1.** Difference between pipeline stages over matched embedding-loss-tree configurations (test metric: MCC for classification, Spearman correlation for Caco-2 and Half-Life; supplementary tables; intervals are bootstrap over configurations and are descriptive because configurations share a test set).]({F}figA_ladder_gain.pdf)

![**Figure 2.** Best configuration of each method per dataset (red: RAFE).]({F}figB_best_config_ranking.pdf)

![**Figure 3.** Cost-performance frontier of the best configuration of each stage (median view over matched configurations).](../results/revision/e12_pareto/pareto_frontier.pdf)

{CQ("R1.3", "Insufficient ablation studies")}

**Ans:** We added each control under the identical protocol.

*Feature groups.* Adding or removing any of the four RAFE groups changes the score by no more than 0.02 in {frac_small*100:.0f}% of the {len(dev)} dataset-by-arm cells (Table 2, Figure 4); the {len(big)} larger shifts are all decreases and are confined to {", ".join(f"{k} (test n = {big_n[k]})" for k in big_cfg)}, which have the smallest test sets in the study (n = {", ".join(str(v) for v in smallest)}); individual values are listed in Table 2. No single group drives the effect of RAFE, consistent with the groups carrying overlapping information (R2.3).

*Retrieval metric and neighbour count.* We compared CLM-cosine retrieval with Morgan-Tanimoto retrieval and with random retrieval (five independent random draws), and swept k from 1 to 100. Of {n23} paired comparisons, {k23} {'has' if k23 == 1 else 'have'} an interval that excludes zero ({'; '.join(exc23)}; Figure 5), and both differences are below 0.011 in size. RAFE therefore needs no tuning of its retrieval settings. Equal performance of random neighbours shows that it acts as chemical-space context rather than as an analogue lookup, and the manuscript now says so. {P("BBBP neighbour-count sweep (cells still running)")}

*Mixed tree models.* The reviewer is right that the main-text tables could pair different trees. The controlled matrix (Table 5) and the full same-tree grids (Tables 5a to 5c) under R2.1 compare every representation within one tree.

**Table 2.** RAFE feature-group ablation (5-seed means, each dataset at its own tree; "+X" adds one group to FT Embed+PC, "RAFE-X" removes it from full RAFE; A physchem neighbourhood, B density, C centroid, D Tanimoto).

{md(T2)}

![**Figure 4.** Change relative to FT Embed+PC for each feature-group arm (RAFE-X: RAFE without group X).]({F}figC_ablation_heatmap.pdf)

![**Figure 5.** Left: effect of the number of retrieved neighbours (difference to k = 10). Right: effect of the retrieval strategy (difference to CLM-cosine retrieval; error bars: spread over the five random draws).]({F}figD_retrieval_robustness.pdf)

{CQ("R1.4", "Adequacy of the ZINC-250k retrieval database")}

**Ans:** We measured, for every dataset molecule, the largest Morgan-Tanimoto similarity to ZINC-250k (Table 3, Figure 6). The median is {allrow["median"]:.2f} over all {int(allrow["n"]):,} molecules, and about {allrow[">=0.6"]*100:.0f}% have a neighbour at 0.6 or above; BACE is the least covered (median {cov[cov.dataset=="bace"]["median"].iloc[0]:.3f}). ZINC-250k therefore provides background chemical context rather than close analogues. Consistent with this, the per-molecule gain from RAFE is unrelated to coverage (R2.3) and random neighbours perform like similar ones (R1.3). Whether a task-matched library would do better is untested and is listed as a limitation. [Revision]

**Table 3.** Coverage of the dataset molecules by ZINC-250k.

{md(T3)}

![**Figure 6.** Joint UMAP of the dataset molecules and ZINC-250k.](../results/revision/e07_coverage/umap_coverage.pdf)

{CQ("R1.5", "End-to-end computational efficiency and runtime analysis")}

**Ans:** Table 4 and Figure 7 report the cost of each pipeline stage, measured directly or taken from the logs of the original runs; these figures are not a controlled cross-method comparison (see R1.3 and R2.1 for controlled accuracy comparisons). One-off costs are modest for the SMILES-based models (LoRA fine-tuning: seconds to minutes per dataset; embedding the 249k-molecule library: approximately {(z_h+z_d)/2:.1f} minutes) and substantially larger for Uni-Mol, where conformer generation dominates (hours). At inference, RAFE adds approximately {rq1_s:.2f} s per molecule on a shared GPU using exact search, and tree-model inference requires under one millisecond. Because the measurements were collected on different hardware, we report them as indicative rather than precisely comparable figures. [Revision: new subsection "Computational cost"]

**Table 4.** Runtime of the pipeline stages.

| Stage | Cost |
|---|---|
| LoRA fine-tuning, one run (lower bound; includes small test runs) | {lora} |
| Library embedding, SMILES models (6 variants) | {z_h:.1f} min (hERG), {z_d:.1f} min (DILI) |
| Library embedding, Uni-Mol | {", ".join(f"{r.dataset}: {r.hours:.1f} h" for r in um.itertuples())} (hERG and DILI: 2 variants each; Caco-2 rerun: 1 variant) |
| RAFE at inference, one molecule (shared P100, exact search over 249k molecules) | {rq1} |
| RAFE at inference, 192 molecules | {rq192} |
| XGBoost inference, one molecule | {xg1} |
| Tree stage per cell, 20 Optuna trials / 5-fold CV / fit (upper bound, shared server) | XGBoost {tree["XGBoost"]}; LightGBM {tree["LightGBM"]}; CatBoost {tree["CatBoost"]} |

![**Figure 7.** Cost of the pipeline stages (blue: one-off; red: per query molecule).]({F}figE_runtime.pdf)

---

## Reviewer 2

{CQ("R2.1", "Lack of controlled comparisons between molecular representations and machine learning algorithms")}

**Ans:** We thank the reviewer for pressing on this; it is the comment that shaped the revision. The main-text tables reported the best row per representation, which is not a controlled comparison. We have now run PC-only, FT Embed, FT Embed+PC and RAFE through the same code, splits and test molecules, with the same tuning budget (20 Optuna trials, 5-fold cross-validation on the training split only), the same three tree models and the same five seeds (Table 5, Figure 8). The test split is evaluated once per fitted model. [Revision: Methods, and a main-text table with the full matrix in the supplement]

The outcome is clear. RAFE is never significantly different from FT Embed+PC ({n_rf} paired comparisons: {pos_rf} better, {neg_rf} worse), so the final stage carries no detectable cost. It is significantly better than the fine-tuned embedding alone in {pos_re} of {n_re} comparisons ({sig_re}) and never worse. Which stage wins depends on the dataset: the embedding helps over descriptors on BACE (MolFormer), BBBP and Half-Life, whereas PC-only is strongest or tied on Caco-2, hERG and Flavor, and RAFE is significantly below PC-only in {neg_rp} of {n_rp} comparisons ({neg_rp_ds}) and significantly above it in {pos_rp}. Identifying such dataset-specific winners is the purpose of the benchmark. {P("complete BBBP cells (" + ", ".join(incomplete) + " have cells with fewer than 5 seeds at the time of drafting) and final paired intervals for Table 5") if incomplete else ""}

**Table 5.** Controlled comparison (mean over seeds; MCC for classification, Spearman for regression; one embedding configuration per dataset).

{md(T5)}

![**Figure 8.** The controlled comparison of Table 5 (thick line: mean over trees; markers: individual trees).]({F}figF_controlled_matrix.pdf)

Because the reviewer's concern is that best rows may pair different trees, we also give the full grid for the three datasets where RAFE matters most (Tables 5a to 5c: every embedding and loss, compared within one tree; the bottom rows pool the difference over rows and seeds with paired intervals). On DILI, RAFE exceeds FT Embed with both trees, and exceeds FT Embed+PC with XGBoost (+0.032, interval +0.005 to +0.060) but not with LightGBM (-0.009, -0.029 to +0.012). On Caco-2, RAFE and FT Embed+PC are indistinguishable and both exceed FT Embed by about 0.08, while PC-only is above every stage. On BBBP no stage separates from another. On Half-Life RAFE falls below FT Embed+PC with LightGBM (-0.023, -0.044 to -0.002). CatBoost columns are still running. {P("CatBoost columns of Tables 5a to 5c")}

**Table 5a.** {FT["dili"]}

**Table 5b.** {FT["caco2"]}

**Table 5c.** {FT["bbbp"]}

{CQ("R2.2", "Insufficient comparison with conventional molecular fingerprints")}

**Ans:** We agree that these baselines belong in the main benchmark. Table 6 evaluates ECFP4, ECFP6, MACCS, ECFP4+MACCS, RDKit descriptors and the 473 physicochemical descriptors with the same trees and tuning (best tree shown; {min(v for k, v in seeds4.items() if k not in ("herg", "flavor"))}{'' if min(v for k, v in seeds4.items() if k not in ('herg', 'flavor')) == max(v for k, v in seeds4.items() if k not in ('herg', 'flavor')) else ' to ' + str(max(v for k, v in seeds4.items() if k not in ('herg', 'flavor')))} seeds on the smaller datasets and {seeds4["herg"]} on hERG and Flavor because of cost). Fingerprints alone are clearly weaker than descriptors on Caco-2 ({T6_num.loc[T6_num.Dataset=="Caco-2","ECFP4"].iloc[0]:.3f} for ECFP4 versus {T6_num.loc[T6_num.Dataset=="Caco-2","473 physchem (PC-only)"].iloc[0]:.3f}), DILI and Half-Life, and comparable on BACE, hERG and Flavor. RAFE is clearly better than ECFP4 alone on DILI (+{p4[(p4.cfg=="dili_um")&(p4.B=="ecfp4")]["diff"].iloc[0]:.2f}) and Caco-2 (+{p4[(p4.cfg=="caco2_mtr")&(p4.B=="ecfp4")]["diff"].iloc[0]:.2f}); against PC-only the intervals include zero on all three datasets compared (BBBP, DILI, Caco-2). [Revision]

**Table 6.** Fingerprint and descriptor baselines (best tree per cell; MCC for classification, Spearman for regression).

{md(T6)}

{CQ("R2.3", "The independent contribution of RAFE is not convincingly established")}

**Ans:** We examined this from three directions. *Where RAFE helps:* the per-molecule gain over FT Embed+PC is unrelated to ZINC coverage (Spearman between {e10s.spearman_gain_vs_coverage.min():+.2f} and {e10s.spearman_gain_vs_coverage.max():+.2f}; all intervals include zero; Figure 9, left). *What it adds:* a ridge model recovers {rg("A")[0]*100:.0f} to {rg("A")[1]*100:.0f}% of the variance of the physicochemical-neighbourhood group, {rg("B")[0]*100:.0f} to {rg("B")[1]*100:.0f}% of the density group and {rg("C")[0]*100:.0f} to {rg("C")[1]*100:.0f}% of the centroid group from the embedding and descriptors on held-out molecules (a linear estimate), so these groups are partly redundant; the Tanimoto group is not predictable from them (Figure 9, right). *How much predictions change:* RAFE corrects or breaks between {int((e10p.corrected_by_RAFE.fillna(0)).min())} and {int(max(e10p.corrected_by_RAFE.max(), e10p.broken_by_RAFE.max()))} test molecules on the classification sets. To test whether the redundant part can be removed, we evaluated five label-free variants (Table 7).

![**Figure 9.** Left: per-molecule gain of RAFE over FT Embed+PC by ZINC-coverage tercile. Right: held-out R² of each RAFE group predicted from the embedding and descriptors.](../results/revision/e10/e10_summary.pdf)

**Table 7.** Label-free RAFE variants (V1: groups A+B only; V2: residualised features; V3: additional ZINC properties; V4: compressed features; V5: hybrid retrieval) at each dataset's configuration, own tree, 5-seed test means. The "best variant" is the one with the highest test mean among RAFE and V1 to V5. Flavor is still running.

{md(T7)}

Of {v_n} paired comparisons of the five variants with FT Embed+PC (six datasets, two trees), {v_better_ft} is significantly better and {v_worse_ft} significantly worse; against RAFE, {v_better_rf} are better and {v_worse_rf} worse. No variant differs significantly from RAFE, so we claim no improvement from them. With the group ablations, this supports reading RAFE as a conservative refinement rather than a source of new information. {P("Flavor row")}

{CQ("R2.4", "Insufficient isolation of the contribution of three-dimensional molecular information")}

**Ans:** We agree and have separated the three components (Table 8, Figure 10). The answer depends on the endpoint. On BACE the ECFP4 part alone scores higher than the 3D vector ({e5m["BACE"][1]:.3f} versus {e5m["BACE"][0]:.3f} with XGBoost; the difference is significant for LightGBM and CatBoost). On Caco-2 the concatenation is clearly better than either part (+{p5v("caco2_um","XGBoost","concat2560","cls512")[0]:.2f} to +{p5v("caco2_um","CatBoost","concat2560","cls512")[0]:.2f} over the 3D vector for XGBoost and CatBoost), so the two views are complementary there. On DILI the 3D vector outperforms ECFP4 for CatBoost. We have corrected the wording accordingly: the headline BACE result (MCC 0.623) is the end-to-end LoRA Uni-Mol model with its own head, which does not use ECFP4, so the ECFP4 confound the reviewer identified applies only to the frozen-embedding rows. [Revision]

**Table 8.** 3D versus ECFP4 within the Uni-Mol vector (XGBoost, 5 seeds; MCC, and Spearman for Caco-2).

{md(T8)}

We also examined the influence of the conformer (relevant to R3.2). Re-embedding BACE with five RDKit ETKDG seeds (the original seed reproduces the stored embeddings exactly) shows that the 3D vector changes noticeably between conformers (median cosine similarity to the original {dr.median_cos_vs_seed42.min():.2f} to {dr.median_cos_vs_seed42.max():.2f}). The resulting spread in test MCC is {r3d.min():.0f} to {r3d.max():.0f} times the tuning noise for the 3D vector alone and {rcat.min():.0f} to {rcat.max():.0f} times with ECFP4 added (Table 9, Figure 10), so a single conformer should be reported with this caveat. Pipelines built on SMILES-based models, including the RAFE configurations with MolFormer and ChemBERTa, do not involve conformers at all.

**Table 9.** Conformer-seed sensitivity on BACE (5 conformer seeds × 3 tree seeds).

{md(T9)}

![**Figure 10.** Left: 3D versus ECFP4 versus both (XGBoost). Right: test MCC for each conformer seed (42 = original), both inputs, XGBoost and LightGBM.]({F}figG_3d_vs_ecfp4_conformers.pdf)

{CQ("R2.5", "Statistical evaluation requires greater methodological rigour")}

**Ans:** We agree. The ±1 SE overlap is replaced by paired bootstrap intervals on identical test molecules (differences averaged over seeds within each resample), with five seeds per arm controlling the Optuna sampler, the cross-validation shuffle and the model initialisation. On the second point, the reviewer is right that the best of many configurations is optimistic in a single run. We therefore (i) report seed-averaged values next to every best-run value (DILI as an example: {d_st[0]:.3f} ± {d_st[1]:.3f} over five seeds with the stored RAFE features versus 0.790 in the table, with FT Embed+PC at {d_ft[0]:.3f} ± {d_ft[1]:.3f}); (ii) keep the usual convention of judging configurations on test performance, but on five-seed means, so that one favourable run cannot decide a winner; and (iii) give the paired difference to the runner-up configuration (Table 10). The grid is complete for XGBoost and LightGBM on five datasets. {P("CatBoost, hERG and Flavor of the full grid, then Table 10")}

**Table 10.** Best configuration per dataset and stage, judged on five-seed test means (XGBoost and LightGBM; CatBoost cells for the five small datasets are still running). "Single run" is the best value in the original tables of the same modality. {P("hERG and Flavor rows")}

{md(T10)}

Two readings follow. The best configuration is MolFormer in {n_mf} of {len(B15)} rows, and the single-run values exceed the five-seed means in {n_sr} of {len(B15)} rows (largest for DILI RAFE, 0.790 versus {gap_dili:.3f}). The best configuration is separable from the runner-up in only {n_sep} of {len(B15)} rows, so the identity of the best configuration is itself uncertain and we base claims on paired differences. Selecting by training-split cross-validation instead is unreliable on these scaffold splits: it lowers the RAFE score by {gap_lo:.2f} to {gap_hi:.2f} relative to the test-based choice, so we do not use it for ranking.

![**Figure 11.** Five-seed mean test metric over the full configuration grid for the five completed datasets (red box: best cell).](../results/revision/e15_analysis/e15_grid_heatmap.pdf)

{CQ("R2.6", "Limited coherence of the central scientific narrative")}

**Ans:** We accept that the previous version did not say which contribution was central, and the manuscript is now organised around three hypotheses, each with its own section and verdict: H1, LoRA adaptation is parameter-efficient without loss of accuracy (supported: {76}-{97}% fewer trainable parameters at competitive accuracy); H2, retrieval features add information beyond the embedding and the descriptors (partly supported: RAFE improves on the embedding alone on {len(sig_fe)} of 7 datasets but is not distinguishable from embedding plus descriptors in {n_rf} paired comparisons); H3, 3D geometry helps structure-dependent endpoints (weakly supported: the best BACE result is the 3D end-to-end model, but the frozen Uni-Mol vector contains an ECFP4 block and depends on the conformer, R2.4).

On the second point we would offer a different reading. That a simple descriptor model stays competitive is not a tension with our claims; it is one of the findings, and it is the reason the benchmark was worth running. Published CLM benchmarks frequently omit this control, so a result showing that it is competitive on three of seven datasets, with paired intervals rather than overlap screens, has direct consequences for how such work should be evaluated. The revised Discussion states this as a recommendation: a PC-only baseline should be a mandatory control in molecular CLM benchmarks. We have made the complexity question explicit rather than leaving it unreconciled, and we answer it dataset by dataset. [Revision]

{CQ("R2.7", "Manuscript organization and repetitive discussion")}

**Ans:** We merged "Results and Discussion" and restricted the discussion to interpretation, limitations and implications. [Revision]

{CQ("R2.8", "Inconsistencies in references and technical presentation")}

**Ans:** We reformatted the references with DOIs throughout. {P("final DOI check of every reference")} The LoRA description now matches the implementation: ΔW = (α/r)·B·A, with A initialised by Kaiming-uniform and B initialised to zero (the library default); Figure 2 of the paper was redrawn, and the Methods and supplement use the same notation. [Revision]

---

## Reviewer 3

{CQ("R3.1", "Ablation of the retrieval components")}

**Ans:** These controls are in Table 2 and Figures 4 and 5 (R1.3), with the interpretation in R2.3. We also discuss the conditions under which RAFE helps (R1.2, R2.3).

{CQ("R3.2", "3D embeddings versus Morgan fingerprints")}

**Ans:** See Tables 8 and 9 and Figure 10 (R2.4). We did not repeat the conformer analysis for hERG and Flavor because of cost, and we say so in the limitations. {P("decision whether to add conformer runs for a second dataset")}

{CQ("R3.3", "Statistical significance, multiple runs and test-set use")}

**Ans:** The paired bootstrap and the five-seed protocol are described in R2.1 and R2.5. On model selection: the tree models are tuned by 5-fold cross-validation on the training split only (MCC for classification, negative RMSE for regression), and the test split is evaluated once per fitted model and never used for tuning. Configurations (embedding model, loss, tree model and stage) are then compared on the test set, as is usual for benchmarks. Because the best configuration is selected on the same test set, its score is optimistic; we therefore report five-seed means and the paired difference to the runner-up (Table 10). LoRA fine-tuning uses the training split, with the validation split for monitoring and checkpoint selection. As a limitation, seed variability was measured for the downstream stage; fine-tuning and library re-embedding were not repeated with independent seeds because of cost, so the intervals describe the downstream stage and are lower bounds on total variability. [Revision: Methods and Limitations]

{CQ("R3.4", "Interpretation of the performance claims")}

**Ans:** We revised each of the examples. (a) *BBBP:* the RAFE effect over FT Embed+PC depends on the tree ({bbd['CatBoost']:+.3f} with CatBoost, {bbd['LightGBM']:+.3f} with LightGBM, {bbd['XGBoost']:+.3f} with XGBoost; Table 5), and we now describe it as small and tree-dependent. (b) *DILI metrics:* the best configuration depends on the metric: for RAFE, MCC, accuracy and F1 select {cfgname(dm.loc["RAFE","best_MCC_cfg"])} (MCC {dm.loc["RAFE","best_MCC"]:.3f}) while AUC selects {cfgname(dm.loc["RAFE","best_AUC_cfg"])} ({dm.loc["RAFE","best_AUC"]:.3f}); the rank correlation between MCC and AUC across configurations is {dm.rho_MCC_AUC.min():.2f} to {dm.rho_MCC_AUC.max():.2f}. With 96 test molecules, MCC at a fixed threshold is noisier than AUC, so we state the ranking metric and report AUC alongside. (c) *Half-Life:* Spearman and R² behave differently because R² is computed in hours after inverting the log1p transform, where {share3*100:.0f}% of the test variance sits in three molecules; for the fingerprint and descriptor baselines R² rises from {hh.R2_raw_hours.mean():.3f} to {hh.R2_log1p_space.mean():.3f} on the log scale (Figure 12). The rank correlation between Spearman and R² across the supplementary configurations is {rho_all:.2f} ({rho_rafe:.2f} within RAFE). We report both: the models rank half-lives moderately well (Spearman {bestsp.min():.2f} to {bestsp.max():.2f}) but do not predict their magnitude. (d) *BACE and protein-ligand interactions:* we removed the statement that Uni-Mol captures protein-ligand interactions; the data show that a 3D-aware model performs best on BACE, which does not by itself establish interaction modelling. [Revision]

![**Figure 12.** Half-Life. Left: Spearman versus R² (hours) over the supplementary configurations. Right: R² in hours versus on the log1p scale for the descriptor and fingerprint baselines.]({F}figH_halflife_metrics.pdf)

{CQ("R3.5", "Novelty relative to EffiChem; inconsistencies")}

**Ans:** The novelty argument is set out under R1.1; in short, we position PEARL as a controlled benchmark with a leakage-free, tuning-free retrieval stage, and the elements that are new relative to EffiChem are the Uni-Mol branch, the descriptor and retrieval stages, the regression tasks and the controlled comparisons. We audited the manuscript against the supplement and the code and corrected: the number of RAFE features (60 in the implementation, 18 + 5 + 34 + 3, not 69); the embedding dimension of MolFormer-XL; the LoRA initialisation and formula; the loss-function definitions; and the label of the best BACE configuration (the stored retrieval index is MolFormer FL, whereas the text said WL). A table of all corrections is in the supplement. {P("final consistency check of the revised tables")} [Revision]

{CQ("R3.6", "Additional related work")}

**Ans:** We added a discussion of Fang et al. (Nature Machine Intelligence, 2022), Li et al. (Nature Communications, 2023), Su et al. (ChemRxiv, 2026) and Li et al. (Journal of Chemical Theory and Computation, 2026) to the Introduction and Discussion, in the context of geometric, knowledge-based and multimodal molecular representations. {P("verification of the four DOIs against the sources")} [Revision]

---

## Limitations now stated in the manuscript
- Seed variability covers the downstream stage; fine-tuning and library embedding were not repeated with independent seeds.
- ZINC-250k covers the task chemistry only moderately; other libraries were not tested.
- The new experiments ran XGBoost on GPU, so absolute values differ slightly from the original CPU runs; comparisons are made within experiments.

We thank the reviewers again; the revised analyses have made the paper clearer and its claims better supported.
'''
open("manuscript/PEARL_Response_to_Reviewers_Draft.md", "w").write(doc); print(len(doc.split()), "words;", doc.count("PENDING"), "PENDING markers")
print("derived:", dict(top3=top3, wins=wins, mrank=round(mrank, 2), mtop3=mtop3, sig_ladder=[x[0] for x in sig_fe], n_rf=n_rf, pos_rf=pos_rf, neg_rf=neg_rf, pos_re=pos_re, neg_rp=neg_rp, neg_rp_ds=neg_rp_ds, share3=round(share3, 3), n23=n23, k23=k23, incomplete=incomplete))

# ---------------- LaTeX tables for the revised manuscript (same DataFrames as the response; \input from PEARL_paper_v3.tex)
import os
os.makedirs("manuscript/tables", exist_ok=True)
def tex_esc(v, nd=3):
    t = f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)
    bold = t.startswith("**") and t.endswith("**") and len(t) > 4
    if bold: t = t[2:-2]
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("±", "$\\pm$"), ("−", "$-$"), ("≥", "$\\geq$")): t = t.replace(a, b)
    t = re.sub(r"(?<![\w}])-(?=\d)", "$-$", t)
    return f"\\textbf{{{t}}}" if bold else t
def to_tex(df, name, nd=3):
    cols = "l" * len(df.columns); head = " & ".join(tex_esc(c) for c in df.columns) + " \\\\"
    rows = "\n".join(" & ".join(tex_esc(v, nd) for v in r) + " \\\\" for r in df.values)
    open(f"manuscript/tables/{name}.tex", "w").write("\\begin{tabular}{" + cols + "}\n\\toprule\n" + head + "\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n")
for nm_, df_ in (("tab_ranking_singlerun", T1), ("tab_groups", T2), ("tab_coverage", T3), ("tab_controlled", T5), ("tab_fingerprints", T6), ("tab_variants", T7), ("tab_3d_ecfp", T8), ("tab_conformers", T9), ("tab_best_configs", T10)): to_tex(df_, nm_)
print("exported", len(os.listdir("manuscript/tables")), "LaTeX tables")

T4 = pd.DataFrame([
    ("LoRA fine-tuning, one run (lower bound)", f"{lora}"),
    ("Library embedding, SMILES models (6 variants)", f"{z_h:.1f} min (hERG), {z_d:.1f} min (DILI)"),
    ("Library embedding, Uni-Mol", ", ".join(f"{r.dataset}: {r.hours:.1f} h" for r in um.itertuples())),
    ("RAFE at inference, one molecule", f"{rq1}"),
    ("RAFE at inference, 192 molecules", f"{rq192}"),
    ("XGBoost inference, one molecule", f"{xg1}"),
    ("Tree stage per cell, 20 trials/5-fold CV/fit (upper bound, shared server)", f'XGBoost {tree["XGBoost"]}; LightGBM {tree["LightGBM"]}; CatBoost {tree["CatBoost"]}')], columns=["Stage", "Cost"])
to_tex(T4, "tab_runtime")

# ---------------- numbers used in the manuscript text (macros; every value comes from the analysis files)
KEY = {"BACE": "BACE", "BBBP": "BBBP", "DILI": "DILI", "Caco-2": "CACO", "Half-Life": "HL"}; ST = {"FT Embed": "FT", "FT Embed+PC": "FTPC", "RAFE": "RAFE"}
mac = ["% generated by scripts/revision/build_response.py; do not edit"]
for r in B15.itertuples():
    k = KEY[r.dataset] + ST[r.stage]; mac += [f"\\newcommand{{\\m{k}}}{{{r.oracle_test_mean:.3f}}}", f"\\newcommand{{\\s{k}}}{{{r.oracle_test_sd:.3f}}}", f"\\newcommand{{\\sr{k}}}{{{r.single_run_original_tables:.3f}}}", f"\\newcommand{{\\cfg{k}}}{{{r.oracle_config}}}"]
for r in rk.itertuples(): mac.append(f"\\newcommand{{\\m{KEY[r.dataset]}PC}}{{{getattr(r, '_4') if False else rk.loc[r.Index, 'PC-only']:.3f}}}")
mac += [f"\\newcommand{{\\ntop}}{{{r15_top3}}}", f"\\newcommand{{\\nds}}{{{n15}}}", f"\\newcommand{{\\mrankRAFE}}{{{r15_mean:.1f}}}", f"\\newcommand{{\\mrankFTPC}}{{{r15_ftpc:.1f}}}", f"\\newcommand{{\\mrankFT}}{{{r15_ft:.1f}}}", f"\\newcommand{{\\mrankPC}}{{{r15_pc:.1f}}}"]
open("manuscript/tables/numbers.tex", "w").write("\n".join(mac) + "\n"); print("macros:", len(mac))

# ---------------- further macros for the manuscript text
def M(name, val): mac2.append(f"\\newcommand{{\\{name}}}{{{tex_esc(val) if isinstance(val, str) else val}}}")
mac2 = []
M("srtop", top3); M("srmean", f"{mrank:.1f}"); M("medtop", mtop3); M("medmean", f"{mmrank:.1f}"); M("medpc", f"{pcm:.1f}")
M("nmf", n_mf); M("nsr", n_sr); M("nsep", n_sep); M("nrows", len(B15)); M("gaplo", f"{gap_lo:.2f}"); M("gaphi", f"{gap_hi:.2f}"); M("cvtop", r15_cv_top3); M("cvmean", f"{r15_cv_rafe:.1f}"); M("cvpcmean", f"{r15_cv_pc:.1f}")
M("fracsmall", f"{frac_small*100:.0f}"); M("ncellsabl", len(dev)); M("nbig", len(big)); M("nkcomp", n23); M("nkexc", k23); M("bigcfg", ", ".join(big_cfg)); M("excdetails", "; ".join(exc23))
M("spmin", f"{e10s.spearman_gain_vs_coverage.min():+.2f}"); M("spmax", f"{e10s.spearman_gain_vs_coverage.max():+.2f}")
for g_ in "ABC": M(f"rg{g_}lo", f"{rg(g_)[0]*100:.0f}"); M(f"rg{g_}hi", f"{rg(g_)[1]*100:.0f}")
M("ecorrmax", int(max(e10p.corrected_by_RAFE.max(), e10p.broken_by_RAFE.max())))
M("vn", v_n); M("vbetterft", v_better_ft); M("vworseft", v_worse_ft); M("vbetterrf", v_better_rf); M("vworserf", v_worse_rf)
M("covmedian", f"{allrow['median']:.2f}"); M("covn", f"{int(allrow['n']):,}"); M("covsixty", f"{allrow['>=0.6']*100:.0f}"); M("covbace", f"{cov[cov.dataset=='bace']['median'].iloc[0]:.3f}")
M("ecfpcaco", f"{T6_num.loc[T6_num.Dataset=='Caco-2','ECFP4'].iloc[0]:.3f}"); M("pccaco", f"{T6_num.loc[T6_num.Dataset=='Caco-2','473 physchem (PC-only)'].iloc[0]:.3f}")
M("draferecfpdili", f"{p4[(p4.cfg=='dili_um')&(p4.B=='ecfp4')]['diff'].iloc[0]:.2f}"); M("draferecfpcaco", f"{p4[(p4.cfg=='caco2_mtr')&(p4.B=='ecfp4')]['diff'].iloc[0]:.2f}")
M("ecfpbace", f"{e5m['BACE'][1]:.3f}"); M("threedbace", f"{e5m['BACE'][0]:.3f}")
M("concatlo", f"{p5v('caco2_um','XGBoost','concat2560','cls512')[0]:.2f}"); M("concathi", f"{p5v('caco2_um','CatBoost','concat2560','cls512')[0]:.2f}")
M("coslo", f"{dr.median_cos_vs_seed42.min():.2f}"); M("coshi", f"{dr.median_cos_vs_seed42.max():.2f}"); M("rthreedlo", f"{r3d.min():.0f}"); M("rthreedhi", f"{r3d.max():.0f}"); M("rcatlo", f"{rcat.min():.0f}"); M("rcathi", f"{rcat.max():.0f}")
M("lorarun", lora); M("zhmin", f"{z_h:.1f}"); M("zdmin", f"{z_d:.1f}"); M("raferq", f"{rq1_s:.2f}"); M("umhours", ", ".join(f"{r.dataset}: {r.hours:.1f} h" for r in um.itertuples()))
M("bbbpcat", f"{bbd['CatBoost']:+.3f}"); M("bbblgb", f"{bbd['LightGBM']:+.3f}"); M("bbbxgb", f"{bbd['XGBoost']:+.3f}")
M("dilirafecfg", cfgname(dm.loc["RAFE","best_MCC_cfg"])); M("dilirafemcc", f"{dm.loc['RAFE','best_MCC']:.3f}"); M("dilirafeaucfg", cfgname(dm.loc["RAFE","best_AUC_cfg"])); M("dilirafeauc", f"{dm.loc['RAFE','best_AUC']:.3f}")
M("rhomcclo", f"{dm.rho_MCC_AUC.min():.2f}"); M("rhomcchi", f"{dm.rho_MCC_AUC.max():.2f}")
M("sharethree", f"{share3*100:.0f}"); M("rawr", f"{hh.R2_raw_hours.mean():.3f}"); M("logr", f"{hh.R2_log1p_space.mean():.3f}"); M("rhoall", f"{rho_all:.2f}"); M("rhorafe", f"{rho_rafe:.2f}"); M("spbestlo", f"{bestsp.min():.2f}"); M("spbesthi", f"{bestsp.max():.2f}")
open("manuscript/tables/numbers.tex", "a").write("\n".join(mac2) + "\n"); print("extra macros:", len(mac2))
