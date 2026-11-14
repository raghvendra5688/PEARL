"""E0/E9: parse the full configuration matrices in the supplementary tables (S2-S8) and quantify
selection effects: per dataset x modality distribution of test metric over all CLM x loss x tree
configurations; best-vs-median gap; and selection-free MATCHED comparisons (same CLM/loss/tree)
of RAFE vs FT Embed+PC, FT Embed+PC vs FT Embed, with Wilcoxon/sign tests."""
import re, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, binomtest
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import RES, REPO
OUT = RES / "e09_selection_audit"; OUT.mkdir(parents=True, exist_ok=True)
tex = open(REPO / "manuscript/PEARL_supplementary_v2.tex").read().splitlines()
names = {"s2_bbbp": "bbbp", "s3_flavor": "flavor", "s4_bace": "bace", "s5_herg": "herg", "s6_dili": "dili", "s7_caco2": "caco2", "s8_halflife": "half_life"}
rows = []; cur = None; header = None
def clean(x): return re.sub(r"\\textsuperscript\{.*?\}|\$\^\{?[+*]\}?\$|\\textbf\{|\}|\$|\\_|\\text\{|\^", "", x).strip()
for ln in tex:
    m = re.search(r"\\label\{tab:(s\d_\w+)\}", ln)
    if m and m.group(1) in names: cur = names[m.group(1)]; header = None; continue
    if cur is None: continue
    if ln.startswith("Modality &") and header is None:
        header = [clean(h) for h in ln.replace("\\\\", "").split("&")]; continue
    if header and "&" in ln and not ln.startswith(("\\multicolumn", "Modality")) and ln.rstrip().endswith("\\\\"):
        f = [clean(x) for x in ln.replace("\\\\", "").split("&")]
        if len(f) == len(header):
            d = dict(zip(header, f)); d["dataset"] = cur; rows.append(d)
D = pd.DataFrame(rows)
for c in D.columns:
    if c not in ("Modality", "CLM", "Loss", "ML", "dataset"): D[c] = pd.to_numeric(D[c].str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
D["Modality"] = D["Modality"].str.replace(r"\s+", " ", regex=True).str.replace("FT Embed+", "FT Embed").str.strip()
D["metric"] = np.where(D.get("MCC").notna(), D.get("MCC"), D.get("Spearman")) if "MCC" in D else D.get("Spearman")
D.to_csv(OUT / "all_configs_parsed.csv", index=False)
print(D.groupby(["dataset", "Modality"]).size().unstack().to_string())

g = []
for (ds, mod), d in D.dropna(subset=["metric"]).groupby(["dataset", "Modality"]):
    g.append(dict(dataset=ds, modality=mod, n_cfg=len(d), min=d.metric.min(), q25=d.metric.quantile(.25), median=d.metric.median(), mean=d.metric.mean(), max=d.metric.max(), best_minus_median=d.metric.max() - d.metric.median()))
G = pd.DataFrame(g); G.round(3).to_csv(OUT / "config_distribution.csv", index=False); print(G.round(3).to_string())

# matched pairs
M = D.dropna(subset=["metric"]).copy()
M["key"] = M.CLM + "|" + M.Loss.fillna("") + "|" + M.ML.fillna("")
res = []
for ds, d in M.groupby("dataset"):
    piv = d.pivot_table(index="key", columns="Modality", values="metric", aggfunc="first")
    for a, b in [("RAFE", "PC+FT"), ("PC+FT", "FT Embed"), ("RAFE", "FT Embed")]:
        a_ = a if a in piv else None; b_ = b if b in piv else None
        if b == "PC+FT" and "PC+FT" not in piv and "FT Embed+PC" in piv: b_ = "FT Embed+PC"
        if a == "PC+FT" and "PC+FT" not in piv and "FT Embed+PC" in piv: a_ = "FT Embed+PC"
        if not a_ or not b_: continue
        x = piv[[a_, b_]].dropna(); df = x[a_] - x[b_]
        if len(df) < 4: continue
        try: p = wilcoxon(df).pvalue
        except Exception: p = np.nan
        res.append(dict(dataset=ds, A=a, B=b, n_pairs=len(df), mean_diff=df.mean(), median_diff=df.median(), frac_A_better=(df > 0).mean(), wilcoxon_p=p))
R = pd.DataFrame(res); R.round(4).to_csv(OUT / "matched_pair_differences.csv", index=False); print(R.round(4).to_string())
