"""E11 analysis: label-free RAFE variants. Per dataset configuration and tree: mean test score over 5 seeds for FT+PC, RAFE and V1-V5, the variant chosen by the
training-only CV score (cv_best, among RAFE and V1-V5), paired bootstrap (seed-pooled) of chosen variant and each variant versus FT+PC, RAFE and PC-only, and win counts.
Outputs (results/revision/e11_analysis/): variants_table.csv (own tree), comparisons.csv (all trees), wins.csv"""
import sys, json, glob, numpy as np, pandas as pd; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import e06b_paired_all as pa
from rev_common import RES, DS
from rafe_lib import CFG
OUT = RES / "e11_analysis"; OUT.mkdir(exist_ok=True); R = str(RES) + "/"
ARMS = ["base_FT+PC", "full_RAFE", "V1_AB", "V2_residual", "V3_wide", "V4_compressed", "V5_hybrid"]; VARS = ARMS[1:]; TREES = ["XGBoost", "LightGBM"]
OWN = {"bbbp_mf": "XGBoost", "dili_um": "XGBoost", "caco2_mtr": "XGBoost", "bace_mf": "LightGBM", "herg_mf": "LightGBM", "half_life_mf": "XGBoost", "flavor_mf": "LightGBM"}
NAME = {"bbbp_mf": "BBBP", "dili_um": "DILI", "caco2_mtr": "Caco-2", "bace_mf": "BACE", "herg_mf": "hERG", "flavor_mf": "Flavor", "half_life_mf": "Half-Life"}
rows = []
for f in glob.glob(R + "e11/*/*.json"):
    j = json.load(open(f)); rows.append(dict(cfg=f.split("/")[-2], arm=j["arm"], clf=j["clf"], seed=j["seed"], v=j.get("MCC", j.get("Spearman")), cv=j.get("cv_best", np.nan)))
d = pd.DataFrame(rows); g = d.groupby(["cfg", "arm", "clf"]).agg(test=("v", "mean"), sd=("v", "std"), cv=("cv", "mean"), n=("v", "count")).reset_index()
complete = [c for c in OWN if ((g.cfg == c) & (g.n == 5)).sum() == len(ARMS) * len(TREES)]
print("complete configurations:", [NAME[c] for c in complete], "| incomplete:", [NAME[c] for c in OWN if c not in complete])
def path(cfg, arm, clf, sd): return RES / ("e1" if arm == "PC_only" else "e11") / cfg / f"{arm}_{clf}_s{sd}"
def cmp_(cfg, A, B, clf):
    pr = [(path(cfg, A, clf, s), path(cfg, B, clf, s)) for s in range(42, 47)]; o, lo, hi, n = pa.cmp(CFG[cfg]["ds"], pr); return o, lo, hi
comp, tab = [], []
for c in complete:
    for clf in TREES:
        x = g[(g.cfg == c) & (g.clf == clf)].set_index("arm"); sel = x.loc[VARS].cv.idxmax()
        for A in VARS + ["selected"]:
            arm = sel if A == "selected" else A
            for B in ("base_FT+PC", "full_RAFE", "PC_only"):
                if arm == B: continue
                o, lo, hi = cmp_(c, arm, B, clf); comp.append(dict(dataset=NAME[c], tree=clf, A=A if A != "selected" else f"selected ({sel})", B=B, diff=o, ci_lo=lo, ci_hi=hi, own_tree=(clf == OWN[c])))
    x = g[(g.cfg == c) & (g.clf == OWN[c])].set_index("arm"); sel = x.loc[VARS].cv.idxmax()
    pcv = np.mean([json.load(open(p)).get("MCC", json.load(open(p)).get("Spearman")) for p in glob.glob(f"{R}e1/{c}/PC_only_{OWN[c]}_s4?.json")])
    tab.append({"Dataset": NAME[c], "Tree": OWN[c], "PC-only": pcv, **{a: x.loc[a].test for a in ARMS}, "Chosen by CV": sel, "Chosen: test score": x.loc[sel].test})
C = pd.DataFrame(comp); C.round(4).to_csv(OUT / "comparisons.csv", index=False); T = pd.DataFrame(tab); T.round(4).to_csv(OUT / "variants_table.csv", index=False)
wins = []
for A in VARS + ["selected"]:
    for B in ("base_FT+PC", "full_RAFE", "PC_only"):
        x = C[(C.A.str.startswith("selected") if A == "selected" else (C.A == A)) & (C.B == B)]
        if len(x): wins.append(dict(variant=A, versus=B, comparisons=len(x), point_estimate_better=int((x["diff"] > 0).sum()), ci_better=int((x.ci_lo > 0).sum()), ci_worse=int((x.ci_hi < 0).sum()), mean_diff=x["diff"].mean()))
W = pd.DataFrame(wins); W.round(4).to_csv(OUT / "wins.csv", index=False)
pd.set_option("display.width", 250); print(T.round(3).to_string(index=False)); print(); print(W.round(3).to_string(index=False))
