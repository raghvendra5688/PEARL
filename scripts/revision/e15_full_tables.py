"""Full same-tree comparison tables (mean +- SD over 5 seeds) from the E15 grid: rows = embedding x loss, column groups = tree, columns = FT Embed | FT Embed+PC | RAFE.
Every comparison is within one embedding, loss and tree. PC-only (embedding independent) comes from E1. Output: results/revision/e15_analysis/full_tables/<ds>.{md,tex}.
A tree with fewer than 5 seeds in any cell of a dataset (e.g. CatBoost still on the cluster) is shown as 'pending'."""
import json, glob, re, numpy as np, pandas as pd
from pathlib import Path
R = Path("results/revision"); O = R / "e15_analysis" / "full_tables"; O.mkdir(parents=True, exist_ok=True)
NAME = {"bace": "BACE", "bbbp": "BBBP", "herg": "hERG", "dili": "DILI", "caco2": "Caco-2", "half_life": "Half-Life", "flavor": "Flavor"}
PC = {"bace": "bace_mf", "bbbp": "bbbp_mf", "dili": "dili_um", "caco2": "caco2_mtr", "half_life": "half_life_mf", "herg": "herg_mf", "flavor": "flavor_mf"}
EMB = {"chemberta_77m_mlm": "ChemBERTa-MLM", "chemberta_77m_mtr": "ChemBERTa-MTR", "molformer_finetuned": "MolFormer", "unimol": "Uni-Mol"}
ARMS = [("FT_only", "FT Embed"), ("base_FT+PC", "FT Embed+PC"), ("full_RAFE", "RAFE")]; TREES = ["XGBoost", "LightGBM", "CatBoost"]
val = lambda j: j.get("MCC", j.get("Spearman"))
rows = []
for f in glob.glob(str(R / "e15/*/*.json")):
    j = json.load(open(f)); ds, emb = f.split("/")[-2].split("@"); rows.append(dict(ds=ds, emb=emb, arm=j["arm"], clf=j["clf"], v=val(j)))
d = pd.DataFrame(rows); a = d.groupby(["ds", "emb", "arm", "clf"]).v.agg(["mean", "std", "count"]).reset_index()
def lab(e): m = re.match(r"(chemberta_77m_mlm|chemberta_77m_mtr|molformer_finetuned|unimol)_(\w+)", e); return EMB[m.group(1)] + " " + m.group(2).upper().replace("HUBER", "Huber")
def pc(ds, clf):
    v = [val(json.load(open(f))) for f in glob.glob(str(R / f"e1/{PC[ds]}/PC_only_{clf}_s4?.json"))]; return (np.mean(v), np.std(v, ddof=1), len(v)) if v else None
summary = []
for ds in NAME:
    x = a[a.ds == ds]
    if x.empty: continue
    embs = sorted(set(x.emb), key=lab); trees = [t for t in TREES if (x.clf == t).any() and ((x[x.clf == t].groupby("emb").apply(lambda y: (y["count"] == 5).all() and len(y) == 3).all()) and x[x.clf == t].emb.nunique() == len(embs))]
    pend = [t for t in TREES if t not in trees]
    if not trees: print(NAME[ds], "skipped: no tree with a complete grid"); continue
    hdr = "| Embedding, loss | " + " | ".join(f"{t}: {l}" for t in trees for _, l in ARMS) + " |\n|---|" + "---|" * (3 * len(trees)) + "\n"
    body = []
    for e in embs:
        cells = []
        for t in trees:
            vals = {arm: x[(x.emb == e) & (x.arm == arm) & (x.clf == t)].iloc[0] for arm, _ in ARMS}; best = max(vals, key=lambda k: vals[k]["mean"])
            for arm, _ in ARMS:
                s = f"{vals[arm]['mean']:.3f} ± {vals[arm]['std']:.3f}"; cells.append(f"**{s}**" if arm == best else s)
        body.append(f"| {lab(e)} | " + " | ".join(cells) + " |")
    pcrow = "| *PC-only (473 descriptors)* | " + " | ".join(((f"{pc(ds, t)[0]:.3f} ± {pc(ds, t)[1]:.3f}" if pc(ds, t) else "n/a") if i == 0 else "") for t in trees for i, _ in enumerate(ARMS)) + " |"
    # per-tree mean over embeddings of the within-cell differences (descriptive)
    drow = []
    for t in trees:
        w = x[x.clf == t].pivot(index="emb", columns="arm", values="mean"); drow += [f"RAFE−FT: {(w.full_RAFE - w.FT_only).mean():+.3f}", f"RAFE−FT+PC: {(w.full_RAFE - w['base_FT+PC']).mean():+.3f}", f"{len(w)} rows"]
        for e_, r_ in w.iterrows(): summary.append(dict(dataset=NAME[ds], tree=t, emb=lab(e_), rafe_minus_ft=r_.full_RAFE - r_.FT_only, rafe_minus_ftpc=r_.full_RAFE - r_["base_FT+PC"]))
    drows = "| *Mean difference over rows* | " + " | ".join(drow) + " |"
    md = f"**{NAME[ds]}**: mean ± SD test {'Spearman' if ds in ('caco2', 'half_life') else 'MCC'} over five seeds (42 to 46). Bold: best of the three stages within the row and tree.\n\n" + hdr + "\n".join(body) + "\n" + pcrow + "\n" + drows + "\n" + (f"\nPending trees: {', '.join(pend)}.\n" if pend else "")
    (O / f"{ds}.md").write_text(md); print(md)
pd.DataFrame(summary).round(4).to_csv(O / "within_cell_differences.csv", index=False)
