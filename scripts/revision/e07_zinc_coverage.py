"""E7: chemical-space coverage of ZINC-250k for the 7 target datasets (R1.4).
Quantitative: per-molecule max Morgan(ECFP4,2048) Tanimoto to ZINC; coverage at 0.4/0.6/0.8.
Qualitative: UMAP (Jaccard on ECFP4 bits) of ZINC subsample + each dataset."""
import sys, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import *
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
RDLogger.DisableLog("rdApp.*")
OUT = RES / "e07_coverage"; OUT.mkdir(parents=True, exist_ok=True)
g = GetMorganGenerator(radius=2, fpSize=2048)

def fps(smiles):
    X = np.zeros((len(smiles), 2048), np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None: X[i] = g.GetFingerprintAsNumPy(m)
    return X

zdf = pd.read_csv(REPO / "data/zinc250k/zinc250k_cleaned.csv")
Z = fps(zdf.smiles.tolist()); print("ZINC", Z.shape, flush=True)
dev = "cuda:0"
Zt = torch.from_numpy(Z).to(dev).half(); zc = Zt.float().sum(1)
rows, per = [], []
allfp = {}
for ds in DS:
    dfs = load_clean(ds)
    for sp in ["train", "valid", "test"]:
        s = dfs[sp][SMI].tolist(); X = fps(s); allfp[(ds, sp)] = X
        best = np.zeros(len(s), np.float32); arg = np.zeros(len(s), np.int64); top5 = np.zeros(len(s), np.float32)
        for a in range(0, len(s), 512):
            q = torch.from_numpy(X[a:a+512]).to(dev).half(); qc = q.float().sum(1)
            inter = (q @ Zt.T).float()
            tan = inter / (qc[:, None] + zc[None, :] - inter + 1e-9)
            v, i = tan.topk(5, dim=1)
            best[a:a+512] = v[:, 0].cpu().numpy(); arg[a:a+512] = i[:, 0].cpu().numpy(); top5[a:a+512] = v.mean(1).cpu().numpy()
        per.append(pd.DataFrame(dict(dataset=ds, split=sp, smiles=s, max_tanimoto=best, top5_mean_tanimoto=top5, nn_zinc_idx=arg)))
    print(ds, "done", flush=True)
P = pd.concat(per); P.to_csv(OUT / "per_molecule_coverage.csv", index=False)
def summ(d):
    return pd.Series(dict(n=len(d), median=d.max_tanimoto.median(), mean=d.max_tanimoto.mean(),
                          **{f">={t}": (d.max_tanimoto >= t).mean() for t in (0.3, 0.4, 0.6, 0.8)}))
T = P.groupby("dataset").apply(summ); T.loc["ALL"] = summ(P)
Tt = P[P.split == "test"].groupby("dataset").apply(summ); Tt.columns = [c + "_test" for c in Tt.columns]
pd.concat([T, Tt], axis=1).round(3).to_csv(OUT / "coverage_table.csv"); print(pd.concat([T, Tt], axis=1).round(3).to_string())

# --- UMAP ---
import umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
zi = rng.choice(len(Z), 30000, replace=False)
mats = [Z[zi]]; lab = ["ZINC"] * len(zi)
for ds in DS:
    X = np.vstack([allfp[(ds, s)] for s in ["train", "valid", "test"]])
    if len(X) > 2500: X = X[rng.choice(len(X), 2500, replace=False)]
    mats.append(X); lab += [ds] * len(X)
M = np.vstack(mats).astype(bool); lab = np.array(lab)
emb = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="jaccard", random_state=0).fit_transform(M)
np.savez_compressed(OUT / "umap.npz", emb=emb, lab=lab)
names = {"bace": "BACE", "bbbp": "BBBP", "herg": "hERG", "dili": "DILI", "caco2": "Caco-2", "half_life": "Half-Life", "flavor": "Flavor"}
fig, ax = plt.subplots(2, 4, figsize=(18, 8.5)); ax = ax.ravel()
zm = lab == "ZINC"
for k, ds in enumerate(DS):
    ax[k].scatter(emb[zm, 0], emb[zm, 1], s=0.4, c="#cccccc", alpha=0.4, rasterized=True)
    m = lab == ds; ax[k].scatter(emb[m, 0], emb[m, 1], s=3, c="#d6336c", alpha=0.7, rasterized=True)
    cv = T.loc[ds]; ax[k].set_title(f"{names[ds]} (n={int(cv.n)}); {100*cv['>=0.4']:.0f}% have ZINC nbr ≥0.4", fontsize=10)
    ax[k].set_xticks([]); ax[k].set_yticks([])
ax[7].hist([P[P.dataset == d].max_tanimoto for d in DS], bins=30, histtype="step", label=[names[d] for d in DS], density=True)
ax[7].set_xlabel("max ECFP4 Tanimoto to ZINC-250k"); ax[7].legend(fontsize=7); ax[7].set_title("Nearest-neighbour similarity")
plt.tight_layout(); plt.savefig(OUT / "umap_coverage.pdf")
print("E7 DONE")
