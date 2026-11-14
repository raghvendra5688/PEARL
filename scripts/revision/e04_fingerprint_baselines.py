"""E4: independent conventional-representation baselines (R2.2): ECFP4, ECFP6, MACCS, ECFP4+MACCS,
RDKit descriptors, full PC vector; XGB/LGBM/CatBoost; same scaffold splits & HPO protocol."""
import sys, argparse, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import *
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
RDLogger.DisableLog("rdApp.*")
OUT = RES / "e04_baselines"

def fp_matrix(smiles, radius, nbits=2048):
    g = GetMorganGenerator(radius=radius, fpSize=nbits)
    X = np.zeros((len(smiles), nbits), np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None: X[i] = g.GetFingerprintAsNumPy(m)
    return X

def maccs_matrix(smiles):
    X = np.zeros((len(smiles), 167), np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fp = MACCSkeys.GenMACCSKeys(m); X[i, list(fp.GetOnBits())] = 1
    return X

def features(ds):
    cache = EX2 / "revision_cache" / "e04" / f"{ds}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True); return {k: z[k].item() for k in z.files}
    dfs = load_clean(ds); pc = load_pc(ds); y, _ = encode_labels(ds, dfs)
    cols = list(pc["train"].columns); iG = cols.index("graph_diameter")
    d = {}
    for sp in ["train", "test"]:
        s = dfs[sp][SMI].tolist()
        e4, e6, mc = fp_matrix(s, 2), fp_matrix(s, 3), maccs_matrix(s)
        d[sp] = dict(ecfp4=e4, ecfp6=e6, maccs=mc, ecfp4_maccs=np.hstack([e4, mc]),
                     rdkit_desc=pc[sp].values[:, :iG], pc473=pc[sp].values, y=y[sp])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, train=np.array(d["train"], dtype=object), test=np.array(d["test"], dtype=object))
    return d

def job(a):
    ds, fs, clf, seed, nj = a
    d = features(ds)
    r = run_cell(OUT / ds / f"{fs}_{clf}_s{seed}", ds, d["train"][fs], d["train"]["y"], d["test"][fs], d["test"]["y"],
                 clf, seed, nj=nj, meta=dict(featureset=fs))
    print(ds, fs, clf, seed, {k: round(v, 3) for k, v in r.items() if k in ("MCC", "Spearman", "AUC", "R2")}, r["seconds"], flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10); ap.add_argument("--nj", type=int, default=4)
    ap.add_argument("--small_seeds", type=int, default=5); ap.add_argument("--datasets", default="dili,caco2,half_life,bace,bbbp,herg,flavor")
    a = ap.parse_args()
    FS = ["ecfp4", "maccs", "ecfp4_maccs", "rdkit_desc", "pc473", "ecfp6"]
    jobs = []
    for ds in a.datasets.split(","):
        features(ds)  # build cache serially
        seeds = [42 + i for i in range(a.small_seeds)] if ds not in ("herg", "flavor") else [42]
        for sd in seeds:
            for fs in FS:
                for c in CLFS:
                    jobs.append((ds, fs, c, sd, a.nj))
    print(len(jobs), "cells", flush=True)
    pool_run(job, jobs, a.workers)
