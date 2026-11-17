"""E11: label-free RAFE variants, evaluated at each dataset's best RAFE configuration with the same protocol as E1 (20 Optuna trials, 5-fold CV on train, seeds 42-46).
Arms (all inputs are [FT embedding | PC descriptors | ...]; no variant uses labels in retrieval or feature construction):
  base_FT+PC      reference: no RAFE
  full_RAFE       reference: the original 60 RAFE features
  V1_AB           RAFE groups A (physchem neighbourhood) and B (density) only
  V2_residual     RAFE features minus their ridge prediction from [embedding | PC] (cross-fitted on train, 5-fold; feature-to-feature regression, no labels)
  V3_wide         full RAFE + neighbour statistics (mean/std at k=10,20) of TPSA, MW, HBD, HBA, rotatable bonds of the ZINC neighbours
  V4_compressed   [embedding | PC | top 16 principal components of the 60 RAFE features] (PCA fit on train, unsupervised)
  V5_hybrid       full RAFE computed from neighbours retrieved by z(CLM cosine)+z(Morgan Tanimoto) instead of CLM cosine only
Selection among variants (analysis step) uses cv_best = training-only 5-fold CV score. Output: results/revision/e11/<cfg>/<arm>_<clf>_s<seed>.json|npz"""
import sys, argparse, pickle; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rafe_lib import *
from rdkit.Chem import Descriptors, rdMolDescriptors
from multiprocessing import Pool
CFGS = ["bbbp_mf", "dili_um", "caco2_mtr", "bace_mf", "herg_mf", "flavor_mf", "half_life_mf"]
ARMS = ["base_FT+PC", "full_RAFE", "V1_AB", "V2_residual", "V3_wide", "V4_compressed", "V5_hybrid"]
E11C = CACHE / "e11"; E11C.mkdir(exist_ok=True)
def _props(smis):
    out = np.full((len(smis), 5), np.nan, np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None: out[i] = [rdMolDescriptors.CalcTPSA(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumRotatableBonds(m)]
    return out
def zinc_extra(meta):
    cp = CACHE / "zinc_props_extra.npz"
    if cp.exists(): return np.load(cp)["p"]
    smis = list(meta["smiles"]); ch = [smis[i:i + 5000] for i in range(0, len(smis), 5000)]
    with Pool(16) as p: P = np.vstack(p.map(_props, ch))
    P = np.where(np.isnan(P), np.nanmedian(P, 0), P); np.savez_compressed(cp, p=P); return P
def variants(name):
    """returns {arm: {'train': X, 'test': X}} (float32) and caches them"""
    cp = E11C / f"{name}.pkl"
    if cp.exists(): return pickle.load(open(cp, "rb"))
    from sklearn.linear_model import RidgeCV, Ridge; from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA; from sklearn.model_selection import KFold
    cfg = CFG[name]; base = get_base(name); R = build_rafe(name, "clm", 0, "orig", 10)
    E = {sp: np.hstack([base[sp]["emb"], base[sp]["pc"]]) for sp in ("train", "valid", "test")}
    F = {sp: R[sp].values.astype(np.float32) for sp in ("train", "valid", "test")}; cols = list(R["train"].columns)
    cat = lambda *a: {sp: np.hstack([x[sp] for x in a]).astype(np.float32) for sp in ("train", "test")}
    out = {"base_FT+PC": cat(E), "full_RAFE": cat(E, F)}
    ab = [i for i, c in enumerate(cols) if group_of(c) in "AB"]; out["V1_AB"] = cat(E, {sp: F[sp][:, ab] for sp in F})
    # V2 residual (train residuals cross-fitted so train and test residuals have the same distribution)
    sc = StandardScaler().fit(E["train"]); S = {sp: np.nan_to_num(sc.transform(E[sp])) for sp in E}
    al = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(S["train"], F["train"]).alpha_
    res = {"train": np.zeros_like(F["train"])}
    for a, b in KFold(5, shuffle=True, random_state=0).split(S["train"]):
        res["train"][b] = F["train"][b] - Ridge(alpha=al).fit(S["train"][a], F["train"][a]).predict(S["train"][b])
    full = Ridge(alpha=al).fit(S["train"], F["train"]); res.update({sp: F[sp] - full.predict(S[sp]) for sp in ("valid", "test")})
    out["V2_residual"] = cat(E, res)
    # V3 wide: extra ZINC property statistics of the same CLM neighbours
    meta, zfp, zscaf = zinc_resources(cfg); vocab = pickle.load(open(CACHE / "zinc_scaf_vocab.pkl", "rb")); Z = l2n(np.load(cfg["zinc"]).astype(np.float32))
    P = zinc_extra(meta); ex = {"TPSA": P[:, 0], "MW": P[:, 1], "HBD": P[:, 2], "HBA": P[:, 3], "RotB": P[:, 4]}; X3 = {}
    for sp in ("train", "valid", "test"):
        Q = l2n(base[sp]["emb"]); idx = retrieve("clm", Q, base[sp]["smiles"], Z, zfp, 20)
        d, _ = compute_features(Q, base[sp]["smiles"], idx, Z, meta, zfp, zscaf, vocab, "orig", 10, props_override=ex)
        X3[sp] = d[[c for c in d.columns if any(f"_{p}_" in c for p in ex)]].values.astype(np.float32)
    out["V3_wide"] = cat(E, F, X3)
    pca = PCA(16, random_state=0).fit(StandardScaler().fit(F["train"]).transform(F["train"])); sc2 = StandardScaler().fit(F["train"])
    out["V4_compressed"] = cat(E, {sp: pca.transform(sc2.transform(F[sp])) for sp in F})
    Rh = build_rafe(name, "hybrid", 0, "orig", 10); out["V5_hybrid"] = cat(E, {sp: Rh[sp].values.astype(np.float32) for sp in ("train", "valid", "test")})
    pickle.dump(out, open(cp, "wb")); print(f"[{name}] variants cached:", {k: v["train"].shape[1] for k, v in out.items()}, flush=True); return out
def job(a):
    name, arm, clf, seed, nj = a; ds = CFG[name]["ds"]; b = get_base(name); X = variants(name)[arm]
    r = run_cell(RES / "e11" / name / f"{arm}_{clf}_s{seed}", ds, X["train"], b["train"]["y"], X["test"], b["test"]["y"], clf, seed, nj=nj, meta=dict(cfg=name, arm=arm))
    print("e11", name, arm, clf, seed, {m: round(v, 3) for m, v in r.items() if m in ("MCC", "Spearman", "AUC")}, r["seconds"], flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cfgs", default=",".join(CFGS)); ap.add_argument("--clfs", default="XGBoost,LightGBM"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--nj", type=int, default=4); a = ap.parse_args()
    os.environ["REV_RETR_DEV"] = "cuda:1"
    for n in a.cfgs.split(","): variants(n)          # serial: feature construction (GPU retrieval) once per configuration
    jobs = [(n, arm, c, 42 + i, a.nj) for n in a.cfgs.split(",") for c in a.clfs.split(",") for i in range(a.seeds) for arm in ARMS]
    print(len(jobs), "cells", flush=True); pool_run(job, jobs, a.workers)
