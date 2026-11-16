"""RAFE re-implementation with pluggable retrieval strategy, k, and feature-group selection.
Mirrors scripts/smiles/rag/rag_feature_extraction.py for the 'orig' mode (60 features:
A=18 physchem, B=5 density, C=34 centroid[PCA32+2], D=3 Tanimoto); validated against stored RAFE CSVs."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import *
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
CACHE = EX2 / "revision_cache"; CACHE.mkdir(parents=True, exist_ok=True)  # big artifacts live in EffiChem_Extras_v2

EMB1 = EX1 / "All_Embeddings"; ZIN1 = EX1 / "weighted_loss_flavor/zinc_embeddings"; ZUM1 = EX1 / "weighted_loss_flavor/zinc_embeddings_unimol"
CFG = {
  "bbbp_mf":   dict(ds="bbbp", clf="XGBoost", emb=EMB1/"BBBP_Embeddings/bbbp_{sp}_embed.csv", ecol="MolFormer_Finetuned_FL_embeddings",
                    zinc=ZIN1/"bbbp/MolFormer_Finetuned_FL.npy", meta=EX1/"rag_indices/bbbp/meta.pkl", stored="MolFormer_Finetuned_FL", rag_dir="bbbp"),
  "dili_um":   dict(ds="dili", clf="XGBoost", emb=EX2/"unimol_embeddings/DILI_Embeddings/dili_{sp}_embed.csv", ecol="UniMol_FL_embeddings",
                    zinc=EX2/"zinc_embeddings_unimol/dili/UniMol_FL.npy", meta=EX2/"rag_indices_unimol/dili/meta.pkl", stored="UniMol_FL", rag_dir="unimol:dili"),
  "caco2_mtr": dict(ds="caco2", clf="CatBoost", emb=EX2/"finetuned_pc_embeddings/CACO2_Embeddings/caco2_{sp}_features.csv", ecol="ChemBERTa_77M_MTR_Huber_embeddings",
                    zinc=EX2/"zinc_embeddings/caco2/ChemBERTa_77M_MTR_Huber.npy", meta=EX2/"rag_indices/caco2/meta.pkl", stored="ChemBERTa_77M_MTR_Huber", rag_dir="caco2"),
  "dili_mf":   dict(ds="dili", clf="XGBoost", emb=EX2/"finetuned_pc_embeddings/DILI_Embeddings/dili_{sp}_features.csv", ecol="MolFormer_Finetuned_WL_embeddings",
                    zinc=EX2/"zinc_embeddings/dili/MolFormer_Finetuned_WL.npy", meta=EX2/"rag_indices/dili/meta.pkl", stored="MolFormer_Finetuned_WL", rag_dir="dili"),
  "bace_um":   dict(ds="bace", clf="XGBoost", emb=EX1/"weighted_loss_flavor/unimol_embeddings/BACE_Embeddings/bace_{sp}_embed.csv", ecol="UniMol_WL_embeddings",
                    zinc=ZUM1/"bace/UniMol_WL.npy", meta=EX1/"weighted_loss_flavor/rag_indices_unimol/bace/meta.pkl", stored="UniMol_WL", rag_dir="unimol:bace"),
  "herg_mf":   dict(ds="herg", clf="LightGBM", emb=EX2/"finetuned_pc_embeddings/HERG_Embeddings/herg_{sp}_features.csv", ecol="MolFormer_Finetuned_FL_embeddings",
                    zinc=EX2/"zinc_embeddings/herg/MolFormer_Finetuned_FL.npy", meta=EX2/"rag_indices/herg/meta.pkl", stored="MolFormer_Finetuned_FL", rag_dir="herg"),
  "flavor_mf": dict(ds="flavor", clf="LightGBM", emb=EX1/"PC_FT_All_Embeddings/flavor_Embeddings/fart_{sp}_features.csv", ecol="Molformer_Finetuned_WL_embeddings",
                    zinc=EX1/"weighted_loss_flavor/zinc_embeddings/flavor/Molformer_Finetuned_WL.npy", meta=EX1/"rag_indices/flavor/meta.pkl", stored="Molformer_Finetuned_WL", rag_dir="flavor"),
  "bace_mf":   dict(ds="bace", clf="LightGBM", emb=EMB1/"BACE_Embeddings/bace_{sp}_embed.csv", ecol="Molformer_Finetuned_WL_embeddings",
                    zinc=ZIN1/"bace/Molformer_Finetuned_WL.npy", meta=EX1/"rag_indices/bace/meta.pkl", stored=None, rag_dir=None),
  "half_life_mf": dict(ds="half_life", clf="XGBoost", emb=EX2/"finetuned_pc_embeddings/HALF_LIFE_Embeddings/half_life_{sp}_features.csv", ecol="MolFormer_Finetuned_Huber_embeddings",
                    zinc=EX2/"zinc_embeddings/half_life/MolFormer_Finetuned_Huber.npy", meta=EX2/"rag_indices/half_life/meta.pkl", stored="MolFormer_Finetuned_Huber", rag_dir="half_life"),
  "bbbp_um":   dict(ds="bbbp", clf="XGBoost", emb=EX1/"weighted_loss_flavor/unimol_embeddings/BBBP_Embeddings/bbbp_{sp}_embed.csv", ecol="UniMol_FL_embeddings", zinc=None, meta=None, stored=None, rag_dir=None),
  "caco2_um":  dict(ds="caco2", clf="CatBoost", emb=EX2/"unimol_embeddings/CACO2_Embeddings/caco2_{sp}_embed.csv", ecol="UniMol_Huber_embeddings", zinc=None, meta=None, stored=None, rag_dir=None),
  "herg_um":   dict(ds="herg", clf="LightGBM", emb=EX2/"unimol_embeddings/HERG_Embeddings/herg_{sp}_embed.csv", ecol="UniMol_FL_embeddings", zinc=None, meta=None, stored=None, rag_dir=None),
}
SPLITMAP = {"train": "train", "valid": "eval", "test": "eval"}

def parse_emb(s):
    s = str(s).strip()
    try: a = np.array(json.loads(s), np.float32)
    except Exception: a = np.array([float(x) for x in s.strip("[]").split(",") if x.strip()], np.float32)
    return np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)

def load_emb(cfg, sp):
    f = str(cfg["emb"]).format(sp={"train": "train", "valid": "eval", "test": "test"}[sp])
    d = pd.read_csv(f, usecols=[SMI, cfg["ecol"]])
    return {s: e for s, e in zip(d[SMI], d[cfg["ecol"]])}

def get_base(name):
    """aligned arrays for train/valid/test: smiles, emb (raw), pc (df), y"""
    cfg = CFG[name]; ds = cfg["ds"]
    cp = CACHE / f"base_{name}.pkl"
    if cp.exists(): return pickle.load(open(cp, "rb"))
    dfs = load_clean(ds); pc = load_pc(ds); y, _ = encode_labels(ds, dfs)
    out = {}
    for sp in ["train", "valid", "test"]:
        m = load_emb(cfg, sp); keep = [i for i, s in enumerate(dfs[sp][SMI]) if s in m]
        if len(keep) < len(dfs[sp]): print(f"[{name}/{sp}] dropped {len(dfs[sp]) - len(keep)} rows without embedding", flush=True)
        sm = dfs[sp][SMI].values[keep]
        out[sp] = dict(smiles=list(sm), emb=np.vstack([parse_emb(m[s]) for s in sm]).astype(np.float32), pc=pc[sp].values[keep].astype(np.float32), y=y[sp][keep])
    pickle.dump(out, open(cp, "wb")); return out

# ---- ZINC resources -------------------------------------------------------------------------
_gen = GetMorganGenerator(radius=2, fpSize=2048)
def zinc_resources(cfg):
    class _U(pickle.Unpickler):   # meta pickles written under numpy 2 -> load under numpy 1.x
        def find_class(self, m, n): return super().find_class(m.replace("numpy._core", "numpy.core"), n)
    meta = _U(open(cfg["meta"], "rb")).load()
    cp = CACHE / "zinc_fp_scaf.npz"
    if cp.exists():
        z = np.load(cp); fp, scaf = z["fp"], z["scaf"]
    else:
        fp = np.zeros((len(meta["smiles"]), 256), np.uint8); sc = {}; scaf = np.zeros(len(meta["smiles"]), np.int64)
        for i, s in enumerate(meta["smiles"]):
            m = Chem.MolFromSmiles(s)
            if m is None: continue
            fp[i] = np.packbits(_gen.GetFingerprintAsNumPy(m))
            scaf[i] = sc.setdefault(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False), len(sc) + 1)
        np.savez_compressed(cp, fp=fp, scaf=scaf); pickle.dump(sc, open(CACHE / "zinc_scaf_vocab.pkl", "wb"))
    return meta, fp, scaf

_POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
def tanimoto_packed(q, nb):
    """q (256,) packed, nb (K,256) packed -> (K,)"""
    inter = _POP[q[None, :] & nb].sum(1).astype(np.float32); un = _POP[q].sum() + _POP[nb].sum(1) - inter
    return inter / np.maximum(un, 1)

def query_fp_scaf(smiles, vocab):
    fp = np.zeros((len(smiles), 256), np.uint8); sc = np.zeros(len(smiles), np.int64)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        fp[i] = np.packbits(_gen.GetFingerprintAsNumPy(m))
        sc[i] = vocab.get(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False), -1)
    return fp, sc

# ---- retrieval --------------------------------------------------------------------------------
def l2n(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def retrieve(strategy, Q, qsmiles, Z, zfp, K, seed=0, dev="cuda:0"):
    """returns idx (N,K) ranked neighbour indices for strategy in {clm, morgan, random}"""
    N, M = len(Q), len(Z)
    if strategy == "random":
        rng = np.random.default_rng(seed); return np.vstack([rng.choice(M, K, replace=False) for _ in range(N)])
    out = np.zeros((N, K), np.int64)
    if strategy == "clm":
        Zt = torch.from_numpy(Z).to(dev)
        for a in range(0, N, 512):
            s = torch.from_numpy(Q[a:a+512]).to(dev) @ Zt.T; out[a:a+512] = s.topk(K, dim=1).indices.cpu().numpy()
        del Zt; return out
    if strategy == "morgan":
        Zb = torch.from_numpy(np.unpackbits(zfp, axis=1)).to(dev).half(); zc = Zb.float().sum(1)
        qfp = np.zeros((N, 2048), np.uint8)
        for i, s in enumerate(qsmiles):
            m = Chem.MolFromSmiles(s)
            if m is not None: qfp[i] = _gen.GetFingerprintAsNumPy(m)
        for a in range(0, N, 512):
            q = torch.from_numpy(qfp[a:a+512]).to(dev).half(); qc = q.float().sum(1); inter = (q @ Zb.T).float()
            t = inter / (qc[:, None] + zc[None, :] - inter + 1e-9); out[a:a+512] = t.topk(K, dim=1).indices.cpu().numpy()
        del Zb; return out
    raise ValueError(strategy)

# ---- feature computation ---------------------------------------------------------------------
def compute_features(Q, qsmiles, idx, Z, meta, zfp, zscaf, vocab, mode="orig", k=10):
    """Q (N,D) normalised, idx (N,K) ranked. mode 'orig': k=10/20 dual (60 feats before PCA→60);
    mode 'single': all statistics at one k. Returns (df_without_pca, centroids (N,D), group map)."""
    N = len(Q); qfp, qsc = query_fp_scaf(qsmiles, vocab)
    props = {"logP": np.asarray(meta["logP"]), "qed": np.asarray(meta["qed"]), "SAS": np.asarray(meta["SAS"])}
    ks, kl = (10, 20) if mode == "orig" else (k, k)
    rows = []; cents = np.zeros_like(Q)
    for i in range(N):
        nb = idx[i]; zs = Z[nb]; sims = zs @ Q[i]
        f = {}
        def wm(v, w): w = np.clip(w, 0, None); return float(np.dot(v, w) / (w.sum() + 1e-9))
        if mode == "orig":
            for p, arr in props.items():
                v10, v20 = arr[nb[:10]], arr[nb[:20]]
                f[f"zinc_{p}_mean_k10"] = v10.mean(); f[f"zinc_{p}_std_k10"] = v10.std(); f[f"zinc_{p}_mean_k20"] = v20.mean()
                f[f"zinc_{p}_std_k20"] = v20.std(); f[f"zinc_{p}_wmean_k10"] = wm(v10, sims[:10]); f[f"zinc_{p}_wmean_k20"] = wm(v20, sims[:20])
            s10, s20 = sims[:10], sims[:20]
            f["zinc_nearest_sim"] = s10[0]; f["zinc_k10_sim_mean"] = s10.mean(); f["zinc_k10_sim_std"] = s10.std()
            f["zinc_k20_sim_mean"] = s20.mean(); f["zinc_k20_sim_std"] = s20.std()
            kc = 10
        else:
            for p, arr in props.items():
                v = arr[nb[:k]]; f[f"zinc_{p}_mean_k{k}"] = v.mean(); f[f"zinc_{p}_std_k{k}"] = v.std(); f[f"zinc_{p}_wmean_k{k}"] = wm(v, sims[:k])
            sk = sims[:k]; f["zinc_nearest_sim"] = sk[0]; f[f"zinc_k{k}_sim_mean"] = sk.mean(); f[f"zinc_k{k}_sim_std"] = sk.std()
            kc = k
        zc = Z[nb[:kc]]; c = zc.mean(0); cents[i] = c; cn = c / (np.linalg.norm(c) + 1e-9)
        f["zinc_sim_to_centroid"] = float(Q[i] @ cn); f["zinc_centroid_spread"] = float(np.mean(np.linalg.norm(zc - c, axis=1)))
        nbk = nb[:kc]; tn = tanimoto_packed(qfp[i], zfp[nbk])
        f["tanimoto_k1"] = tn[0]; f["tanimoto_k5_mean"] = tn[:5].mean()
        f[f"scaffold_match_frac_k{kc}"] = float(np.mean(zscaf[nbk] == qsc[i])) if qsc[i] > 0 else 0.0
        rows.append(f)
    return pd.DataFrame(rows), cents

def group_of(col):
    if col.startswith("zinc_centroid_pca") or col in ("zinc_sim_to_centroid", "zinc_centroid_spread"): return "C"
    if col.startswith("zinc_") and any(t in col for t in ("logP", "qed", "SAS")): return "A"
    if col.startswith("zinc_") and ("sim" in col): return "B"
    if col.startswith("tanimoto") or col.startswith("scaffold"): return "D"
    raise ValueError(col)

def build_rafe(name, strategy="clm", seed=0, mode="orig", k=10, force=False):
    """computes (and caches) RAFE features for train/valid/test. PCA(32) fit on train centroids."""
    tag = f"{name}_{strategy}_r{seed}_{mode}{k if mode != 'orig' else ''}"
    cp = CACHE / f"rafe_{tag}.pkl"
    if cp.exists() and not force: return pickle.load(open(cp, "rb"))
    from sklearn.decomposition import PCA
    cfg = CFG[name]; base = get_base(name); meta, zfp, zscaf = zinc_resources(cfg); vocab = pickle.load(open(CACHE / "zinc_scaf_vocab.pkl", "rb"))
    Z = l2n(np.load(cfg["zinc"]).astype(np.float32)); K = 20 if mode == "orig" else k
    out = {}; pca = None
    for sp in ["train", "valid", "test"]:
        Q = l2n(base[sp]["emb"]); idx = retrieve(strategy, Q, base[sp]["smiles"], Z, zfp, K, seed=seed * 1000 + {"train": 1, "valid": 2, "test": 3}[sp])
        df, cents = compute_features(Q, base[sp]["smiles"], idx, Z, meta, zfp, zscaf, vocab, mode, k)
        if pca is None: pca = PCA(32, random_state=42).fit(cents)
        cp_ = pca.transform(cents)
        for j in range(32): df[f"zinc_centroid_pca_{j:02d}"] = cp_[:, j]
        out[sp] = df
        print(f"[{tag}/{sp}] {df.shape}", flush=True)
    pickle.dump(out, open(cp, "wb")); return out

def assemble(name, rafe, groups, base=None, include_emb=True, include_pc=True):
    """X for each split = [emb | PC | selected RAFE groups]"""
    base = base or get_base(name); X = {}
    cols = [c for c in rafe["train"].columns if group_of(c) in groups] if rafe is not None else []
    for sp in ["train", "valid", "test"]:
        parts = ([base[sp]["emb"]] if include_emb else []) + ([base[sp]["pc"]] if include_pc else [])
        if cols: parts.append(rafe[sp][cols].values.astype(np.float32))
        X[sp] = np.hstack(parts)
    return X, cols
