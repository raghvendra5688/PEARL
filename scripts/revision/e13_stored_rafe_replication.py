"""E13 (audit): does the paper's STORED RAFE feature table reproduce the supplementary-table gain when re-run under the controlled harness?
Arms: storedRAFE = [FT emb | PC | RAFE columns read from data/rag_features*/..._rag.csv]; compared with my base_FT+PC and recomputed full_RAFE (E1).
Same Optuna/CV/seed protocol as E1 (run_cell). Output: results/revision/e13_stored/{cfg}/storedRAFE_{clf}_s{seed}.json|npz"""
import sys, argparse; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rafe_lib import *
STORED = {"bbbp_mf": "data/rag_features/bbbp/MolFormer_Finetuned_FL", "dili_um": "data/rag_features_unimol/dili/UniMol_FL",
          "herg_mf": "data/rag_features/herg/MolFormer_Finetuned_FL", "bace_mf": "data/rag_features/bace/MolFormer_Finetuned_FL"}
def stored(name, sp, smiles):
    d = pd.read_csv(REPO / f"{STORED[name]}_{ {'train': 'train', 'test': 'test'}[sp] }_rag.csv").drop_duplicates(SMI).set_index(SMI)
    feat = [c for c in d.columns if c != DS[CFG[name]["ds"]]["label"] and not c.startswith("Unnamed")]
    miss = [s for s in smiles if s not in d.index]; assert not miss, (name, sp, len(miss))
    return d.loc[smiles, feat].values.astype(np.float32)
def job(a):
    name, clf, seed, nj = a; b = get_base(name); ds = CFG[name]["ds"]
    X = {sp: np.hstack([b[sp]["emb"], b[sp]["pc"], stored(name, sp, b[sp]["smiles"])]) for sp in ("train", "test")}
    r = run_cell(RES / "e13_stored" / name / f"storedRAFE_{clf}_s{seed}", ds, X["train"], b["train"]["y"], X["test"], b["test"]["y"], clf, seed, nj=nj, meta=dict(cfg=name, arm="storedRAFE"))
    print("e13", name, clf, seed, {m: round(v, 3) for m, v in r.items() if m in ("MCC", "Spearman", "AUC")}, r["seconds"], flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cfgs", default="dili_um,bbbp_mf"); ap.add_argument("--clfs", default="XGBoost"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--nj", type=int, default=4); a = ap.parse_args()
    jobs = [(n, c, 42 + i, a.nj) for n in a.cfgs.split(",") for c in a.clfs.split(",") for i in range(a.seeds)]
    print(len(jobs), "cells", flush=True); pool_run(job, jobs, a.workers)
