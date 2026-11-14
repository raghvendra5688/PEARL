"""E5: separate Uni-Mol finetuned 3D CLS (first 512 dims) from the raw ECFP4 (last 2048 dims) of the 2560-d hybrid.
Arms: cls512 | ecfp2048 | concat2560 (no PC, no RAFE) with XGB/LGBM/CatBoost, 5 seeds."""
import sys, argparse; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rafe_lib import *
SL = {"cls512": slice(0, 512), "ecfp2048": slice(512, 2560), "concat2560": slice(0, 2560)}
def job(a):
    name, arm, clf, seed, nj = a; b = get_base(name); ds = CFG[name]["ds"]
    assert b["train"]["emb"].shape[1] == 2560, b["train"]["emb"].shape
    X = {sp: b[sp]["emb"][:, SL[arm]] for sp in ("train", "test")}
    r = run_cell(RES / "e05" / name / f"{arm}_{clf}_s{seed}", ds, X["train"], b["train"]["y"], X["test"], b["test"]["y"], clf, seed, nj=nj, meta=dict(cfg=name, arm=arm))
    print("e5", name, arm, clf, seed, {m: round(v, 3) for m, v in r.items() if m in ("MCC", "Spearman", "AUC")}, r["seconds"], flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cfgs", default="bace_um,dili_um,bbbp_um,caco2_um"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--nj", type=int, default=4); a = ap.parse_args()
    jobs = []
    for n in a.cfgs.split(","):
        get_base(n)
        jobs += [(n, arm, c, 42 + i, a.nj) for i in range(a.seeds) for arm in SL for c in CLFS]
    print(len(jobs), "cells", flush=True); pool_run(job, jobs, a.workers)
