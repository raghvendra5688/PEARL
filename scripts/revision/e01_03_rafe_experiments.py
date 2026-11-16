"""E1 (feature-group ablation), E2 (retrieval strategy), E3 (k sweep). One driver; idempotent cells.
usage: e01_03_rafe_experiments.py --exp e1|e2|e3 --cfgs bbbp_mf,dili_um,caco2_mtr --seeds 5 --workers 8"""
import sys, argparse; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rafe_lib import *

def arms(exp):
    """list of (arm_name, strategy, rseed, mode, k, groups)"""
    if exp == "e1":
        # controlled representation matrix (Reviewer 2.1): PC_only, FT_only, FT+PC, RAFE all through the same code path
        a = [("PC_only", None, 0, "orig", 10, ""), ("FT_only", None, 0, "orig", 10, ""), ("base_FT+PC", None, 0, "orig", 10, "")]
        for g in "ABCD": a.append((f"+{g}", "clm", 0, "orig", 10, g))
        a.append(("full_RAFE", "clm", 0, "orig", 10, "ABCD"))
        for g in "ABCD": a.append((f"full-{g}", "clm", 0, "orig", 10, "ABCD".replace(g, "")))
        return a
    if exp == "e2":
        return [("clm", "clm", 0, "orig", 10, "ABCD"), ("morgan", "morgan", 0, "orig", 10, "ABCD")] + \
               [(f"random_r{r}", "random", r, "orig", 10, "ABCD") for r in range(5)]
    if exp == "e3":
        return [(f"k{k}", "clm", 0, "single", k, "ABCD") for k in (1, 5, 10, 20, 50, 100)]

def job(a):
    exp, name, arm, strat, rs, mode, k, groups, seed, nj, clf = a
    cfg = CFG[name]; ds = cfg["ds"]; base = get_base(name)
    rafe = build_rafe(name, strat, rs, mode, k) if strat else None
    X, cols = assemble(name, rafe, groups, base, include_emb=(arm != "PC_only"), include_pc=(arm != "FT_only"))
    clf = clf or cfg["clf"]
    r = run_cell(RES / exp / name / f"{arm}_{clf}_s{seed}", ds, X["train"], base["train"]["y"], X["test"], base["test"]["y"], clf, seed, nj=nj,
                 meta=dict(cfg=name, arm=arm, n_rafe=len(cols), strategy=strat, k=k, mode=mode, groups=groups))
    print(exp, name, arm, clf, seed, {m: round(v, 3) for m, v in r.items() if m in ("MCC", "Spearman", "AUC", "R2")}, r["seconds"], flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True); ap.add_argument("--cfgs", default="bbbp_mf,dili_um,caco2_mtr")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--nj", type=int, default=4)
    ap.add_argument("--extra_clfs", action="store_true", help="e1 only: also run the other two trees (seed 42) for base & full")
    ap.add_argument("--clfs", default="", help="comma list of trees to run for every arm/seed (default: the config's own tree)")
    ap.add_argument("--shard", default="0/1", help="K/N: run only jobs K, K+N, ... (SLURM array splitting)")
    a = ap.parse_args(); jobs = []
    for name in a.cfgs.split(","):
        for (arm, strat, rs, mode, k, g) in arms(a.exp):           # build features serially first (GPU)
            if strat: build_rafe(name, strat, rs, mode, k)
        for (arm, strat, rs, mode, k, g) in arms(a.exp):
            seeds = [42 + i for i in range(a.seeds)]   # random-retrieval arms: every retrieval draw x every model seed (retrieval seed rs is independent of the model seed)
            for sd in seeds:
                for c in (a.clfs.split(",") if a.clfs else [None]): jobs.append((a.exp, name, arm, strat, rs, mode, k, g, sd, a.nj, c))
            if a.extra_clfs and a.exp == "e1" and arm in ("base_FT+PC", "full_RAFE", "FT_only", "PC_only"):
                for c in CLFS:
                    if c != CFG[name]["clf"]: jobs.append((a.exp, name, arm, strat, rs, mode, k, g, 42, a.nj, c))
    ks, ns = map(int, a.shard.split("/")); jobs = jobs[ks::ns]
    print(len(jobs), "cells", flush=True); pool_run(job, jobs, a.workers)
