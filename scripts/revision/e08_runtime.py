"""E8 (R1.5): end-to-end cost of the PEARL pipeline. Mix of (i) durations parsed from the logs of the original pipeline runs,
(ii) wandb LoRA training runtimes, (iii) fresh micro-benchmarks of RAFE retrieval + feature construction and tree inference, (iv) tree-fit seconds from E1/E4 cells.
Hardware differs between sources (H200 node for the original logs, shared P100 here); all numbers are wall-clock and indicative. Output: results/revision/e08/"""
import sys, re, glob, json, time, datetime as dt, numpy as np, pandas as pd; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import *
from rafe_lib import CFG, CACHE, get_base, zinc_resources, l2n, retrieve, compute_features
import pickle
OUT = RES / "e08"; OUT.mkdir(exist_ok=True); rows = []
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
def span(f):
    t = [dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") for l in open(f, errors="ignore") if (m := TS.match(l))]
    return (t[-1] - t[0]).total_seconds() if len(t) > 1 else np.nan
LOG = REPO / "logs"
def blocks(f, marker):
    """durations (hours) of consecutive blocks in a log, split at lines containing `marker` with dataset=<name>; last block ends at the last timestamp"""
    ev = [(dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), l) for l in open(f, errors="ignore") if (m := TS.match(l))]
    st = [(t, re.search(r"dataset=(\w+)", l).group(1)) for t, l in ev if marker in l and "dataset=" in l]
    return [(d, (st[k + 1][0] if k + 1 < len(st) else ev[-1][0]) - t0) for k, (t0, d) in enumerate(st)]
LOG = REPO / "logs"
# Only the ZINC-embedding blocks can be attributed reliably from the old logs (other logs mix parallel datasets and appended reruns)
for d, td in blocks(LOG / "embed_zinc250k_new_datasets_herg.log", "ZINC-250k HF Embedding")[:2]:
    rows.append(dict(source="original pipeline log (H200 node)", stage="ZINC-250k embedding, 6 fine-tuned SMILES-CLM variants (ChemBERTa-MLM/MTR, MolFormer x FL/WL), one dataset", dataset=d, hours=round(td.total_seconds() / 3600, 3), file="embed_zinc250k_new_datasets_herg.log"))
for d, td in blocks(LOG / "embed_zinc250k_unimol_new_datasets_herg.log", "ZINC-250k Uni-Mol Embedding")[:3]:
    rows.append(dict(source="original pipeline log (H200 node)", stage="ZINC-250k embedding with fine-tuned Uni-Mol (3D conformer generation dominates); 2 variants (FL, WL) for herg/dili, 1 for caco2 rerun", dataset=d, hours=round(td.total_seconds() / 3600, 3), file="embed_zinc250k_unimol_new_datasets_herg.log"))
# LoRA training runtimes from wandb summaries
tr = []
for f in glob.glob("/export/qcai-omics/Raghvendra/EffiChem_Extras_v2/wandb*/**/wandb-summary.json", recursive=True):
    j = json.load(open(f))
    if "train_runtime" in j: tr.append(j["train_runtime"])
if tr: rows.append(dict(source="wandb summaries", stage=f"LoRA fine-tuning (one run; n={len(tr)} runs, median / IQR / max in seconds)", dataset="all", hours=round(float(np.median(tr)) / 3600, 4), file=f"median {np.median(tr):.0f} s, IQR {np.percentile(tr,25):.0f}-{np.percentile(tr,75):.0f} s, max {np.max(tr):.0f} s"))
# tree-stage seconds from E1/E4/E15 cells (shared server: inflated by load)
sec = []
for f in glob.glob(str(RES / "e1" / "*" / "*.json")) + glob.glob(str(RES / "e04_baselines" / "*" / "*.json")):
    j = json.load(open(f)); sec.append(dict(clf=j["clf"], n_feat=j.get("n_feat", np.nan), n_train=j["n_train"], seconds=j["seconds"]))
sd = pd.DataFrame(sec)
for c, g in sd.groupby("clf"): rows.append(dict(source="E1/E4 cells (loaded shared server)", stage=f"tree stage per cell: 20 Optuna trials x 5-fold CV + final fit ({c})", dataset=f"n={len(g)} cells", hours=round(g.seconds.median() / 3600, 3), file=f"median {g.seconds.median():.0f} s, IQR {g.seconds.quantile(.25):.0f}-{g.seconds.quantile(.75):.0f} s"))
# micro-benchmark: retrieval + RAFE features for a query batch, and single-molecule latency (BBBP, MolFormer, exact GPU search over 249k ZINC molecules)
name = "bbbp_mf"; cfg = CFG[name]; base = get_base(name); meta, zfp, zscaf = zinc_resources(cfg); vocab = pickle.load(open(CACHE / "zinc_scaf_vocab.pkl", "rb"))
t0 = time.perf_counter(); Z = l2n(np.load(cfg["zinc"]).astype(np.float32)); t_load = time.perf_counter() - t0
rows.append(dict(source="measured now (shared P100)", stage="load ZINC-250k embedding matrix (one-off per dataset/model)", dataset=name, hours=round(t_load / 3600, 5), file=f"{t_load:.1f} s"))
Q = l2n(base["test"]["emb"]); sm = base["test"]["smiles"]; res = {}
def run(q, s):
    t = time.perf_counter(); idx = retrieve("clm", q, s, Z, zfp, 20); t1 = time.perf_counter(); df, _ = compute_features(q, s, idx, Z, meta, zfp, zscaf, vocab, "orig", 10); return t1 - t, time.perf_counter() - t1
run(Q[:8], sm[:8])   # warm-up
for label, n in (("single molecule", 1), ("batch of 192 (BBBP test set)", len(Q))):
    tt = [run(Q[:n], sm[:n]) for _ in range(5 if n == 1 else 3)]; r = np.median([a for a, _ in tt]); f_ = np.median([b for _, b in tt])
    rows.append(dict(source="measured now (shared P100)", stage=f"RAFE at inference, {label}: exact retrieval over 249k ZINC molecules / feature construction", dataset=name, hours=round((r + f_) / 3600, 6), file=f"retrieval {r*1000:.1f} ms, features {f_*1000:.1f} ms, total {(r+f_)*1000:.1f} ms ({(r+f_)*1000/n:.2f} ms per molecule)"))
# tree inference latency
import xgboost as xgb
X = np.hstack([base["train"]["emb"], base["train"]["pc"]]); y = base["train"]["y"]
m = xgb.XGBClassifier(n_estimators=300, max_depth=6, tree_method="hist", n_jobs=4).fit(X, y); Xt = np.hstack([base["test"]["emb"], base["test"]["pc"]])
for n in (1, len(Xt)):
    tt = [(lambda t: (m.predict_proba(Xt[:n]), time.perf_counter() - t)[1])(time.perf_counter()) for _ in range(20)]
    rows.append(dict(source="measured now", stage=f"XGBoost inference (300 trees, {X.shape[1]}-d input), {'single molecule' if n == 1 else 'batch of 192'}", dataset=name, hours=round(np.median(tt) / 3600, 7), file=f"{np.median(tt)*1000:.1f} ms"))
pd.DataFrame(rows).to_csv(OUT / "runtime_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120); print(pd.DataFrame(rows)[["source", "stage", "dataset", "hours", "file"]].to_string())
