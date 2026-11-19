"""E5b (R3.2): sensitivity of the fine-tuned Uni-Mol representation to the conformer (RDKit ETKDG seed) on BACE.
Re-embeds BACE train/valid/test with the saved fine-tuned weights (weighted_loss_BACE, same model as the UniMol_WL embeddings) for conformer seeds 42-46
(42 = the seed used for the original embeddings, so it must reproduce them), then
 (a) measures embedding drift (cosine similarity of the 512-d CLS vector to seed 42),
 (b) trains trees on cls512 and concat2560 for every conformer seed (tree seeds 42-44) and decomposes test-metric variance into conformer vs tree-seed variance.
Embeddings are cached in EffiChem_Extras_v2/revision_cache/e05b; cells in results/revision/e05b/bace_um/"""
import os, sys, argparse; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unimol" / "finetuning"))
import numpy as np, pandas as pd, torch
import unimol_lora_trainer as ult
from unimol_tools.data.datahub import DataHub
from rafe_lib import get_base, CACHE
from rev_common import *
OUT = RES / "e05b" / "bace_um"; OUT.mkdir(parents=True, exist_ok=True); C = CACHE / "e05b"; C.mkdir(exist_ok=True)
MODEL = Path("/export/qcai-omics/Raghvendra/EffiChem_Extras/weighted_loss_BACE/dptech__Uni__Mol_LoRA_Finetuned")
SEEDS = list(range(42, 47))
def embed_split(model, smiles, seed):
    def pre(smiles_list, repr_model):
        san = [ult._canonicalize(s) for s in smiles_list]
        hub = DataHub(data=san, task="repr", is_train=False, data_type=repr_model.params.get("data_type", "molecule"), remove_hs=repr_model.params.get("remove_hs", False),
                      model_name=repr_model.params.get("model_name", "unimolv1"), seed=seed)
        return hub.data["unimol_input"]
    ult.preprocess_smiles_for_unimol = pre
    return model.get_embeddings(list(smiles), batch_size=64)
def job(a):
    arm, clf, cs, ts, nj = a; b = get_base("bace_um"); E = {sp: np.load(C / f"emb_s{cs}_{sp}.npy") for sp in ("train", "test")}
    sl = slice(0, 512) if arm == "cls512" else slice(0, 2560)
    r = run_cell(OUT / f"conf{cs}_{arm}_{clf}_s{ts}", "bace", E["train"][:, sl], b["train"]["y"], E["test"][:, sl], b["test"]["y"], clf, ts, nj=nj, meta=dict(cfg="bace_um", arm=arm, conformer_seed=cs))
    print("e05b conf", cs, arm, clf, ts, {m: round(v, 3) for m, v in r.items() if m in ("MCC", "AUC")}, r["seconds"], flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--nj", type=int, default=4); ap.add_argument("--limit", type=int, default=0, help="debug: first N molecules per split")
    a = ap.parse_args(); b = get_base("bace_um")
    if not all((C / f"emb_s{s}_{sp}.npy").exists() for s in SEEDS for sp in ("train", "valid", "test")) or a.limit:
        model = ult.load_finetuned_unimol(MODEL, device=torch.device("cuda")); model.eval()
        for s in SEEDS:
            for sp in ("train", "valid", "test"):
                f = C / f"emb_s{s}_{sp}.npy"
                if f.exists() and not a.limit: continue
                sm = b[sp]["smiles"][: a.limit or None]; e = embed_split(model, sm, s)
                if not a.limit: np.save(f, e)
                else: print(sp, s, "seed-42 check vs stored CLS cosine:" if s == 42 else "", float(np.median((e[:, :512] * b[sp]["emb"][:len(sm), :512]).sum(1) / (np.linalg.norm(e[:, :512], axis=1) * np.linalg.norm(b[sp]["emb"][:len(sm), :512], axis=1)))), flush=True)
                print("embedded", sp, "conformer seed", s, e.shape, flush=True)
        del model; torch.cuda.empty_cache()
        if a.limit: sys.exit(0)
    rows = []
    for s in SEEDS:   # (a) drift vs seed 42 and reproduction of the stored embeddings
        for sp in ("test",):
            e0, es = np.load(C / f"emb_s42_{sp}.npy")[:, :512], np.load(C / f"emb_s{s}_{sp}.npy")[:, :512]; st = b[sp]["emb"][:, :512]
            cos = lambda x, y: (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
            rows.append(dict(split=sp, conformer_seed=s, median_cos_vs_seed42=float(np.median(cos(es, e0))), p5_cos_vs_seed42=float(np.percentile(cos(es, e0), 5)), median_cos_vs_stored=float(np.median(cos(es, st)))))
    pd.DataFrame(rows).round(4).to_csv(RES / "e05b" / "embedding_drift.csv", index=False); print(pd.DataFrame(rows).round(4).to_string(), flush=True)
    jobs = [(arm, clf, cs, ts, a.nj) for cs in SEEDS for ts in (42, 43, 44) for arm in ("cls512", "concat2560") for clf in ("XGBoost", "LightGBM")]
    print(len(jobs), "cells", flush=True); pool_run(job, jobs, a.workers)
    R = []
    for f in OUT.glob("*.json"):
        j = json.load(open(f)); R.append(dict(arm=j["arm"], clf=j["clf"], conformer_seed=j["conformer_seed"], tree_seed=j["seed"], MCC=j["MCC"], AUC=j["AUC"]))
    d = pd.DataFrame(R); g = []
    for (arm, clf), x in d.groupby(["arm", "clf"]):
        cm = x.groupby("conformer_seed").MCC.mean(); tm = x.groupby("tree_seed").MCC.mean()
        g.append(dict(arm=arm, clf=clf, n=len(x), mean_MCC=x.MCC.mean(), sd_across_conformers=cm.std(ddof=1), sd_across_tree_seeds=tm.std(ddof=1), range_over_conformers=cm.max() - cm.min()))
    pd.DataFrame(g).round(4).to_csv(RES / "e05b" / "conformer_vs_tree_variance.csv", index=False); print(pd.DataFrame(g).round(4).to_string())
