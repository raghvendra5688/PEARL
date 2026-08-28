"""
Rank all 8 PEARL methods (PC-only, Chemprop, GCN, E2E LoRA -- HF, E2E LoRA --
Uni-Mol, FT Embed, FT Embed+PC, RAFE) on each of the 7 benchmark datasets
(BACE, BBBP, FLAVOR, HERG, DILI, CACO2, HALF_LIFE) by its primary metric (MCC
for classification, Spearman for regression), take the top 5 per dataset, and
upload the corresponding saved model artifacts into ONE consolidated Hugging
Face Hub model repo, under `<DATASET>/rank<N>_<method>/` subfolders.

Model artifacts live in two external stores plus the repo's own results/ tree:
- /export/qcai-omics/Raghvendra/EffiChem_Extras     (BACE, BBBP, FLAVOR)
- /export/qcai-omics/Raghvendra/EffiChem_Extras_v2  (HERG, DILI, CACO2, HALF_LIFE)
- PEARL/results/{pc_only,gnn/chemprop,gnn/gcn,ft_embeddings,rag,rag_unimol}

Every score is read live from each method's saved metrics file (JSON, or for
older EffiChem_Extras tree-model runs, a pickled .npy dict) or from the
LoRA/Uni-Mol test-result CSVs -- nothing is hardcoded, so reruns stay correct
if results change. Directory-name casing is genuinely inconsistent across the
project (e.g. `flavor_FT_Results` next to `FLAVOR_PC_Only_Results`), which is
why `effichem_case` exists below instead of a derived rule.

Known gap: some RAFE tree models for BACE/BBBP/FLAVOR were never retained on
disk (only their metrics were). If RAFE's single best-scoring config lacks a
saved artifact, the resolver falls back to the next-best RAFE config that
does have one; only if none of them survive does the script warn and promote
the next method instead -- so "top 5" always means 5 models that actually get
uploaded.

Usage:
    # Inspect the ranking and resolved artifact paths only -- uploads nothing.
    python upload_top5_to_hf.py

    # Actually create the repo and upload (requires `huggingface-cli login`
    # first, or HF_TOKEN set in the environment).
    python upload_top5_to_hf.py --hf-user <your-hf-username> --do-upload

    # Restrict to specific datasets while testing.
    python upload_top5_to_hf.py --datasets BACE HERG
"""

import argparse
import functools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EFFICHEM = Path("/export/qcai-omics/Raghvendra/EffiChem_Extras")
EFFICHEM_V2 = Path("/export/qcai-omics/Raghvendra/EffiChem_Extras_v2")
RESULTS = Path(__file__).resolve().parent.parent.parent / "results"

TREE_MODELS = ("XGBoost", "LightGBM", "CatBoost")
OLD_STORE_DATASETS = ("BACE", "BBBP", "FLAVOR")  # live in EFFICHEM, not EFFICHEM_V2

# `effichem_case` is the exact casing EffiChem_Extras(_v2) uses for this
# dataset's LoRA/FT/RAFE directories -- verified by listing each store
# directly, not derived (BACE/BBBP are upper-case there, flavor is lower-case).
DATASETS = {
    "BACE":      dict(task="classification", metric="MCC",      store=EFFICHEM,    effichem_case="BACE"),
    "BBBP":      dict(task="classification", metric="MCC",      store=EFFICHEM,    effichem_case="BBBP"),
    "FLAVOR":    dict(task="classification", metric="MCC",      store=EFFICHEM,    effichem_case="flavor"),
    "HERG":      dict(task="classification", metric="MCC",      store=EFFICHEM_V2, effichem_case="HERG"),
    "DILI":      dict(task="classification", metric="MCC",      store=EFFICHEM_V2, effichem_case="DILI"),
    "CACO2":     dict(task="regression",     metric="Spearman", store=EFFICHEM_V2, effichem_case="CACO2"),
    "HALF_LIFE": dict(task="regression",     metric="Spearman", store=EFFICHEM_V2, effichem_case="HALF_LIFE"),
}

LOSS_DIR_PREFIX = {"fl": "focal_loss", "wl": "weighted_loss", "huber": "huber_loss"}
LOSS_LABEL_PREFIX = {"Focal Loss": "focal_loss", "Weighted Loss": "weighted_loss", "Huber Loss": "huber_loss"}


# --------------------------------------------------------------------------- #
# Shared helpers for reading metrics and picking the best tree model in a dir
# --------------------------------------------------------------------------- #

def read_metric_file(path):
    """Read one metrics file: modern plain JSON, or an older pickled-.npy dict."""
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True).item()
    with open(path) as f:
        return json.load(f)


def best_of_tree_dir(metrics_dir, models_dir, metric, model_prefix=""):
    """Given a `metrics/` dir with one file per tree-model type, return
    (score, path_to_winning_.pkl) for whichever of XGBoost/LightGBM/CatBoost
    scores highest on `metric`. `model_prefix` covers the old-store convention
    where filenames repeat the embedding name (`MolFormer_Finetuned_FL_CatBoost.pkl`)
    instead of the plain `CatBoost.pkl` used everywhere else."""
    best = None
    for model_name in TREE_MODELS:
        for ext in (".json", ".npy"):
            mfile = metrics_dir / f"{model_prefix}{model_name}_metrics{ext}"
            if mfile.exists():
                score = read_metric_file(mfile).get(metric)
                if score is not None and (best is None or score > best[0]):
                    best = (score, models_dir / f"{model_prefix}{model_name}.pkl")
                break
    return best


def ranked_in_embedding_root(root, metric, prefixed=False):
    """Scan a directory of embedding/loss subdirectories (each holding its own
    `metrics/` + `models/`) and return every (score, path, subdirectory_name)
    hit found under `root`, best first."""
    if not root.exists():
        return []
    hits = []
    for sub in root.iterdir():
        if not sub.is_dir() or sub.name in ("evaluation_plots", "logs"):
            continue
        prefix = f"{sub.name}_" if prefixed else ""
        hit = best_of_tree_dir(sub / "metrics", sub / "models", metric, model_prefix=prefix)
        if hit:
            hits.append((hit[0], hit[1], sub.name))
    hits.sort(key=lambda h: h[0], reverse=True)
    return hits


def best_in_embedding_root(root, metric, prefixed=False):
    """The single best (score, path, subdirectory_name) under `root`, or None."""
    hits = ranked_in_embedding_root(root, metric, prefixed=prefixed)
    return hits[0] if hits else None


def best_of(*candidates):
    """max() over (score, ...) tuples, ignoring any None candidates."""
    candidates = [c for c in candidates if c is not None]
    return max(candidates, key=lambda c: c[0]) if candidates else None


# --------------------------------------------------------------------------- #
# Per-method-family resolvers. Each returns:
#   {"method": <label>, "score": float, "path": Path, "kind": str, "detail": str}
# or None if unavailable for that dataset. `kind` tells upload_entry() how to
# push the artifact: "hf_checkpoint" (whole folder), "pickle"/"torch_ckpt"
# (single file), or "missing" (scored, but no artifact survives on disk).
# --------------------------------------------------------------------------- #

def _lora_csv_candidates(csv_path, metric):
    """Parse one LoRA test-results CSV into (score, model_family) pairs.
    Three schemas exist across the project's history; detect by column names
    rather than by dataset, since that's what actually varies."""
    df = pd.read_csv(csv_path)
    if "Model" in df.columns:
        # Old-store (BACE/BBBP/FLAVOR): explicit model-family column.
        col = metric if metric in df.columns else "MCC"
        return list(zip(df[col], df["Model"]))

    # New-store: a "BEST_MODEL"/"best_checkpoint" path column instead of a
    # plain name; the model family is the path segment before the run-id.
    path_col = "BEST_MODEL" if "BEST_MODEL" in df.columns else "best_checkpoint"
    col = "MCC" if "MCC" in df.columns else "eval_Spearman"
    families = (re.search(r"/([^/]+)/[^/]+/?$", str(p)) for p in df[path_col])
    return [(score, m.group(1)) for score, m in zip(df[col], families) if m]


def resolve_lora_hf(dataset, cfg):
    """E2E LoRA across the 3 HuggingFace CLM families (ChemBERTa-MLM,
    ChemBERTa-MTR, MolFormer): best (model, loss) pair, resolved to its
    consolidated `<loss>_<DATASET>/<Model>_LoRA_Finetuned/` checkpoint."""
    lora_root = RESULTS / "lora_finetuning" / dataset.lower()
    candidates = []  # (score, model_family, loss_key)
    for csv_path in lora_root.glob("test_results_*/*.csv"):
        loss_key = csv_path.parent.name.removeprefix("test_results_")  # fl / wl / huber
        candidates += [(score, family, loss_key) for score, family in _lora_csv_candidates(csv_path, cfg["metric"])]

    best = best_of(*candidates)
    if best is None:
        return None
    score, model_family, loss_key = best
    ckpt_dir = cfg["store"] / f"{LOSS_DIR_PREFIX[loss_key]}_{cfg['effichem_case']}" / f"{model_family}_LoRA_Finetuned"
    if not ckpt_dir.exists():
        print(f"    [warn] resolved E2E LoRA-HF winner but checkpoint missing: {ckpt_dir}", file=sys.stderr)
        return None
    return {"method": "E2E LoRA -- HF", "score": score, "path": ckpt_dir,
            "kind": "hf_checkpoint", "detail": f"{model_family}, {loss_key}"}


def resolve_lora_unimol(dataset, cfg):
    """E2E LoRA on Uni-Mol: best loss variant by primary metric."""
    csv_path = RESULTS / "unimol_finetuning" / dataset.lower() / "unimol_lora_metrics.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    col = cfg["metric"] if cfg["metric"] in df.columns else "MCC"
    row = df.loc[df[col].idxmax()]
    loss_prefix = LOSS_LABEL_PREFIX.get(row["ML_Model"])
    if loss_prefix is None:
        return None
    ckpt_dir = cfg["store"] / f"{loss_prefix}_{cfg['effichem_case']}" / "dptech__Uni__Mol_LoRA_Finetuned"
    if not ckpt_dir.exists():
        print(f"    [warn] resolved E2E LoRA-UniMol winner but checkpoint missing: {ckpt_dir}", file=sys.stderr)
        return None
    return {"method": "E2E LoRA -- Uni-Mol", "score": row[col], "path": ckpt_dir,
            "kind": "hf_checkpoint", "detail": str(row["ML_Model"])}


def resolve_ft_embed(dataset, cfg, pc):
    """FT-Embed / FT-Embed+PC: best of the HF-CLM branch and the Uni-Mol
    branch, matching the paper's "best of HF and Uni-Mol" convention.

    HF-CLM results for BACE/BBBP/FLAVOR live under EffiChem_Extras with
    embedding-prefixed filenames (non-PC variant only); everything else
    lives under PEARL/results/ft_embeddings with plain filenames."""
    suffix = "PC_FT_Results" if pc else "FT_Results"
    if dataset in OLD_STORE_DATASETS:
        hf_root, hf_prefixed = EFFICHEM / f"{cfg['effichem_case']}_{suffix}", not pc
    else:
        hf_root, hf_prefixed = RESULTS / "ft_embeddings" / f"{dataset}_{suffix}", False

    hf_best = best_in_embedding_root(hf_root, cfg["metric"], prefixed=hf_prefixed)
    um_best = best_in_embedding_root(RESULTS / "ft_embeddings" / f"UniMol_{dataset}_{suffix}", cfg["metric"])

    best = best_of(hf_best, um_best)
    if best is None:
        return None
    score, path, detail = best
    return {"method": "FT Embed+PC" if pc else "FT Embed", "score": score,
            "path": path, "kind": "pickle", "detail": detail}


def resolve_pc_only(dataset, cfg):
    root = RESULTS / "pc_only" / f"{dataset}_PC_Only_Results"
    hit = best_of_tree_dir(root / "metrics", root / "models", cfg["metric"])
    if not hit:
        return None
    score, path = hit
    return {"method": "PC-only", "score": score, "path": path, "kind": "pickle", "detail": path.stem}


def _resolve_graph_baseline(dataset, cfg, method, subdir, ckpt_relpath, extra_relpaths=()):
    """`extra_relpaths` are sidecar files (relative to the same `<dataset>_<method>_Results`
    root as `ckpt_relpath`) required to actually reload the checkpoint -- e.g.
    Chemprop's `args.json`, which chemprop.predict needs alongside model.pt."""
    root = RESULTS / "gnn" / subdir / f"{dataset}_{method}_Results"
    metrics_file = root / "metrics" / "test_metrics.json"
    ckpt = root / ckpt_relpath
    if not (metrics_file.exists() and ckpt.exists()):
        return None
    score = read_metric_file(metrics_file).get(cfg["metric"])
    if score is None:
        return None
    extra = [root / p for p in extra_relpaths if (root / p).exists()]
    return {"method": method, "score": score, "path": ckpt, "kind": "torch_ckpt", "extra": extra,
            "detail": "D-MPNN" if method == "Chemprop" else "GINConv"}


resolve_chemprop = functools.partial(_resolve_graph_baseline, method="Chemprop", subdir="chemprop",
                                      ckpt_relpath=Path("final_model/fold_0/model_0/model.pt"),
                                      extra_relpaths=(Path("final_model/args.json"),))
resolve_gcn = functools.partial(_resolve_graph_baseline, method="GCN", subdir="gcn",
                                 ckpt_relpath=Path("final_model/model.pt"))


def resolve_rafe(dataset, cfg):
    """RAFE = FT-Embed+PC + ZINC-250k retrieval features, under results/rag/
    (HF-CLM) and results/rag_unimol/ (Uni-Mol). Some historical configs'
    .pkl files were never retained on disk (only their metrics were) -- when
    the single best-scoring config is one of those, fall back to the next-best
    RAFE config that still has a saved artifact, rather than giving up on RAFE
    entirely. Only if none of them have a surviving artifact is `kind` set to
    "missing" so the caller can skip and promote the next method.

    Casing here is its own thing, independent of `effichem_case`: rag/rag_unimol
    use plain lower-case for BACE/BBBP/FLAVOR (verified via direct listing --
    `results/rag/bbbp`, not `results/rag/BBBP`, even though BBBP's LoRA/FT
    dirs in EffiChem_Extras ARE upper-case) and the dataset's own upper-case
    key for HERG/DILI/CACO2/HALF_LIFE."""
    case = dataset.lower() if dataset in OLD_STORE_DATASETS else dataset
    candidates = (ranked_in_embedding_root(RESULTS / "rag" / case, cfg["metric"])
                  + ranked_in_embedding_root(RESULTS / "rag_unimol" / case, cfg["metric"]))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)

    for score, path, detail in candidates:
        if path.exists():
            return {"method": "RAFE", "score": score, "detail": detail, "path": path, "kind": "pickle"}

    score, path, detail = candidates[0]  # best config's artifact is gone; report it for the skip warning
    return {"method": "RAFE", "score": score, "detail": detail, "path": None, "kind": "missing"}


RESOLVERS = [
    resolve_pc_only,
    resolve_chemprop,
    resolve_gcn,
    resolve_lora_hf,
    resolve_lora_unimol,
    functools.partial(resolve_ft_embed, pc=False),
    functools.partial(resolve_ft_embed, pc=True),
    resolve_rafe,
]


def get_ranked_candidates(dataset):
    """Every method's (score, artifact) for `dataset`, best first. Methods
    with no resolvable score are dropped; a scored-but-artifact-missing
    method (RAFE for BACE/BBBP/FLAVOR) is kept with kind="missing" so
    top5_uploadable() can skip and promote correctly."""
    cfg = DATASETS[dataset]
    rows = []
    for resolver in RESOLVERS:
        try:
            hit = resolver(dataset, cfg)
        except Exception as e:
            print(f"    [error] {resolver} failed for {dataset}: {e}", file=sys.stderr)
            hit = None
        if hit is not None:
            rows.append(hit)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def top5_uploadable(dataset):
    """Rank all methods, then take the first 5 with an artifact that actually
    exists on disk, logging + skipping any higher-ranked method that doesn't."""
    chosen = []
    for rank, row in enumerate(get_ranked_candidates(dataset), start=1):
        if row["kind"] == "missing":
            print(f"  [warn] {dataset}: rank {rank} ({row['method']}, "
                  f"{DATASETS[dataset]['metric']}={row['score']:.4f}) has no saved "
                  f"artifact on disk -- skipping, promoting next method.")
            continue
        chosen.append(row)
        if len(chosen) == 5:
            break
    return chosen


# --------------------------------------------------------------------------- #
# Hugging Face upload
# --------------------------------------------------------------------------- #

def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def make_model_card(dataset, cfg, rank, entry):
    return f"""# PEARL -- {dataset} -- rank {rank}: {entry['method']}

Part of the PEARL benchmark (Parameter-Efficient Adaptation with Retrieval-Augmented
Learning for Molecular Property Prediction). This is the #{rank} best-performing
method on **{dataset}** ({cfg['task']}), ranked by {cfg['metric']}.

- **Method**: {entry['method']}
- **Configuration**: {entry['detail']}
- **{cfg['metric']}**: {entry['score']:.4f}

Code: https://github.com/raghvendra5688/PEARL
"""


def make_repo_readme(all_rankings):
    """Top-level README for the consolidated repo: one table per dataset
    linking to each rank's subfolder."""
    lines = ["---", "tags:", "- pearl", "- molecular-property-prediction", "---", "",
             "# PEARL Benchmark -- Top-5 Models per Dataset", "",
             "Parameter-Efficient Adaptation with Retrieval-Augmented Learning for "
             "Molecular Property Prediction. Each dataset folder below holds its "
             "top-5 best-performing methods, ranked by the dataset's primary metric.",
             "", "Code: https://github.com/raghvendra5688/PEARL", ""]
    for dataset, top5 in all_rankings.items():
        cfg = DATASETS[dataset]
        lines.append(f"## {dataset} ({cfg['task']}, ranked by {cfg['metric']})")
        lines.append("")
        lines.append(f"| Rank | Method | {cfg['metric']} | Configuration | Path |")
        lines.append("|---|---|---|---|---|")
        for rank, entry in enumerate(top5, start=1):
            subdir = f"{dataset}/rank{rank}_{slugify(entry['method'])}"
            lines.append(f"| {rank} | {entry['method']} | {entry['score']:.4f} | "
                          f"{entry['detail']} | [{subdir}](./{subdir}) |")
        lines.append("")
    return "\n".join(lines)


def _stage_as_safetensors(folder_path, staging_dir):
    """Copy an HF-checkpoint folder into `staging_dir`, converting any plain
    `.pt` state dicts (e.g. Uni-Mol's `pytorch_model.pt` / `unimol_encoder.pt`,
    which predate `save_pretrained`/safetensors) into `.safetensors` files.
    Files that aren't pure tensor-only state dicts are copied through as-is."""
    import shutil
    import torch
    from safetensors.torch import save_file

    staging_dir.mkdir(parents=True, exist_ok=True)
    for src in Path(folder_path).iterdir():
        if src.suffix != ".pt":
            shutil.copy2(src, staging_dir / src.name)
            continue
        state_dict = torch.load(src, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and all(torch.is_tensor(v) for v in state_dict.values()):
            save_file(state_dict, staging_dir / f"{src.stem}.safetensors")
        else:
            shutil.copy2(src, staging_dir / src.name)  # not a plain state dict -- leave untouched
    return staging_dir


def upload_entry(api, repo_id, dataset, cfg, rank, entry):
    """Upload one method's artifact into `<dataset>/rank<rank>_<method>/` inside
    the single consolidated `repo_id` repo."""
    from huggingface_hub import upload_file, upload_folder

    subdir = f"{dataset}/rank{rank}_{slugify(entry['method'])}"

    readme_path = Path(f"/tmp/README_{slugify(repo_id)}_{slugify(subdir)}.md")
    readme_path.write_text(make_model_card(dataset, cfg, rank, entry))
    upload_file(path_or_fileobj=str(readme_path), path_in_repo=f"{subdir}/README.md", repo_id=repo_id)

    if entry["kind"] == "hf_checkpoint":
        folder = _stage_as_safetensors(entry["path"], Path(f"/tmp/hf_stage_{slugify(repo_id)}_{slugify(subdir)}"))
        upload_folder(folder_path=str(folder), repo_id=repo_id, path_in_repo=subdir)
    else:
        upload_file(path_or_fileobj=str(entry["path"]), path_in_repo=f"{subdir}/{entry['path'].name}", repo_id=repo_id)
        for extra_path in entry.get("extra", []):
            upload_file(path_or_fileobj=str(extra_path), path_in_repo=f"{subdir}/{extra_path.name}", repo_id=repo_id)
    return subdir


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                     help="Restrict to specific datasets (default: all 7).")
    ap.add_argument("--hf-user", default=None, help="Hugging Face username/org to upload under.")
    ap.add_argument("--repo-name", default="pearl-benchmark-models",
                     help="Name of the single consolidated repo (default: pearl-benchmark-models).")
    ap.add_argument("--do-upload", action="store_true",
                     help="Actually create the repo and upload. Without this flag, the script only prints the ranking (dry run).")
    ap.add_argument("--public", action="store_true", help="Create a public repo (default: private).")
    args = ap.parse_args()

    if args.do_upload and not args.hf_user:
        ap.error("--do-upload requires --hf-user")

    api = None
    repo_id = f"{args.hf_user}/{args.repo_name}" if args.hf_user else None
    if args.do_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"Logged in to Hugging Face as: {api.whoami()['name']}")
        api.create_repo(repo_id, private=not args.public, exist_ok=True)
        print(f"Uploading everything into: https://huggingface.co/{repo_id}\n")

    all_rankings = {}
    for dataset in args.datasets:
        cfg = DATASETS[dataset]
        print(f"=== {dataset} ({cfg['task']}, ranked by {cfg['metric']}) ===")
        top5 = top5_uploadable(dataset)
        all_rankings[dataset] = top5
        for rank, entry in enumerate(top5, start=1):
            print(f"  {rank}. {entry['method']:<22} {cfg['metric']}={entry['score']:.4f}  "
                  f"[{entry['detail']}]\n     -> {entry['path']}")
            if args.do_upload:
                subdir = upload_entry(api, repo_id, dataset, cfg, rank, entry)
                print(f"     uploaded -> https://huggingface.co/{repo_id}/tree/main/{subdir}")
        if len(top5) < 5:
            print(f"  [warn] only {len(top5)} uploadable methods found for {dataset} (expected 5).")
        print()

    if args.do_upload:
        from huggingface_hub import upload_file
        readme_path = Path("/tmp/README_pearl_benchmark_models.md")
        readme_path.write_text(make_repo_readme(all_rankings))
        upload_file(path_or_fileobj=str(readme_path), path_in_repo="README.md", repo_id=repo_id)
        print(f"Top-level README uploaded. Repo: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
