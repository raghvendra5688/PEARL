"""
Each PEARL dataset's top 5 contains exactly one RAFE entry (whichever single
embedding config won RAFE's own internal ranking -- see resolve_rafe() in
upload_top5_to_hf.py). For every dataset where that RAFE entry made the top
5, upload its ZINC-250k FAISS index into the same consolidated Hugging Face
repo produced by upload_top5_to_hf.py, alongside its `rank<N>_rafe/` model
folder -- so the retrieval corpus needed to actually run RAFE inference
travels with the model.

Which embedding "wins" RAFE for each dataset (and thus which FAISS index is
needed) is read live from upload_top5_to_hf.py's own ranking logic -- nothing
is hardcoded here.

FAISS indices live under two external stores, always lower-cased by dataset,
split into HF-CLM vs. Uni-Mol subtrees:
    {EFFICHEM|EFFICHEM_V2}/rag_indices/<dataset>/<embedding>.index         (HF-CLM)
    {EFFICHEM|EFFICHEM_V2}/rag_indices_unimol/<dataset>/<embedding>.index (Uni-Mol)

Usage:
    # Inspect which datasets need an index upload -- uploads nothing.
    python upload_rafe_faiss_indices.py

    # Actually upload (requires `huggingface-cli login` first, or HF_TOKEN set).
    python upload_rafe_faiss_indices.py --hf-user <your-hf-username> --do-upload
"""

import argparse
import sys

from upload_top5_to_hf import DATASETS, EFFICHEM, EFFICHEM_V2, OLD_STORE_DATASETS, slugify, top5_uploadable


def faiss_index_path(dataset, embedding_name):
    store = EFFICHEM if dataset in OLD_STORE_DATASETS else EFFICHEM_V2
    subdir = "rag_indices_unimol" if embedding_name.lower().startswith("unimol") else "rag_indices"
    return store / subdir / dataset.lower() / f"{embedding_name}.index"


def find_rafe_in_top5(dataset):
    """(rank, entry) for this dataset's one RAFE candidate if it made the
    uploadable top 5 (at whatever rank), else (None, None)."""
    for rank, entry in enumerate(top5_uploadable(dataset), start=1):
        if entry["method"] == "RAFE":
            return rank, entry
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                     help="Restrict to specific datasets (default: all 7).")
    ap.add_argument("--hf-user", default=None, help="Hugging Face username/org the repo lives under.")
    ap.add_argument("--repo-name", default="pearl-benchmark-models",
                     help="Consolidated repo to upload into (default: pearl-benchmark-models).")
    ap.add_argument("--do-upload", action="store_true",
                     help="Actually upload. Without this flag, the script only prints what it would do.")
    args = ap.parse_args()

    if args.do_upload and not args.hf_user:
        ap.error("--do-upload requires --hf-user")

    repo_id = f"{args.hf_user}/{args.repo_name}" if args.hf_user else None
    api = None
    if args.do_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"Logged in to Hugging Face as: {api.whoami()['name']}")
        print(f"Uploading into: https://huggingface.co/{repo_id}\n")

    for dataset in args.datasets:
        rank, entry = find_rafe_in_top5(dataset)
        if entry is None:
            print(f"{dataset}: RAFE not in top 5 -- skipping.")
            continue

        index_path = faiss_index_path(dataset, entry["detail"])
        if not index_path.exists():
            print(f"{dataset}: RAFE is rank {rank} ({entry['detail']}) but no FAISS index found at "
                  f"{index_path} -- skipping.", file=sys.stderr)
            continue

        size_mb = index_path.stat().st_size / 1e6
        subdir = f"{dataset}/rank{rank}_{slugify(entry['method'])}/faiss_index"
        print(f"{dataset}: RAFE rank {rank} ({entry['detail']}), index={index_path} ({size_mb:.0f} MB)")
        print(f"  -> {subdir}/{index_path.name}")

        if args.do_upload:
            from huggingface_hub import upload_file
            upload_file(path_or_fileobj=str(index_path), path_in_repo=f"{subdir}/{index_path.name}", repo_id=repo_id)
            print(f"  uploaded -> https://huggingface.co/{repo_id}/tree/main/{subdir}")


if __name__ == "__main__":
    main()
