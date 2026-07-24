#!/bin/bash -l
# Regenerate FLAVOR's RAG feature CSVs (data/rag_features/flavor/*.csv), which no
# longer exist on disk -- needed to backfill Precision/Recall/AUC onto FLAVOR's
# RAFE metrics (rag_modelling_flavor.py originally only computed Accuracy/F1/MCC/
# per-class AUPR for its multiclass path; patched to also compute macro Precision/
# Recall/AUC, matching pc_only_modelling.py's convention).
#
# Runs on numpy 1.26.4 (this env's numpy), so uses a small hybrid PEARL_EXTRAS
# override at EffiChem_Extras_v2/legacy_flavor_fix instead of the original
# EffiChem_Extras directly: FLAVOR's existing rag_indices/flavor/meta.pkl and the
# 6 saved rag_pca/flavor/*_pca.pkl models were pickled under numpy 2.x and fail to
# unpickle under numpy 1.26.4 ("ModuleNotFoundError: No module named
# 'numpy._core.numeric'"). The hybrid dir symlinks in everything that loads fine
# regardless of numpy version (FAISS .index files, plain-CSV embeddings/PC
# features) and supplies a freshly rebuilt meta.pkl. No old PCA file exists at
# this new path, so rag_feature_extraction.py will fit fresh PCA-32 models on the
# train split (same random_state=42, same deterministic FAISS retrieval, so
# centroids -- and the resulting PCA -- should closely reproduce the originals)
# and save them under this same hybrid dir.
#
# Submit before run_rag_modelling_flavor.sh.
#SBATCH -J rag_features_flavor
#SBATCH -o out_rag_features_flavor.log
#SBATCH -e out_rag_features_flavor.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=120000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd002

module load cuda12.6/toolkit/12.6.2
python --version
nvidia-smi
nvcc --version

export MAMBA_EXE='/export/home/rmall/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/export/home/rmall/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"
fi
unset __mamba_setup

micromamba env list
micromamba activate effichem
python3 -c "import numpy; print('numpy', numpy.__version__)"

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

export PEARL_EXTRAS=/export/qcai-omics/Raghvendra/EffiChem_Extras_v2/legacy_flavor_fix

SCRIPT_DIR="$REPO_ROOT/scripts/smiles/rag"
python3 "$SCRIPT_DIR/rag_feature_extraction.py" --dataset flavor --gpu-id 0
