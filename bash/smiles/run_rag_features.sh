#!/bin/bash -l
# Extract ZINC-250k RAG features for all 4 datasets (72 CSVs total:
# 6 models × 3 splits × 4 datasets). Also fits and saves PCA models
# to EffiChem_Extras/rag_pca/. Submit after run_build_zinc_index.sh completes.
#SBATCH -J zinc_rag_features
#SBATCH -o out_rag_features.txt
#SBATCH -e out_rag_features.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=120000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd001

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

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

SCRIPT_DIR="$REPO_ROOT/scripts/smiles/rag"

for DATASET in bace bbbp clintox flavor; do
    echo "========================================"
    echo "RAG feature extraction: $DATASET"
    echo "========================================"
    python3 "$SCRIPT_DIR/rag_feature_extraction.py" --dataset "$DATASET" --gpu-id 0
done

echo "All RAG features extracted."
