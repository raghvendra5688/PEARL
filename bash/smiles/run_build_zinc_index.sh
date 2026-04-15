#!/bin/bash -l
# Build FAISS-GPU FlatIP indices for all 4 datasets (24 indices total).
# Runs sequentially per dataset; each dataset takes ~5-10 min on A100.
# Submit after ALL 4 embed jobs have completed.
#SBATCH -J zinc_build_index
#SBATCH -o out_zinc_build_index.txt
#SBATCH -e out_zinc_build_index.err
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
    echo "Building FAISS index: $DATASET"
    echo "========================================"
    python3 "$SCRIPT_DIR/build_zinc_index.py" --dataset "$DATASET" --gpu-id 0
done

echo "All FAISS indices built."
