#!/bin/bash -l
# Extract Uni-Mol embeddings for all 4 datasets (train/eval/test splits).
# Produces data/unimol_embeddings/{Dataset}_Embeddings/{prefix}_{split}_embed.csv
# with UniMol_FL_embeddings and UniMol_WL_embeddings columns (2560-dim each).
#SBATCH -J unimol_embeddings_flavor
#SBATCH -o out_unimol_embeddings_flavor.log
#SBATCH -e out_unimol_embeddings_flavor.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=120000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -w crirdchpxd006

module load cuda12.6/toolkit/12.6.2
nvidia-smi

export MAMBA_EXE='/export/home/rmall/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/export/home/rmall/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"
fi
unset __mamba_setup

micromamba activate effichem

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

SCRIPT_DIR="$REPO_ROOT/scripts/unimol/finetuning"

for DATASET in flavor; do
    echo "========================================"
    echo "Uni-Mol embedding extraction: $DATASET"
    echo "========================================"
    python3 "$SCRIPT_DIR/unimol_embeddings.py" --dataset "$DATASET"
done

echo "All Uni-Mol embeddings extracted."
