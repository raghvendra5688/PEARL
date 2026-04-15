#!/bin/bash -l
# Embed ZINC-250k with BACE-finetuned Uni-Mol models (FL and WL).
# Output: EffiChem_Extras/zinc_embeddings_unimol/bace/UniMol_{FL,WL}.npy
# Shape: (249455, 2560)  — batch_size=16, ~4–6 hours on H200
#SBATCH -J embed_zinc_unimol_bace
#SBATCH -o out_embed_zinc_unimol_bace.log
#SBATCH -e out_embed_zinc_unimol_bace.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=128000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd003

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

SCRIPT_DIR="$REPO_ROOT/scripts/unimol/rag"
python3 "$SCRIPT_DIR/embed_zinc250k_unimol.py" --dataset bace
