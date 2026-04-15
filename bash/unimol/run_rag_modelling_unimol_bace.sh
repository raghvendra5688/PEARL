#!/bin/bash -l
#SBATCH -J rag_unimol_bace
#SBATCH -o out_rag_unimol_bace.log
#SBATCH -e out_rag_unimol_bace.err
#SBATCH -p gpu-H200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=512000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -w crirdchpxd002

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
python3 "$SCRIPT_DIR/rag_modelling_bace_unimol.py"
