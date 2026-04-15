#!/bin/bash -l
#SBATCH -J unimol_bbbp_fl
#SBATCH -o logs/out_unimol_bbbp_fl.log
#SBATCH -e logs/out_unimol_bbbp_fl.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd004

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

mkdir -p logs

SCRIPT_DIR="$REPO_ROOT/scripts/unimol/finetuning"
python3 "$SCRIPT_DIR/finetune_unimol_bbbp_fl.py"
