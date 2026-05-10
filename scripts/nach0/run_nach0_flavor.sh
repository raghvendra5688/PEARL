#!/bin/bash -l
#SBATCH -J nach0_flavor
#SBATCH -o /export/cse/rmall/Raghvendra/PEARL/scripts/nach0/logs/out_nach0_flavor.log
#SBATCH -e /export/cse/rmall/Raghvendra/PEARL/scripts/nach0/logs/out_nach0_flavor.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd001

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

micromamba activate effichem

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/logs"

# Full Nach0 instruction fine-tuning (SMILES text only, HuggingFace Seq2SeqTrainer).
# Model weights → EffiChem_Extras/nach0/flavor/
# Small outputs  → results/nach0/flavor/
#
# Override defaults via environment variables, e.g.:
#   EPOCHS=5 BATCH=8 sbatch run_nach0_flavor.sh
python3 "$SCRIPT_DIR/nach0_flavor.py" \
    --epochs      "${EPOCHS:-10}"    \
    --batch-size  "${BATCH:-16}"     \
    --lr          "${LR:-1e-4}"      \
    --weight-decay "${WD:-0.01}"     \
    --max-input-len "${MAX_LEN:-512}" \
    --num-workers "${NUM_WORKERS:-4}"
