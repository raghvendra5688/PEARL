#!/bin/bash -l
#SBATCH -J nach0_flavor_ft
#SBATCH -o /export/cse/rmall/Raghvendra/PEARL/scripts/nach0/logs/out_nach0_flavor_ft.log
#SBATCH -e /export/cse/rmall/Raghvendra/PEARL/scripts/nach0/logs/out_nach0_flavor_ft.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
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
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "logs"

# Nach0 LoRA instruction fine-tuning (SMILES text only, manual training loop).
# LoRA adapter weights → EffiChem_Extras/nach0/flavor_ft/
# Small outputs        → results/nach0/flavor_ft/
#
# Override defaults via environment variables, e.g.:
#   EPOCHS=5 LORA_R=8 sbatch run_nach0_flavor_finetune.sh
#python3 "nach0_flavor_finetune.py" \
#    --epochs      "${EPOCHS:-10}"    \
#    --batch-size  "${BATCH:-16}"     \
#    --lr          "${LR:-1e-4}"      \
#    --weight-decay "${WD:-0.01}"     \
#    --grad-clip   "${GRAD_CLIP:-1.0}" \
#    --lora-r      "${LORA_R:-16}"    \
#    --max-input-len "${MAX_LEN:-512}" \
#    --num-workers "${NUM_WORKERS:-4}"

python3 "nach0_flavor_finetune.py" \
    --sweep \
    --sweep-trials "${SWEEP_TRIALS:-20}" \
    --sweep-epochs "${SWEEP_EPOCHS:-10}" \
    --epochs "${EPOCHS:-10}" \
    --batch-size "${BATCH:-16}" \
    --max-input-len "${MAX_LEN:-512}" \
    --wandb-project "${WANDB_PROJECT:-nach0-flavor-ft}"
