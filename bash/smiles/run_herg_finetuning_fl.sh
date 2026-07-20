#!/bin/bash -l
#SBATCH -J herg_ft_fl
#SBATCH -o /export/qcai-omics/Raghvendra/PEARL/logs/smiles/finetuning/out_herg_ft_fl.log
#SBATCH -e /export/qcai-omics/Raghvendra/PEARL/logs/smiles/finetuning/out_herg_ft_fl.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=140000
#SBATCH -A H200
#SBATCH -q h200_qos
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
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<

micromamba env list
micromamba activate effichem
export LD_LIBRARY_PATH="$MAMBA_ROOT_PREFIX/envs/effichem/lib:$LD_LIBRARY_PATH"

REPO_ROOT="/export/qcai-omics/Raghvendra/PEARL"
python3 "$REPO_ROOT/scripts/smiles/finetuning/finetune_herg_fl.py"
