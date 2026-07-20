#!/bin/bash -l
#SBATCH -J bace_gcn
#SBATCH -o /export/qcai-omics/Raghvendra/PEARL/logs/gnn/gcn/out_bace_gcn.log
#SBATCH -e /export/qcai-omics/Raghvendra/PEARL/logs/gnn/gcn/out_bace_gcn.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd002

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
python3 "$REPO_ROOT/scripts/gnn/gcn_baseline.py" --dataset bace
