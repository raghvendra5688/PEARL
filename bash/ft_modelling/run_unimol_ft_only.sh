#!/bin/bash -l
#SBATCH -J unimol_ft_only
#SBATCH -o /export/qcai-omics/Raghvendra/PEARL/logs/ft_modelling/out_unimol_ft_only.log
#SBATCH -e /export/qcai-omics/Raghvendra/PEARL/logs/ft_modelling/out_unimol_ft_only.err
#SBATCH -p gpu-H200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
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
export LD_LIBRARY_PATH="$MAMBA_ROOT_PREFIX/envs/effichem/lib:$LD_LIBRARY_PATH"
export N_JOBS=80

REPO_ROOT="/export/qcai-omics/Raghvendra/PEARL"
python3 "$REPO_ROOT/scripts/smiles/ml/ft_modelling_new_datasets.py" --modality unimol --mode ft_only --dataset all
