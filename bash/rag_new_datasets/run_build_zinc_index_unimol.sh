#!/bin/bash -l
# Submit after run_embed_zinc_unimol.sh completes.
#SBATCH -J zinc_index_unimol_new
#SBATCH -o /export/qcai-omics/Raghvendra/PEARL/logs/rag_new_datasets/out_build_index_unimol.log
#SBATCH -e /export/qcai-omics/Raghvendra/PEARL/logs/rag_new_datasets/out_build_index_unimol.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=120000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
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
    alias micromamba="$MAMBA_EXE"
fi
unset __mamba_setup

micromamba env list
micromamba activate effichem
export LD_LIBRARY_PATH="$MAMBA_ROOT_PREFIX/envs/effichem/lib:$LD_LIBRARY_PATH"

REPO_ROOT="/export/qcai-omics/Raghvendra/PEARL"
export PEARL_EXTRAS_V2="/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"

python3 "$REPO_ROOT/scripts/unimol/rag/build_zinc_index_unimol_new_datasets.py" --dataset all
