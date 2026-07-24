#!/bin/bash -l
# RAG-augmented modelling (HF): [embedding + PC + RAFE] -> XGBoost/LightGBM/CatBoost
# for herg/dili (classification) and caco2/half_life (regression).
# Submit after run_rag_features_hf.sh completes.
#SBATCH -J rag_model_hf_new
#SBATCH -o /export/qcai-omics/Raghvendra/PEARL/logs/rag_new_datasets/out_rag_model_hf.log
#SBATCH -e /export/qcai-omics/Raghvendra/PEARL/logs/rag_new_datasets/out_rag_model_hf.err
#SBATCH -p gpu-H200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=512000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -w crirdchpxd001

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
export PEARL_EXTRAS_V2="/export/qcai-omics/Raghvendra/EffiChem_Extras_v2"

python3 "$REPO_ROOT/scripts/smiles/ml/rag_modelling_new_datasets.py" --dataset all
