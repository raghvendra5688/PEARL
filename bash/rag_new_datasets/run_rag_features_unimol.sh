#!/bin/bash -l
# Extract ZINC-250k RAG features for herg/dili/caco2/half_life (Uni-Mol modality).
# herg/dili: 2 models x 3 splits; caco2/half_life: 1 model x 3 splits.
# Also fits and saves PCA models to $PEARL_EXTRAS_V2/rag_pca_unimol/.
# Submit after run_build_zinc_index_unimol.sh completes.
#SBATCH -J rag_features_unimol_new
#SBATCH -o /export/qcai-omics/Raghvendra/PEARL/logs/rag_new_datasets/out_rag_features_unimol.log
#SBATCH -e /export/qcai-omics/Raghvendra/PEARL/logs/rag_new_datasets/out_rag_features_unimol.err
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

SCRIPT_DIR="$REPO_ROOT/scripts/unimol/rag"

for DATASET in herg dili caco2 half_life; do
    echo "========================================"
    echo "Uni-Mol RAG feature extraction: $DATASET"
    echo "========================================"
    python3 "$SCRIPT_DIR/rag_feature_extraction_unimol_new_datasets.py" --dataset "$DATASET" --gpu-id 0
done

echo "All Uni-Mol RAG features extracted (new datasets)."
