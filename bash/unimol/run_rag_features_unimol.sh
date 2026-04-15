#!/bin/bash -l
# Extract Uni-Mol ZINC-250k RAG features for all 4 datasets.
# Run AFTER run_build_zinc_index_unimol.sh completes.
# Also fits and saves PCA models to EffiChem_Extras/rag_pca_unimol/.
# Output: data/rag_features_unimol/{dataset}/UniMol_{FL,WL}_{train,eval,test}_rag.csv
#SBATCH -J rag_features_unimol
#SBATCH -o out_rag_features_unimol.log
#SBATCH -e out_rag_features_unimol.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=120000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -w crirdchpxd005

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

SCRIPT_DIR="$REPO_ROOT/scripts/unimol/rag"

for DATASET in bace bbbp clintox flavor; do
    echo "========================================"
    echo "Uni-Mol RAG feature extraction: $DATASET"
    echo "========================================"
    python3 "$SCRIPT_DIR/rag_feature_extraction_unimol.py" --dataset "$DATASET" --gpu-id 0
done

echo "All Uni-Mol RAG features extracted."
