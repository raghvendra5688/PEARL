#!/bin/bash -l
#SBATCH -J rag_model_flavor
#SBATCH -o out_rag_model_flavor.log
#SBATCH -e out_rag_model_flavor.err
#SBATCH -p gpu-H200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd006

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
python3 -c "import numpy; print('numpy', numpy.__version__)"

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

# Must match run_rag_features_flavor.sh's PEARL_EXTRAS override: this env's numpy
# (1.26.4) can't unpickle FLAVOR's original meta.pkl/PCA models (pickled under
# numpy 2.x), so that script rebuilt them under this hybrid dir. The
# PC_FT_All_Embeddings CSVs symlinked in here are identical to the originals
# either way (plain text, no pickle version issue).
export PEARL_EXTRAS=/export/qcai-omics/Raghvendra/EffiChem_Extras_v2/legacy_flavor_fix

SCRIPT_DIR="$REPO_ROOT/scripts/smiles/rag"
python3 "$SCRIPT_DIR/rag_modelling_flavor.py"
