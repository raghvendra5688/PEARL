#!/bin/bash
# Source at the top of every HPC job. Same micromamba setup as the other bash/*/run_*.sh scripts.
export MAMBA_EXE='/export/home/rmall/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/export/home/rmall/micromamba'
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__mamba_setup"; else alias micromamba="$MAMBA_EXE"; fi
unset __mamba_setup
micromamba activate effichem
export PY=$(which python)
# compute nodes ship an older libstdc++ (CXXABI_1.3.15 missing -> sqlite3/optuna import fails): use the env's own libraries first
export LD_LIBRARY_PATH="$MAMBA_ROOT_PREFIX/envs/effichem/lib:${LD_LIBRARY_PATH:-}"
$PY -c "import sqlite3, optuna, xgboost, lightgbm, catboost, torch, faiss, sklearn, rdkit" || { echo "ENV CHECK FAILED on $(hostname)"; exit 1; }
# Paths: defaults assume the HPC mounts the same /export/qcai-omics filesystem as the local server (as the other bash/ scripts do).
# If it does not, set REPO, EXTRAS_V1, EXTRAS_V2 before submitting (see EffiChem_Extras_v2/PEARL_revision/PEARL_HPC_Commands.md, section 2).
export REPO=${REPO:-/export/qcai-omics/Raghvendra/PEARL}
export EXTRAS_V1=${EXTRAS_V1:-/export/qcai-omics/Raghvendra/EffiChem_Extras}
export EXTRAS_V2=${EXTRAS_V2:-/export/qcai-omics/Raghvendra/EffiChem_Extras_v2}
export REV_NGPU=${REV_NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}   # 0 -> XGBoost falls back to CPU
export OMP_NUM_THREADS=1
cd "$REPO"
mkdir -p logs/revision results/revision
