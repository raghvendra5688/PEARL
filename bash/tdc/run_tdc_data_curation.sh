#!/bin/bash -l
#SBATCH -J tdc_data_curation
#SBATCH -o out_tdc_data_curation.log
#SBATCH -e out_tdc_data_curation.err
#SBATCH -p gpu-H200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64000
#SBATCH -A H200
#SBATCH -q h200_qos

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

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
python3 "$REPO_ROOT/scripts/data_curation/tdc_data_curation.py"
