#!/bin/bash -l
#SBATCH -J unimol_ft_modelling_flavor
#SBATCH -o logs/out_unimol_ft_modelling_flavor.log
#SBATCH -e logs/out_unimol_ft_modelling_flavor.err
#SBATCH -p gpu-H200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256000
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

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

mkdir -p logs

SCRIPT_DIR="$REPO_ROOT/scripts/unimol/ml"

# Run flavor dataset × both configs (FL and WL) — 6 models total
# Override via environment variables if needed:
#   OPTUNA_TRIALS=30 sbatch run_unimol_ft_modelling.sh
#   N_JOBS=32        sbatch run_unimol_ft_modelling.sh
python3 "$SCRIPT_DIR/unimol_modelling.py" \
    --dataset flavor \
    --config both \
    --trials "${OPTUNA_TRIALS:-20}" \
    --jobs "${N_JOBS:-96}"
