#!/bin/bash -l
# Generate 3D conformers (ETKDGv3 + MMFF94) for all EffiChem-2.0 datasets.
# CPU-only job — no GPU needed.
#SBATCH -J conformer_gen
#SBATCH -o out_conformer_gen.log
#SBATCH -e out_conformer_gen.err
#SBATCH -p gpu-H200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd004

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

SCRIPT_DIR="$REPO_ROOT/scripts/unimol/finetuning"

for DATASET in bace bbbp clintox flavor; do
    echo "========================================"
    echo "Conformer generation: $DATASET"
    echo "========================================"
    python3 "$SCRIPT_DIR/smiles_to_conformers.py" --dataset "$DATASET"
done

echo "All conformers generated."
