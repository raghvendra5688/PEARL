#!/bin/bash -l
#SBATCH -J pearl_e15l
#SBATCH --array=1-6                # tasks 1-6 of the E15 dataset list: 1=herg 2=bace 3=dili 4=bbbp 5=caco2 6=half_life (task 0, Flavor, is a separate job)
#SBATCH -p gpu-H200
#SBATCH -A H200_Collaborator
#SBATCH -q H200_Collaborator_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         # light request: 8 parallel cells x 4 threads
#SBATCH --mem=64000
#SBATCH --time=72:00:00            # cells are idempotent, so resubmit to resume if a job times out
#SBATCH --output=logs/revision/hpc_e15l_%A_%a.log
# E15 (full configuration grid), light version. The XGBoost and LightGBM cells of BACE, DILI, BBBP, Caco-2 and Half-Life already exist from the local runs,
# so tasks 2-6 run CatBoost only (CPU; no GPU requested). hERG (task 1) runs all three trees and needs one GPU for XGBoost:
#   sbatch --array=1   --gres=gpu:1 bash/revision/21_e15_light.sbatch
#   sbatch --array=2-6 bash/revision/21_e15_light.sbatch
# Submit from the repo root after: mkdir -p logs/revision
source /export/qcai-omics/Raghvendra/PEARL/bash/revision/00_env_hpc.sh
DSS=(flavor herg bace dili bbbp caco2 half_life); DS=${DSS[$SLURM_ARRAY_TASK_ID]}
CFGS=$($PY -c "import json;print(','.join(k for k in json.load(open('scripts/revision/grid_catalog.json')) if k.startswith('$DS@')))")
if [ "$DS" = "herg" ]; then CLFS=XGBoost,LightGBM,CatBoost; else CLFS=CatBoost; fi
echo "task=$SLURM_ARRAY_TASK_ID ds=$DS clfs=$CLFS cpus=$SLURM_CPUS_PER_TASK gpus=$REV_NGPU host=$(hostname) configs=$CFGS"
$PY -u scripts/revision/e01_03_rafe_experiments.py --exp e15 --cfgs $CFGS --seeds 5 \
    --clfs $CLFS --workers $((SLURM_CPUS_PER_TASK / 4)) --nj 4
