#!/bin/bash -l
#SBATCH -J pearl_e15
#SBATCH --array=0-6                # 7 independent jobs, one per dataset (see DSS below)
#SBATCH -p gpu-H200
#SBATCH -A H200_Collaborator
#SBATCH -q H200_Collaborator_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96         # CPU allotment per job (max available)
#SBATCH --gres=gpu:1               # one H200 per job (XGBoost + retrieval use it; the trees on CPU use the 96 cores)
#SBATCH --mem=256000
#SBATCH --time=72:00:00            # cells are idempotent, so resubmit to resume if a job times out
#SBATCH --output=logs/revision/hpc_e15_%A_%a.log
# E15: full configuration grid with seeds. For ONE dataset the job runs every (embedding model x loss) configuration:
#   classification datasets 8 configs (ChemBERTa-MLM, ChemBERTa-MTR, MolFormer, Uni-Mol, each with FL and WL), regression datasets 4 configs (Huber),
#   x 3 stages (FT Embed, FT Embed+PC, RAFE) x 3 trees x 5 seeds = 360 cells (classification) or 180 cells (regression) per dataset.
# Purpose: the paper's principle is to find the best configuration per dataset; seed-averaging makes that choice robust, and cv_best (train-only CV score) allows selection without test data.
# Submit from the repo root (mkdir -p logs/revision first):  sbatch bash/revision/20_e15_full_grid.sbatch
source /export/qcai-omics/Raghvendra/PEARL/bash/revision/00_env_hpc.sh
DSS=(flavor herg bace dili bbbp caco2 half_life); DS=${DSS[$SLURM_ARRAY_TASK_ID]}
CFGS=$($PY -c "import json;print(','.join(k for k in json.load(open('scripts/revision/grid_catalog.json')) if k.startswith('$DS@')))")
echo "task=$SLURM_ARRAY_TASK_ID ds=$DS cpus=$SLURM_CPUS_PER_TASK configs=$CFGS"
$PY -u scripts/revision/e01_03_rafe_experiments.py --exp e15 --cfgs $CFGS --seeds 5 \
    --clfs XGBoost,LightGBM,CatBoost --workers 16 --nj 4
