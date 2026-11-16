#!/bin/bash -l
#SBATCH -J pearl_e1e3
#SBATCH -p gpu-H200
#SBATCH -A H200_Collaborator
#SBATCH -q H200_Collaborator_qos
#SBATCH --ntasks=1
#SBATCH --array=0-7                # 8 independent Slurm jobs, one per dataset configuration (see CFGS below)
#SBATCH --cpus-per-task=96         # CPU allotment per job (max available)
#SBATCH --gres=gpu:1               # one H200 per job (XGBoost + retrieval use it; the trees on CPU use the 96 cores)
#SBATCH --mem=256000
#SBATCH --time=72:00:00            # cells are idempotent, so resubmit to resume if a job times out
#SBATCH --output=logs/revision/hpc_e1e3_%A_%a.log
# Controlled representation matrix + RAFE ablations (E1), retrieval-strategy control (E2) and k sweep (E3) on all 7 datasets (8 configurations).
# E1 arms: PC_only, FT_only, FT+PC, +A/+B/+C/+D, full RAFE, full-A/-B/-C/-D; E2: CLM-cosine, Morgan-Tanimoto, 5 random draws x 5 seeds; E3: k=1,5,10,20,50,100.
# Each job runs E1, E2, E3 for its dataset in turn: 3 trees x 5 seeds per arm, 24 cells in parallel x 4 threads = 96 CPUs.
# Submit from the repo root (mkdir -p logs/revision first):  sbatch bash/revision/10_e1_e3_scaleup.sbatch
source /export/qcai-omics/Raghvendra/PEARL/bash/revision/00_env_hpc.sh
CFGS=(flavor_mf herg_mf bace_mf bace_um dili_um bbbp_mf caco2_mtr half_life_mf); CFG=${CFGS[$SLURM_ARRAY_TASK_ID]}
echo "task=$SLURM_ARRAY_TASK_ID cfg=$CFG gpus=$REV_NGPU cpus=$SLURM_CPUS_PER_TASK"
for EXP in e1 e2 e3; do
  $PY -u scripts/revision/e01_03_rafe_experiments.py --exp $EXP --cfgs $CFG --seeds 5 \
      --clfs XGBoost,LightGBM,CatBoost --workers 16 --nj 4
done
