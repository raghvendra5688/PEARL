# PEARL revision: commands to run on the HPC (H200)

Scope: E1 (controlled representation matrix + RAFE group ablation), E2 (retrieval control) and E3 (k sweep) on all 7 datasets (8 configurations:
`flavor_mf herg_mf bace_mf bace_um dili_um bbbp_mf caco2_mtr half_life_mf`), 3 trees (XGBoost, LightGBM, CatBoost) x 5 seeds, identical splits and tuning budget
(Optuna 20 trials, 5-fold CV on the training split; test set used once per fit). Cells already finished locally are skipped. E6b was dropped; E5b has no script yet.

## 1. Setup (the H200 nodes share /export/qcai-omics, as the other bash/*/run_*.sh scripts assume)
```bash
cd /export/qcai-omics/Raghvendra/PEARL
mkdir -p logs/revision
```
`bash/revision/00_env_hpc.sh` activates micromamba env `effichem` and sets `EXTRAS_V1`, `EXTRAS_V2`, `REV_NGPU`. If the filesystem is not shared, copy the repo, `EffiChem_Extras`, `EffiChem_Extras_v2` and set those variables.

## 2. Smoke test (about 10 minutes, one cell on one GPU)
```bash
srun -p gpu-H200 -A <ACCOUNT> -q <QOS> --gres=gpu:1 -c 8 --mem=32G --time=01:00:00 --pty bash
cd /export/qcai-omics/Raghvendra/PEARL && source bash/revision/00_env_hpc.sh
$PY -u scripts/revision/e01_03_rafe_experiments.py --exp e1 --cfgs herg_mf --seeds 1 --clfs LightGBM --workers 2 --nj 4 --shard 0/40
```
(use the account and QoS written in the `#SBATCH` lines of `10_e1_e3_scaleup.sbatch`).

## 3. Full run
```bash
sbatch bash/revision/10_e1_e3_scaleup.sbatch              # Slurm array --array=0-7: 8 independent jobs
sbatch --array=0-1 bash/revision/10_e1_e3_scaleup.sbatch  # only Flavor and hERG first
sbatch --cpus-per-task=64 --mem=160000 bash/revision/10_e1_e3_scaleup.sbatch   # smaller request (script sets --workers 16 = 64 CPUs / 4)
squeue -u $USER ; squeue -j <JOBID> --start
tail -n 5 logs/revision/hpc_e1e3_<JOBID>_<TASK>.log
ls results/revision/e1/flavor_mf | grep -c json           # finished E1 cells (target 180 per configuration)
scancel <JOBID>
```
- **How the work is split:** array task *i* handles configuration *i* from `CFGS`. Inside a task, `--workers` cells run in parallel (4 threads each); XGBoost cells share the single H200, LightGBM/CatBoost use the CPUs. The task runs E1, E2, E3 in order.
- **Cells per configuration:** E1 12 arms (PC_only, FT_only, FT+PC, +A, +B, +C, +D, full RAFE, full-A, full-B, full-C, full-D) x 3 trees x 5 seeds = 180; E2 (CLM-cosine, Morgan) 30 + random 5 draws x 3 x 5 seeds = 75, total 105; E3 6 values of k x 3 x 5 = 90. Total 375 per configuration, about 3,000 over 8.
- **Why:** E1 matrix = Reviewer 2.1 (same algorithm, partitions, tuning budget for every representation); ablation = R1.3, R2.3, R3.1; E2 = R3.1 (random and Morgan retrieval); E3 = R1.3 (number of neighbours); 3 trees x 5 seeds = R2.1, R3.3.
- **Resume:** cells are idempotent. After a time-out, resubmit the same command (or `--array=<failed tasks>`).
- **Cheaper:** edit `--clfs XGBoost,LightGBM,CatBoost` to `XGBoost,LightGBM` (CatBoost is the long pole).

## 4. Collect and analyse
```bash
bash bash/revision/30_collect_and_bootstrap.sh      # pooled paired bootstrap for E1-E3, 50 MB file guard
```
Files above 50 MB belong in `/export/qcai-omics/Raghvendra/EffiChem_Extras_v2/revision_large/`, not in the repo.

## 5. Time estimates (not measured on this cluster)
Per cell (20 trials x 5-fold CV): XGBoost on GPU 5-15 min, LightGBM 2-5 min, CatBoost 15-60 min. With 24 workers (96 CPUs): about 8-20 h for small datasets, 16-40 h for Flavor and hERG; with 16 workers add 50%. If all 8 jobs start together: 1-2 days; one slice at a time: about a week.
