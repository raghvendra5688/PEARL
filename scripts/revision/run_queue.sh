#!/bin/bash
# sequential queue; each step waits for E1 to finish, then runs. idempotent (cells skip if done).
source /export/qcai-omics/Raghvendra/PEARL/scripts/revision/env.sh
cd /export/qcai-omics/Raghvendra/PEARL
while pgrep -f e01_03_rafe_experiments.py.*--exp.e1 >/dev/null; do sleep 30; done
$PY -u scripts/revision/e05_unimol_3d_vs_ecfp.py --workers 6 > logs/revision/e05.log 2>&1
$PY -u scripts/revision/e01_03_rafe_experiments.py --exp e2 --workers 8 --seeds 5 > logs/revision/e02.log 2>&1
$PY -u scripts/revision/e01_03_rafe_experiments.py --exp e3 --workers 8 --seeds 5 > logs/revision/e03.log 2>&1
$PY -u scripts/revision/e06_paired_bootstrap.py > logs/revision/e06.log 2>&1
echo QUEUE_DONE > logs/revision/queue.done
