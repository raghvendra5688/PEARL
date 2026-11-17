#!/bin/bash
# Local supervisor: runs the queued commands one after another; a command that exits non-zero (killed, crashed) is retried up to 20 times.
# All cells are idempotent (finished cells are skipped), so a restart only repeats unfinished work. Started with setsid, so it survives the Claude session.
source /export/qcai-omics/Raghvendra/PEARL/scripts/revision/env.sh
cd /export/qcai-omics/Raghvendra/PEARL
LOG=logs/revision/supervisor.log
ec() { $PY -u scripts/revision/e01_03_rafe_experiments.py --exp e15 --cfgs "$1" --seeds 5 --clfs XGBoost,LightGBM --workers 10 --nj 4; }
grid() { ds=$1; $PY -c "import json;print(','.join(k for k in json.load(open('scripts/revision/grid_catalog.json')) if k.startswith('$ds@')))"; }
JOBS=()
for ds in half_life caco2 bbbp dili bace; do JOBS+=("$ds"); done
for ds in "${JOBS[@]}"; do
  for try in $(seq 1 20); do
    echo "$(date) start E15 $ds try $try" >> $LOG
    ec "$(grid $ds)" > logs/revision/e15_${ds}_local.log 2>&1 && { echo "$(date) done E15 $ds" >> $LOG; break; }
    echo "$(date) E15 $ds exited non-zero, retrying in 60 s" >> $LOG; sleep 60
  done
done
echo "$(date) supervisor finished all local jobs" >> $LOG
