#!/bin/bash
# E5b conformer sensitivity (BACE), detached and auto-retrying; idempotent. Log: logs/revision/e05b.log
source /export/qcai-omics/Raghvendra/PEARL/scripts/revision/env.sh
cd /export/qcai-omics/Raghvendra/PEARL
for try in $(seq 1 10); do
  echo "$(date) E5b try $try" >> logs/revision/e05b_supervisor.log
  $PY -u scripts/revision/e05b_conformer_sensitivity.py --workers 8 --nj 4 >> logs/revision/e05b.log 2>&1 && { echo "$(date) E5b done" >> logs/revision/e05b_supervisor.log; break; }
  sleep 60
done
