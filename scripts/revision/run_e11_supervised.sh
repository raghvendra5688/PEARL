#!/bin/bash
# E11 variants, detached and auto-retrying (cells are idempotent). Log: logs/revision/e11.log, supervisor: logs/revision/e11_supervisor.log
source /export/qcai-omics/Raghvendra/PEARL/scripts/revision/env.sh
cd /export/qcai-omics/Raghvendra/PEARL
for try in $(seq 1 20); do
  echo "$(date) E11 try $try" >> logs/revision/e11_supervisor.log
  $PY -u scripts/revision/e11_variants.py --workers 10 --nj 4 >> logs/revision/e11.log 2>&1 && { echo "$(date) E11 done" >> logs/revision/e11_supervisor.log; break; }
  echo "$(date) E11 exited non-zero, retry in 60 s" >> logs/revision/e11_supervisor.log; sleep 60
done
