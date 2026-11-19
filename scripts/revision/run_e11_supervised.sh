#!/bin/bash
# E11 variants, detached and auto-retrying (cells are idempotent). Stage 1: small Half-Life (8 workers); stage 2: large hERG/Flavor with 3 workers (shared-GPU memory limits).
source /export/qcai-omics/Raghvendra/PEARL/scripts/revision/env.sh
cd /export/qcai-omics/Raghvendra/PEARL
run() { for try in $(seq 1 30); do
  echo "$(date) E11 [$1] workers $2 try $try" >> logs/revision/e11_supervisor.log
  $PY -u scripts/revision/e11_variants.py --cfgs "$1" --workers $2 --nj 4 >> logs/revision/e11.log 2>&1 && { echo "$(date) E11 [$1] done" >> logs/revision/e11_supervisor.log; return; }
  sleep 120; done; }
run half_life_mf 8
run herg_mf 3
run flavor_mf 3
