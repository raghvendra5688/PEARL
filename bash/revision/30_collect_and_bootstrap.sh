#!/bin/bash
# After all array tasks finish: paired bootstrap on the new datasets (CPU, ~1 h). Run from the repo root.
source bash/revision/00_env_hpc.sh
$PY -u scripts/revision/e06_paired_bootstrap.py  > logs/revision/e06_hpc.log  2>&1
$PY -u scripts/revision/e06b_paired_all.py e1 e2 e3 > logs/revision/e06b_hpc.log 2>&1
bash scripts/revision/guard_large_files.sh   # no output file may exceed 50 MB
echo "now copy results/revision/{e1,e2,e3,e06} back to the local repo (rsync -av)"
