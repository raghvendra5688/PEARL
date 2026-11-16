#!/bin/bash
# runs daily_status.py once a day at ~23:30 server time
source /export/qcai-omics/Raghvendra/PEARL/scripts/revision/env.sh
while true; do
  now=$(date +%s); t=$(date -d "today 23:30" +%s); [ $t -le $now ] && t=$(date -d "tomorrow 23:30" +%s)
  sleep $((t-now)); $PY /export/qcai-omics/Raghvendra/PEARL/scripts/revision/daily_status.py >> /export/qcai-omics/Raghvendra/PEARL/logs/revision/daily_status.log 2>&1
  sleep 120
done
