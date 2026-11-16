"""Append an end-of-day status entry to manuscript/PEARL_Revision_Experiment_Plan.md (idempotent per date)."""
import glob, os, datetime, subprocess, re
ROOT="/export/qcai-omics/Raghvendra/PEARL"; MD=f"{ROOT}/manuscript/PEARL_Revision_Experiment_Plan.md"
today=datetime.date.today().isoformat()
hdr="## 12. Daily status log"
txt=open(MD).read()
if hdr not in txt: txt+=f"\n\n{hdr}\n(Auto-appended at ~23:30 server time by `scripts/revision/daily_status.py`; narrative notes added by hand in live sessions.)\n"
if f"### {today}" in txt.split(hdr)[1]: raise SystemExit("already logged")
n=lambda d: len(glob.glob(f"{ROOT}/results/revision/{d}/**/*.json",recursive=True))
tot={"e1":162,"e04_baselines":314}
rows=[(d,n(d)) for d in ["e1","e04_baselines","e05","e2","e3","e06","e07_coverage","e09_selection_audit","e12_pareto"]]
run=subprocess.run("ps -eo cmd | grep -E 'scripts/revision/(e0|run_queue)' | grep -v grep | sed 's#.*scripts/revision/##' | cut -c1-70",shell=True,capture_output=True,text=True).stdout.strip().replace("\n","; ")
q=os.path.exists(f"{ROOT}/logs/revision/queue.done")
big=subprocess.run(f"tail -n1 {ROOT}/logs/revision/guard.log",shell=True,capture_output=True,text=True).stdout.strip()
git=subprocess.run(f"cd {ROOT} && git log -1 --format='%h %s' && git status --short | wc -l",shell=True,capture_output=True,text=True).stdout.strip().replace("\n"," | uncommitted entries: ")
e=f"\n### {today}\n- Result files (json cells / outputs): "+", ".join(f"{d}={c}"+(f"/{tot[d]}" if d in tot else "") for d,c in rows)+f"\n- Running now: {run or 'nothing'}\n- Queue finished: {'yes' if q else 'no'}\n- Large-file guard: {big}\n- Git: {git}\n"
open(MD,"w").write(txt.rstrip("\n")+"\n"+e)
print(e)
