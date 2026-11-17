# PEARL revision: daily status log
Auto-appended at about 23:30 server time by `scripts/revision/daily_status.py`.

### 2026-11-14
- Result files (json cells / outputs): e1=162/162, e04_baselines=314/314, e05=123, e2=0, e3=0, e06=0, e07_coverage=0, e09_selection_audit=0, e12_pareto=0
- Running now: e05_unimol_3d_vs_ecfp.py --workers 6; run_queue.sh
- Queue finished: no
- Large-file guard: guard ok Sat Nov 14 02:19:44 PM +03 2026
- Git: 5f368c2 Add revision experiments: RAFE ablation (E1), fingerprint baselines (E4), ZINC coverage (E7), selection audit (E9), Pareto plot (E12), LoRA figure v3 | uncommitted entries: 7

### 2026-11-15
- Result files (json cells / outputs): e1=162/162, e04_baselines=314/314, e05=180, e2=45, e3=90, e06=0, e07_coverage=0, e09_selection_audit=0, e12_pareto=0
- Running now: nothing
- Queue finished: yes
- Large-file guard: guard ok Sun Nov 15 11:21:38 PM +03 2026
- Git: 5f368c2 Add revision experiments: RAFE ablation (E1), fingerprint baselines (E4), ZINC coverage (E7), selection audit (E9), Pareto plot (E12), LoRA figure v3 | uncommitted entries: 12

### 2026-11-16
- Result files (json cells / outputs): e1=546/162, e04_baselines=314/314, e05=180, e2=130, e3=178, e06=0, e07_coverage=0, e09_selection_audit=0, e12_pareto=0
- Running now: e01_03_rafe_experiments.py --exp $e --cfgs bace_mf --seeds 5 --clfs Li; e01_03_rafe_experiments.py --exp e3 --cfgs bace_mf --seeds 5 --clfs Li
- Queue finished: yes
- Large-file guard: guard ok Mon Nov 16 11:23:15 PM +03 2026
- Git: b798ff1 Revision: reconstructed E1-E3 design, HPC scripts, paired-bootstrap analysis, matched Pareto, stored-RAFE audit | uncommitted entries: 413
