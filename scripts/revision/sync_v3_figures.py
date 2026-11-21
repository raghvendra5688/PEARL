"""Keep manuscript/Figures_v3/ in step with the generated figures in results/revision/.

The v3 manuscript embeds its own copies of the response figures. Those copies go stale whenever a
figure is regenerated, which has silently shipped an out-of-date figure more than once. This copies
any that differ and reports what it did, so it can be run after every figure rebuild.
"""
import hashlib, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES, V3 = REPO / "results" / "revision", REPO / "manuscript" / "Figures_v3"

MAP = {
    "ladder_gain_v3.pdf":           "figures_response/figA_ladder_gain.pdf",
    "best_config_ranking_v3.pdf":   "figures_response/figB_best_config_ranking.pdf",
    "pareto_frontier_v3.pdf":       "e12_pareto/pareto_frontier.pdf",
    "ablation_groups_v3.pdf":       "figures_response/figC_ablation_heatmap.pdf",
    "retrieval_robustness_v3.pdf":  "figures_response/figD_retrieval_robustness.pdf",
    "umap_coverage_v3.pdf":         "e07_coverage/umap_coverage.pdf",
    "runtime_v3.pdf":               "figures_response/figE_runtime.pdf",
    "controlled_matrix_v3.pdf":     "figures_response/figF_controlled_matrix.pdf",
    "rafe_complementarity_v3.pdf":  "e10/e10_summary.pdf",
    "3d_ecfp4_conformers_v3.pdf":   "figures_response/figG_3d_vs_ecfp4_conformers.pdf",
    "grid_heatmap_v3.pdf":          "e15_analysis/e15_grid_heatmap.pdf",
    "halflife_metrics_v3.pdf":      "figures_response/figH_halflife_metrics.pdf",
}

def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None

def main():
    V3.mkdir(exist_ok=True)
    updated, missing = [], []
    for dst, rel in MAP.items():
        src = RES / rel
        if not src.exists():
            missing.append(rel); continue
        if md5(src) != md5(V3 / dst):
            shutil.copy2(src, V3 / dst); updated.append(dst)
    print(f"{len(updated)} figure(s) refreshed" + (": " + ", ".join(updated) if updated else " (all already current)"))
    for m in missing:
        print(f"  MISSING SOURCE: {m}")
    return 1 if missing else 0

if __name__ == "__main__":
    sys.exit(main())
