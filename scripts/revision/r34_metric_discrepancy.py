"""R3.4: metric discrepancies from the supplement tables: (a) Half-Life Spearman vs R2/MAE; (b) DILI MCC vs AUC/Accuracy/F1 (ranking agreement and winner per metric)."""
import pandas as pd, numpy as np; from scipy.stats import spearmanr
R = "results/revision/e09_selection_audit/"; OUT = "results/revision/r34/"; import os; os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(R + "all_configs_parsed.csv")
h = d[d.dataset == "half_life"].copy(); h = h.rename(columns={"R": "R2"})
print(h[["Modality", "CLM", "ML", "R2", "MAE", "Spearman", "MAE (h)"]].head(6).to_string())
rows = [dict(scope="all configs", n=len(h), spearman_vs_R2=spearmanr(h.Spearman, h.R2).correlation, spearman_vs_negMAE=spearmanr(h.Spearman, -h["MAE (h)"]).correlation)]
for m, g in h.groupby("Modality"): rows.append(dict(scope=m, n=len(g), spearman_vs_R2=spearmanr(g.Spearman, g.R2).correlation, spearman_vs_negMAE=spearmanr(g.Spearman, -g["MAE (h)"]).correlation, max_Spearman=g.Spearman.max(), R2_at_best_Spearman=g.loc[g.Spearman.idxmax(), "R2"], max_R2=g.R2.max()))
h2 = h.sort_values("Spearman", ascending=False)[["Modality", "CLM", "ML", "Spearman", "R2", "MAE (h)"]].head(8)
pd.DataFrame(rows).round(3).to_csv(OUT + "half_life_metric_agreement.csv", index=False); h2.round(3).to_csv(OUT + "half_life_top_configs.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string()); print(h2.round(3).to_string())
di = d[d.dataset == "dili"]; rows = []
for m in ("RAFE", "PC+FT", "FT Embed", "E2E LoRA"):
    g = di[di.Modality == m]; r = dict(modality=m, n=len(g))
    for a in ("AUC", "Acc", "F1_mac", "MCC"): r[f"best_{a}"] = round(g[a].max(), 3); r[f"best_{a}_cfg"] = "-".join(map(str, g.loc[g[a].idxmax(), ["CLM", "Loss", "ML"]]))
    r["rho_MCC_AUC"] = spearmanr(g.MCC, g.AUC).correlation if len(g) > 3 else np.nan; rows.append(r)
pd.DataFrame(rows).round(3).to_csv(OUT + "dili_metric_agreement.csv", index=False); print(pd.DataFrame(rows).round(3).T.to_string())
