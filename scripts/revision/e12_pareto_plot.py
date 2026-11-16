"""Pareto frontier (performance vs added pipeline cost) per dataset, from the supplementary-table audit (E9 parse).
x = pipeline stage cost within the frozen-embedding family: FT Embed -> +PC (473 descriptors) -> RAFE (+ZINC retrieval).
y = median over all reported configs (CLM x loss x ML) of the headline metric (MCC / Spearman); bars = IQR; dots = every config.
Frontier uses MEDIANS (not best-of-grid) to avoid selection bias. E2E LoRA shown as a dashed reference."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R="results/revision/e09_selection_audit/"
d=pd.read_csv(R+"all_configs_parsed.csv"); mp=pd.read_csv(R+"matched_pair_differences.csv")
order=["FT Embed","PC+FT","RAFE"]; xs={m:i for i,m in enumerate(order)}
names={"bace":"BACE","bbbp":"BBBP","herg":"hERG","dili":"DILI","caco2":"Caco-2","half_life":"Half-Life","flavor":"Flavor"}
col={"FT Embed":"#8c8c8c","PC+FT":"#4c78a8","RAFE":"#d62728"}
import sys
SEL=sys.argv[1].split(",") if len(sys.argv)>1 else list(names)
tag="" if len(sys.argv)<=1 else "_"+"_".join(SEL)
names={k:names[k] for k in SEL}
n=len(names)
fig,axs=plt.subplots(1,n+1,figsize=(4.3*(n+1),3.9)) if n<7 else plt.subplots(2,4,figsize=(13,6.2)); axs=np.atleast_1d(axs).ravel(); rng=np.random.default_rng(0)
summ=[]
for ax,ds in zip(axs,names):
    s0=d[d.dataset==ds]; keys=None
    for m in order:   # MATCHED configs only: (CLM, loss, tree) present for all three stages, so medians are comparable
        k=set(zip(*[s0[s0.Modality==m][c] for c in ("CLM","Loss","ML")])); keys=k if keys is None else keys&k
    s=s0[[ (m_ in order and (a,b,c) in keys) or m_=="E2E LoRA" for m_,a,b,c in zip(s0.Modality,s0.CLM,s0.Loss,s0.ML)]]
    n_match=len(keys); med={};q={}
    for m in order:
        v=s[s.Modality==m].metric.dropna().values; med[m]=np.median(v); q[m]=np.percentile(v,[25,75])
        ax.scatter(xs[m]+rng.uniform(-.16,.16,len(v)),v,s=9,color=col[m],alpha=.28,lw=0,zorder=1)
    # frontier on medians: stage is on frontier if strictly better than all cheaper stages
    front=[];best=-9
    for m in order:
        if med[m]>best+0.005: front.append(m);best=med[m]
    ax.plot([xs[m] for m in front],[med[m] for m in front],color="k",lw=1.4,ls="-",zorder=2)
    # staircase
    for m in order:
        on=m in front
        ax.errorbar(xs[m],med[m],yerr=[[med[m]-q[m][0]],[q[m][1]-med[m]]],fmt="o",ms=11 if on else 8,mfc=col[m] if on else "white",
                    mec=col[m],mew=1.8,ecolor=col[m],elinewidth=1.6,capsize=3,zorder=3)
    lo=s[s.Modality=="E2E LoRA"].metric.median(); ax.axhline(lo,color="#2ca02c",ls="--",lw=1,zorder=0); 
    ax.text(2.42,lo,"E2E LoRA",color="#2ca02c",fontsize=6.5,va="bottom",ha="right")
    r=mp[(mp.dataset==ds)&(mp.A=="RAFE")&(mp.B=="PC+FT")]
    t=names[ds]+f" (n={n_match} matched configs)"+("  ★ RAFE on frontier" if "RAFE" in front else "")
    ax.set_title(t,fontsize=8.5,fontweight="bold",color=col["RAFE"] if "RAFE" in front else "k")
    if len(r): 
        r=r.iloc[0]; ax.text(.03,.04,f"RAFE−PC+FT: {r.mean_diff:+.3f}\nWilcoxon p={r.wilcoxon_p:.3f}",transform=ax.transAxes,fontsize=7,va="bottom")
    ax.set_xticks(range(3)); ax.set_xticklabels(["FT Embed","+PC","+RAFE"],fontsize=8); ax.set_xlim(-.5,2.5)
    ax.set_ylabel("MCC" if ds in("bace","bbbp","herg","dili","flavor") else "Spearman",fontsize=8); ax.tick_params(labelsize=7)
    ax.grid(alpha=.25,lw=.5); [ax.spines[k].set_visible(False) for k in("top","right")]
    summ.append(dict(dataset=ds,n_matched=n_match,frontier="→".join(front),**{f"median_{m}":round(med[m],3) for m in order}))
ax=axs[n]; ax.axis("off")
ax.text(0,.95,"How to read",fontweight="bold",fontsize=9,va="top")
ax.text(0,.85,"x: added pipeline cost (left→right)\ny: median over MATCHED configs\n(same CLM, loss, tree in all stages)\nbars: IQR, dots: individual configs\nblack line: Pareto frontier of medians\n(a stage is on it only if it beats\n every cheaper stage by >0.005)\nfilled = on frontier, hollow = dominated\ngreen dashed: E2E LoRA median",fontsize=7.5,va="top")
fig.suptitle("PEARL cost–performance Pareto frontier",fontsize=12,fontweight="bold")
fig.tight_layout(rect=[0,0,1,.93]); fig.savefig(R.replace("e09_selection_audit","e12_pareto")+f"pareto_frontier{tag}.pdf") if __import__("os").makedirs("results/revision/e12_pareto",exist_ok=True) is None else 0
pd.DataFrame(summ).to_csv(f"results/revision/e12_pareto/frontier_summary{tag}.csv",index=False); print(pd.DataFrame(summ).to_string())
