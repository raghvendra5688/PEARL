"""LaTeX longtables for the revised supplement (paired intervals E1, E11 variants). Output: manuscript/tables/supp_*.tex"""
import pandas as pd, numpy as np, re
def esc(v):
    t = str(v)
    for a, b in (("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")): t = t.replace(a, b)
    return t
def ci(r): return f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]".replace("-", "$-$")
def longtable(df, cols, spec, cap, label, name):
    head = " & ".join(cols) + " \\\\"
    rows = "\n".join(" & ".join(esc(v) for v in r) + " \\\\" for r in df.values)
    open(f"manuscript/tables/{name}.tex", "w").write(f"\\begin{{longtable}}{{{spec}}}\n\\caption{{{cap}}}\\label{{{label}}}\\\\\n\\toprule\n{head}\n\\midrule\n\\endfirsthead\n\\toprule\n{head}\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n{rows}\n\\end{{longtable}}\n")
p = pd.read_csv("results/revision/e06/paired_e1.csv"); p = p[p.A.isin(["full_RAFE", "base_FT+PC"]) & p.B.isin(["FT_only", "base_FT+PC", "PC_only"]) & (p.A != p.B)]
nm = {"full_RAFE": "RAFE", "base_FT+PC": "FT Embed+PC", "FT_only": "FT Embed", "PC_only": "PC-only"}
d = pd.DataFrame([(r.cfg, r.clf, f"{nm[r.A]} minus {nm[r.B]}", r.metric, ci(r._asdict() if False else r)) for _, r in p.iterrows()], columns=["Configuration", "Tree", "Comparison", "Metric", "Difference [95\\% CI]"])
longtable(d, list(d.columns), "llllp{4.2cm}", "Controlled comparison (E1): paired bootstrap differences (five seeds pooled, 3{,}000 resamples) for each dataset configuration and tree.", "tab:supp_paired", "supp_paired")
c = pd.read_csv("results/revision/e11_analysis/comparisons.csv"); c = c[c.own_tree]
d2 = pd.DataFrame([(r.dataset, r.tree, r.A, r.B.replace("_", "-"), ci(r)) for _, r in c.iterrows()], columns=["Dataset", "Tree", "Variant", "Versus", "Difference [95\\% CI]"])
longtable(d2, list(d2.columns), "lllp{2.6cm}p{4.2cm}", "Label-free RAFE variants (E11), own tree of each dataset: paired bootstrap differences versus FT Embed+PC, RAFE and PC-only.", "tab:supp_variants", "supp_variants")
print(len(d), len(d2))
