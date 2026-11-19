#!/bin/bash
# Markdown -> LaTeX for the response letter. [PENDING: ...] markers become the red \placeholder{...} macro. Compile from manuscript/ (figure paths are relative to it).
cd /export/qcai-omics/Raghvendra/PEARL/manuscript
pandoc PEARL_Response_to_Reviewers_Draft.md -f markdown+pipe_tables-smart -s -t latex -V documentclass=article -V fontsize=10pt -V geometry:margin=2.2cm -V colorlinks=true \
  -H ../scripts/revision/response_header.tex -o /tmp/resp_raw_$$.tex
python3 - <<PY
import re
t = open("/tmp/resp_raw_$$.tex").read()
t = re.sub(r"\\\\textbf\{\{\[\}PENDING(.*?)\{\]\}\}", lambda m: "\\\\placeholder{" + m.group(1) + "}", t, flags=re.S)
t = t.replace("\\\\includegraphics{../results", "\\\\includegraphics[width=\\\\linewidth,height=0.42\\\\textheight,keepaspectratio]{../results")
open("PEARL_Response_to_Reviewers_Draft.tex", "w").write(t)
b = re.sub(r"\\\\[{}]", "", t)
print("placeholders:", t.count("\\\\placeholder{") - 1, "| figures:", t.count("\\\\includegraphics["), "| tables:", t.count("\\\\begin{longtable}"), "| braces", b.count("{"), b.count("}"), "| envs", len(re.findall(r"\\\\begin\{", t)), len(re.findall(r"\\\\end\{", t)))
PY
rm -f /tmp/resp_raw_$$.tex
