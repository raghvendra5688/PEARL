"""Audit pass of the academic-humanizer rules (Layers 1-2) on a LaTeX file. Lists tells with line numbers; does not edit.
Usage: humanizer_audit.py file.tex [--integrity prehumanized.tex] (integrity: numbers, \\cite keys, \\ref/\\label, equations must be identical as multisets)"""
import re, sys, collections
f = sys.argv[1]; L = open(f).read().split("\n")
BAN = r"delve|underscore|intricate|tapestry|testament|pivotal|showcas|foster|leverag|realm|seamless|groundbreaking|paves? the way|sheds? light|revolutioni|transformative|cutting-edge|notably|importantly|it is worth noting|extensive experiments|comprehensive|a wide range of|to the best of our knowledge|for the first time|in recent years|with the rapid development|moreover|furthermore|additionally|genuinely|candid|considerably|remarkabl|crucial"
OVER = r"\b(demonstrate[sd]?|prove[sd]?|establish(es|ed)?|guarantee[sd]?|confirm(s|ed)?)\b"
VAGUE = r"\b(somewhat|relatively|fairly|to some extent|quite)\b"
def prose(l): return re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?|[{}$~]|%.*$", " ", l)
out = collections.Counter()
for n, l in enumerate(L, 1):
    if l.strip().startswith("%") or "\\bibliography" in l: continue
    t = prose(l)
    for name, pat in (("em-dash", r"—|\s--\s|\s-\s"), ("banned/AI vocabulary", BAN), ("over-claiming verb", OVER), ("vague hedge", VAGUE), ("novelty word", r"\bnovel\b")):
        for m in re.finditer(pat, l if name == "em-dash" else t, flags=re.I): out[name] += 1; print(f"{n:4d} [{name}] ...{(l if name=='em-dash' else t)[max(0,m.start()-40):m.end()+40].strip()}")
    for m in re.finditer(r"\bsignificant(ly)?\b", t, flags=re.I):
        ctx = l[max(0, m.start() - 160):m.end() + 160]
        if not re.search(r"interval|\bp\s*[<=]|p\$|Wilcoxon|CI|excludes", ctx): out["significant without test"] += 1; print(f"{n:4d} [significant without test] ...{t[max(0,m.start()-50):m.end()+50].strip()}")
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", t):
        w = len(sent.split())
        if w > 40: out["sentence >40 words"] += 1; print(f"{n:4d} [{w} words] {sent.strip()[:110]}...")
print("\nsummary:", dict(out))
if "--integrity" in sys.argv:
    A = open(sys.argv[sys.argv.index("--integrity") + 1]).read(); B = open(f).read()
    def feats(x):
        x = re.sub(r"%.*", "", x)
        return {"numbers": collections.Counter(re.findall(r"(?<![A-Za-z\\])-?\d+(?:\.\d+)?", re.sub(r"\\cite[a-z]*\{[^}]*\}|\\(?:ref|label)\{[^}]*\}", "", x))), "cite": collections.Counter(k.strip() for c in re.findall(r"\\cite[a-z]*\{([^}]*)\}", x) for k in c.split(",")),
                "ref/label": collections.Counter(re.findall(r"\\(?:ref|label)\{[^}]*\}", x)), "equations": collections.Counter(re.findall(r"\\begin\{equation\}.*?\\end\{equation\}", x, flags=re.S))}
    fa, fb = feats(A), feats(B); bad = False
    for k in fa:
        if fa[k] != fb[k]: bad = True; print("INTEGRITY FAIL", k, "removed:", dict(fa[k] - fb[k]), "added:", dict(fb[k] - fa[k]))
    print("integrity:", "FAILED" if bad else "OK (numbers, citations, refs, equations unchanged)")
