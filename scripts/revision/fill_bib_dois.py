"""Look up a DOI for every bibliography entry that lacks one, and report what it found.

Reviewer R2.8 asked for consistent bibliographic detail with DOIs throughout. This queries Crossref by
title (plus first author and year when present) and only accepts a match when the returned title is a
close match to ours, so that a wrong DOI is never silently written. Nothing is edited in place unless
--write is passed; by default it prints a report for inspection.

Usage: python scripts/revision/fill_bib_dois.py [--write]
"""
import re, sys, json, time, urllib.parse, urllib.request
from pathlib import Path
from difflib import SequenceMatcher

BIB = Path(__file__).resolve().parents[2] / "manuscript" / "sn-bibliography.bib"
UA = "PEARL-bibliography-check/1.0 (mailto:noreply@example.org)"  # generic, no personal detail


def norm(t):
    t = re.sub(r"[{}\\$]", "", t).lower()
    return re.sub(r"[^a-z0-9 ]+", " ", t).strip()


def entries(src):
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", src, flags=re.S):
        yield m.group(0), m.group(1), m.group(2).strip(), m.group(3)


def field(body, name):
    m = re.search(rf"{name}\s*=\s*\{{+(.+?)\}}+\s*,\s*\n", body, flags=re.S | re.I)
    if not m:
        m = re.search(rf"{name}\s*=\s*\{{+(.+?)\}}+\s*\n", body, flags=re.S | re.I)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def crossref(title, author, year):
    q = {"query.bibliographic": title, "rows": "3", "select": "DOI,title,author,issued,container-title"}
    if author:
        q["query.author"] = author
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25) as r:
        items = json.load(r)["message"]["items"]
    best = None
    for it in items:
        t = (it.get("title") or [""])[0]
        sim = SequenceMatcher(None, norm(title), norm(t)).ratio()
        yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        if best is None or sim > best[0]:
            best = (sim, it.get("DOI"), t, yr, (it.get("container-title") or [""])[0])
    return best


def main():
    write = "--write" in sys.argv
    src = BIB.read_text()
    todo = [(raw, key, body) for raw, typ, key, body in entries(src) if "doi" not in body.lower()]
    print(f"{len(todo)} entries without a DOI\n")
    found, unsure, missing = [], [], []
    for raw, key, body in todo:
        title = field(body, "title")
        if not title:
            missing.append((key, "no title")); continue
        author = (field(body, "author") or "").split(" and ")[0].split(",")[0].strip() or None
        year = field(body, "year")
        try:
            sim, doi, t, yr, jrnl = crossref(title, author, year)
        except Exception as e:
            missing.append((key, f"lookup failed: {e}")); continue
        yr_ok = (not year or not yr or abs(int(yr) - int(re.sub(r"\D", "", year) or 0)) <= 1)
        # only accept a recognised publisher / preprint-server prefix; aggregator DOIs
        # (e.g. 10.52202, 10.65215) point at a reprint rather than the canonical record
        TRUSTED = ("10.1021/", "10.1038/", "10.1093/", "10.1016/", "10.1109/", "10.1007/",
                   "10.1162/", "10.1124/", "10.18653/", "10.1101/", "10.21203/", "10.2139/",
                   "10.26434/", "10.1186/", "10.1039/", "10.1145/", "10.1371/", "10.1002/",
                   "10.1080/", "10.13", "10.1126/", "10.1609/")
        if sim >= 0.90 and yr_ok and doi and doi.startswith(TRUSTED):
            found.append((key, doi, t, sim))
        elif sim >= 0.90 and yr_ok:
            unsure.append((key, doi, title, t, sim, yr, year))
        else:
            unsure.append((key, doi, title, t, sim, yr, year))
        time.sleep(0.4)  # be polite to the API

    print(f"=== confident matches ({len(found)}) ===")
    for k, doi, t, sim in found:
        print(f"  {k:28s} {doi:40s} sim={sim:.2f}  {t[:60]}")
    print(f"\n=== needs a human look ({len(unsure)}) ===")
    for k, doi, ours, theirs, sim, yr, year in unsure:
        print(f"  {k:28s} sim={sim:.2f} year {year} vs {yr}\n      ours:   {ours[:85]}\n      found:  {(theirs or '')[:85]}  {doi}")
    print(f"\n=== could not look up ({len(missing)}) ===")
    for k, why in missing:
        print(f"  {k:28s} {why}")

    if write and found:
        out = src
        for k, doi, _t, _s in found:
            pat = re.compile(r"(@\w+\{" + re.escape(k) + r",)", re.S)
            out = pat.sub(lambda m: m.group(1) + f"\n  doi     = {{{doi}}},", out, count=1)
        BIB.write_text(out)
        print(f"\nwrote {len(found)} DOIs into {BIB.name}")
    elif found:
        print("\n(dry run; pass --write to insert the confident matches)")


if __name__ == "__main__":
    main()
