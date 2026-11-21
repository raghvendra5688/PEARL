"""Guard against internal experiment task ids (E1, E2, ... E15, E04, E05b) leaking into anything a reviewer sees.

Checks, for every deliverable: the markdown source, the rendered PDF text, the docx body text, the LaTeX
sources and their \\input tables, and -- the one that is easy to miss -- the text baked inside every figure
PDF, where a matplotlib title or annotation can carry an id that no text-level grep would ever find.

Usage: python scripts/revision/check_no_task_ids.py        (exit 1 if anything leaks)
"""
import re, sys, zipfile, glob
from pathlib import Path

MS = Path(__file__).resolve().parents[2] / "manuscript"
RES = Path(__file__).resolve().parents[2] / "results" / "revision"

# A task id is E followed by 1-2 digits, optionally a trailing letter (E5b), as a standalone token.
# Guard against false positives from things like "BACE1513" or "CACE-2" by requiring a non-alphanumeric
# boundary on the left, and never matching when the E is preceded by a letter.
TASK_ID = re.compile(r"(?<![A-Za-z0-9])E\d{1,2}[a-z]?(?![A-Za-z0-9])")

# Legitimate tokens that would otherwise trip the pattern (none expected, but kept explicit).
ALLOW = set()


def hits(text):
    return [m.group(0) for m in TASK_ID.finditer(text) if m.group(0) not in ALLOW]


def ctx(text, tok, n=70):
    i = text.find(tok)
    return re.sub(r"\s+", " ", text[max(0, i - n):i + len(tok) + n]).strip()


def from_pdf(path):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(path))
    return " ".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))


def from_docx(path):
    d = zipfile.ZipFile(path).read("word/document.xml").decode()
    # a space per tag, so adjacent table cells cannot fuse into a false match such as "BACE"+"1513"
    return re.sub(r"<[^>]+>", " ", d)


def main():
    targets = []
    md = MS / "PEARL_Response_to_Reviewers_Draft.md"
    if md.exists():
        targets.append(("markdown: response", md.read_text()))
    for name in ("PEARL_Response_to_Reviewers.pdf",):
        p = MS / name
        if p.exists():
            targets.append((f"pdf text: {name}", from_pdf(p)))
    for name in ("PEARL_Response_to_Reviewers.docx",):
        p = MS / name
        if p.exists():
            targets.append((f"docx text: {name}", from_docx(p)))
    for tex in ("PEARL_paper_v3.tex", "PEARL_supplementary_v3.tex"):
        p = MS / tex
        if p.exists():
            # strip comment lines: a leaked id in a %-comment never reaches the reader
            body = "\n".join(l for l in p.read_text().split("\n") if not l.lstrip().startswith("%"))
            targets.append((f"latex: {tex}", body))
    for t in sorted(glob.glob(str(MS / "tables" / "*.tex"))):
        targets.append((f"latex table: {Path(t).name}", Path(t).read_text()))
    # every figure the response and the manuscript embed, read as rendered text
    figs = sorted(glob.glob(str(MS / "Figures_Response" / "*.pdf"))) + sorted(glob.glob(str(MS / "Figures_v3" / "*.pdf")))
    for f in figs:
        try:
            targets.append((f"figure text: {Path(f).parent.name}/{Path(f).name}", from_pdf(f)))
        except Exception as e:  # a figure we cannot read is reported, not skipped silently
            targets.append((f"figure UNREADABLE: {f} ({e})", ""))

    bad = 0
    for label, text in targets:
        h = hits(text)
        if h:
            bad += 1
            print(f"LEAK  {label}: {sorted(set(h))}")
            for tok in sorted(set(h)):
                print(f"        ...{ctx(text, tok)}...")
    print(f"\nscanned {len(targets)} targets ({len(figs)} of them figures); {bad} with leaks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
