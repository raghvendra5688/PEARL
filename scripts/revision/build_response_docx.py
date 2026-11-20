"""Response letter as .docx for collaborators: figures rasterised from the vector PDFs, tables kept as real Word tables.
The PDFs stay the canonical figures; the PNGs here exist only for the Word file. Run after build_response.py."""
import re, subprocess, sys, shutil
from pathlib import Path
import pypdfium2 as pdfium
REPO = Path(__file__).resolve().parents[2]; MS = REPO / "manuscript"
md = (MS / "PEARL_Response_to_Reviewers_Draft.md").read_text()
png = REPO.parent / "EffiChem_Extras_v2" / "PEARL_revision" / "response_png"   # rasterised copies live outside the repo
png.mkdir(parents=True, exist_ok=True)
DPI = int(sys.argv[1]) if len(sys.argv) > 1 else 220
def raster(rel):
    src = (MS / rel).resolve()
    if not src.exists(): print("  missing:", rel); return None
    out = png / (src.stem + ".png")
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        doc = pdfium.PdfDocument(str(src)); im = doc[0].render(scale=DPI / 72).to_pil().convert("RGB")
        if im.width > 2400: im = im.resize((2400, round(im.height * 2400 / im.width)))
        im.save(out, "PNG", optimize=True)
    return out
n = 0
def sub(m):
    global n
    p = raster(m.group(2))
    if p is None: return m.group(0)
    n += 1; return f"![{m.group(1)}]({p})"
md2 = re.sub(r"!\[(.*?)\]\((\.\./[^)]+\.pdf)\)", sub, md, flags=re.S)
tmp = MS / "_response_docx.md"; tmp.write_text(md2)
docx = MS / "PEARL_Response_to_Reviewers.docx"
cmd = ["pandoc", tmp.name, "-f", "markdown+pipe_tables-smart", "-t", "docx", "--resource-path", f"{MS}:{png}", "-o", docx.name]
ref = MS / "response_reference.docx"
if ref.exists(): cmd += ["--reference-doc", ref.name]
subprocess.run(cmd, cwd=MS, check=True); tmp.unlink()
print(f"wrote {docx} ({docx.stat().st_size/1e6:.2f} MB); {n} figures embedded at {DPI} dpi")
