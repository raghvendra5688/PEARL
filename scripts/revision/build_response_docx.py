"""Response letter as .docx for collaborators: figures rasterised from the vector PDFs, tables kept as real Word tables.
The PDFs stay the canonical figures; the PNGs here exist only for the Word file. Run after build_response.py."""
import re, subprocess, sys, shutil, zipfile
from pathlib import Path
import pypdfium2 as pdfium
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from docx.shared import Pt
REPO = Path(__file__).resolve().parents[2]; MS = REPO / "manuscript"
md = (MS / "PEARL_Response_to_Reviewers.md").read_text()
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

# pandoc writes the source PNG's absolute local path (including the server directory tree) into each image's
# pic:cNvPr descr attribute (distinct from wp:docPr descr, which correctly holds the figure caption and is kept).
# Strip it so no filesystem path reaches the shared document.
with zipfile.ZipFile(docx) as z:
    names = z.namelist(); doc_xml = z.read("word/document.xml").decode("utf-8")
    other = {nm: z.read(nm) for nm in names if nm != "word/document.xml"}
leaks = len(re.findall(r'(<pic:cNvPr\b[^>]*?)descr="[^"]*"', doc_xml))
doc_xml = re.sub(r'(<pic:cNvPr\b[^>]*?)descr="[^"]*"', r'\1descr=""', doc_xml)
with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as z:
    for nm, data in other.items(): z.writestr(nm, data)
    z.writestr("word/document.xml", doc_xml)

# compact the widest tables (same sizes chosen by hand for readability: 9pt for the 11-column
# feature-group ablation, 10pt for the full per-tree grids and the best-configuration table).
# Matched by the "Table N." caption that pandoc always places immediately before its table.
COMPACT = {"2": 9, "5a": 10, "5b": 10, "5c": 10, "10": 10}
dd = Document(docx)
body = dd.element.body
items = [Paragraph(c, dd) if c.tag == qn("w:p") else Table(c, dd) if c.tag == qn("w:tbl") else None for c in body]
n_compacted = 0
for i, it in enumerate(items):
    if isinstance(it, Paragraph):
        m = re.match(r"^Table (\d+[a-z]?)\.", it.text)
        if m and m.group(1) in COMPACT and i + 1 < len(items) and isinstance(items[i + 1], Table):
            pt = COMPACT[m.group(1)]
            for row in items[i + 1].rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for r in p.runs:
                            r.font.size = Pt(pt)
            n_compacted += 1
dd.save(docx)
print(f"wrote {docx} ({docx.stat().st_size/1e6:.2f} MB); {n} figures embedded at {DPI} dpi; stripped {leaks} leaked file paths; compacted {n_compacted}/{len(COMPACT)} tables")
