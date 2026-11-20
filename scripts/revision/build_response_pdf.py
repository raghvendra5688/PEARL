"""Response letter as PDF for sharing/printing: markdown -> HTML (pandoc) -> PDF (weasyprint), figures rasterised from the
vector PDFs (reuses the PNG cache built by build_response_docx.py). The PDFs stay canonical; this is a rendered copy.
Run after build_response.py (and build_response_docx.py, which creates/updates the PNG cache)."""
import re, subprocess, sys
from pathlib import Path
import pypdfium2 as pdfium
REPO = Path(__file__).resolve().parents[2]; MS = REPO / "manuscript"
md = (MS / "PEARL_Response_to_Reviewers_Draft.md").read_text()
png = REPO.parent / "EffiChem_Extras_v2" / "PEARL_revision" / "response_png"; png.mkdir(parents=True, exist_ok=True)
DPI = int(sys.argv[1]) if len(sys.argv) > 1 else 300
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
tmp = MS / "_response_pdf.md"; tmp.write_text(md2)
html_body = subprocess.run(["pandoc", tmp.name, "-f", "markdown+pipe_tables-smart", "-t", "html"], cwd=MS, check=True, capture_output=True, text=True).stdout
tmp.unlink()
css = """
@page { size: A4; margin: 1.5cm 1.6cm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 7.5pt; color: #555; } }
body { font-family: "Liberation Serif","DejaVu Serif",serif; font-size: 8.7pt; line-height: 1.24; color: #111; }
h1 { font-size: 14pt; margin: 0 0 6pt 0; }
h2 { font-size: 11pt; border-bottom: 1px solid #999; padding-bottom: 1.5pt; margin: 14pt 0 6pt 0; }
h3 { font-size: 9.3pt; margin: 9pt 0 3pt 0; }
p { margin: 5pt 0; }
ol, ul { margin: 4pt 0; padding-left: 1.3em; }
li { margin: 2pt 0; }
strong { color: #111; }
blockquote { background: #f2f2f2; border-left: 3px solid #888; margin: 4pt 0; padding: 3pt 8pt; font-style: italic; font-size: 8.2pt; line-height: 1.2; }
table { border-collapse: collapse; width: 100%; font-size: 7pt; margin: 5pt 0; }
table, th, td { border: 0.5pt solid #888; }
th, td { padding: 1.8pt 3.5pt; text-align: left; line-height: 1.15; }
thead { background: #e8e8e8; }
img { max-width: 92%; display: block; margin: 5pt auto; page-break-inside: avoid; }
hr { border: none; border-top: 0.5pt solid #bbb; margin: 8pt 0; }
code { background: #f2f2f2; padding: 0.5pt 2pt; font-size: 7.6pt; }
a { color: #0645ad; text-decoration: none; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{html_body}</body></html>"
pdf_path = MS / "PEARL_Response_to_Reviewers.pdf"
from weasyprint import HTML
HTML(string=html, base_url=str(MS)).write_pdf(str(pdf_path))
print(f"wrote {pdf_path} ({pdf_path.stat().st_size/1e6:.2f} MB); {n} figures embedded at {DPI} dpi")
