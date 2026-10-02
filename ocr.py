"""
ocr.py — Local OCR for scanned (image-only) PDFs (R6).

Pages are rendered with pdfium and sent to a local vision model served by
Ollama (default: GLM-OCR, 0.9B, MIT). Nothing leaves the machine.

The model is asked for table recognition, because health-check reports are
mostly tables; its markdown/HTML table output is converted to plain rows so
the rest of the pipeline (header-aware table parsing in abnormal.py) is reused.
"""

from __future__ import annotations

import base64
import io
import re
from html import unescape
from typing import Dict, List

import requests

import config

OCR_MODEL = "glm-ocr"
OCR_PROMPT = "Table Recognition:"
RENDER_SCALE = 2.0  # ~144 dpi; enough for printed lab tables


def is_scanned(content: Dict, min_chars: int = 50) -> bool:
    return len((content.get("text") or "").strip()) < min_chars


def render_pages(pdf_bytes: bytes, scale: float = RENDER_SCALE, max_pages: int = 6) -> List[bytes]:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_bytes)
    out = []
    for i in range(min(len(pdf), max_pages)):
        img = pdf[i].render(scale=scale).to_pil().convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _ocr_page(png: bytes, model: str, base_url: str) -> str:
    payload = {"model": model, "stream": False, "options": {"temperature": 0, "num_ctx": 8192},
               "messages": [{"role": "user", "content": OCR_PROMPT,
                             "images": [base64.b64encode(png).decode()]}]}
    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=config.LLM_TIMEOUT_S)
    r.raise_for_status()
    return r.json()["message"]["content"]


# OCR output quirks, measured with eval/ocr_eval.py (R11):
#  * full-width digits come back spaced out with a middle dot: "１５８．０" -> "1 5 8 · 0";
#  * report words sometimes come back in simplified characters: 参考值, 红血球, 检查项目;
#  * a page is often one small table (the header) plus plain text lines, and the text lines
#    hold the lab values.
_SPACED_DIGITS = re.compile(r"(?<![\w.])\d(?: [\d.·•・‧∙])+(?![\w.])")
_MID_DOT = re.compile(r"(?<=\d)\s*[·•・‧∙]\s*(?=\d)")
_S2T = str.maketrans("参检项结单红验围标记类组别总压缩体岁质细数计胆肾离转碱钙钾钠铁镁维杂",
                     "參檢項結單紅驗圍標記類組別總壓縮體歲質細數計膽腎離轉鹼鈣鉀鈉鐵鎂維雜")


def clean_ocr_text(s: str) -> str:
    """Undo OCR quirks in a cell or line. Only applied to OCR output, never to a text layer."""
    s = unescape(s).translate(_S2T)
    s = _SPACED_DIGITS.sub(lambda m: m.group(0).replace(" ", ""), s)
    return _MID_DOT.sub(".", s)


def _rows_from_markup(text: str) -> List[List[str]]:
    """Parse HTML <tr>/<td> and markdown pipe tables into rows. A single response can mix both
    (R12: a half-width OCR crop returned an HTML table for the header and markdown tables for the
    lab sections), so both are collected rather than one short-circuiting the other."""
    rows: List[List[str]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S | re.I):
        cells = [clean_ocr_text(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)]
        if any(cells):
            rows.append(cells)
    for line in text.splitlines():
        if line.strip().startswith("|") and not re.fullmatch(r"[\s|:\-]+", line):
            cells = [clean_ocr_text(c).strip() for c in line.strip().strip("|").split("|")]
            if any(cells):
                rows.append(cells)
    return rows


def _text_outside_tables(text: str) -> str:
    """The OCR output with HTML tables and markdown table lines removed (tags stripped, cleaned)."""
    rest = re.sub(r"<table[^>]*>.*?</table>", "\n", text, flags=re.S | re.I)
    rest = "\n".join(ln for ln in rest.splitlines() if not ln.strip().startswith("|"))
    return clean_ocr_text(re.sub(r"<[^>]+>", " ", rest))


# A "two-column" page layout (R12): two independent small item/value/range tables side by side
# per section. GLM-OCR reads the whole page as one wide table and the left/right halves land at
# different, inconsistent cell offsets per row (the header row gets an extra leading cell the data
# rows don't have), which breaks header-aware column lookup downstream. Fixed by cropping the page
# image in half and OCR-ing each half on its own, where it degrades to the already-reliable
# single-column case.
#
# Detection must be specific to *that* layout. A standalone "項目" cell appearing twice in one row
# only happens when two full item-name columns were merged side by side (confirmed on the twocol
# layout). A different, single-table layout prints findings as "檢查項目/結果/單位" on the left and
# that same table's reference-range columns "單位/參考值/判定" on the right (no second "項目" at
# all) — splitting *that* one in half throws away the row's item name on the right half, turning
# reference-range cells into orphaned unit-only rows that the item-name matcher can misfire on.
# Checking other header words (結果/參考值/單位) too caught this second layout as a false positive
# (R12 first pass: scan abnormal false positives rose from R11's 4/1/0 to 12/12/7) — "項目" alone
# is the reliable signal.
_DOUBLEWIDE_TOKEN = "項目"


def _is_doublewide(rows: List[List[str]]) -> bool:
    return any(sum(c.strip() == _DOUBLEWIDE_TOKEN for c in row) >= 2 for row in rows)


def _split_halves(png: bytes) -> List[bytes]:
    """Crop a rendered page image into overlapping left/right halves, as PNG bytes each."""
    from PIL import Image
    img = Image.open(io.BytesIO(png))
    w, h = img.size
    halves = [img.crop((0, 0, int(w * 0.53), h)), img.crop((int(w * 0.47), 0, w, h))]
    out = []
    for half in halves:
        buf = io.BytesIO()
        half.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _ocr_one_page(png: bytes, model: str, base_url: str) -> tuple[List[List[str]], str]:
    """OCR one page image; re-OCR as two half-width crops if it reads as a merged two-column table."""
    raw = _ocr_page(png, model, base_url)
    rows = _rows_from_markup(raw)
    rest = _text_outside_tables(raw).strip()
    if _is_doublewide(rows):
        rows, parts = [], []
        for half_png in _split_halves(png):
            half_raw = _ocr_page(half_png, model, base_url)
            rows += _rows_from_markup(half_raw)
            half_rest = _text_outside_tables(half_raw).strip()
            if half_rest:
                parts.append(half_rest)
        rest = "\n".join(parts)
    return rows, rest


def ocr_pdf(pdf_bytes: bytes, model: str = OCR_MODEL,
            base_url: str = config.OLLAMA_BASE_URL) -> Dict:
    """Return {"text", "tables", "pages", "ocr": True} like pdf_utils.extract_pdf_content.
    Table rows go to `tables`; the text outside the tables is kept too (it often holds the values)."""
    texts, tables = [], []
    pages = render_pages(pdf_bytes)
    for png in pages:
        rows, rest = _ocr_one_page(png, model, base_url)
        if rows:
            tables.append(rows)
        parts = ["\n".join("  ".join(r) for r in rows)] if rows else []
        if rest:
            parts.append(rest)
        texts.append("\n".join(parts))
    return {"text": "\n\n".join(texts), "tables": tables, "pages": len(pages), "ocr": True}
