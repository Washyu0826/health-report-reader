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
    """Parse HTML <tr>/<td> or markdown pipe tables into rows."""
    rows: List[List[str]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S | re.I):
        cells = [clean_ocr_text(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)]
        if any(cells):
            rows.append(cells)
    if rows:
        return rows
    for line in text.splitlines():
        if line.strip().startswith("|") and not re.fullmatch(r"[\s|:\-]+", line):
            rows.append([clean_ocr_text(c).strip() for c in line.strip().strip("|").split("|")])
    return rows


def _text_outside_tables(text: str) -> str:
    """The OCR output with HTML tables and markdown table lines removed (tags stripped, cleaned)."""
    rest = re.sub(r"<table[^>]*>.*?</table>", "\n", text, flags=re.S | re.I)
    rest = "\n".join(ln for ln in rest.splitlines() if not ln.strip().startswith("|"))
    return clean_ocr_text(re.sub(r"<[^>]+>", " ", rest))


def ocr_pdf(pdf_bytes: bytes, model: str = OCR_MODEL,
            base_url: str = config.OLLAMA_BASE_URL) -> Dict:
    """Return {"text", "tables", "pages", "ocr": True} like pdf_utils.extract_pdf_content.
    Table rows go to `tables`; the text outside the tables is kept too (it often holds the values)."""
    texts, tables = [], []
    pages = render_pages(pdf_bytes)
    for png in pages:
        raw = _ocr_page(png, model, base_url)
        rows = _rows_from_markup(raw)
        if rows:
            tables.append(rows)
        parts = ["\n".join("  ".join(r) for r in rows)] if rows else []
        rest = _text_outside_tables(raw).strip()
        if rest:
            parts.append(rest)
        texts.append("\n".join(parts))
    return {"text": "\n\n".join(texts), "tables": tables, "pages": len(pages), "ocr": True}
