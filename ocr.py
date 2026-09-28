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


def _rows_from_markup(text: str) -> List[List[str]]:
    """Parse HTML <tr>/<td> or markdown pipe tables into rows."""
    rows: List[List[str]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S | re.I):
        cells = [unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)]
        if any(cells):
            rows.append(cells)
    if rows:
        return rows
    for line in text.splitlines():
        if line.strip().startswith("|") and not re.fullmatch(r"[\s|:\-]+", line):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def ocr_pdf(pdf_bytes: bytes, model: str = OCR_MODEL,
            base_url: str = config.OLLAMA_BASE_URL) -> Dict:
    """Return {"text", "tables", "pages", "ocr": True} like pdf_utils.extract_pdf_content."""
    texts, tables = [], []
    pages = render_pages(pdf_bytes)
    for png in pages:
        raw = _ocr_page(png, model, base_url)
        rows = _rows_from_markup(raw)
        if rows:
            tables.append(rows)
            texts.append("\n".join("  ".join(r) for r in rows))
        else:
            texts.append(re.sub(r"<[^>]+>", " ", raw))
    return {"text": "\n\n".join(texts), "tables": tables, "pages": len(pages), "ocr": True}
