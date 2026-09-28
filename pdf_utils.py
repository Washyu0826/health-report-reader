"""
pdf_utils.py — PDF text + table extraction.

extract_pdf_content() returns a dict with:
  - text: best-effort linearized text (for LLM context)
  - tables: list of tables, each as list of rows (list of cell strings)
  - pages: page count

extract_text_from_pdf() is kept as a thin wrapper for callers that only want text.
"""

import io
from typing import Dict, List


def extract_pdf_content(pdf_bytes: bytes) -> Dict:
    """Try pdfplumber (better tables/layout), fall back to pypdf for text only."""
    result = _extract_pdfplumber(pdf_bytes)
    if result["text"] and len(result["text"].strip()) > 50:
        return result

    text = _extract_pypdf_text(pdf_bytes)
    return {"text": text, "tables": [], "pages": 0}


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Backwards-compatible: return only the linearized text."""
    return extract_pdf_content(pdf_bytes)["text"]


def _extract_pdfplumber(pdf_bytes: bytes) -> Dict:
    try:
        import pdfplumber
    except ImportError:
        return {"text": "", "tables": [], "pages": 0}

    text_parts: List[str] = []
    tables: List[List[List[str]]] = []
    page_count = 0
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)

                for table in page.extract_tables() or []:
                    cleaned = [
                        [(c or "").strip() for c in row]
                        for row in table if row
                    ]
                    cleaned = [row for row in cleaned if any(row)]
                    # page.extract_text() already contains the table cells, so
                    # tables are returned separately and NOT re-appended to text
                    # (that duplicated every value in the LLM prompt).
                    if cleaned:
                        tables.append(cleaned)
    except Exception as e:
        print(f"pdfplumber error: {e}")
        return {"text": "", "tables": [], "pages": page_count}

    return {
        "text": "\n\n".join(text_parts),
        "tables": tables,
        "pages": page_count,
    }


def _extract_pypdf_text(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        print(f"pypdf error: {e}")
        return ""


def tables_as_markdown(tables: List[List[List[str]]], max_rows: int = 60) -> str:
    """Render extracted tables as markdown — useful for LLM context."""
    if not tables:
        return ""
    blocks = []
    for ti, table in enumerate(tables):
        if not table:
            continue
        rows = table[:max_rows]
        ncols = max(len(r) for r in rows)
        normalized = [r + [""] * (ncols - len(r)) for r in rows]
        header = normalized[0]
        body = normalized[1:] if len(normalized) > 1 else []
        sep = ["---"] * ncols
        lines = [
            f"### Table {ti + 1}",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        lines.extend("| " + " | ".join(r) + " |" for r in body)
        if len(table) > max_rows:
            lines.append(f"_({len(table) - max_rows} more rows truncated)_")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
