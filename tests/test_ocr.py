"""OCR post-processing (ocr.py): quirks measured by eval/ocr_eval.py. No model calls."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ocr  # noqa: E402
from abnormal import evaluate_findings, extract_lab_values, load_checkitem_lookup  # noqa: E402


@pytest.mark.parametrize("raw,clean", [
    ("1 5 8 · 0", "158.0"),                     # full-width digits come back spaced, with a middle dot
    ("5 3 · 9", "53.9"),
    ("1 8 · 5 ~ 2 4 · 0", "18.5 ~ 24.0"),      # a range keeps its separator
    ("< 8 0", "< 80"),
    ("3 · 7 2", "3.72"),
    ("8.2 10^3/uL", "8.2 10^3/uL"),            # value + unit: untouched
    ("男:3.5~7.2 女:2.6~6.0", "男:3.5~7.2 女:2.6~6.0"),
    ("2025-06-18", "2025-06-18"),
    ("参考值 红血球 检查项目", "參考值 紅血球 檢查項目"),  # simplified -> traditional
    ("&lt;85", "<85"),
])
def test_clean_ocr_text(raw, clean):
    assert ocr.clean_ocr_text(raw) == clean


def test_table_cells_are_cleaned():
    html = ("<table><tr><th>檢查項目</th><th>結果</th><th>参考值</th></tr>"
            "<tr><td>身高 Height</td><td>1 5 8 · 0</td><td></td></tr></table>")
    assert ocr._rows_from_markup(html) == [["檢查項目", "結果", "參考值"], ["身高 Height", "158.0", ""]]


def test_text_outside_a_header_table_is_kept(monkeypatch):
    """A page that is a small header table plus plain lab lines (the 'plain' layout) used to lose
    every lab line: only the table rows were kept."""
    raw = ("<table><tr><td>姓名</td><td>測試員</td></tr></table>\n"
           "【一般體格檢查】\n身高：164.2 cm\n收縮壓：156↑ mmHg 参考值 90~139\n")
    monkeypatch.setattr(ocr, "render_pages", lambda pdf_bytes: [b"page"])
    monkeypatch.setattr(ocr, "_ocr_page", lambda png, model, base_url: raw)
    out = ocr.ocr_pdf(b"%PDF")
    assert out["tables"] == [[["姓名", "測試員"]]]
    assert "收縮壓：156↑ mmHg 參考值 90~139" in out["text"]
    f = {x["canonical_key"]: x for x in evaluate_findings(extract_lab_values(out["text"], out["tables"]),
                                                            load_checkitem_lookup())}
    assert f["sbp"]["value"] == 156 and f["sbp"]["status"] == "high"
