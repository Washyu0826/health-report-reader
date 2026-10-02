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


def test_is_doublewide_true_for_two_merged_item_columns():
    """Two independent item-name columns merged side by side (the twocol layout): a standalone
    '項目' cell appears twice in one row."""
    header = ["【一般體格檢查】", "項目", "結果", "参考值", "項目", "結果", "参考值"]
    assert ocr._is_doublewide([header]) is True


def test_is_doublewide_false_for_one_table_with_a_repeated_unit_column():
    """R12 regression: a different, single-table layout prints '檢查項目/結果/單位' on the left and
    that same table's '單位/參考值/判定' reference-range columns on the right — no second '項目' at
    all. Splitting this one in half throws away the item name on the right half (orphaned
    unit-only rows), which pushed scan false positives from 4/1/0 up to 12/12/7. Only a standalone
    '項目' repeat (not '結果' or '單位') may trigger the split."""
    merged = ["檢查項目", "結果", "單位", "單位", "參考值", "判定"]
    assert ocr._is_doublewide([merged]) is False


def test_doublewide_split_only_fires_for_merged_item_columns(monkeypatch):
    """ocr_pdf splits into left/right crops when a row has '項目' twice, and leaves a single-table
    page (repeated '單位' only) alone."""
    doublewide_raw = ("<table><tr><th>項目</th><th>結果</th><th>項目</th><th>結果</th></tr>"
                       "<tr><td>身高</td><td>158</td><td>體重</td><td>54</td></tr></table>")
    single_table_raw = ("<table><tr><th>檢查項目</th><th>結果</th><th>單位</th>"
                         "<th>單位</th><th>參考值</th><th>判定</th></tr>"
                         "<tr><td>身高</td><td>158</td><td>cm</td><td>cm</td><td></td><td></td></tr></table>")
    half_left = "<table><tr><th>項目</th><th>結果</th></tr><tr><td>身高</td><td>158</td></tr></table>"
    half_right = "<table><tr><th>項目</th><th>結果</th></tr><tr><td>體重</td><td>54</td></tr></table>"

    monkeypatch.setattr(ocr, "render_pages", lambda pdf_bytes: [b"page"])
    monkeypatch.setattr(ocr, "_split_halves", lambda png: [b"left", b"right"])

    calls = iter([half_left, half_right])
    monkeypatch.setattr(ocr, "_ocr_page", lambda png, model, base_url: (
        doublewide_raw if png == b"page" else next(calls)))
    out = ocr.ocr_pdf(b"%PDF")
    assert out["tables"] == [[["項目", "結果"], ["身高", "158"], ["項目", "結果"], ["體重", "54"]]]

    monkeypatch.setattr(ocr, "_ocr_page", lambda png, model, base_url: single_table_raw)
    out = ocr.ocr_pdf(b"%PDF")
    assert out["tables"] == [[["檢查項目", "結果", "單位", "單位", "參考值", "判定"],
                               ["身高", "158", "cm", "cm", "", ""]]]


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
