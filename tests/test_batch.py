"""Batch CLI (batch.py) in --no-llm mode: no Ollama, no network."""

import csv
import json
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import batch  # noqa: E402

SAMPLES = ROOT / "eval" / "synth" / "samples"
TEXT_PDFS = ["syn_002.pdf", "syn_048.pdf"]
SCANNED_PDF = "syn_021.pdf"
pytestmark = pytest.mark.skipif(not (SAMPLES / SCANNED_PDF).exists(), reason="synthetic sample PDFs missing")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(self, address):
        raise AssertionError(f"network access attempted: {address}")
    monkeypatch.setattr(socket.socket, "connect", refuse)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_batch_writes_per_report_files_and_standardised_tables(tmp_path):
    rc = batch.main([str(SAMPLES / n) for n in TEXT_PDFS] + ["--no-llm", "--out", str(tmp_path)])
    assert rc == 0
    summary = read_csv(tmp_path / "summary.csv")
    assert [r["file"] for r in summary] == TEXT_PDFS and all(r["status"] == "ok" for r in summary)
    s002 = summary[0]
    assert int(s002["abnormal"]) > 0 and "代謝症候群" in s002["conditions"] and s002["advice_tags"] == "0"

    findings = read_csv(tmp_path / "findings_all.csv")
    assert {r["file"] for r in findings} == set(TEXT_PDFS)
    assert all(r["canonical_key"] for r in findings)                  # every row on the canonical item list
    assert sum(r["file"] == "syn_002.pdf" for r in findings) == int(s002["findings"])

    d = json.loads((tmp_path / "syn_002.json").read_text(encoding="utf-8"))
    assert d["mode"] == "no-llm" and d["findings"] and d["tags"]["conditions"]
    assert "text" not in d                                           # no report text in the export
    assert (tmp_path / "syn_002.csv").exists()


def test_batch_folder_input_and_scanned_pdf_reports_error(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    for n in [TEXT_PDFS[0], SCANNED_PDF]:
        (src / n).write_bytes((SAMPLES / n).read_bytes())
    (src / "notes.txt").write_text("ignored", encoding="utf-8")
    rc = batch.main([str(src), "--no-llm", "--out", str(tmp_path / "out")])
    assert rc == 1                                                    # one report failed
    rows = {r["file"]: r for r in read_csv(tmp_path / "out" / "summary.csv")}
    assert set(rows) == {TEXT_PDFS[0], SCANNED_PDF}
    assert rows[SCANNED_PDF]["status"] == "error" and "without --no-llm" in rows[SCANNED_PDF]["error"]
    assert rows[TEXT_PDFS[0]]["status"] == "ok"


def test_batch_rejects_non_pdf_inputs(tmp_path):
    with pytest.raises(SystemExit):
        batch.main([str(tmp_path / "missing.docx"), "--no-llm", "--out", str(tmp_path)])
