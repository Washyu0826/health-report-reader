"""Smoke test for the Gradio UI's analysis path in demo-stub mode.

No network, no LLM, no embeddings: stub resources use checkitem fact sheets
only, and the web fallback (enabled here to exercise the audit path) uses a
canned in-memory search function.
"""

import csv
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import storage  # noqa: E402
import ui  # noqa: E402

SAMPLE = ROOT / "eval" / "synth" / "samples" / "syn_048.pdf"


@pytest.fixture(scope="module")
def res():
    return ui.build_resources(stub=True)


@pytest.fixture(scope="module")
def analysis(res, tmp_path_factory, monkeypatch_module):
    if not SAMPLE.exists():
        pytest.skip("synthetic sample PDFs not generated")

    def no_network(*a, **k):
        raise AssertionError("network call attempted in stub mode")

    import requests
    monkeypatch_module.setattr(requests, "get", no_network)
    monkeypatch_module.setattr(requests, "post", no_network)
    db = str(tmp_path_factory.mktemp("runs") / "runs.sqlite3")
    stages = []
    a = ui.run_analysis(SAMPLE.read_bytes(), res, alias="smoke", allow_web=True,
                        on_stage=stages.append, runs_db=db)
    return a, stages, db


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_findings_and_table(analysis):
    a, stages, _ = analysis
    assert not a.result.error
    assert len(a.result.findings) > 20 and a.result.abnormals
    html_ = ui.render_findings(a)
    assert "<table" in html_ and "參考值衝突" in html_          # syn_048 has range conflicts
    assert stages[:4] == ["extract", "abnormal", "retrieve", "verify"]


def test_tags_html_escaped(analysis):
    a, _, _ = analysis
    a.tags["summary"] = '<img src=x onerror="alert(1)">'
    a.tags["food"] = [{"text": "<script>x</script>", "src": "inferred", "conf": 0.5}]
    out = ui.render_tags(a)
    assert "<script>" not in out and "<img src=x" not in out
    assert "&lt;script&gt;" in out and "&lt;img" in out
    assert 'class="chip' in out and "規則" in out                # rule tags rendered with badge
    assert "觸發依據" in out and "引用段落" in out                  # evidence + cited passage


def test_csv_and_json_export(analysis):
    a, _, _ = analysis
    a.tags["avoid"] = [{"text": '=HYPERLINK("x"),"quoted"', "src": "inferred", "conf": 0.4}]
    text = ui.export_csv(a)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][0] == "type"
    assert any(r[0] == "finding" for r in rows) and any(r[0] == "tag" for r in rows)
    bad = [r for r in rows if "HYPERLINK" in r[2]][0]
    assert bad[2].startswith("'=")                                # formula neutralised
    data = ui.export_dict(a)
    json.dumps(data, ensure_ascii=False)
    assert data["findings"] and "raw_name" not in data["findings"][0]


def test_timing_and_verification(analysis, res):
    a, _, _ = analysis
    for k in ("extract_s", "abnormal_s", "retrieve_s", "llm_s", "verify_s", "total_s"):
        assert k in a.result.timing
    assert "_verification" in a.tags
    perf = ui.render_perf(a, res)
    assert "各階段耗時" in perf and "引用支持率" in perf


def test_web_audit_and_trend_storage(analysis):
    a, _, db = analysis
    assert a.web_audit["enabled"] and a.web_audit["sent"]
    assert all("衛教" in q for q in a.web_audit["sent"])
    assert "本次送出的查詢" in ui.render_audit(a)
    # trend store: only canonical findings, no file name, numeric series only
    runs = storage.list_runs_for_patient(a.patient_id, db_path=db)
    assert len(runs) == 1 and not runs[0]["file_name"]
    stored = json.loads(runs[0]["findings_json"])
    assert all(f.get("canonical_key") for f in stored)
    assert all("raw_name" not in f for f in stored)
    keys = [k for _, k in ui.trend_choices(a.patient_id, db)]
    assert "hb" in keys and "urine_protein" not in keys           # qualitative excluded
    assert storage.delete_patient(a.patient_id, db_path=db) == 1
    assert storage.list_runs_for_patient(a.patient_id, db_path=db) == []


def test_report_date_extraction():
    assert ui.report_date_from_text("出生日期 1967-01-01 檢查日期 2025-09-21") == "2025-09-21"
    assert ui.report_date_from_text("受檢日期：113/05/02") == "2024-05-02"
    assert ui.report_date_from_text("出生日期 1967-01-01") is None
