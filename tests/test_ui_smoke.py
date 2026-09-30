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


# ─── nurse review (demo-stub mode) ───────────────────────────────────────────

REVIEW_SAMPLE = ROOT / "eval" / "synth" / "samples" / "syn_002.pdf"   # several rule conditions


def test_review_pauses_then_resumes_without_the_removed_condition(res, tmp_path):
    if not REVIEW_SAMPLE.exists():
        pytest.skip("synthetic sample PDFs not generated")
    db = str(tmp_path / "runs.sqlite3")
    stages, partials = [], []
    a = ui.run_analysis(REVIEW_SAMPLE.read_bytes(), res, runs_db=db, review=True,
                        on_stage=stages.append, on_partial=partials.append)
    assert a.paused and not a.cached and stages == ["extract", "abnormal", "retrieve"]
    assert partials and a.tags["conditions"]                          # findings + rules already shown
    choices, values = ui.review_choices(a)
    assert len(choices) == len(values) and "conditions::過重" in values
    keep = [v for v in values if v not in ("conditions::過重", "risks::痛風")]

    more = []
    final = ui.resume_analysis(a, keep, res, on_stage=more.append)
    assert not final.paused and more[-1] == "verify"
    conds = [t["text"] for t in final.tags["conditions"]]
    assert "過重" not in conds and "血脂異常" in conds
    assert "痛風" not in [t["text"] for t in final.tags["risks"]]
    assert final.result.review["removed"] == {"conditions": ["過重"], "risks": ["痛風"]}
    assert "護理師審核" in ui.render_audit(final) and "過重" in ui.render_audit(final)
    assert ui.export_dict(final)["review"]["removed"]["conditions"] == ["過重"]
    with pytest.raises(ValueError):
        ui.resume_analysis(final, keep, res)                          # nothing left to resume


def test_review_mode_bypasses_the_report_cache(res, tmp_path, monkeypatch):
    if not REVIEW_SAMPLE.exists():
        pytest.skip("synthetic sample PDFs not generated")
    from report_cache import ReportCache
    res.cache, old = ReportCache(str(tmp_path / "cache"), enabled=True), res.cache
    try:
        monkeypatch.setattr(res.cache, "get", lambda k: pytest.fail("cache read in review mode"))
        a = ui.run_analysis(REVIEW_SAMPLE.read_bytes(), res, runs_db=str(tmp_path / "r.db"), review=True)
        final = ui.resume_analysis(a, ui.review_choices(a)[1], res)     # keep everything
        assert final.result.review["removed"] == {"conditions": [], "risks": []}
        assert "全部保留" in ui.render_audit(final)
        assert not list((tmp_path / "cache").glob("*.json"))
    finally:
        res.cache = old


def test_fhir_export_has_no_report_text(analysis, tmp_path):
    a, _, _ = analysis
    b = ui.export_fhir(a)
    blob = json.dumps(b, ensure_ascii=False)
    patient = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Patient")
    assert set(patient) <= {"resourceType", "id", "gender"}
    header = [ln for ln in a.result.text.splitlines() if "姓名" in ln]
    assert header and not any(tok in blob for tok in header[0].split() if len(tok) >= 3 and "姓名" not in tok)
    assert Path(ui.write_fhir_export(a)).exists()
