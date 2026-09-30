"""FHIR R4 export (fhir_export.py) + LOINC map (data/loinc_map.json). Offline.

Runs the rule-only pipeline (run_llm=False, ocr=False) on two synthetic
reports, builds bundles and checks structure, coding, interpretation, privacy
and determinism. When fhir.resources is installed (requirements-dev) the
bundles are also validated against its R4B models.
"""

import json
import re
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fhir_export import (  # noqa: E402
    EXAM_KEYS, INTERPRETATION_SYSTEM, LOINC_SYSTEM, UCUM_SYSTEM, VITAL_SIGN_KEYS, to_fhir_bundle, write_fhir,
)

SYNTH = ROOT / "eval" / "synth" / "samples"
PDFS = ["syn_002.pdf", "syn_048.pdf"]
LOINC_MAP = json.loads((ROOT / "data" / "loinc_map.json").read_text(encoding="utf-8"))
SYNONYM_KEYS = {it["key"] for it in json.loads((ROOT / "data" / "lab_synonyms.json")
                                               .read_text(encoding="utf-8"))["items"]}
LOINC_RX = re.compile(r"^\d{1,7}-\d$")
FHIR_ID_RX = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")
EXPECTED_INTERP = {"high": "H", "low": "L", "normal": "N", "positive": "POS", "negative": "NEG"}
VITALS = {"height", "weight", "bmi", "waist", "sbp", "dbp", "pulse", "temperature"}
assert VITALS == VITAL_SIGN_KEYS and not (VITALS & EXAM_KEYS) and EXAM_KEYS <= SYNONYM_KEYS


class _NoEgress:
    """Fail on any socket connect / DNS lookup while active (the export and the
    rule-only pipeline must not touch the network or a local model server)."""

    def __enter__(self):
        self._connect, self._gai = socket.socket.connect, socket.getaddrinfo

        def deny(*a, **kw):
            raise AssertionError(f"network access attempted: {a[1:] if len(a) > 1 else a}")

        socket.socket.connect = deny
        socket.getaddrinfo = deny
        return self

    def __exit__(self, *exc):
        socket.socket.connect, socket.getaddrinfo = self._connect, self._gai
        return False


def _analyze(name):
    from pipeline import analyze_pdf
    with _NoEgress():
        return analyze_pdf((SYNTH / name).read_bytes(), run_llm=False, ocr=False)


def _bundle_for(r):
    from ui import report_date_from_text
    return to_fhir_bundle(r.findings, sex=r.sex, report_date=report_date_from_text(r.text))


@pytest.fixture(scope="module")
def analyses():
    return {name: _analyze(name) for name in PDFS}


@pytest.fixture(scope="module")
def bundles(analyses):
    with _NoEgress():
        return {name: _bundle_for(r) for name, r in analyses.items()}


def _resources(bundle, rtype):
    return [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == rtype]


def _loinc_check_digit_ok(code: str) -> bool:
    """LOINC mod-10 check digit (Luhn over the part before the hyphen)."""
    base, check = code.split("-")
    total = 0
    for i, ch in enumerate(reversed(base)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return (10 - total % 10) % 10 == int(check)


# ─── data/loinc_map.json ─────────────────────────────────────────────────────

def test_loinc_map_shape():
    assert LOINC_MAP["version"] == 1 and LOINC_MAP["source"]
    items = LOINC_MAP["items"]
    assert set(items) <= SYNONYM_KEYS
    for key, it in items.items():
        assert set(it) >= {"loinc", "display", "ucum", "confidence", "source_url", "note"}, key
        assert it["note"], f"{key}: every entry explains itself"
        if it["loinc"] is None:
            assert it["confidence"] is None and it["source_url"] is None, key
            continue
        assert LOINC_RX.match(it["loinc"]), f"{key}: bad LOINC format {it['loinc']}"
        assert _loinc_check_digit_ok(it["loinc"]), f"{key}: LOINC check digit wrong {it['loinc']}"
        assert it["confidence"] in ("high", "medium"), key
        assert it["display"], key
        assert it["source_url"] == f"https://loinc.org/{it['loinc']}", key


def test_loinc_map_covers_common_items():
    items = LOINC_MAP["items"]
    expected = {"glucose_ac": "1558-6", "chol": "2093-3", "hdl": "2085-9", "tg": "2571-8", "hb": "718-7",
                "wbc": "6690-2", "alt": "1742-6", "ast": "1920-8", "creatinine": "2160-0",
                "height": "8302-2", "weight": "29463-7", "bmi": "39156-5", "sbp": "8480-6",
                "dbp": "8462-4", "pulse": "8867-4", "waist": "8280-0"}
    for key, code in expected.items():
        assert items[key]["loinc"] == code, key


# ─── bundle structure ────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", PDFS)
def test_bundle_structure(name, analyses, bundles):
    r, b = analyses[name], bundles[name]
    assert b["resourceType"] == "Bundle" and b["type"] == "collection"
    assert FHIR_ID_RX.match(b["id"])
    full_urls = [e["fullUrl"] for e in b["entry"]]
    assert len(set(full_urls)) == len(full_urls)
    assert all(u.startswith("urn:uuid:") for u in full_urls)
    for e in b["entry"]:
        assert FHIR_ID_RX.match(e["resource"]["id"]), e["resource"]["id"]

    (patient,) = _resources(b, "Patient")
    (report,) = _resources(b, "DiagnosticReport")
    observations = _resources(b, "Observation")
    assert len(observations) == len(r.findings) > 0
    p_url = next(e["fullUrl"] for e in b["entry"] if e["resource"] is patient)

    assert report["status"] == "final"
    assert report["category"][0]["coding"][0] == {
        "system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "LAB", "display": "Laboratory"}
    assert report["code"] == {"text": "Health check-up report"}
    assert report["subject"] == {"reference": p_url}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", report["effectiveDateTime"])
    obs_urls = [e["fullUrl"] for e in b["entry"] if e["resource"]["resourceType"] == "Observation"]
    assert [x["reference"] for x in report["result"]] == obs_urls

    for o in observations:
        assert o["status"] == "final"
        assert o["subject"] == {"reference": p_url}
        cat = o["category"][0]["coding"][0]
        assert cat["system"] == "http://terminology.hl7.org/CodeSystem/observation-category"
        assert cat["code"] in ("laboratory", "vital-signs", "exam")
        assert o["code"].get("text")
        values = [k for k in o if k.startswith("value")]
        assert len(values) == 1 and values[0] in ("valueQuantity", "valueString"), o["id"]
        if "valueQuantity" in o:
            q = o["valueQuantity"]
            assert isinstance(q["value"], (int, float)) and not isinstance(q["value"], bool)
            if "code" in q:
                assert q["system"] == UCUM_SYSTEM
            if "comparator" in q:
                assert q["comparator"] in ("<", "<=", ">", ">=")
        for rr in o.get("referenceRange", []):
            assert rr.get("text") or "low" in rr or "high" in rr


@pytest.mark.parametrize("name", PDFS)
def test_observation_content_matches_findings(name, analyses, bundles):
    r, b = analyses[name], bundles[name]
    for f, o in zip(r.findings, _resources(b, "Observation")):
        key = f["canonical_key"]
        assert o["id"].endswith(key.replace("_", "-"))
        assert o["code"]["text"] == f["display_name"]

        entry = LOINC_MAP["items"].get(key) or {}
        codings = o["code"].get("coding", [])
        assert all(c["system"] == LOINC_SYSTEM for c in codings)
        if entry.get("loinc"):
            assert [c["code"] for c in codings] == [entry["loinc"]], key
            assert codings[0]["display"] == entry["display"]
        else:
            assert codings == [], key

        want_cat = "vital-signs" if key in VITALS else ("exam" if key in EXAM_KEYS else "laboratory")
        assert o["category"][0]["coding"][0]["code"] == want_cat, key

        if isinstance(f["value"], (int, float)):
            assert o["valueQuantity"]["value"] == pytest.approx(f["value"])
            assert o["valueQuantity"].get("comparator", "") == (f.get("censored") or "")
        else:
            assert o["valueString"] == f["value_text"]

        want = EXPECTED_INTERP.get(f["status"])
        interp = [c for i in o.get("interpretation", []) for c in i["coding"]]
        if want:
            assert [(c["system"], c["code"]) for c in interp] == [(INTERPRETATION_SYSTEM, want)], key
        else:
            assert interp == [], key

        ranges = o.get("referenceRange", [])
        if f.get("report_range"):
            assert ranges[0]["text"] == f["report_range"]
            assert "type" not in ranges[0]
            if f["range_source"] == "report" and f["ref_high"] is not None:
                assert ranges[0]["high"]["value"] == pytest.approx(f["ref_high"])
            if f["range_source"] == "report" and f["ref_low"] is not None:
                assert ranges[0]["low"]["value"] == pytest.approx(f["ref_low"])
        if f.get("kb_range"):
            pub = ranges[-1]
            assert pub["type"] == {"text": "public reference range"} and pub["text"] == f["kb_range"]

        notes = [n["text"] for n in o.get("note", [])]
        if f.get("range_conflict"):
            assert f["range_conflict_note"] in notes, key


@pytest.mark.parametrize("name", PDFS)
def test_every_abnormal_finding_is_flagged(name, analyses, bundles):
    r, b = analyses[name], bundles[name]
    obs = _resources(b, "Observation")
    abnormal = [(f, o) for f, o in zip(r.findings, obs) if f["direction"] in ("high", "low")]
    assert abnormal, "the synthetic samples contain abnormal findings"
    for f, o in abnormal:
        codes = {c["code"] for i in o.get("interpretation", []) for c in i["coding"]}
        assert codes and codes <= {"H", "L", "POS"}, (f["canonical_key"], codes)


def test_known_values_syn_002(bundles):
    obs = {o["id"].split("-", 2)[2]: o for o in _resources(bundles["syn_002.pdf"], "Observation")}
    glu = obs["glucose-ac"]
    assert glu["code"]["coding"][0]["code"] == "1558-6"
    assert glu["valueQuantity"] == {"value": 117, "unit": "mg/dL", "system": UCUM_SYSTEM, "code": "mg/dL"}
    assert glu["interpretation"][0]["coding"][0]["code"] == "H"
    assert glu["referenceRange"][0]["low"]["value"] == 70 and glu["referenceRange"][0]["high"]["value"] == 100
    hdl = obs["hdl"]
    assert hdl["interpretation"][0]["coding"][0]["code"] == "L"
    assert obs["urine-protein"]["valueString"] == "陰性"
    assert obs["urine-protein"]["interpretation"][0]["coding"][0]["code"] == "NEG"
    assert obs["sbp"]["category"][0]["coding"][0]["code"] == "vital-signs"


def test_positive_qualitative_syn_048(bundles):
    obs = {o["id"].split("-", 2)[2]: o for o in _resources(bundles["syn_048.pdf"], "Observation")}
    prot = obs["urine-protein"]
    assert prot["valueString"] == "3+"
    assert prot["interpretation"][0]["coding"][0]["code"] == "POS"
    assert obs["pulse"]["valueQuantity"]["code"] == "/min"


# ─── privacy ─────────────────────────────────────────────────────────────────

def _header_pii(text: str):
    """Identifying fields printed in the synthetic report header (page 1)."""
    pats = {
        "name": r"姓名\s*(\S+(?:\s\S)?)\s+性別",
        "id_number": r"身分證字號\s*(\S+)",
        "birth_date": r"出生日期\s*(\S+)",
        "chart_no": r"病歷號碼\s*(\S+)",
        "phone": r"聯絡電話\s*(\S+)",
        "address": r"地址\s*(\S+)",
    }
    out = {}
    for k, p in pats.items():
        m = re.search(p, text)
        assert m, f"syn_002 header field {k} not found - synthetic sample changed?"
        out[k] = m.group(1)
    out["name_first_token"] = out["name"].split()[0]
    return out


def test_no_header_pii_in_bundle(analyses, bundles):
    pii = _header_pii(analyses["syn_002.pdf"].text)
    assert pii["id_number"] == "A000000002" and pii["phone"] == "0900-000-002"
    dumped = json.dumps(bundles["syn_002.pdf"], ensure_ascii=False)
    for field, value in pii.items():
        assert value not in dumped, f"{field} leaked into the FHIR bundle"
    for label in ("姓名", "身分證", "出生日期", "病歷號碼", "聯絡電話", "地址"):
        assert label not in dumped


@pytest.mark.parametrize("name", PDFS)
def test_patient_is_minimal(name, analyses, bundles):
    (patient,) = _resources(bundles[name], "Patient")
    assert set(patient) == {"resourceType", "id", "gender"}
    assert patient["gender"] == {"M": "male", "F": "female"}[analyses[name].sex]
    assert "name" not in patient and "identifier" not in patient and "birthDate" not in patient
    (report,) = _resources(bundles[name], "DiagnosticReport")
    assert "identifier" not in report


def test_report_id_is_hashed_not_written(analyses):
    r = analyses["syn_002.pdf"]
    b1 = to_fhir_bundle(r.findings, sex=r.sex, report_id="SYN-000002")
    b2 = to_fhir_bundle(r.findings, sex=r.sex, report_id="another-report")
    assert "SYN-000002" not in json.dumps(b1, ensure_ascii=False)
    assert b1["id"] != b2["id"]
    assert _resources(b1, "Patient")[0]["gender"] == "female"
    assert _resources(to_fhir_bundle(r.findings), "Patient")[0]["gender"] == "unknown"
    assert "effectiveDateTime" not in _resources(b1, "DiagnosticReport")[0]


# ─── determinism / io ────────────────────────────────────────────────────────

def test_deterministic_output(bundles, tmp_path):
    again = _bundle_for(_analyze("syn_002.pdf"))
    first = json.dumps(bundles["syn_002.pdf"], ensure_ascii=False, sort_keys=True)
    assert json.dumps(again, ensure_ascii=False, sort_keys=True) == first
    p1 = write_fhir(tmp_path / "a" / "b1.json", bundles["syn_002.pdf"])
    p2 = write_fhir(tmp_path / "b2.json", again)
    assert Path(p1).read_bytes() == Path(p2).read_bytes()
    assert json.loads(Path(p1).read_text(encoding="utf-8")) == bundles["syn_002.pdf"]


# ─── synthetic edge cases ────────────────────────────────────────────────────

def _finding(**kw):
    base = {"canonical_key": "glucose_ac", "display_name": "飯前血糖", "name": "飯前血糖", "value": 117.0,
            "value_text": "117", "unit": "mg/dL", "status": "high", "direction": "high",
            "range_source": "report", "report_range": "70~100", "kb_range": "≧70 且 <100 mg/dL",
            "ref_low": 70.0, "ref_high": 100.0, "ref_unit": "mg/dL", "range_conflict": False,
            "range_conflict_note": "", "censored": ""}
    base.update(kw)
    return base


def test_censored_value_keeps_comparator():
    f = _finding(canonical_key="hs_crp", display_name="高敏感度C反應蛋白", value=0.02, value_text="<0.02",
                 unit="mg/dL", status="normal", direction="normal", censored="<", report_range="<0.3",
                 ref_low=None, ref_high=0.3)
    (o,) = _resources(to_fhir_bundle([f]), "Observation")
    assert o["valueQuantity"] == {"value": 0.02, "comparator": "<", "unit": "mg/dL",
                                  "system": UCUM_SYSTEM, "code": "mg/dL"}
    assert o["code"]["coding"][0]["code"] == "30522-7"
    assert o["referenceRange"][0] == {"high": {"value": 0.3, "unit": "mg/dL", "system": UCUM_SYSTEM,
                                               "code": "mg/dL"}, "text": "<0.3"}


def test_unit_contradicting_loinc_property_drops_code():
    f = _finding(value=6.5, value_text="6.5", unit="mmol/L", report_range="3.9~5.5", ref_low=3.9, ref_high=5.5,
                 ref_unit="mmol/L")
    (o,) = _resources(to_fhir_bundle([f]), "Observation")
    assert "coding" not in o["code"] and o["code"]["text"] == "飯前血糖"
    assert o["valueQuantity"]["code"] == "mmol/L"
    assert any("1558-6" in n["text"] for n in o["note"])


def test_unmapped_item_has_text_only_and_reference_range_source():
    f = _finding(canonical_key="visceral_fat", display_name="內臟脂肪等級", value=12.0, value_text="12", unit="",
                 status="high", direction="high", range_source="reference", report_range=None,
                 kb_range="1~9", ref_low=1.0, ref_high=9.0, ref_unit="")
    (o,) = _resources(to_fhir_bundle([f], report_date="2025-04-06"), "Observation")
    assert o["code"] == {"text": "內臟脂肪等級"}
    assert o["category"][0]["coding"][0]["code"] == "exam"
    assert o["effectiveDateTime"] == "2025-04-06"
    assert len(o["referenceRange"]) == 1
    assert o["referenceRange"][0]["type"] == {"text": "public reference range"}
    assert o["referenceRange"][0]["high"]["value"] == 9


def test_bad_report_date_rejected():
    with pytest.raises(ValueError):
        to_fhir_bundle([_finding()], report_date="06/04/2025")


# ─── schema validation (fhir.resources, dev dependency) ──────────────────────

@pytest.mark.parametrize("name", PDFS)
def test_validates_against_fhir_models(name, bundles):
    pytest.importorskip("fhir.resources")
    from fhir.resources.R4B.bundle import Bundle
    Bundle.model_validate(bundles[name])  # raises on any schema violation
