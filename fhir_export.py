"""
fhir_export.py — Export analysed lab findings as a FHIR R4 Bundle.

    bundle = to_fhir_bundle(result.findings, sex=result.sex,
                            report_date=report_date_from_text(result.text))
    write_fhir("report.fhir.json", bundle)

Bundle (type "collection"):
  * Patient — ONLY `gender` and an opaque, per-bundle id (no name, identifier,
    birth date, address or telecom: nothing from the report header).
  * DiagnosticReport — status final, category LAB, code.text "Health check-up
    report", subject -> Patient, effectiveDateTime = report_date (if given),
    result -> every Observation.
  * one Observation per finding — LOINC coding from data/loinc_map.json when
    the item is mapped (code.text = the Chinese display name), valueQuantity
    (UCUM, censored values keep their comparator) or valueString for
    qualitative results, interpretation H/L/N/POS/NEG from `status`,
    referenceRange = [printed report range, public reference range], and a
    note when the two ranges disagree.

Ids and fullUrls are derived deterministically (same input -> identical JSON).
report_id, when given, only seeds those ids through SHA-256; it is never
written out. Standard library only; no network access.

The LOINC mapping is for interoperability demos and must be reviewed before any
clinical use (see data/loinc_map.README.md).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

_HERE = Path(__file__).resolve().parent
LOINC_MAP_PATH = _HERE / "data" / "loinc_map.json"
SYNONYMS_PATH = _HERE / "data" / "lab_synonyms.json"

LOINC_SYSTEM = "http://loinc.org"
UCUM_SYSTEM = "http://unitsofmeasure.org"
OBS_CATEGORY_SYSTEM = "http://terminology.hl7.org/CodeSystem/observation-category"
V2_0074_SYSTEM = "http://terminology.hl7.org/CodeSystem/v2-0074"
INTERPRETATION_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"

REPORT_CODE_TEXT = "Health check-up report"
PUBLIC_RANGE_TYPE_TEXT = "public reference range"

# Items recorded as vital signs; physical examinations that are not lab tests
# (body composition, eyes, hearing, bone density, risk scores) use "exam";
# everything else is "laboratory".
VITAL_SIGN_KEYS = frozenset({"height", "weight", "bmi", "waist", "sbp", "dbp", "pulse", "temperature"})
EXAM_KEYS = frozenset({
    "body_fat_pct", "bone_mass", "metabolic_age", "bmr", "visceral_fat", "fat_right_arm", "fat_right_leg",
    "fat_left_arm", "fat_left_leg", "fat_trunk", "body_water", "ideal_weight", "muscle_mass", "bmd",
    "va_naked_right", "va_naked_left", "va_corr_right", "va_corr_left", "iop_right", "iop_left",
    "hearing_r_500", "hearing_r_1k", "hearing_r_2k", "hearing_l_500", "hearing_l_1k", "hearing_l_2k",
    "cvd_10y_risk", "framingham_score",
})
_CATEGORY_DISPLAY = {"vital-signs": "Vital Signs", "exam": "Exam", "laboratory": "Laboratory"}

_INTERPRETATION = {  # finding["status"] -> v3-ObservationInterpretation
    "high": ("H", "High"),
    "low": ("L", "Low"),
    "normal": ("N", "Normal"),
    "positive": ("POS", "Positive"),
    "negative": ("NEG", "Negative"),
}
_GENDER = {"M": "male", "F": "female"}
_COMPARATORS = {"<", "<=", ">", ">="}

# Printed unit (normalised by _unit_key) -> UCUM code; used when the printed unit
# differs from the item's canonical unit (whose UCUM code is in loinc_map.json).
_UCUM_BY_UNIT = {
    "cm": "cm", "m": "m", "kg": "kg", "g": "g", "kg/m2": "kg/m2", "%": "%",
    "mmhg": "mm[Hg]", "次/分": "/min", "次/min": "/min", "bpm": "/min", "/min": "/min", "beats/min": "/min",
    "°c": "Cel", "℃": "Cel", "kcal": "kcal", "kcal/day": "kcal/d", "kcal/d": "kcal/d",
    "10^3/ul": "10*3/uL", "10*3/ul": "10*3/uL", "x10^3/ul": "10*3/uL", "10^3/µl": "10*3/uL",
    "10^6/ul": "10*6/uL", "10*6/ul": "10*6/uL", "x10^6/ul": "10*6/uL", "10^6/µl": "10*6/uL",
    "10^9/l": "10*9/L", "10*9/l": "10*9/L", "10^12/l": "10*12/L", "10*12/l": "10*12/L", "/ul": "/uL",
    "g/dl": "g/dL", "g/l": "g/L", "mg/dl": "mg/dL", "mg/l": "mg/L", "ug/dl": "ug/dL", "µg/dl": "ug/dL",
    "ug/l": "ug/L", "µg/l": "ug/L", "ug/ml": "ug/mL", "µg/ml": "ug/mL", "ng/ml": "ng/mL", "ng/dl": "ng/dL",
    "pg/ml": "pg/mL", "fl": "fL", "pg": "pg",
    "mmol/l": "mmol/L", "umol/l": "umol/L", "µmol/l": "umol/L", "nmol/l": "nmol/L", "pmol/l": "pmol/L",
    "meq/l": "meq/L", "mmol/mol": "mmol/mol",
    "u/l": "U/L", "iu/l": "[IU]/L", "uiu/ml": "u[IU]/mL", "µiu/ml": "u[IU]/mL", "miu/ml": "m[IU]/mL",
    "miu/l": "m[IU]/L", "iu/ml": "[IU]/mL",
    "ml/min/1.73m2": "mL/min/{1.73_m2}", "ml/min/1.73m^2": "mL/min/{1.73_m2}",
    "/hpf": "/[HPF]", "/lpf": "/[LPF]", "db": "dB",
}

# UCUM code -> coarse LOINC property class. When the reported unit and the mapped
# item's unit fall in different classes (e.g. glucose in mmol/L vs a mass-
# concentration LOINC code) the LOINC coding is dropped: a wrong code is worse
# than none. Units not listed here are not checked.
_UNIT_CLASS = {}
for _cls, _codes in {
    "mass-conc": ("g/dL", "g/L", "mg/dL", "mg/L", "ug/dL", "ug/L", "ug/mL", "ng/mL", "ng/dL", "pg/mL"),
    "substance-conc": ("mmol/L", "umol/L", "nmol/L", "pmol/L", "meq/L"),
    "number-conc": ("10*3/uL", "10*6/uL", "10*9/L", "10*12/L", "/uL"),
    "fraction": ("%",),
    "substance-ratio": ("mmol/mol",),
}.items():
    for _c in _codes:
        _UNIT_CLASS[_c] = _cls


# ─── Reference data ──────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def load_loinc_map(path: Union[str, Path] = LOINC_MAP_PATH) -> Dict[str, Dict]:
    """{canonical_key: {loinc, display, ucum, confidence, source_url, note}}."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["items"]


@lru_cache(maxsize=4)
def _catalog_units(path: Union[str, Path] = SYNONYMS_PATH) -> Dict[str, Tuple[str, ...]]:
    """{canonical_key: (canonical unit, *numerically identical units)} as unit keys."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)["items"]
    return {it["key"]: tuple(_unit_key(u) for u in [it.get("unit") or ""] + list(it.get("unit_equiv") or []))
            for it in items}


def _unit_key(u) -> str:
    s = unicodedata.normalize("NFKC", str(u or "")).strip().lower().replace(" ", "")
    return s.replace("μ", "µ").replace("²", "2")


# ─── Small builders ──────────────────────────────────────────────────────────

def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _fhir_id(s: str) -> str:
    """FHIR id: [A-Za-z0-9-.]{1,64}."""
    return (re.sub(r"[^A-Za-z0-9.\-]+", "-", s).strip("-") or "x")[:64]


def _full_url(seed: str, rtype: str, rid: str) -> str:
    return "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:health-report-tagger:{seed}/{rtype}/{rid}"))


def _coding(system: str, code: str, display: Optional[str] = None) -> Dict:
    c = {"system": system, "code": code}
    if display:
        c["display"] = display
    return c


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(v: float, text: str = ""):
    """Keep integers as JSON integers when the report printed no decimals."""
    if isinstance(v, float) and v.is_integer() and "." not in (text or ""):
        return int(v)
    return v


def _date_str(d) -> Optional[str]:
    if d is None or d == "":
        return None
    if isinstance(d, datetime):
        return d.isoformat()
    if isinstance(d, date):
        return d.isoformat()
    s = str(d).strip()
    if not re.fullmatch(r"\d{4}(-\d{2}(-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2}))?)?)?", s):
        raise ValueError(f"report_date must be an ISO date/dateTime, got {s!r}")
    return s


def _unit_for(key: str, unit_text: str, entry: Optional[Dict]) -> Tuple[str, Optional[str]]:
    """(display unit, UCUM code or None) for a value of item `key` printed with `unit_text`."""
    uk = _unit_key(unit_text)
    canonical = _catalog_units().get(key, ("",))
    map_ucum = (entry or {}).get("ucum")
    if map_ucum and (not uk or uk in canonical):
        return (unit_text or map_ucum), map_ucum
    return unit_text, _UCUM_BY_UNIT.get(uk)


def _quantity(value, unit_text: str, ucum: Optional[str], text: str = "", comparator: str = "") -> Dict:
    q = {"value": _num(value, text)}
    if comparator in _COMPARATORS:
        q["comparator"] = comparator
    if unit_text:
        q["unit"] = unit_text
    if ucum:
        q["system"] = UCUM_SYSTEM
        q["code"] = ucum
    return q


def _category(key: str) -> str:
    if key in VITAL_SIGN_KEYS:
        return "vital-signs"
    if key in EXAM_KEYS:
        return "exam"
    return "laboratory"


# ─── Observation ─────────────────────────────────────────────────────────────

def _observation(f: Dict, oid: str, subject_ref: str, eff: Optional[str], lmap: Dict[str, Dict]) -> Dict:
    key = f.get("canonical_key") or ""
    entry = lmap.get(key)
    name = f.get("display_name") or f.get("name") or key
    value = f.get("value")
    value_text = str(f.get("value_text") or "")
    notes: List[str] = []

    # value[x]
    unit_text = f.get("unit") or (f.get("ref_unit") if f.get("range_source") == "report" else "") or ""
    ucum: Optional[str] = None
    value_part: Dict = {}
    if _is_number(value):
        unit_disp, ucum = _unit_for(key, unit_text, entry)
        value_part["valueQuantity"] = _quantity(value, unit_disp, ucum, value_text, f.get("censored") or "")
    elif value_text or (value not in (None, "")):
        value_part["valueString"] = value_text or str(value)
    else:
        value_part["dataAbsentReason"] = {"coding": [_coding(
            "http://terminology.hl7.org/CodeSystem/data-absent-reason", "unknown", "Unknown")]}

    # code (LOINC when mapped and the reported unit does not contradict it)
    code: Dict = {}
    if entry and entry.get("loinc"):
        cls_val, cls_map = _UNIT_CLASS.get(ucum or ""), _UNIT_CLASS.get(entry.get("ucum") or "")
        if cls_val and cls_map and cls_val != cls_map:
            notes.append(f"LOINC {entry['loinc']} omitted: reported unit {unit_text} does not match "
                         f"the mapped term ({entry.get('display')}).")
        else:
            code["coding"] = [_coding(LOINC_SYSTEM, entry["loinc"], entry.get("display"))]
    code["text"] = name

    cat = _category(key)
    obs: Dict = {
        "resourceType": "Observation",
        "id": oid,
        "status": "final",
        "category": [{"coding": [_coding(OBS_CATEGORY_SYSTEM, cat, _CATEGORY_DISPLAY[cat])]}],
        "code": code,
        "subject": {"reference": subject_ref},
    }
    if eff:
        obs["effectiveDateTime"] = eff
    obs.update(value_part)

    interp = _INTERPRETATION.get(f.get("status") or "")
    if interp:
        obs["interpretation"] = [{"coding": [_coding(INTERPRETATION_SYSTEM, *interp)]}]

    if f.get("range_conflict_note"):
        notes.insert(0, str(f["range_conflict_note"]))
    if f.get("qualitative_result"):
        notes.append(f"報告判讀：{f['qualitative_result']}")
    if notes:
        obs["note"] = [{"text": t} for t in notes]

    ranges = _reference_ranges(f, key, entry, unit_text)
    if ranges:
        obs["referenceRange"] = ranges
    return obs


def _bounds(f: Dict, key: str, entry: Optional[Dict], unit_text: str) -> Dict:
    lo, hi = f.get("ref_low"), f.get("ref_high")
    unit_disp, ucum = _unit_for(key, f.get("ref_unit") or unit_text, entry)
    out = {}
    if _is_number(lo):
        out["low"] = _quantity(lo, unit_disp, ucum)
    if _is_number(hi):
        out["high"] = _quantity(hi, unit_disp, ucum)
    return out


def _reference_ranges(f: Dict, key: str, entry: Optional[Dict], unit_text: str) -> List[Dict]:
    """[printed report range, public reference range] — each only when present.
    Numeric low/high go on the range that was actually used for `status`."""
    src = f.get("range_source")
    out = []
    if f.get("report_range"):
        rr = _bounds(f, key, entry, unit_text) if src == "report" else {}
        rr["text"] = str(f["report_range"])
        out.append(rr)
    if f.get("kb_range"):
        rr = _bounds(f, key, entry, unit_text) if src in ("reference", "kb") else {}
        rr["type"] = {"text": PUBLIC_RANGE_TYPE_TEXT}
        rr["text"] = str(f["kb_range"])
        out.append(rr)
    return out


# ─── Bundle ──────────────────────────────────────────────────────────────────

def _seed(findings: List[Dict], sex, report_date: Optional[str], report_id) -> str:
    if report_id not in (None, ""):
        return _digest("report_id", str(report_id))
    sig = [[f.get("canonical_key"), str(f.get("value_text") or f.get("value")), f.get("unit") or "",
            f.get("status") or ""] for f in findings]
    return _digest("findings", json.dumps(sig, ensure_ascii=False, sort_keys=True), str(sex), str(report_date))


def to_fhir_bundle(findings: Iterable[Dict], *, sex: Optional[str] = None,
                   report_date: Union[str, date, None] = None, report_id: Optional[str] = None,
                   loinc_map: Optional[Dict[str, Dict]] = None) -> Dict:
    """FHIR R4 Bundle (type "collection") for one report's findings.

    findings: abnormal.evaluate_findings() output (PipelineResult.findings).
    sex: "M" / "F" / None -> Patient.gender male / female / unknown.
    report_date: exam date (ISO "YYYY-MM-DD", e.g. ui.report_date_from_text());
      becomes DiagnosticReport/Observation.effectiveDateTime.
    report_id: optional stable key for this report (e.g. a file hash); only its
      SHA-256 digest seeds the resource ids. Without it the ids derive from the
      findings, so identical input always yields identical JSON.
    loinc_map: override data/loinc_map.json items (tests)."""
    findings = list(findings or [])
    lmap = load_loinc_map() if loinc_map is None else loinc_map
    eff = _date_str(report_date)
    seed = _seed(findings, sex, eff, report_id)
    tag = seed[:16]

    pid, rid = f"patient-{tag}", f"report-{tag}"
    p_url, r_url = _full_url(seed, "Patient", pid), _full_url(seed, "DiagnosticReport", rid)
    patient = {"resourceType": "Patient", "id": pid, "gender": _GENDER.get(sex or "", "unknown")}

    obs_entries = []
    for n, f in enumerate(findings, 1):
        oid = _fhir_id(f"obs-{n}-{f.get('canonical_key') or 'item'}")
        obs_entries.append({"fullUrl": _full_url(seed, "Observation", oid),
                            "resource": _observation(f, oid, p_url, eff, lmap)})

    report = {
        "resourceType": "DiagnosticReport",
        "id": rid,
        "status": "final",
        "category": [{"coding": [_coding(V2_0074_SYSTEM, "LAB", "Laboratory")]}],
        "code": {"text": REPORT_CODE_TEXT},
        "subject": {"reference": p_url},
    }
    if eff:
        report["effectiveDateTime"] = eff
    if obs_entries:
        report["result"] = [{"reference": e["fullUrl"]} for e in obs_entries]

    return {
        "resourceType": "Bundle",
        "id": f"bundle-{tag}",
        "type": "collection",
        "entry": [{"fullUrl": p_url, "resource": patient}, {"fullUrl": r_url, "resource": report}] + obs_entries,
    }


def bundle_json(bundle: Dict) -> str:
    return json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"


def write_fhir(path: Union[str, Path], bundle: Dict) -> str:
    """Write the bundle as UTF-8 JSON (application/fhir+json); returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(bundle_json(bundle), encoding="utf-8", newline="\n")
    return str(p)
