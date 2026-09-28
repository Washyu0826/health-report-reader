"""
R2 regression tests for abnormal flagging v2 (abnormal.py, data/lab_synonyms.json,
data/reference_ranges.json, pdf table extraction). Run:  .venv/Scripts/python -m pytest tests -q

No Ollama needed: the LLM name-normaliser is exercised with fakes / a patched
requests.post.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import abnormal as ab  # noqa: E402

RANGES = ROOT / "data" / "reference_ranges.json"


@pytest.fixture(scope="module")
def cat():
    return ab.load_checkitem_lookup(str(RANGES))


def run(text="", tables=None, cat=None, **kw):
    dbg = {}
    f = ab.evaluate_findings(ab.extract_lab_values(text, tables or []), cat, debug=dbg, **kw)
    return f, dbg


def one(text="", tables=None, cat=None, **kw):
    f, _ = run(text, tables, cat, **kw)
    assert len(f) == 1, f
    return f[0]


# ─── 1. canonical item model ────────────────────────────────────────────────

def test_catalog_has_no_synonym_collisions(cat):
    assert cat.collisions == []


def test_every_canonical_item_has_a_reference_range_or_a_documented_skip(cat):
    ref = json.loads(RANGES.read_text(encoding="utf-8"))
    ranged = {e["key"] for e in ref["items"]}
    skipped = {e["key"] for e in ref["skipped"]}
    syn = json.loads((ROOT / "data" / "lab_synonyms.json").read_text(encoding="utf-8"))
    keys = {d["key"] for d in syn["items"]}
    assert ranged | skipped == keys and not ranged & skipped
    assert set(cat.items) == keys                     # no items beyond the curated table
    for e in ref["items"]:
        assert e.get("source"), f"{e['key']}: reference range without a source"
        assert cat.get(e["key"]).ref_origin == "reference"


def test_default_catalog_is_the_public_reference_file():
    assert Path(ab.DEFAULT_RANGES_PATH).name == "reference_ranges.json"
    assert ab.load_catalog() is ab.load_checkitem_lookup(str(RANGES))


def test_canonical_item_fields(cat):
    g = cat.get("glucose_ac")
    assert g.zh and g.en and g.unit == "mg/dL"
    assert g.kb_spec["kind"] == "range" and g.kb_spec["lo"] == 70 and g.kb_spec["hi"] == 100
    assert g.kb_spec["hi_strict"] is True and not g.kb_spec["lo_strict"]   # 100 is already high
    assert g.unit_factor("mmol/L") == pytest.approx(18.0)
    hb = cat.get("hb")
    assert set(hb.sex_spec) == {"M", "F"}


@pytest.mark.parametrize("name,key", [
    ("AC sugar飯前血糖", "glucose_ac"), ("AC sugar", "glucose_ac"), ("空腹血糖值AC", "glucose_ac"),
    ("飯前血糖檢查AC", "glucose_ac"), ("GPT", "alt"), ("ALT", "alt"), ("SGPT", "alt"), ("GPT(ALT)", "alt"),
    ("ALT(GPT)", "alt"), ("丙酮轉胺基 ALT(GPT)", "alt"), ("GOT麩草轉氨基脢", "ast"), ("草酸轉胺基 AST(GOT)", "ast"),
    ("Hb", "hb"), ("HGB", "hb"), ("血紅素", "hb"), ("Hb血色素", "hb"), ("平均血紅素MCH", "mch"),
    ("MCHC平均血色素濃度", "mchc"), ("K", "k"), ("鉀", "k"), ("P", "p"), ("磷", "p"), ("鈉", "na"),
    ("Na", "na"), ("鈣", "ca"), ("氯", "cl"), ("Potassium鉀", "k"),
    ("高敏感度C-反應蛋白hs-CRP", "hs_crp"), ("HS-CRP高敏感度C反應蛋白", "hs_crp"), ("高敏感度C-反應蛋白", "hs_crp"),
    ("CRP C反應蛋白質", "crp"), ("WBC尿白血球", "urine_wbc"), ("WBC白血球總數", "wbc"),
    ("尿白血球[尿沉渣]", "urine_wbc_sed"), ("Bilirubin尿液膽紅素", "urine_bil"), ("Bilirubin", "bil_total"),
    ("T-BILI總膽紅素", "bil_total"), ("D-BILI直接膽紅素", "bil_direct"), ("T4 四碘甲狀腺素", "t4"),
    ("游離四碘甲狀腺素Free T4", "free_t4"), ("LY#淋巴球數", "lymph_abs"), ("淋巴球Lym%", "lymph_pct"),
    ("LY淋巴球百分比", "lymph_pct"), ("嗜中性球Neut %", "neut_pct"), ("NE#嗜中性球數", "neut_abs"),
    ("眼壓[右眼] I.O.P. R", "iop_right"), ("左眼壓", "iop_left"), ("(三頻)左耳精密聽力1K HZ", "hearing_l_1k"),
    ("右耳精密聽力2K HZ", "hearing_r_2k"), ("矯正後視力[右眼] Corrected R", "va_corr_right"),
    ("LDH乳酸脫氫脢", "ldh"), ("乳酸脫氫 LDH", "ldh"), ("Anti-HBs results", "anti_hbs"),
    ("HBsAgB型肝炎表面抗原", "hbsag"), ("總膽固醇/高密度膽固醇CHOL/HDL", "chol_hdl_ratio"),
    ("低/高膽固醇比", "ldl_hdl_ratio"), ("白蛋白/球蛋白比", "ag_ratio"), ("Albumin/Globulin", "ag_ratio"),
    ("Uric Acid尿酸", "uric_acid"), ("CA-199胰臟癌", "ca199"), ("cyfra21-1肺癌", "cyfra"),
    ("Sp.gr尿比重檢查", "urine_sg"), ("PROTEIN尿蛋白", "urine_protein"), ("Sugar尿糖檢查", "urine_glucose"),
    ("麥胺酸轉移侫γ-GT(GGT)", "ggt"), ("HbAlC醣化血色素", "hba1c"), ("收縮壓Systolic B.P.", "sbp"),
    ("Risk-factor動脈硬化指數", "risk_factor"), ("糞便潛血免疫分析(定量)", "fobt"),
])
def test_synonym_resolution(cat, name, key):
    it = cat.resolve(name, has_pct="%" in name)
    assert it is not None and it.key == key, (name, it and it.key)


@pytest.mark.parametrize("name", ["Lactate", "陳大文收縮壓", "年齡", "受檢日期", "電話", "Risk", "factor",
                                  "台北市信義區腰圍", "MASS", "HZ", "results"])
def test_unresolvable_names_are_not_guessed(cat, name):
    assert cat.resolve(name) is None


# ─── 2. normalisation ────────────────────────────────────────────────────────

def test_nfkc_full_width_range_and_value(cat):
    assert ab.parse_reference_range("０．３～１．０") == (0.3, 1.0)
    assert ab.parse_reference_range("＜200") == (None, 200.0)
    f = one("總膽紅素 １．２ mg/dL （０．３～１．０）", cat=cat)
    assert f["value"] == 1.2 and f["status"] == "high" and f["range_source"] == "report"


def test_micro_sign_unification_keeps_ug_distinct_from_g():
    assert ab.unit_key("μg/dL") == ab.unit_key("µg/dL") == "ug/dl"
    assert ab.unit_key("μg/dL") != ab.unit_key("g/dL")
    assert ab.unit_key("10^3/μL") == ab.unit_key("x10^3/uL") == ab.unit_key("^3/μL")


def test_t4_in_ug_per_dl_is_t4_not_free_t4(cat):
    f = one("T4 8.0 μg/dL", cat=cat)
    assert f["canonical_key"] == "t4" and f["status"] == "normal" and f["unit"] == "μg/dL"


@pytest.mark.parametrize("line,val,flag", [
    ("AST 45 H (13-39 U/L)", 45.0, "H"), ("＊T-BILI總膽紅素： 1.2↑ 參考值 (0.3～1.0 mg/dL)", 1.2, "H"),
    ("GOT麩草轉氨基脢： 12↓ 參考值 (13～39 U/L)", 12.0, "L"), ("LDL-C 165 H", 165.0, "H"),
])
def test_trailing_flags_are_stripped(cat, line, val, flag):
    f = one(line, cat=cat)
    assert f["value"] == val and f["report_flag"] == flag


def test_bracketed_names(cat):
    assert one("GPT(ALT) 55 U/L (7~52)", cat=cat)["status"] == "high"
    assert one("ALT(GPT) 55", cat=cat)["canonical_key"] == "alt"


@pytest.mark.parametrize("line", ["血壓 148/92 mmHg", "B.P. 148/92", "收縮壓/舒張壓 148/92",
                                  "血壓(收縮壓/舒張壓)：148/92"])
def test_blood_pressure_split_into_two_items(cat, line):
    f, _ = run(line, cat=cat)
    by = {x["canonical_key"]: x for x in f}
    assert set(by) == {"sbp", "dbp"}
    assert by["sbp"]["value"] == 148 and by["sbp"]["status"] == "high"
    assert by["dbp"]["value"] == 92 and by["dbp"]["status"] == "high"


# ─── 3. range parsing ────────────────────────────────────────────────────────

@pytest.mark.parametrize("ref,kind,lo,hi", [
    ("70～100 mg/dL", "range", 70, 100), ("(0.60～1.20 mg/dL)", "range", 0.6, 1.2), ("-1～4", "range", -1, 4),
    ("7-25mg/dL", "range", 7, 25), ("<130mg/dL", "upper", None, 130), ("≤2.37 ng/mL", "upper", None, 2.37),
    ("≦2.37", "upper", None, 2.37), (">60 ml/min/1.73m2", "lower", 60, None), ("≧40", "lower", 40, None),
    (">=60", "lower", 60, None), ("1.0 以下", "upper", None, 1.0), ("5.0以上", "lower", 5.0, None),
    ("(>90", "lower", 90, None),
])
def test_parse_ref_numeric(ref, kind, lo, hi):
    sp = ab.parse_ref(ref)
    assert sp["kind"] == kind
    assert sp.get("lo") == lo and sp.get("hi") == hi


def test_parse_ref_strictness():
    assert ab.parse_ref("<130")["strict"] is True
    assert ab.parse_ref("≦130")["strict"] is False
    assert ab.parse_ref(">50")["strict"] is True
    assert ab.parse_ref("≧50")["strict"] is False


@pytest.mark.parametrize("ref", ["男:13-18 女:12-16", "男 13.5~17.5 / 女 12.0~16.0 g/dL", "男<7.0 女<6.0",
                                 "女:2.3-6.6 男:4.4-7.6 mg/dL", "M: 13-18, F: 12-16"])
def test_parse_ref_sex_specific_never_low_gt_high(ref):
    sp = ab.parse_ref(ref)
    assert sp["kind"] == "sex"
    for s in ("M", "F"):
        b = ab._spec_bounds(sp[s])
        assert b is not None
        if b[0] is not None and b[1] is not None:
            assert b[0] <= b[1]
    assert ab.parse_reference_range("男<7.0 女<6.0") is None  # legacy helper no longer returns (7.0, 6.0)


@pytest.mark.parametrize("ref,kind", [("陰性", "qual_neg"), ("(-)", "qual_neg"), ("Negative", "qual_neg"),
                                      ("neg", "qual_neg"), ("(-～- (+/-))", "qual_neg"), ("-~- (+/-", "qual_neg"),
                                      ("(mg/dL)", "unit_only"), ("(/HPF)", "unit_only"),
                                      ("網膜正常|無明顯異常", "none"), ("ALT", "none")])
def test_parse_ref_qualitative(ref, kind):
    assert ab.parse_ref(ref)["kind"] == kind


@pytest.mark.parametrize("value,ref,sex,name,expected", [
    ("7.2", "男:4.4-7.6 女:2.3-6.6 mg/dL", "F", "尿酸", "high"),
    ("7.2", "男:4.4-7.6 女:2.3-6.6 mg/dL", "M", "尿酸", "normal"),
    ("7.2", "男:4.4-7.6 女:2.3-6.6 mg/dL", None, "尿酸", "normal"),   # inside the M range -> not flagged
    ("8.0", "男:4.4-7.6 女:2.3-6.6 mg/dL", None, "尿酸", "high"),     # outside both
    ("6.5", "男<7.0 女<6.0", "F", "HbA1c", "high"),
    ("6.5", "男<7.0 女<6.0", "M", "HbA1c", "normal"),
])
def test_sex_specific_report_range(value, ref, sex, name, expected):
    assert ab.classify_value(value, ref, sex=sex, name=name) == expected


# ─── 4. value parsing ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,kind,grade", [
    ("陰性", "neg", None), ("陽性", "pos", "+"), ("+", "pos", "+"), ("-", "neg", None), ("±", "trace", None),
    ("+/-", "trace", None), ("1+", "pos", "1+"), ("2+", "pos", "2+"), ("3+", "pos", "3+"),
    (">=1000(3+)", "pos", "3+"), ("80(2+)", "pos", "2+"), ("Trace", "trace", None), ("Negative", "neg", None),
    ("Calcium Oxalate 2+", "pos", "2+"), ("12.5", "num", None), ("<0.05", "num", None), ("3-5", "text", None),
])
def test_parse_value_kinds(raw, kind, grade):
    pv = ab.parse_value(raw)
    assert pv["kind"] == kind
    if grade:
        assert pv["grade"] == grade


def test_censored_values(cat):
    f = one("CRP <0.05 mg/dL", cat=cat)
    assert f["status"] == "normal" and f["censored"] == "<" and f["value"] == 0.05
    assert ab.classify_value("<0.01", "(0.41～6.69 ng/mL)", name="AMH") == "low"
    assert ab.classify_value("<0.80", "(0～35 U/ml)") == "normal"
    assert ab.classify_value("<5", "(0～2 )") == "unknown"   # detection limit above the upper bound


@pytest.mark.parametrize("line,status", [
    ("尿蛋白 +", "positive"), ("尿糖 陰性", "negative"), ("HBsAg 陽性", "positive"), ("尿潛血 80(2+) H", "positive"),
    ("尿糖 >=1000(3+) (mg/dL)", "positive"), ("尿蛋白 ± (-)", "negative"), ("尿潛血 Trace", "negative"),
    ("尿酮體 3+ (-)", "positive"),
])
def test_qualitative_items(cat, line, status):
    f = one(line, cat=cat)
    assert f["status"] == status
    assert f["direction"] == ("high" if status == "positive" else "normal")


def test_protective_antibody_is_never_high(cat):
    f = one("Anti-HBs results 125.0 (<10 IU/L)", cat=cat)
    assert f["direction"] == "normal"


# ─── units ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("line,key,status", [
    ("Glucose 5.8 mmol/L", "glucose_ac", "high"), ("Glucose 5.0 mmol/L", "glucose_ac", "normal"),
    ("Cholesterol 6.0 mmol/L", "chol", "high"), ("TG 1.0 mmol/L", "tg", "normal"), ("TG 2.5 mmol/L", "tg", "high"),
    ("Creatinine 88.4 μmol/L", "creatinine", "normal"), ("Uric acid 480 μmol/L", "uric_acid", "high"),
    ("鈣 2.2 mmol/L", "ca", "normal"),
])
def test_unit_conversion_against_kb(cat, line, key, status):
    f = one(line, cat=cat)
    assert f["canonical_key"] == key and f["status"] == status and f["range_source"] == "reference"


def test_incompatible_unit_is_unknown_not_guessed(cat):
    f = one("Glucose 5.8 g/L", cat=cat)
    assert f["status"] == "unknown" and "不相容" in f["range_conflict_note"]


# ─── sex ─────────────────────────────────────────────────────────────────────

def test_detect_sex_from_header():
    assert ab.detect_sex("受檢者：王○○　性別：女　年齡：40") == "F"
    assert ab.detect_sex("Sex: M  Age 50") == "M"
    assert ab.detect_sex("性別：—") is None


def test_sex_unknown_kb_range_flags_only_outside_both(cat):
    f = one("血紅素 12.5 g/dL", cat=cat)
    assert f["status"] == "normal" and f["sex_unknown"] is True and f["sex_used"] is None
    f = one("血紅素 11.5 g/dL", cat=cat)
    assert f["status"] == "low" and f["sex_unknown"] is True


def test_sex_param_and_header(cat):
    assert one("血紅素 12.5 g/dL", cat=cat, sex="M")["status"] == "low"
    f = one("性別：女\n血紅素 12.5 g/dL", cat=cat)
    assert f["status"] == "normal" and f["sex_used"] == "F" and not f["sex_unknown"]


def test_sex_inferred_from_printed_sex_specific_ranges(cat):
    text = "Uric Acid尿酸 3.6 (2.3～6.6 mg/dL)\nHb血色素 12.5 (12～16 g/dL)\n血紅素 x"
    _, dbg = run(text, cat=cat)
    assert dbg["sex"] == "F" and dbg["sex_source"] == "range_inference"


# ─── report range vs KB range ────────────────────────────────────────────────

def test_report_range_is_primary_and_conflict_marked(cat):
    f = one("Uric Acid尿酸 3.6 (2.3～6.6 mg/dL)", cat=cat, sex="M")
    assert f["range_source"] == "report" and f["status"] == "normal"
    assert f["kb_status"] == "low" and f["range_conflict"] is True and "參考範圍" in f["range_conflict_note"]
    assert f["report_range"] == "2.3～6.6 mg/dL" or f["report_range"] == "2.3~6.6 mg/dL"


def test_kb_range_used_when_report_has_none(cat):
    f = one("LDL 165", cat=cat)
    assert f["range_source"] == "reference" and f["status"] == "high" and f["kb_range"]
    assert ab.RANGE_SOURCE_ALIASES["kb"] == "reference"   # older results used "kb"


# ─── public reference ranges: exclusive bounds, sex-specific, qualitative ─────

@pytest.mark.parametrize("line,key,status", [
    ("Glucose 100", "glucose_ac", "high"), ("Glucose 99.9", "glucose_ac", "normal"),
    ("Glucose 70", "glucose_ac", "normal"), ("Glucose 69", "glucose_ac", "low"),
    ("BMI 24.0", "bmi", "high"), ("BMI 23.9", "bmi", "normal"), ("BMI 18.5", "bmi", "normal"),
    ("LDL 130", "ldl", "high"), ("LDL 129", "ldl", "normal"),
    ("ALT 41", "alt", "normal"), ("ALT 42", "alt", "high"),              # inclusive bound
    ("HbA1c 5.7 %", "hba1c", "high"), ("HbA1c 5.6 %", "hba1c", "normal"),
])
def test_reference_bounds_respect_exclusivity(cat, line, key, status):
    f = one(line, cat=cat)
    assert f["canonical_key"] == key and f["status"] == status and f["range_source"] == "reference"


def test_exclusive_bound_applies_to_sex_specific_ranges(cat):
    assert one("腰圍 80 cm", cat=cat, sex="F")["status"] == "high"
    assert one("腰圍 80 cm", cat=cat, sex="M")["status"] == "normal"
    assert one("腰圍 90 cm", cat=cat, sex="M")["status"] == "high"
    f = one("腰圍 85 cm", cat=cat)                      # sex unknown: abnormal for F only
    assert f["status"] == "normal" and f["sex_unknown"] is True


def test_reference_range_text_is_readable(cat):
    assert one("Glucose 120", cat=cat)["kb_range"] == "≧70 且 <100 mg/dL"
    assert one("ALT 50", cat=cat)["kb_range"] == "0～41 U/L"
    f = one("血紅素 12.5 g/dL", cat=cat)
    assert f["kb_range"].startswith("男 13～17.2") and "女 12～15.2" in f["kb_range"]


def test_qualitative_normal_from_reference_file(cat):
    assert cat.get("hbsag").qual_expected == "negative"
    assert cat.get("hpv_rlu").qual_expected == "negative"     # only the reference file says so
    assert one("HBsAg 陽性", cat=cat)["status"] == "positive"
    assert one("HBsAg 0.5 COI", cat=cat)["status"] == "normal"
    assert one("HBsAg 1.0 COI", cat=cat)["status"] == "high"  # high_exclusive cut-off


def test_one_sex_only_reference_range(cat):
    # PSA has no female range and progesterone no general one: never guessed.
    assert "F" not in cat.get("psa").sex_spec
    assert one("PSA 6.0 ng/mL", cat=cat, sex="M")["status"] == "high"
    assert one("黃體酮 5.0 ng/mL", cat=cat)["status"] == "unknown"


# ─── 5. table extraction ─────────────────────────────────────────────────────

def test_header_aware_table_ignores_row_number_and_category(cat):
    table = [["序號", "檢驗項目", "結果", "單位", "參考值"],
             ["1", "ALT", "55", "U/L", "7-52"],
             ["2", "AST", "45 H", "U/L", "13-39"],
             ["生化", "Glucose", "115", "mg/dL", "70-100"]]
    f, _ = run("", [table], cat=cat)
    by = {x["canonical_key"]: x for x in f}
    assert by["alt"]["value"] == 55 and by["alt"]["status"] == "high"
    assert by["ast"]["value"] == 45 and by["ast"]["status"] == "high" and by["ast"]["report_flag"] == "H"
    assert by["glucose_ac"]["value"] == 115 and by["glucose_ac"]["status"] == "high"
    assert all(x["source"] == "table" for x in f)


def test_header_variants_and_flag_column(cat):
    table = [["檢查項目", "檢驗值", "單位", "參考範圍", "判定"],
             ["Creatinine 肌酸酐", "0.58", "mg/dL", "0.60～1.20", "L"]]
    f = one("", [table], cat=cat)
    assert f["status"] == "low" and f["report_range"].startswith("0.60") and f["report_flag"] == "L"


def test_two_column_side_by_side_table(cat):
    table = [["項目", "結果", "參考值", "項目", "結果", "參考值"],
             ["總膽固醇Cholesterol", "225 H", "(<200 mg/dl)", "HDL高密度脂蛋白", "38", "(>40 mg/dL)"],
             ["三酸甘油脂TG", "100", "(<150 mg/dl)", "LDL低密度脂蛋白", "141 H", "(<130 mg/dl)"]]
    f, _ = run("", [table], cat=cat)
    by = {x["canonical_key"]: x["status"] for x in f}
    assert by == {"chol": "high", "hdl": "low", "tg": "normal", "ldl": "high"}


def test_headerless_rows(cat):
    f, _ = run("", [[["AC sugar飯前血糖", "92", "(70～100 mg/dL)"]], [["尿糖", ">=1000(3+)", "(mg/dL)"]]], cat=cat)
    by = {x["canonical_key"]: x for x in f}
    assert by["glucose_ac"]["status"] == "normal"
    assert by["urine_glucose"]["status"] == "positive" and "3+" in by["urine_glucose"]["value_text"]


def test_split_range_across_unit_and_ref_columns_is_repaired(cat):
    table = [["檢查項目", "結果", "單位", "參考範圍"], ["BUN 尿素氮", "11.3", "-25mg/dL", "7"],
             ["LDH乳酸脫氫脢", "137", "-271U/L", "140"]]
    f, _ = run("", [table], cat=cat)
    by = {x["canonical_key"]: x for x in f}
    assert by["bun"]["status"] == "normal" and by["ldh"]["status"] == "low"


def test_multiline_cells(cat):
    table = [["項目", "結果", "參考值"], ["尿糖", ">=1000(3\n+) H", "(mg/dL)"],
             ["高敏感度C-反應蛋白hs-CR\nP", "0.157", "(<0.3 mg/dL)"]]
    f, _ = run("", [table], cat=cat)
    by = {x["canonical_key"]: x for x in f}
    assert by["urine_glucose"]["status"] == "positive"
    assert by["hs_crp"]["status"] == "normal"


def test_text_line_with_two_items(cat):
    f, _ = run("矯正後視力[右眼] 0.5 L (0.8～2.0 ) (三頻)右耳精密聽力1K HZ 30 H (0～25 dB)", cat=cat)
    by = {x["canonical_key"]: x for x in f}
    assert by["va_corr_right"]["status"] == "low" and by["hearing_r_1k"]["value"] == 30
    assert by["hearing_r_1k"]["status"] == "high"


def test_table_layout_text_line_value_unit_range(cat):
    f = one("BUN 尿素氮 6.8 mg/dL 7~25", cat=cat)
    assert f["status"] == "low" and f["unit"] == "mg/dL"
    f = one("WBC白血球總數 3.8 ^3/μL 3.8~10.0 10", cat=cat)
    assert f["status"] == "normal" and f["value"] == 3.8


def test_pdf_utils_table_path(cat):
    """pdf_utils + extraction on a tiny generated PDF (skipped if reportlab is absent)."""
    reportlab = pytest.importorskip("reportlab")  # noqa: F841
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    import io
    from pdf_utils import extract_pdf_content
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.drawString(50, 800, "Glucose 125 mg/dL (70-100)")
    c.drawString(50, 780, "LDL 99 mg/dL (<130)")
    c.save()
    content = extract_pdf_content(buf.getvalue())
    f, _ = run(content["text"], content["tables"], cat=cat)
    by = {x["canonical_key"]: x["status"] for x in f}
    assert by == {"glucose_ac": "high", "ldl": "normal"}


def test_bare_range_after_reference_word(cat):
    f = one("血球容積比 Hct： 47.0 % 參考值 36～48", cat=cat, sex="F")
    assert f["range_source"] == "report" and f["status"] == "normal" and f["kb_status"] == "high"
    assert f["range_conflict"] is True
    f = one("＊ALT(GPT) 丙胺酸轉胺酶： 42↑ U/L 參考值 <41", cat=cat)
    assert f["status"] == "high" and f["range_source"] == "report"


def test_names_naming_several_items_are_ambiguous(cat):
    assert cat.resolve("LH:FSH 比值") is None
    assert cat.resolve("AST/ALT") is None


# ─── 6. de-duplication ───────────────────────────────────────────────────────

def test_numeric_index_and_qualitative_interpretation_merge(cat):
    table = [["項目", "結果", "參考值"], ["HBsAg results", "0.38", "(0.00～0.90 COI)"],
             ["B型肝炎表面抗原HBsAg", "Negative", "(陰性)"]]
    f = one("", [table], cat=cat)
    assert f["canonical_key"] == "hbsag" and f["value"] == 0.38 and f["qualitative_result"] == "Negative"
    # contradicting interpretation -> both kept, marked
    table[2][1] = "Positive"
    f, _ = run("", [table], cat=cat)
    assert len(f) == 2 and all(x["duplicate_conflict"] for x in f)



def test_dedupe_prefers_table_over_text(cat):
    text = "CHOL總膽固醇 394 (0-200 mg/dL)"
    table = [["檢查項目", "結果", "單位", "參考範圍"], ["CHOL總膽固醇", "394", "mg/dL", "0~200"]]
    f, _ = run(text, [table], cat=cat)
    assert len(f) == 1 and f[0]["source"] == "table" and not f[0]["duplicate_conflict"]


def test_dedupe_keeps_conflicting_values_and_marks_them(cat):
    text = "CHOL總膽固醇 394 (0-200 mg/dL)\n總膽固醇 180 (0-200 mg/dL)"
    f, _ = run(text, cat=cat)
    assert len(f) == 2 and all(x["duplicate_conflict"] for x in f)


def test_dedupe_same_value_different_names(cat):
    f, _ = run("總膽紅素 1.5 (0.3～1.0)\nT-BILI 1.5 (0.3～1.0)", cat=cat)
    assert len(f) == 1


# ─── 7. unmatched names ──────────────────────────────────────────────────────

def test_unmatched_and_admin_fields_are_dropped(cat):
    text = "受檢日期：2026-03-15\n年齡：42\n電話 0912345678\n陳大文收縮壓 150\nLactate 12\nGlucose 99"
    f, dbg = run(text, cat=cat)
    assert [x["canonical_key"] for x in f] == ["glucose_ac"]
    assert {"年齡", "電話", "陳大文收縮壓", "Lactate"} <= set(dbg["unmatched"])


# ─── 8. finding schema ───────────────────────────────────────────────────────

LEGACY_KEYS = {"name", "value", "unit", "direction", "ref_low", "ref_high", "ref_unit", "matched_name", "source"}
NEW_KEYS = {"canonical_key", "display_name", "status", "range_source", "report_range", "kb_range",
            "range_conflict", "range_conflict_note", "sex_used", "sex_unknown", "raw_name"}


def test_finding_keys(cat):
    f = one("LDL低密度脂蛋白 309 (<130mg/dL)", cat=cat)
    assert LEGACY_KEYS | NEW_KEYS <= set(f)
    assert f["direction"] == "high" and f["ref_high"] == 130 and f["ref_low"] is None
    assert f["matched_name"] and f["display_name"] and f["raw_name"] == "LDL低密度脂蛋白"
    assert isinstance(f["range_conflict"], bool)


def test_legacy_value_dicts_and_helpers(cat):
    f = ab.evaluate_findings([{"name": "LDL", "value": 150.0, "unit": "mg/dL"}], cat)
    assert f[0]["direction"] == "high"
    txt = ab.format_findings_for_llm(one("尿蛋白 2+ (-)", cat=cat) and [one("尿蛋白 2+ (-)", cat=cat)])
    assert "POSITIVE" in txt
    assert ab.abnormal_only([one("尿蛋白 2+ (-)", cat=cat)])


# ─── LLM fallback (Q11) ──────────────────────────────────────────────────────

class FakeNormalizer:
    def __init__(self, mapping):
        self.mapping, self.calls = mapping, []

    def __call__(self, names, catalog):
        self.calls.append(list(names))
        return {n: self.mapping.get(n) for n in names}


def test_llm_normalizer_only_for_unresolved_names(cat):
    fake = FakeNormalizer({"血清肌酸酐值X": "creatinine"})
    text = "Glucose 99\n血清肌酸酐值X 2.5 (0.7-1.3)\n年齡：42"
    f, dbg = run(text, cat=cat, llm_normalizer=fake)
    assert fake.calls == [["血清肌酸酐值X"]]          # one batched call, admin field not sent
    by = {x["canonical_key"]: x for x in f}
    assert by["creatinine"]["match_method"] == "llm" and by["creatinine"]["status"] == "high"
    assert by["creatinine"]["kb_range"] is None      # never judged against the KB range
    assert by["glucose_ac"]["match_method"] == "synonym"


def test_llm_normalizer_not_called_when_everything_matches(cat):
    fake = FakeNormalizer({})
    run("Glucose 99\nLDL 120", cat=cat, llm_normalizer=fake)
    assert fake.calls == []


def test_llm_mapping_rejected_on_unit_conflict_or_duplicate_key(cat):
    fake = FakeNormalizer({"奇怪血糖": "glucose_ac", "葡萄糖值Y": "ldl"})
    f, dbg = run("LDL 120\n奇怪血糖 5.5 g/L\n葡萄糖值Y 99", cat=cat, llm_normalizer=fake)
    assert [x["canonical_key"] for x in f] == ["ldl"]
    assert {"奇怪血糖", "葡萄糖值Y"} <= set(dbg["unmatched"])


def test_make_llm_normalizer_batches_caches_and_uses_schema(monkeypatch, cat):
    import requests
    calls = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": json.dumps({"n0": "creatinine", "n1": "UNKNOWN"})}}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        calls.append((url, json))
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    ab._LLM_CACHE.clear()
    fn = ab.make_llm_normalizer(base_url="http://127.0.0.1:11434", model="m")
    out = fn(["肌酸酐Z", "未知Q"], cat)
    assert out == {"肌酸酐Z": "creatinine", "未知Q": None}
    assert len(calls) == 1 and calls[0][0].startswith("http://127.0.0.1:11434")
    schema = calls[0][1]["format"]
    enum = schema["properties"]["n0"]["enum"]
    assert "UNKNOWN" in enum and "glucose_ac" in enum and len(enum) == len(cat.items) + 1
    fn(["肌酸酐Z"], cat)
    assert len(calls) == 1  # cached in-process
    ab._LLM_CACHE.clear()


def test_make_llm_normalizer_failure_is_harmless(monkeypatch, cat):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    ab._LLM_CACHE.clear()
    assert ab.make_llm_normalizer()(["肌酸酐Z"], cat) == {"肌酸酐Z": None}
    ab._LLM_CACHE.clear()


def test_pipeline_exposes_sex_and_llm_switch():
    import pipeline
    params = inspect.signature(pipeline.analyze_pdf).parameters
    assert "sex" in params and "llm_normalize" in params


# ─── real samples (skipped when the de-identified eval PDFs are absent) ──────

SAMPLES = sorted((ROOT / "eval" / "samples").glob("*.pdf"))


@pytest.mark.skipif(not SAMPLES, reason="eval/samples not present")
@pytest.mark.parametrize("pdf", SAMPLES, ids=lambda p: p.stem)
def test_samples_one_finding_per_item(cat, pdf):
    from pdf_utils import extract_pdf_content
    c = extract_pdf_content(pdf.read_bytes())
    f, _ = run(c["text"], c["tables"], cat=cat)
    keys = [x["canonical_key"] for x in f if not x["duplicate_conflict"]]
    assert len(keys) == len(set(keys))
    assert not any(x["duplicate_conflict"] for x in f)


# ─── 7. numeric results next to a printed "negative" (2026-09-28) ───────────

@pytest.mark.parametrize("ref,hi", [("陰性(<1.0)", 1.0), ("Negative (<1.0 COI)", 1.0), ("(-)(<0.9)", 0.9)])
def test_negative_with_cutoff_parses_as_upper_bound(ref, hi):
    sp = ab.parse_ref(ref)
    assert sp["kind"] == "upper" and sp["hi"] == hi and sp["qual_neg"]


@pytest.mark.parametrize("line,status", [
    ("B型肝炎表面抗原 HBsAg 0.36 COI (陰性)", "normal"),         # "(陰性)" is the lab's reading -> public <1.0
    ("C型肝炎抗體 Anti-HCV 0.05 S/CO (陰性(<1.0))", "normal"),   # printed cut-off used
    ("B型肝炎表面抗原 HBsAg 250.3 COI (陰性(<1.0))", "high"),
    ("B型肝炎表面抗原 HBsAg 陽性 (陰性(<1.0))", "positive"),
])
def test_numeric_serology_next_to_printed_negative(cat, line, status):
    assert one(line, cat=cat)["status"] == status


def test_below_detection_limit_is_not_positive():
    assert ab.classify_value("<0.1", "陰性", name="尿蛋白") == "normal"
    assert ab.classify_value("30", "陰性", name="尿蛋白") == "high"  # dipstick without numeric range: unchanged


# ─── 8. header fields never leak into item names (2026-09-28) ──────────────

def test_patient_name_before_an_item_is_trimmed(cat):
    f, _ = run("姓名：王小明 身高：170 cm 體重：92 kg\n王小明 BMI：31.8\n", cat=cat)
    assert {x["canonical_key"] for x in f} == {"height", "weight", "bmi"}
    assert not any("王小明" in (x["name"] + x["raw_name"]) for x in f)


def test_multiword_item_names_are_kept_whole(cat):
    f = one("HDL Cholesterol 35 (>40 mg/dL)", cat=cat)
    assert f["canonical_key"] == "hdl" and f["name"] == "HDL Cholesterol"
