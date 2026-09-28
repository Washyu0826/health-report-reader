"""
test_abnormal.py — Unit tests for abnormal-value flagging (abnormal.py).

Three groups:
  R  reference-range parsing + direction   (value, reference[, sex][, name])
  E  extraction: can extract_lab_values() pull the right value out of a
     realistic report line / table row?
  P  end-to-end pipeline: extract_lab_values -> evaluate_findings with the
     public reference ranges (data/reference_ranges.json), on report lines that
     mostly carry their own reference range (the report's range is the ground
     truth, not the generic one).

Failures are EXPECTED on the R0 baseline — they are the to-do list.

Runs standalone (no pytest needed) and prints a pass/fail table:
    .venv/Scripts/python eval/test_abnormal.py            # table + summary
    .venv/Scripts/python eval/test_abnormal.py --json out.json
Also pytest-compatible:  pytest eval/test_abnormal.py

Adapter for group R: if abnormal.py grows a function
    classify_value(value: str, reference: str, sex: str|None = None, name: str = "") -> "high"|"low"|"normal"|"unknown"
it is used directly. Otherwise the R0 behaviour is emulated exactly as the
pipeline does it: parse_reference_range(ref) -> evaluate_findings() with a
one-entry KB (non-numeric values cannot enter the pipeline -> "unknown").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import abnormal  # noqa: E402

# (id, value, reference, sex, name, expected)
R_CASES = [
    ("R01 '<130' upper bound, high",          "309",   "(<130mg/dL)",            None, "LDL",  "high"),
    ("R02 '<130' just below",                 "129",   "(<130 mg/dl)",           None, "LDL",  "normal"),
    ("R03 '<130' boundary is high (strict)",  "130",   "(<130 mg/dl)",           None, "LDL",  "high"),
    ("R04 '≦2.37' inclusive, above",          "2.38",  "(≦2.37 ng/mL)",          None, "CYFRA", "high"),
    ("R05 '≦2.37' boundary normal",           "2.37",  "(≦2.37 ng/mL)",          None, "CYFRA", "normal"),
    ("R06 full-width ～ low",                 "39.2",  "(40～150 mg/dL)",        None, "HDL",  "low"),
    ("R07 full-width ～ low (BUN)",           "6.8",   "(7～25 mg/dL)",          None, "BUN",  "low"),
    ("R08 '>50' lower bound normal",          "75.5",  "(>50 mg/dL)",            None, "HDL",  "normal"),
    ("R09 '>50' boundary is low (strict)",    "50",    "(>50 mg/dL)",            None, "HDL",  "low"),
    ("R10 '≧40' boundary normal",             "40",    "(≧40 mg/dL)",            None, "HDL",  "normal"),
    ("R11 '≧40' below",                       "38",    "(≧40 mg/dL)",            None, "HDL",  "low"),
    ("R12 '>60' unit with digits (1.73m2)",   "132.54", "(>60 ml/min/1.73m2)",   None, "eGFR", "normal"),
    ("R13 negative lower bound -1～4",        "-2.33", "(-1～4 )",               None, "BMD",  "low"),
    ("R14 unit glued to range '140-271U/L'",  "137",   "(140-271U/L)",           None, "LDH",  "low"),
    ("R15 unit glued '7-25mg/dL' normal",     "14.6",  "(7-25mg/dL)",            None, "BUN",  "normal"),
    ("R16 unit has hyphen 'mm-Hg'",           "21",    "(1～20 mm-Hg)",          None, "IOP",  "high"),
    ("R17 range upper boundary inclusive",    "5",     "(0～5 )",                None, "Risk", "normal"),
    ("R18 unit '10^3/ul' contains digits",    "3.9",   "(4.0～11.0 10^3/ul)",    None, "WBC",  "low"),
    ("R19 missing unit, normal",              "1.5",   "(1.0～2.2 )",            None, "A/G",  "normal"),
    ("R20 ASCII tilde",                       "25.1",  "(4.79~23.3 ng/ml)",      None, "PRL",  "high"),
    ("R21 sex-specific, female high",         "7.2",   "男:4.4-7.6 女:2.3-6.6 mg/dL", "F", "尿酸", "high"),
    ("R22 sex-specific, male normal",         "7.2",   "男:4.4-7.6 女:2.3-6.6 mg/dL", "M", "尿酸", "normal"),
    ("R23 sex-specific, male low (Hb)",       "13.0",  "男 13.5~17.5 / 女 12.0~16.0 g/dL", "M", "Hb", "low"),
    ("R24 qualitative 陰性 vs 陰性",          "陰性",  "陰性",                   None, "HBsAg", "normal"),
    ("R25 qualitative 陽性 vs 陰性",          "陽性",  "陰性",                   None, "HBsAg", "high"),
    ("R26 qualitative '+' vs '(-)'",          "+",     "(-)",                    None, "尿蛋白", "high"),
    ("R27 qualitative '-' vs '(-)'",          "-",     "(-)",                    None, "尿蛋白", "normal"),
    ("R28 dipstick '3+' vs '(-)'",            "3+",    "(-)",                    None, "尿酮體", "high"),
    ("R29 dipstick '>=1000(3+)' unit-only ref", ">=1000(3+)", "(mg/dL)",        None, "尿糖",  "high"),
    ("R30 trace '+/-' vs '(-～- (+/-))'",     "+/-",   "(-～- (+/-))",           None, "尿蛋白", "normal"),
    ("R31 'Negative' vs 'Negative'",          "Negative", "Negative",            None, "HCV",  "normal"),
    ("R32 censored '<0.01' below range",      "<0.01", "(0.41～6.69 ng/mL)",     None, "AMH",  "low"),
    ("R33 censored '<0.80' within range",     "<0.80", "(0～35 U/ml)",           None, "CA19-9", "normal"),
    ("R34 protective antibody titre",         "125.0", "(<10 IU/L)",             None, "Anti-HBs", "normal"),
    ("R35 numeric vs '(neg)' reference",      "15",    "(neg)",                  None, "EP cell", "high"),
    # R2 additions
    ("R36 sex unknown, outside both ranges",  "8.0",   "男:4.4-7.6 女:2.3-6.6 mg/dL", None, "尿酸", "high"),
    ("R37 sex unknown, inside one range",     "7.2",   "男:4.4-7.6 女:2.3-6.6 mg/dL", None, "尿酸", "normal"),
    ("R38 full-width digits/point range",     "0.8",   "０．３～１．０",         None, "總膽紅素", "normal"),
    ("R39 dipstick '1+' vs '(-)'",            "1+",    "(-)",                    None, "尿蛋白", "high"),
    ("R40 'Trace' vs 陰性",                   "Trace", "陰性",                   None, "尿潛血", "normal"),
]

# (id, text_line_or_None, table_row_or_None, name_substring, expected_value)
E_CASES = [
    ("E01 value before '(<130mg/dL)'", "LDL低密度脂蛋白 309 (<130mg/dL)", None, "LDL|低密度", 309.0),
    ("E02 full-width colon + '參考值'", "AC sugar飯前血糖： 92 參考值 (70～100 mg/dL)", None, "血糖", 92.0),
    ("E03 arrow flag glued '1.2↑'", "＊T-BILI總膽紅素： 1.2↑ 參考值 (0.3～1.0 mg/dL)", None, "膽紅素", 1.2),
    ("E04 table cell with ' H' flag", None, ["總膽固醇Cholesterol", "225 H", "(<200 mg/dl)"], "膽固醇", 225.0),
    ("E05 table: negative value -2.33", None, ["超音波骨質密度檢查Bone Mineral Densitometry", "-2.33", "(-1～4 )"], "骨質", -2.33),
    ("E06 table: qualitative '>=1000(3+)'", None, ["尿糖", ">=1000(3+)", "(mg/dL)"], "尿糖", "3+"),
    ("E07 name with space 'BUN 尿素氮'", "BUN 尿素氮 6.8 (7～25 mg/dL)", None, "尿素氮", 6.8),
    ("E08 table: 4-col with unit column", None, ["Creatinine 肌酸酐", "0.58", "mg/dL", "0.60～1.20"], "肌酸酐", 0.58),
    # R2 additions
    ("E09 blood pressure 148/92 -> diastolic", "血壓 148/92 mmHg", None, "舒張", 92.0),
    ("E10 two items on one text line", "矯正後視力[右眼] 0.5 L (0.8～2.0 ) (三頻)右耳精密聽力1K HZ 30 H (0～25 dB)",
     None, "聽力", 30.0),
    ("E11 table row-number column is not the value", None, ["1", "ALT", "55", "U/L", "7-52"], "ALT", 55.0),
]

# (id, report_line, expected_direction, item_value) — the finding is picked by VALUE,
# so a mangled name (e.g. 'results', 'B.P.') still counts as extracted.
P_CASES = [
    ("P01 creatinine vs report range (F)", "Creatinine 肌酸酐 0.58 (0.60～1.20 mg/dL)", "low", 0.58),
    ("P02 LDH low vs report 140-271", "LDH乳酸脫氫脢 137 (140-271U/L)", "low", 137.0),
    ("P03 Anti-HBs 125 must not be high", "Anti-HBs results 125.0 (<10 IU/L)", "normal", 125.0),
    ("P04 HDL 39.2 low", "HDL高密度脂蛋白 39.2 (40～150 mg/dL)", "low", 39.2),
    ("P05 LDL 309 high", "LDL低密度脂蛋白 309 (<130mg/dL)", "high", 309.0),
    ("P06 HS-CRP 0.112 vs report 0～0.1", "HS-CRP高敏感度C反應蛋白 0.112 (0～0.1 mg/dl)", "high", 0.112),
    ("P07 T-BIL 1.55 vs report <1.2", "總膽紅素T-BIL 1.55 (<1.2 mg/dl)", "high", 1.55),
    ("P08 SBP 124 vs report 90～120", "收縮壓Systolic B.P. 124 (90～120 mmHg)", "high", 124.0),
    # R0 crash: tie in candidates_in.sort() compares dicts -> TypeError
    ("P09 no crash on 'AC sugar飯前血糖' (sort tie)", "TABLE:AC sugar飯前血糖|92|(70～100 mg/dL)", "normal", 92.0),
    ("P10 no crash on 'Uric Acid尿酸' (sort tie)", "TABLE:Uric Acid尿酸|3.3|(4.4～7.6 mg/dL)", "low", 3.3),
    # R2 additions (no printed range -> reference range, unit conversion, one-character names, sex unknown)
    ("P11 glucose 5.8 mmol/L -> 104 mg/dL high", "Glucose 5.8 mmol/L", "high", 5.8),
    ("P12 one-character name 鉀", "鉀 5.9 mEq/L", "high", 5.9),
    ("P13 Hb 12.5, sex unknown -> not flagged", "血紅素 12.5 g/dL", "normal", 12.5),
    ("P14 bare range after 參考值", "血球容積比 Hct： 36.6 % 參考值 36～48", "normal", 36.6),
]


# ─── adapters ────────────────────────────────────────────────────────────────

def classify_r0(value: str, ref: str, sex=None, name: str = "") -> str:
    fn = getattr(abnormal, "classify_value", None)
    if callable(fn):
        return fn(value, ref, sex=sex, name=name)
    try:
        v = float(value)
    except ValueError:
        return "unknown"  # R0 pipeline cannot represent non-numeric values at all
    rng = abnormal.parse_reference_range(ref)
    kb = [{"names": ["__item__"], "range": rng, "unit": "", "raw_ref": ref}]
    out = abnormal.evaluate_findings([{"name": "__item__", "value": v, "unit": ""}], kb)
    return out[0]["direction"]


def _pick(values, name_sub):
    alts = [a.lower() for a in name_sub.split("|")]
    return [f for f in values if any(a in f["name"].lower() for a in alts)]


def run_all():
    results = []
    for cid, value, ref, sex, name, exp in R_CASES:
        try:
            got = classify_r0(value, ref, sex, name)
        except Exception as e:  # noqa: BLE001
            got = f"EXC {type(e).__name__}: {e}"
        results.append({"id": cid, "group": "R", "input": f"{value} | {ref} | sex={sex}",
                        "expected": exp, "got": got, "pass": got == exp})

    for cid, line, row, name_sub, exp in E_CASES:
        try:
            vals = abnormal.extract_lab_values(line or "", [[row]] if row else [])
            cand = _pick(vals, name_sub)
            if isinstance(exp, float):
                got = [c["value"] for c in cand]
                ok = any(abs(g - exp) < 1e-9 for g in got)
            else:  # qualitative — any representation containing the grade
                got = [str(c.get("value_text", c["value"])) for c in cand]
                ok = any(exp in g for g in got)
        except Exception as e:  # noqa: BLE001
            got, ok = f"EXC {type(e).__name__}: {e}", False
        results.append({"id": cid, "group": "E", "input": line or row, "expected": exp,
                        "got": got, "pass": ok})

    kb = abnormal.load_checkitem_lookup(str(ROOT / "data" / "reference_ranges.json"))
    for cid, line, exp, val in P_CASES:
        try:
            if line.startswith("TABLE:"):  # a pdfplumber table row, cells split by '|'
                vals = abnormal.extract_lab_values("", [[line[6:].split("|")]])
            else:
                vals = abnormal.extract_lab_values(line, [])
            findings = abnormal.evaluate_findings(vals, kb)
            cand = [f for f in findings if abs(f["value"] - val) < 1e-9]
            got = [f"{c['name']}={c['value']}:{c['direction']}(ref {c['ref_low']}~{c['ref_high']})" for c in cand]
            ok = bool(cand) and all(c["direction"] == exp for c in cand)
        except Exception as e:  # noqa: BLE001
            got, ok = f"EXC {type(e).__name__}: {e}", False
        results.append({"id": cid, "group": "P", "input": line, "expected": exp, "got": got, "pass": ok})
    return results


# ─── pytest entry points ─────────────────────────────────────────────────────

def _mk_test(r):
    def _t():
        assert r["pass"], f"{r['id']}: expected {r['expected']!r}, got {r['got']!r}"
    return _t


try:
    for _r in run_all():
        globals()["test_" + _r["id"].split()[0]] = _mk_test(_r)
except Exception:  # pragma: no cover
    pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    res = run_all()
    for r in res:
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"[{mark}] {r['id']:45s} expected={r['expected']!r:10}  got={r['got']!r}")
    by = {}
    for r in res:
        g = by.setdefault(r["group"], [0, 0])
        g[0] += r["pass"]
        g[1] += 1
    tot = sum(r["pass"] for r in res)
    print("\nSUMMARY  " + "  ".join(f"{g}: {p}/{n}" for g, (p, n) in sorted(by.items()))
          + f"  | total {tot}/{len(res)} ({tot / len(res):.0%})")
    if args.json:
        Path(args.json).write_text(json.dumps({"results": res, "summary": by}, ensure_ascii=False, indent=1,
                                              default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
