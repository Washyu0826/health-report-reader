"""
gt_rules.py — Deterministic, pipeline-INDEPENDENT rules used to derive ground
truth (abnormal direction + metric tags) from the structured source rows.

Deliberately does NOT import abnormal.py: the ground truth must not inherit the
bugs of the code under test.

Rules (documented so they can be argued with):
  * Reference strings are NFKC-normalised (full-width -> half-width).
  * "lo~hi" / "lo-hi" / "lo～hi"   -> inclusive normal band.
  * "<x" / "≦x" / "≤x"              -> upper bound. "<x" is strict (v >= x is high),
                                        "≦x" is inclusive (v > x is high).
  * ">x" / "≧x" / "≥x"              -> lower bound, strict / inclusive analogously.
  * "-", "neg", "陰性", "negative", "(-~- (+/-))" -> qualitative-negative reference.
  * Values "<x"/"≦x" are censored: treated as x - tiny; ">x"/">=x" as x + tiny.
  * Qualitative positive grades ("+", "1+".."4+", "Positive", "陽性", "80(2+)",
    ">=1000(3+)") are HIGH when the reference is qualitative-negative OR the
    reference only carries a unit (dipstick style, e.g. "(mg/dL)").
    "+/-" (trace) is normal. Negative words are normal.
  * Numeric value against a qualitative-negative reference: > 0 is HIGH.
  * Protective antibody titres (Anti-HBs / B肝表面抗體) above the cut-off are
    NORMAL (immunity), not high.
  * Anything else (free text, "20-29" sediment counts with unit-only refs,
    percentages with prose refs) -> "unknown" and excluded from metric GT.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

EPS = 1e-9
NUM = r"-?\d+(?:\.\d+)?"

_QUAL_NEG_REF = {"-", "neg", "negative", "陰性", "(-)", "無", "-~-(+/-)", "-~- (+/-)"}
_NEG_VALUES = {"-", "neg", "negative", "陰性", "無", "normal", "正常", "(-)", "nonreactive"}
_TRACE_VALUES = {"+/-", "±", "trace", "微量"}


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()


def _strip_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s


def parse_ref(ref: str) -> Dict:
    """Return {'kind': range|upper|lower|qual_neg|unit_only|none, ...}."""
    raw = nfkc(ref)
    s = _strip_parens(raw)
    if not s:
        return {"kind": "none"}
    low = s.lower().replace(" ", "")
    if low in {x.replace(" ", "") for x in _QUAL_NEG_REF} or low.startswith("-~-"):
        return {"kind": "qual_neg"}
    m = re.match(rf"^(<=|≦|≤|<)\s*({NUM})", s)
    if m:
        return {"kind": "upper", "hi": float(m.group(2)), "strict": m.group(1) == "<"}
    m = re.match(rf"^(>=|≧|≥|>)\s*({NUM})", s)
    if m:
        return {"kind": "lower", "lo": float(m.group(2)), "strict": m.group(1) == ">"}
    m = re.match(rf"^({NUM})\s*[~〜\-–—到至]\s*({NUM})", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo <= hi:
            return {"kind": "range", "lo": lo, "hi": hi}
    if not re.search(r"\d", s.replace("m2", "").replace("^", "")) or re.fullmatch(r"[/A-Za-zµμ%\.\d\^]+", s):
        # unit only, e.g. "(mg/dL)", "(/HPF)", "(cell/uL)", "cm"
        if not re.search(NUM + r"\s*[~\-]", s):
            return {"kind": "unit_only"}
    return {"kind": "none"}


def parse_value(value: str) -> Dict:
    """Return {'kind': num|qual_pos|qual_neg|trace|text, 'v': float?}."""
    s = nfkc(value)
    low = s.lower()
    if low in _NEG_VALUES:
        return {"kind": "qual_neg"}
    if low in _TRACE_VALUES:
        return {"kind": "trace"}
    if re.search(r"(\d\+|^\+{1,4}$|positive|陽性)", low):
        g = re.search(r"(\d)\+", s)
        return {"kind": "qual_pos", "grade": (g.group(1) + "+") if g else "+"}
    m = re.fullmatch(rf"(<=|≦|≤|<|>=|≧|≥|>)?\s*({NUM})\s*%?", s)
    if m:
        v = float(m.group(2))
        op = m.group(1) or ""
        if op in ("<", "<=", "≦", "≤"):
            v -= 1e-6
        elif op in (">", ">=", "≧", "≥"):
            v += 1e-6
        return {"kind": "num", "v": v, "censored": bool(op)}
    return {"kind": "text"}


# Protective antibody titres: a value ABOVE the "<10" cut-off means immunity,
# i.e. normal/desirable (source system agrees: flag 0).
_PROTECTIVE_AB = re.compile(r"anti-?hbs|表面抗體", re.I)


def classify(value: str, ref: str, name: str = "") -> str:
    """high / low / normal / unknown."""
    d = _classify(value, ref)
    if d == "high" and name and _PROTECTIVE_AB.search(name):
        return "normal"
    return d


def _classify(value: str, ref: str) -> str:
    r = parse_ref(ref)
    v = parse_value(value)
    if v["kind"] == "num":
        x = v["v"]
        if r["kind"] == "range":
            if x < r["lo"] - EPS:
                return "low"
            if x > r["hi"] + EPS:
                return "high"
            return "normal"
        if r["kind"] == "upper":
            return "high" if (x >= r["hi"] if r["strict"] else x > r["hi"] + EPS) else "normal"
        if r["kind"] == "lower":
            return "low" if (x <= r["lo"] if r["strict"] else x < r["lo"] - EPS) else "normal"
        if r["kind"] == "qual_neg":
            return "high" if x > 0 else "normal"
        return "unknown"
    if v["kind"] in ("qual_neg", "trace"):
        return "normal" if r["kind"] in ("qual_neg", "unit_only") else "unknown"
    if v["kind"] == "qual_pos":
        return "high" if r["kind"] in ("qual_neg", "unit_only") else "unknown"
    return "unknown"


# ─── Names / aliases for metric matching ─────────────────────────────────────

# Canonical synonym groups: if any member appears in an item name, all members
# become aliases (lower-cased). Mainstream Taiwanese report vocabulary.
ALIAS_GROUPS: List[List[str]] = [
    ["總膽固醇", "chol", "cholesterol", "tc", "膽固醇"],
    ["低密度脂蛋白", "低密度膽固醇", "ldl", "ldl-c", "壞膽固醇"],
    ["高密度脂蛋白", "高密度膽固醇", "hdl", "hdl-c", "好膽固醇"],
    ["三酸甘油脂", "三酸甘油酯", "tg", "t-g", "triglyceride", "中性脂肪"],
    ["尿酸", "uric acid", "ua"],
    ["飯前血糖", "空腹血糖", "ac sugar", "glucose", "血糖"],
    ["收縮壓", "systolic", "sbp", "血壓"],
    ["舒張壓", "diastolic", "dbp", "血壓"],
    ["總膽紅素", "t-bili", "t-bil", "bilirubin", "膽紅素"],
    ["直接膽紅素", "d-bili", "d-bil", "direct bilirubin", "膽紅素"],
    ["乳酸脫氫", "ldh"],
    ["平均血球血色素", "mch"],
    ["鹼性磷酸", "alk-p", "alp"],
    ["麩草", "got", "ast", "sgot"],
    ["麩丙", "gpt", "alt", "sgpt"],
    ["體脂肪", "fat%", "體脂率", "body fat"],
    ["眼壓", "i.o.p", "iop"],
    ["矯正後視力", "矯正視力", "視力"],
    ["聽力", "hearing"],
    ["骨質密度", "bone mineral", "bmd", "t-score", "t值"],
    ["肌酸酐", "creatinine", "cre"],
    ["胰澱粉", "amylase", "澱粉酶"],
    ["磷", "phosphorus"],
    ["高敏感度c反應蛋白", "hs-crp", "crp", "c反應蛋白"],
    ["催乳激素", "prolactin", "泌乳激素"],
    ["白血球", "wbc"],
    ["抗穆勒氏管", "amh"],
    ["cyfra", "cyfra21-1", "肺癌指標"],
    ["尿素氮", "bun"],
    ["尿糖", "尿葡萄糖", "urine glucose", "糖尿"],
    ["尿潛血", "潛血", "occult blood", "血尿"],
    ["尿酮體", "ketone", "酮體"],
    ["上皮細胞", "ep cell", "epithelial"],
    ["細菌", "bacteria"],
]

_UNIT_WORDS = {"hz", "k hz", "db", "mg/dl", "u/l", "l", "r"}


def name_aliases(item_name: str, fix_name: str = "", fix_en: str = "") -> List[str]:
    s = nfkc(" ".join([item_name, fix_name or "", fix_en or ""]))
    out: List[str] = []
    for seg in re.findall(r"[A-Za-z][A-Za-z0-9\-\.#%/]*(?:\s[A-Za-z][A-Za-z0-9\-\.#%/]*)*", s):
        out.append(seg)
    for seg in re.findall(r"[一-鿿]+", s):
        out.append(seg)
    low = s.lower()
    for grp in ALIAS_GROUPS:
        if any(g in low for g in grp if len(g) >= 2):
            out.extend(grp)
    seen, clean = set(), []
    for a in out:
        a2 = a.strip().lower()
        if (len(a2) < 2 and a2.isascii()) or a2 in _UNIT_WORDS or a2 in seen:
            continue
        seen.add(a2)
        clean.append(a2)
    return clean


def display_name(item_name: str) -> str:
    """Short, human-ish display name used as the expected metric tag text."""
    s = nfkc(item_name).strip()
    if "HZ" in s.upper():  # hearing items: keep side + frequency
        return s
    zh = re.findall(r"[一-鿿][一-鿿\[\]/]*", s)
    m = re.match(r"^([A-Za-z][A-Za-z0-9\-\.#%]{0,7})(?=[\s一-鿿])", s)
    if zh:
        longest = max(zh, key=len)
        return f"{m.group(1)} {longest}" if m else longest
    return s


def value_tokens(value: str) -> List[str]:
    v = parse_value(value)
    s = nfkc(value)
    if v["kind"] == "num":
        m = re.search(NUM, s)
        return [m.group()] if m else [s]
    if v["kind"] == "qual_pos":
        return [v["grade"], "陽性", "positive", s.lower()]
    return [s.lower()]


def extract_unit(ref: str) -> str:
    s = _strip_parens(nfkc(ref))
    m = re.search(r"([A-Za-zµμ%/\^\d\.\-]*[A-Za-zµμ%/][A-Za-zµμ%/\^\d\.\-]*|公分|次/分)\s*$", s)
    if not m:
        return ""
    u = m.group(1)
    # strip leading number glued to the unit ("25mg/dL")
    u = re.sub(r"^-?\d+(?:\.\d+)?", "", u)
    return u.strip()
