"""
abnormal.py — Detect lab values that fall outside reference ranges (R2 "v2").

Pipeline
  1. load_checkitem_lookup(path) -> LabCatalog: canonical items built from
     data/lab_synonyms.json (curated synonyms, units, conversions)
     + a reference-range source. Default: data/reference_ranges.json (public
     adult ranges with citations; config.REFERENCE_RANGES_PATH). A legacy
     KB file in the {"Content": "【檢查項目】…【正常參考範圍】…"} list format is
     still accepted when passed explicitly (internal use).
  2. extract_lab_values(text, tables) -> raw records {name, value, unit,
     ref_text (the range PRINTED in the report), flag, source}. Tables are read
     header-aware (項目/結果/單位/參考值/判定, side-by-side groups); text lines
     are scanned left-to-right so one line may carry several items.
  3. evaluate_findings(records, catalog, sex=...) -> one finding per canonical
     item. Names are resolved by EXACT lookup of the normalised name / its
     segments in the synonym table (optional batched LLM fallback for names
     the table cannot resolve). Status is computed against the report range
     AND the reference range; the report range is primary, disagreements are
     marked.
     Unmatched names are dropped (kept in a debug list), never guessed.

Finding keys (backward compatible): name, value, unit, direction, ref_low,
ref_high, ref_unit, matched_name, source.  Added: canonical_key, display_name,
status (high/low/normal/positive/negative/unknown), range_source
("report"/"reference"/"report_flag"/None; older results may say "kb" for
"reference"), report_range, kb_range (= the reference range text; the key name
is kept for compatibility), range_conflict,
range_conflict_note, sex_used, sex_unknown, raw_name, value_text, flag,
match_method, duplicate_conflict.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent


def _resolve(path: str) -> str:
    """Relative paths resolve against the CWD first, then this module's directory."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p.resolve())
    return str((_HERE / p).resolve())


try:
    import config as _config
    _RANGES_SETTING = _config.REFERENCE_RANGES_PATH
except Exception:  # noqa: BLE001 — abnormal.py stays importable on its own
    _RANGES_SETTING = os.environ.get("REFERENCE_RANGES_PATH", "data/reference_ranges.json")
DEFAULT_RANGES_PATH = _resolve(_RANGES_SETTING)
DEFAULT_KB_PATH = DEFAULT_RANGES_PATH  # backward-compatible name
RANGE_SOURCE_REFERENCE = "reference"
RANGE_SOURCE_ALIASES = {"kb": RANGE_SOURCE_REFERENCE}  # older results / internal callers
DEFAULT_SYNONYMS_PATH = os.environ.get("LAB_SYNONYMS_PATH", str(_HERE / "data" / "lab_synonyms.json"))

EPS = 1e-9
NUM = r"[-+]?\d+(?:\.\d+)?"

# ─── Normalisation ───────────────────────────────────────────────────────────

_PUA = re.compile("[\ue000-\uf8ff]")
_CJK = "\u3400-\u9fff"
_CJK_RE = re.compile(f"[{_CJK}]")


def nfkc(s) -> str:
    """NFKC (full-width digits/letters/punct -> ASCII, µ -> μ), private-use
    glyphs -> space, whitespace collapsed."""
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = _PUA.sub(" ", s)
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", s).strip()


# Variant characters that appear in real reports (脢/酶/侫, 胺/氨, 醣/糖 ...).
_CHAR_MAP = str.maketrans({
    "脢": "酶", "侫": "酶", "胺": "氨", "醣": "糖", "酯": "脂", "沈": "沉", "佈": "布",
    "[": "(", "]": ")", "〈": "(", "〉": ")", "【": "(", "】": ")",
})
_KEY_STRIP = re.compile(r"[\s\.\-_·'’*:,;=]")


def nkey(s) -> str:
    """Lookup key for names: NFKC, lower, variant chars unified, spaces and
    . - _ · ' * : , removed, redundant outer brackets dropped. # % / + kept."""
    s = nfkc(s).lower().translate(_CHAR_MAP)
    s = _KEY_STRIP.sub("", s)
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")" and _balanced(s[1:-1]):
        s = s[1:-1]
    return s


def _balanced(s: str) -> bool:
    d = 0
    for ch in s:
        if ch == "(":
            d += 1
        elif ch == ")":
            d -= 1
            if d < 0:
                return False
    return d == 0


def unit_key(u) -> str:
    """Canonical comparison key for a unit string ('' when empty)."""
    s = nfkc(u).lower().strip()
    s = s.strip("()").strip()
    s = s.replace("µ", "u").replace("μ", "u").replace(" ", "")
    s = s.replace("mm~hg", "mmhg").replace("mm-hg", "mmhg")
    s = s.replace("e.u./dl", "eu/dl").replace("e.u/dl", "eu/dl").replace("e.u.", "eu")
    s = s.replace("公分", "cm").replace("公斤", "kg").replace("次/min", "次/分")
    if s.startswith("x10"):
        s = s[1:]
    s = re.sub(r"^\^(\d)", r"10^\1", s)
    s = re.sub(r"^10\*(\d)", r"10^\1", s)
    s = s.replace("/1.73m²", "/1.73m2")
    if s in ("iu/l",):
        s = "u/l"
    if s in ("mg/dl", "mg%"):
        s = "mg/dl"
    return s


# ─── Reference ranges ────────────────────────────────────────────────────────

_QUAL_NEG_REF = {"-", "neg", "negative", "陰性", "無", "nonreactive", "non-reactive", "notdetected",
                 "未檢出", "陰性(-)", "(-)", "normal", "正常"}
_OPS_LT = ("<", "<=", "≦", "≤", "＜")
_OPS_GT = (">", ">=", "≧", "≥", "＞")
_RANGE_SEP = r"[~\-–—到至]"
_SEX_PAIR = re.compile(
    r"(女性?|男性?|female|male|\b[MF](?=\s*[:：]))\s*[:：]?\s*"
    rf"((?:<=|>=|≦|≧|≤|≥|<|>)\s*{NUM}|{NUM}\s*{_RANGE_SEP}\s*{NUM})", re.I)
_UNIT_ONLY = re.compile(r"^[A-Za-zμu%°/\.\^\-]*[A-Za-zμu%°/][A-Za-zμu%°/\.\^\d\-]*$")
_QUAL_NEG_BOUND = re.compile(
    rf"^(?:陰性|negative|neg|non-?reactive|\(-\)|-)\s*\(?\s*((?:<=|≦|≤|<)\s*{NUM}[^()]*?)\s*\)?$", re.I)


def _strip_parens(s: str) -> str:
    s = s.strip()
    changed = True
    while changed and s:
        changed = False
        if s[0] == "(" and s[-1] == ")" and _balanced(s[1:-1]):
            s, changed = s[1:-1].strip(), True
        elif s[0] == "(" and s.count("(") > s.count(")"):  # unclosed "(>90"
            s, changed = s[1:].strip(), True
        elif s[-1] == ")" and s.count(")") > s.count("("):
            s, changed = s[:-1].strip(), True
    return s


def _clean_unit(rest: str) -> str:
    rest = rest.strip().strip("()").strip()
    if not rest or not re.search(r"[A-Za-zμ%°/]|次|公分|公斤", rest):
        return ""
    return rest


def parse_ref(ref) -> Dict:
    """Parse a printed reference into a spec dict.

    kind: range(lo, hi) | upper(hi, strict) | lower(lo, strict) | qual_neg |
          unit_only | sex(M, F) | none.  Always carries 'unit' and 'text'.
    """
    raw = nfkc(ref)
    s = re.sub(r"^(參考值|參考範圍|參考區間|正常值|正常範圍|ref(erence)?)\s*[:：]?\s*", "", raw, flags=re.I)
    s = _strip_parens(s)
    out = {"kind": "none", "unit": "", "text": raw}
    if not s:
        return out

    pairs = _SEX_PAIR.findall(s)
    if pairs:
        by_sex = {}
        for who, spec in pairs:
            w = who.lower()
            sx = "F" if (w.startswith("女") or w.startswith("f")) else "M"
            by_sex.setdefault(sx, spec)
        tail = s[_last_match_end(_SEX_PAIR, s):]
        unit = _clean_unit(re.sub(r"^[\s/,;、]+", "", tail))
        if len(by_sex) == 2:
            specs = {k: parse_ref(f"{v} {unit}") for k, v in by_sex.items()}
            if all(sp["kind"] in ("range", "upper", "lower") for sp in specs.values()):
                return {"kind": "sex", "M": specs["M"], "F": specs["F"], "unit": unit, "text": raw}

    # "陰性(<1.0)" / "Negative (<1.0 COI)": the printed cut-off that defines "negative" is the
    # usable reference for a numeric index; qualitative results still read it as "expected negative".
    m = _QUAL_NEG_BOUND.match(s)
    if m:
        sp = parse_ref(m.group(1))
        if sp["kind"] == "upper":
            return {**sp, "text": raw, "qual_neg": True}

    low = s.lower().replace(" ", "")
    if low in _QUAL_NEG_REF or low.startswith("-~-") or low.startswith("陰性"):
        return {**out, "kind": "qual_neg"}

    m = re.match(rf"^(<=|≦|≤|<)\s*({NUM})(.*)$", s)
    if m:
        return {**out, "kind": "upper", "hi": float(m.group(2)), "strict": m.group(1) == "<",
                "unit": _clean_unit(m.group(3))}
    m = re.match(rf"^(>=|≧|≥|>)\s*({NUM})(.*)$", s)
    if m:
        return {**out, "kind": "lower", "lo": float(m.group(2)), "strict": m.group(1) == ">",
                "unit": _clean_unit(m.group(3))}
    m = re.match(rf"^({NUM})\s*([A-Za-zμ%/\.\d\-\^]*)\s*(以下|以上)$", s)
    if m:
        v = float(m.group(1))
        if m.group(3) == "以下":
            return {**out, "kind": "upper", "hi": v, "strict": False, "unit": _clean_unit(m.group(2))}
        return {**out, "kind": "lower", "lo": v, "strict": False, "unit": _clean_unit(m.group(2))}
    m = re.match(rf"^({NUM})\s*{_RANGE_SEP}\s*({NUM})(.*)$", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo <= hi:
            return {**out, "kind": "range", "lo": lo, "hi": hi, "unit": _clean_unit(m.group(3))}
        return out
    if _looks_unit(s):
        return {**out, "kind": "unit_only", "unit": s.strip()}
    return out


_COMMON_UNIT_KEYS = None


def _looks_unit(s: str) -> bool:
    """A bare unit such as 'mg/dL', '/HPF', '%', 'COI', 'fL' (not an item name like 'ALT')."""
    global _COMMON_UNIT_KEYS
    t = s.replace(" ", "")
    if not t or not _UNIT_ONLY.match(t) or re.search(rf"{NUM}\s*{_RANGE_SEP}\s*{NUM}", t):
        return False
    if "/" in t or "%" in t:
        return True
    if _COMMON_UNIT_KEYS is None:
        _COMMON_UNIT_KEYS = {unit_key(u) for u in _COMMON_UNITS}
    return unit_key(t) in _COMMON_UNIT_KEYS


def _last_match_end(rx, s) -> int:
    end = 0
    for m in rx.finditer(s):
        end = m.end()
    return end


def parse_reference_range(ref: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Legacy helper: (low, high) with None for an open side; None if unparseable."""
    sp = parse_ref(ref)
    return _spec_bounds(sp)


def _spec_bounds(sp: Optional[Dict]) -> Optional[Tuple[Optional[float], Optional[float]]]:
    if not sp:
        return None
    if sp["kind"] == "range":
        return (sp["lo"], sp["hi"])
    if sp["kind"] == "upper":
        return (None, sp["hi"])
    if sp["kind"] == "lower":
        return (sp["lo"], None)
    return None


def _spec_equal(a: Dict, b: Dict) -> bool:
    if a["kind"] != b["kind"]:
        return False
    return all(abs((a.get(k) or 0) - (b.get(k) or 0)) < 1e-6 for k in ("lo", "hi"))


# ─── Values ──────────────────────────────────────────────────────────────────

_NEG_VALUES = {"-", "neg", "negative", "陰性", "(-)", "無", "nonreactive", "non-reactive", "notdetected",
               "未檢出", "normal", "正常", "陰性(-)"}
_TRACE_VALUES = {"+/-", "±", "trace", "微量", "+-", "tr", "(+/-)", "(±)"}
_POS_WORDS = ("陽性", "positive", "reactive")
_FLAG_TAIL = re.compile(r"(?:\s+(HH|LL|H|L|A)|\s*([↑↓]+|\*+))\s*$")
_GRADE = re.compile(r"([1-4])\s*\+")


def parse_value(raw) -> Dict:
    """Parse a result cell / token.

    kind: num | pos | neg | trace | text.  num carries v and op ('<', '<=',
    '>', '>=', ''); pos carries grade ('+', '1+'..'4+') and optionally v (the
    number printed with the grade, e.g. '>=1000(3+)').  flag: printed H/L/↑/↓/*.
    """
    s = nfkc(raw)
    flag = ""
    s = re.sub(r"^[*]+\s*", "", s)
    while True:
        m = _FLAG_TAIL.search(s)
        if not m or m.start() == 0:
            break
        f = m.group(1) or m.group(2)
        if f.startswith("↑"):
            f = "H"
        elif f.startswith("↓"):
            f = "L"
        elif f.startswith("*"):
            f = "*"
        flag = flag or f
        s = s[:m.start()].rstrip()
    out = {"kind": "text", "v": None, "op": "", "grade": None, "text": s, "flag": flag, "unit": ""}
    if not s:
        return out
    low = s.lower().replace(" ", "")
    if low in _NEG_VALUES:
        return {**out, "kind": "neg"}
    if low in _TRACE_VALUES:
        return {**out, "kind": "trace"}
    m = re.fullmatch(rf"(<=|>=|≦|≧|≤|≥|<|>)?\s*({NUM})\s*\(\s*([1-4]\+|\+{{1,4}})\s*\)", s)  # ">=1000(3+)"
    if m:
        return {**out, "kind": "pos", "grade": m.group(3), "v": float(m.group(2)), "op": _norm_op(m.group(1) or "")}
    m = re.fullmatch(r"\(?\s*([1-4]\+|\+{1,4})\s*\)?", s)  # "2+", "+", "(3+)"
    if m:
        return {**out, "kind": "pos", "grade": m.group(1)}
    if any(w in low for w in _POS_WORDS) and not low.startswith("non"):
        return {**out, "kind": "pos", "grade": "+"}
    m = re.fullmatch(rf"(<=|>=|≦|≧|≤|≥|<|>)?\s*({NUM})\s*(%)?", re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s))
    if m:
        return {**out, "kind": "num", "v": float(m.group(2)), "op": _norm_op(m.group(1) or ""),
                "unit": "%" if m.group(3) else ""}
    g = _GRADE.search(s)  # "Calcium Oxalate 2+"
    if g and not re.search(r"\d\s*[~\-]\s*\d", s):
        return {**out, "kind": "pos", "grade": g.group(1) + "+"}
    return out


def _norm_op(op: str) -> str:
    return {"≦": "<=", "≤": "<=", "≧": ">=", "≥": ">="}.get(op, op)


# ─── Catalog ─────────────────────────────────────────────────────────────────

@dataclass
class LabItem:
    key: str
    zh: str
    en: str
    synonyms: List[str]
    unit: str = ""
    kb_item: Optional[str] = None
    kb_index: Optional[int] = None
    kb_ref_text: str = ""
    kb_spec: Optional[Dict] = None
    sex_ref_text: Dict[str, str] = field(default_factory=dict)
    sex_spec: Dict[str, Dict] = field(default_factory=dict)
    conversions: Dict[str, float] = field(default_factory=dict)
    unit_equiv: List[str] = field(default_factory=list)
    qual_expected: Optional[str] = None
    specific_over: List[str] = field(default_factory=list)
    protective: bool = False
    # where kb_spec / sex_spec came from: "reference" (public reference_ranges.json)
    # or "kb" (legacy KB file); plus the citation / caveat of a public range.
    ref_origin: str = "kb"
    ref_source: str = ""
    ref_note: str = ""

    @property
    def display_name(self) -> str:
        return self.zh or self.en or self.key

    @property
    def kb_zh(self) -> str:
        if self.kb_item and self.ref_origin == "kb":
            return self.kb_item.split("（")[0].strip() or self.display_name
        return self.display_name

    def unit_factor(self, unit: str) -> Optional[float]:
        """Factor to convert a value in `unit` into this item's unit (None = incompatible)."""
        uk = unit_key(unit)
        if not uk or not self.unit:
            return 1.0
        if uk == unit_key(self.unit) or uk in {unit_key(u) for u in self.unit_equiv}:
            return 1.0
        for u, f in self.conversions.items():
            if uk == unit_key(u):
                return float(f)
        return None


_KB_NAME_RE = re.compile(r"【檢查項目】([^\n]*)")
_KB_REF_RE = re.compile(r"【正常參考範圍】([^\n]+)")


class LabCatalog:
    """Canonical lab items + exact-match synonym index."""

    def __init__(self, items: List[LabItem]):
        self.items: Dict[str, LabItem] = {it.key: it for it in items}
        self.index: Dict[str, List[str]] = {}
        self.collisions: List[Tuple[str, List[str]]] = []
        for it in items:
            for syn in [it.zh, it.en] + list(it.synonyms):
                k = nkey(syn)
                if not k:
                    continue
                keys = self.index.setdefault(k, [])
                if it.key not in keys:
                    keys.append(it.key)
        for k, keys in self.index.items():
            if len(keys) > 1:
                self.collisions.append((k, keys))
        self.known_units = {unit_key(u) for u in _COMMON_UNITS}
        for it in items:
            for u in [it.unit] + list(it.unit_equiv) + list(it.conversions):
                if u:
                    self.known_units.add(unit_key(u))

    # list-like behaviour (callers only pass the object through / count it)
    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items.values())

    def get(self, key: str) -> Optional[LabItem]:
        return self.items.get(key)

    def lookup_exact(self, s: str) -> List[str]:
        return self.index.get(nkey(s), [])

    # ── name resolution ──────────────────────────────────────────────────
    def resolve(self, name: str, has_pct: bool = False) -> Optional[LabItem]:
        for tier in _name_candidates(name):
            hits: List[Tuple[str, int, bool]] = []
            for cand, is_cjk in tier:
                variants = [cand + "%", cand] if has_pct and not cand.endswith("%") else [cand]
                for i, v in enumerate(variants):
                    keys = self.index.get(nkey(v), [])
                    for k in keys:
                        hits.append((k, len(nkey(v)), is_cjk))
                    if keys and i == 0 and len(variants) == 2:
                        break  # "淋巴球%" matched -> don't also take plain "淋巴球"
            if not hits:
                continue
            keys = {h[0] for h in hits}
            if len(keys) > 1:
                dominated = {k2 for k in keys for k2 in self.items[k].specific_over}
                if keys - dominated:
                    keys = keys - dominated
            if len(keys) == 1:
                return self.items[next(iter(keys))]
            return None  # e.g. "LH:FSH 比值", "AST/ALT": several unrelated items -> never guess
        return None

    def is_unit(self, tok: str) -> bool:
        uk = unit_key(tok)
        if not uk:
            return False
        if uk in self.known_units:
            return True
        return bool(re.fullmatch(r"[a-zu%°]+(/[a-z0-9\.\^]+)+", uk)) or bool(re.fullmatch(r"10\^\d+/[a-z]+", uk))


_COMMON_UNITS = [
    "mg/dL", "g/dL", "U/L", "IU/L", "ng/mL", "pg/mL", "%", "fL", "pg", "mmHg", "mm-Hg", "cm", "kg", "dB",
    "mEq/L", "mmol/L", "umol/L", "ug/dL", "ng/dL", "uIU/mL", "mIU/mL", "mIU/L", "COI", "S/CO", "/HPF", "/LPF",
    "EU/dL", "kg/m2", "kcal", "bpm", "次/分", "mg/L", "g/L", "mL/min/1.73m2", "U/mL", "IU/mL", "cells/uL",
    "cell/uL", "/uL", "10^3/uL", "10^6/uL", "10^9/L", "10^12/L", "RLU/Cutoff", "ug/L", "nmol/L", "pmol/L",
    "mm/hr", "sec", "°C", "公分", "公斤",
]

_CATALOG_CACHE: Dict[Tuple[str, str, float, float], LabCatalog] = {}


def _fmt_bound(x: float) -> str:
    return f"{float(x):g}"


def bounds_spec(lo: Optional[float], hi: Optional[float], lo_excl: bool = False, hi_excl: bool = False,
                unit: str = "") -> Optional[Dict]:
    """Spec dict (same shape as parse_ref) from numeric bounds; None when both are open.

    Exclusive bounds mean the bound value itself is already abnormal
    (e.g. fasting glucose < 100: 100 counts as high)."""
    if lo is None and hi is None:
        return None
    u = f" {unit}" if unit else ""
    if lo is None:
        text = f"{'<' if hi_excl else '≦'}{_fmt_bound(hi)}{u}"
        return {"kind": "upper", "hi": float(hi), "strict": bool(hi_excl), "unit": unit, "text": text}
    if hi is None:
        text = f"{'>' if lo_excl else '≧'}{_fmt_bound(lo)}{u}"
        return {"kind": "lower", "lo": float(lo), "strict": bool(lo_excl), "unit": unit, "text": text}
    if lo_excl or hi_excl:
        text = f"{'>' if lo_excl else '≧'}{_fmt_bound(lo)} 且 {'<' if hi_excl else '≦'}{_fmt_bound(hi)}{u}"
    else:
        text = f"{_fmt_bound(lo)}～{_fmt_bound(hi)}{u}"
    return {"kind": "range", "lo": float(lo), "hi": float(hi), "lo_strict": bool(lo_excl),
            "hi_strict": bool(hi_excl), "unit": unit, "text": text}


def reference_entry_specs(entry: Dict) -> Tuple[Optional[Dict], Dict[str, Dict]]:
    """reference_ranges.json item -> (general spec, {sex: spec}).

    A sex with both bounds open (e.g. PSA for women) gets no sex spec."""
    unit = entry.get("unit") or ""
    lx, hx = bool(entry.get("low_exclusive")), bool(entry.get("high_exclusive"))
    general = bounds_spec(entry.get("low"), entry.get("high"), lx, hx, unit)
    by_sex = {}
    for sx, pair in (entry.get("sex_specific") or {}).items():
        if sx not in ("M", "F") or not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        sp = bounds_spec(pair[0], pair[1], lx, hx, unit)
        if sp:
            by_sex[sx] = sp
    return general, by_sex


def _read_json(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def is_reference_file(data) -> bool:
    """True for the reference_ranges.json layout ({"items": [{"key": ...}]})."""
    return isinstance(data, dict) and isinstance(data.get("items"), list)


def load_catalog(ranges_path: str = DEFAULT_RANGES_PATH, synonyms_path: str = DEFAULT_SYNONYMS_PATH) -> LabCatalog:
    """Build the catalog from the synonym table + one reference-range source.

    ranges_path: data/reference_ranges.json (public, default) or, for internal
    use, a legacy KB list file; the format is detected from the content."""
    ranges_path, synonyms_path = _resolve(str(ranges_path)), _resolve(str(synonyms_path))
    try:
        mt = (Path(ranges_path).stat().st_mtime if Path(ranges_path).exists() else 0.0,
              Path(synonyms_path).stat().st_mtime if Path(synonyms_path).exists() else 0.0)
    except OSError:
        mt = (0.0, 0.0)
    ck = (str(ranges_path), str(synonyms_path), mt[0], mt[1])
    if ck in _CATALOG_CACHE:
        return _CATALOG_CACHE[ck]

    data = _read_json(ranges_path) if Path(ranges_path).exists() else None
    syn = (_read_json(synonyms_path) if Path(synonyms_path).exists() else None) or {"items": []}
    if is_reference_file(data):
        cat = _catalog_from_reference(data, syn)
    else:
        cat = _catalog_from_legacy_kb(data if isinstance(data, list) else [], syn)
    _CATALOG_CACHE[ck] = cat
    return cat


def _item_from_synonyms(d: Dict) -> LabItem:
    return LabItem(
        key=d["key"], zh=d.get("zh", ""), en=d.get("en", ""), synonyms=list(d.get("synonyms", [])),
        unit=d.get("unit", "") or "", kb_item=d.get("kb_item"),
        conversions=dict(d.get("conversions", {})), unit_equiv=list(d.get("unit_equiv", [])),
        qual_expected=d.get("qual_expected"), specific_over=list(d.get("specific_over", [])),
        protective=bool(d.get("protective", False)),
    )


_QUAL_NORMAL_NEG = {"陰性", "negative", "neg", "-", "(-)", "nonreactive", "non-reactive", "未檢出"}


def _catalog_from_reference(ref: Dict, syn: Dict) -> LabCatalog:
    """Public mode: every range comes from reference_ranges.json (keyed by canonical key)."""
    by_key = {e["key"]: e for e in ref.get("items", []) if e.get("key")}
    items: List[LabItem] = []
    for d in syn.get("items", []):
        it = _item_from_synonyms(d)
        it.ref_origin = RANGE_SOURCE_REFERENCE
        e = by_key.get(it.key)
        if e:
            general, by_sex = reference_entry_specs(e)
            it.kb_spec = general
            it.kb_ref_text = general["text"] if general else ""
            for sx, sp in by_sex.items():
                it.sex_spec[sx] = sp
                it.sex_ref_text[sx] = sp["text"]
            qn = nfkc(e.get("qualitative_normal") or "").lower().replace(" ", "")
            if qn in _QUAL_NORMAL_NEG:
                it.qual_expected = "negative"
            it.ref_source, it.ref_note = e.get("source", "") or "", e.get("note", "") or ""
        items.append(it)
    return LabCatalog(items)


def _catalog_from_legacy_kb(data: List[Dict], syn: Dict) -> LabCatalog:
    """Internal mode: KB entries with 【檢查項目】/【正常參考範圍】 lines, linked via kb_item."""
    kb_entries: Dict[str, Tuple[int, str]] = {}
    for idx, entry in enumerate(data):
        content = entry.get("Content", "") or ""
        nm = _KB_NAME_RE.search(content)
        rf = _KB_REF_RE.search(content)
        if nm:
            kb_entries[nm.group(1).strip()] = (idx, rf.group(1).strip() if rf else "")

    items: List[LabItem] = []
    used_kb = set()
    for d in syn.get("items", []):
        it = _item_from_synonyms(d)
        ref_text = d.get("ref") or ""
        if it.kb_item and it.kb_item in kb_entries:
            it.kb_index, kb_ref = kb_entries[it.kb_item]
            used_kb.add(it.kb_item)
            ref_text = ref_text or kb_ref
        it.kb_ref_text = ref_text
        it.kb_spec = parse_ref(ref_text) if ref_text else None
        for sx, txt in (d.get("sex_ref") or {}).items():
            it.sex_ref_text[sx] = txt
            it.sex_spec[sx] = parse_ref(txt)
        items.append(it)

    # KB items the curated table does not cover still become (exact-name) items.
    for kb_name, (idx, kb_ref) in kb_entries.items():
        if kb_name in used_kb:
            continue
        zh = kb_name.split("（")[0].strip()
        en = kb_name[len(zh):].strip().lstrip("（").rstrip("）").strip()
        syns = [s for s in (zh, en) if s and (len(s) >= 2 or _CJK_RE.search(s))]
        sp = parse_ref(kb_ref) if kb_ref else None
        items.append(LabItem(key=f"kb_{idx}", zh=zh, en=en, synonyms=syns, kb_item=kb_name, kb_index=idx,
                             kb_ref_text=kb_ref, kb_spec=sp, unit=(sp or {}).get("unit", "")))
    return LabCatalog(items)


def load_checkitem_lookup(path: str = DEFAULT_RANGES_PATH) -> LabCatalog:
    """Backward-compatible entry point (pipeline / UI / eval). `path` is a
    reference-range source: reference_ranges.json (default) or a legacy KB file."""
    return load_catalog(path)


def _as_catalog(kb) -> LabCatalog:
    if isinstance(kb, LabCatalog):
        return kb
    return load_catalog()


# ─── Name candidates ─────────────────────────────────────────────────────────

_PREFIXES = ("(三頻)", "(單頻)", "(雙頻)", "(定量)", "血清", "血中", "血液")
_SUFFIXES = ("檢查", "篩檢", "檢驗", "測定", "results", "result", "結果", "(定量)", "(定性)", "指數檢查")
_BRACKETS = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
_CJK_RUN = re.compile(f"[{_CJK}\\[\\]()/]*[{_CJK}][{_CJK}\\[\\]()/]*")
_LETTER_CJK_RUN = re.compile(f"[A-Za-z]-?[{_CJK}][{_CJK}\\[\\]()/]*")
_MIXED_SPAN = re.compile(f"(?:[A-Za-z]-?)?[{_CJK}][{_CJK}A-Za-z\\-]*[{_CJK}]")
_LATIN_RUN = re.compile(r"[A-Za-z0-9α-ωΑ-Ωγβ#%/+\.\-()\[\]',][A-Za-z0-9α-ωΑ-Ωγβ#%/+\.\-()\[\]', ]*")


def _strip_affix(s: str) -> str:
    t = s.strip()
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if t.startswith(p) and len(t) > len(p):
                t, changed = t[len(p):].strip(), True
        for suf in _SUFFIXES:
            if t.lower().endswith(suf) and len(t) > len(suf):
                t, changed = t[: len(t) - len(suf)].strip(), True
    return t


def _name_candidates(name: str) -> List[List[Tuple[str, bool]]]:
    """Tiers of candidate strings (tier order = priority); (string, is_cjk)."""
    s = nfkc(name)
    s_nb = _BRACKETS.sub(" ", s).strip()
    t0 = [(x, bool(_CJK_RE.search(x))) for x in (s, s_nb) if x]          # exact whole name first
    t1 = [(x, bool(_CJK_RE.search(x))) for x in (_strip_affix(s), _strip_affix(s_nb)) if x]
    t2, t3 = [], []
    for src in (s, s_nb):
        for rx in (_CJK_RUN, _LETTER_CJK_RUN, _MIXED_SPAN):
            for r in rx.findall(src):
                for x in (r, _strip_affix(r)):
                    if x:
                        t2.append((x, True))
        for r in _LATIN_RUN.findall(src):
            r = r.strip(" ,")
            if not re.search(r"[A-Za-zα-ωΑ-Ωγβ]", r):
                continue
            t2.append((r, False))
            t2.append((_strip_affix(r), False))
            words = r.split()
            for w in words:
                t3.append((w, False))
                wb = _BRACKETS.sub("", w)
                if wb and wb != w:
                    t3.append((wb, False))
            if len(words) >= 2:
                t3.append((" ".join(words[:2]), False))
    for b in re.findall(r"\(([^()]+)\)|\[([^\[\]]+)\]", s):
        for x in b:
            if x:
                t3.append((x, bool(_CJK_RE.search(x))))
    return [t0, t1, t2, t3]


# ─── Extraction ──────────────────────────────────────────────────────────────

_SEX_RE = re.compile(r"(?:性\s*別|sex|gender)\s*[:：]?\s*(男|女|male|female|M|F)(?![A-Za-z])", re.I)


def detect_sex(text: str) -> Optional[str]:
    """'M' / 'F' from a 性別/Sex header, else None."""
    m = _SEX_RE.search(nfkc(text or ""))
    if not m:
        return None
    v = m.group(1).lower()
    return "F" if v in ("女", "female", "f") else "M"


_VALUE_START = re.compile(
    r"(?:<=|>=|≦|≧|≤|≥|<|>)?\s*[-+]?\d+(?:\.\d+)?\s*\(\s*[1-4]\+\s*\)"   # 80(2+), >=1000(3+)
    r"|[1-4]\+(?![\d+])"                                                  # 2+
    r"|(?:<=|>=|≦|≧|≤|≥|<|>)?\s*[-+]?\d+(?:\.\d+)?"
    r"|陰性|陽性|微量|negative|positive|trace|non-?reactive|reactive|neg\b|pos\b"
    r"|\+/-|±|\([1-4]\+\)|[1-4]\+|\+{1,4}|\(-\)|-(?=\s|$|\()", re.I)
_FLAG_AFTER = re.compile(r"\s*([↑↓]+|\*+)|\s+(HH|LL|H|L)(?![A-Za-z0-9])")
_UNIT_AFTER = re.compile(
    r"\s*((?:x?10\^?\d+|\^\d+)\s*/\s*[uμm]?[lL]|[A-Za-zμ%°/][A-Za-zμ%°/\.\^\d\-]*|次/分|公分|公斤)")
_BARE_REF = re.compile(
    rf"\s*((?:<=|>=|≦|≧|≤|≥|<|>)\s*{NUM}|{NUM}\s*[~\-–—]\s*{NUM}|-~-\s*\(\+/-\)?)")
_BP_RE = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})(?!\d)(?:\s*(mmHg|mm-Hg|mm~Hg))?", re.I)
_BP_NAME = re.compile(r"(血壓|b\.?\s*p\.?|blood\s*pressure|收縮壓\s*/\s*舒張壓|sbp\s*/\s*dbp)\s*[:：]?\s*(?:\([^)]*\))?\s*[:：]?\s*$", re.I)
_LEAD_JUNK = re.compile(r"^[\s*•·※\-–>)\]]*(?:\(?\d{1,2}[\.、)]\s*(?=\D))?")
_TRAIL_JUNK = re.compile(r"[\s:=]*(參考值|參考範圍)?[\s:=]*$")


def clean_name(raw: str) -> str:
    s = nfkc(raw)
    s = _LEAD_JUNK.sub("", s)
    s = _TRAIL_JUNK.sub("", s)
    return s.strip(" :=")


def _name_ok(name: str) -> bool:
    return bool(name) and len(name) <= 80 and bool(re.search(f"[A-Za-z{_CJK}]", name))


_PERSON_LIKE = re.compile(f"[{_CJK}]{{2,4}}")


def _trim_name_prefix(name: str, cat: LabCatalog, has_pct: bool = False) -> str:
    """Drop a leading header field from a text-line item name. The name is everything since
    the previous value, so it can carry e.g. the patient's name: "姓名:王小明 身高" -> "身高",
    "王小明 BMI" -> "BMI". A leading token is dropped only if it names no catalog item and
    looks like a header field — a "label:value" pair or a bare 2–4 character CJK word (a
    personal name) — and the rest still resolves to the SAME item. Name parts such as
    "Urine Glucose 尿糖", "ALT(GPT) 丙胺酸轉胺酶" or "HDL Cholesterol" are left whole."""
    toks = name.split()
    item = cat.resolve(name, has_pct=has_pct) if len(toks) > 1 else None
    if item is None:
        return name
    best = name
    for i in range(1, len(toks)):  # drop toks[:i]
        t = toks[i - 1]
        if not (":" in t or _PERSON_LIKE.fullmatch(t)) or cat.resolve(t, has_pct=has_pct) is not None:
            break
        it = cat.resolve(" ".join(toks[i:]), has_pct=has_pct)
        if it is not None and it.key == item.key:
            best = " ".join(toks[i:])
    return best


def _match_paren(s: str, i: int) -> int:
    """s[i] == '(' -> index after the matching ')' (or len(s) if unclosed)."""
    d = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            d += 1
        elif s[j] == ")":
            d -= 1
            if d == 0:
                return j + 1
    return len(s)


def _scan_line(line: str, cat: LabCatalog, push) -> None:
    s = nfkc(line)
    if not s or len(s) > 400:
        return
    # Blood pressure "130/85" -> two items
    for m in list(_BP_RE.finditer(s)):
        pre = s[:m.start()]
        nm = _BP_NAME.search(pre)
        if nm:
            unit = m.group(3) or "mmHg"
            raw = clean_name(pre[nm.start():]) or nm.group(1)
            push({"name": "收縮壓", "raw_name": raw, "value_text": m.group(1), "unit": unit, "ref_text": "",
                  "flag": "", "hint_key": "sbp"})
            push({"name": "舒張壓", "raw_name": raw, "value_text": m.group(2), "unit": unit, "ref_text": "",
                  "flag": "", "hint_key": "dbp"})
            s = s[:nm.start()] + " " * (m.end() - nm.start()) + s[m.end():]
    pos, prev_end = 0, 0
    n = len(s)
    while pos < n:
        m = _VALUE_START.search(s, pos)
        if not m:
            break
        st, en = m.start(), m.end()
        before = s[st - 1] if st > 0 else " "
        if before not in " :=":
            pos = st + 1
            continue
        after = s[en] if en < n else " "
        tok = m.group()
        if after in "~/-" and en + 1 < n and s[en + 1].isdigit():  # a range "3-5", a date, a ratio
            pos = en + 1
            continue
        if after == ":" and en + 1 < n and s[en + 1].isdigit():  # time
            pos = en + 1
            continue
        j = en
        flag = ""
        fm = _FLAG_AFTER.match(s, j)
        if fm:
            flag = fm.group(1) or fm.group(2)
            j = fm.end()
        unit = ""
        um = _UNIT_AFTER.match(s, j)
        if um and cat.is_unit(um.group(1)):
            unit = um.group(1)
            j = um.end()
            if not flag:
                fm = _FLAG_AFTER.match(s, j)
                if fm:
                    flag = fm.group(1) or fm.group(2)
                    j = fm.end()
        # the token must end at a boundary
        nxt = s[j] if j < n else " "
        if j == en and not (nxt in " (（" or j >= n):
            pos = st + 1
            continue
        name = clean_name(s[prev_end:st])
        # reference: optional "參考值", then "(...)" or a bare range
        k = j
        rm = re.match(r"\s*(參考值|參考範圍|正常值)?\s*[:：]?\s*", s[k:])
        k2 = k + (rm.end() if rm else 0)
        ref = ""
        if k2 < n and s[k2] == "(":
            e = _match_paren(s, k2)
            ref = s[k2:e]
            k = e
        else:
            bm = _BARE_REF.match(s, k2)  # "... 參考值 36～48" / "... mg/dL 7~25"
            if bm:
                ref = bm.group(1)
                k = bm.end()
                um2 = _UNIT_AFTER.match(s, k)
                if um2 and cat.is_unit(um2.group(1)):
                    ref += " " + um2.group(1)
                    k = um2.end()
            elif rm and rm.group(1):
                k = k2
        if _name_ok(name):
            name = _trim_name_prefix(name, cat, has_pct="%" in name + unit + ref)
            push({"name": name, "raw_name": name, "value_text": tok.strip() + (f" {flag}" if flag else ""),
                  "unit": unit, "ref_text": ref, "flag": ""})
        prev_end = pos = max(k, j, en)


# ── tables ──

_HEADER_WORDS = {
    "name": ["檢查項目", "檢驗項目", "項目名稱", "項目", "item", "items", "test", "testname", "名稱", "檢查名稱"],
    "value": ["檢查結果", "檢驗結果", "檢驗值", "結果", "result", "results", "value", "數值", "測定值", "本次結果",
              "檢查值"],
    "unit": ["單位", "unit", "units"],
    "ref": ["參考值", "參考範圍", "參考區間", "正常值", "正常範圍", "reference", "ref", "referencerange",
            "參考標準", "標準值", "生物參考區間"],
    "flag": ["判定", "備註", "flag", "異常", "標記", "判讀", "h/l", "異常標記", "結果判定"],
    "index": ["序號", "編號", "no", "#", "項次"],
    "category": ["類別", "分類", "category", "組別"],
}
_HEADER_INDEX = {nkey(w): role for role, ws in _HEADER_WORDS.items() for w in ws}


def _header_role(cell: str) -> Optional[str]:
    k = nkey(cell)
    if not k:
        return None
    if k in _HEADER_INDEX:
        return _HEADER_INDEX[k]
    if "結果" in k and len(k) <= 12:
        return "value"
    if ("參考" in k or "正常值" in k) and len(k) <= 8:
        return "ref"
    return None


def _join_cell(c, for_name: bool) -> str:
    if c is None:
        return ""
    parts = [p.strip() for p in str(c).split("\n")]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if not for_name:
        return nfkc("".join(parts) if any(re.fullmatch(r"[+)(\d]+\)?\s*[HL]?", p) for p in parts[1:])
                    else " ".join(parts))
    out = parts[0]
    for p in parts[1:]:
        glue = "" if (len(p) <= 2 and out[-1:].isalpha() and p[:1].isalpha()) else " "
        out += glue + p
    return nfkc(out)


def _parse_table(table: List[List[str]], cat: LabCatalog, push) -> None:
    roles: Optional[List[Optional[str]]] = None
    for row in table or []:
        if not row:
            continue
        raw_cells = list(row)
        rl = [_header_role(_join_cell(c, True)) for c in raw_cells]
        if sum(1 for r in rl if r) >= 2 and "name" in rl and \
                not any(parse_value(_join_cell(c, False))["kind"] == "num" for c in raw_cells):
            roles = rl
            continue
        if roles and len(roles) == len(raw_cells):
            groups = []
            starts = [i for i, r in enumerate(roles) if r == "name"]
            for gi, st in enumerate(starts):
                en = starts[gi + 1] if gi + 1 < len(starts) else len(roles)
                groups.append(range(st, en))
            for g in groups:
                rec = {"name": "", "value_text": "", "unit": "", "ref_text": "", "flag": ""}
                for i in g:
                    r = roles[i]
                    if r == "name":
                        rec["name"] = _join_cell(raw_cells[i], True)
                    elif r == "value" and not rec["value_text"]:
                        rec["value_text"] = _join_cell(raw_cells[i], False)
                    elif r == "unit":
                        rec["unit"] = _join_cell(raw_cells[i], False)
                    elif r == "ref":
                        rec["ref_text"] = _join_cell(raw_cells[i], False)
                    elif r == "flag":
                        rec["flag"] = _join_cell(raw_cells[i], False)
                if rec["name"] and rec["value_text"] and parse_value(rec["value_text"])["kind"] != "text":
                    _push_table_rec(rec, push)
                elif rec["name"]:
                    _heuristic_cells([raw_cells[i] for i in g], cat, push)
        else:
            _heuristic_cells(raw_cells, cat, push)


def _repair_split_range(rec: Dict) -> None:
    """pdfplumber sometimes splits a right-aligned range '7-25mg/dL' into the
    unit column ('-25mg/dL') and the reference column ('7'). Re-join it."""
    ref, unit = nfkc(rec.get("ref_text", "")), nfkc(rec.get("unit", ""))
    m = re.fullmatch(rf"[~\-–]\s*({NUM.replace('[-+]?', '')})\s*(.*)", unit)
    if m and re.fullmatch(NUM, ref.strip()):
        rec["ref_text"] = f"{ref.strip()}-{m.group(1)} {m.group(2)}".strip()
        rec["unit"] = m.group(2).strip()
        return
    m = re.fullmatch(rf"({NUM})\s*[~\-–]", ref)  # '7-' in ref, '25 mg/dL' in unit
    m2 = re.fullmatch(rf"({NUM.replace('[-+]?', '')})\s*(.*)", unit)
    if m and m2:
        rec["ref_text"] = f"{m.group(1)}-{m2.group(1)} {m2.group(2)}".strip()
        rec["unit"] = m2.group(2).strip()


def _push_table_rec(rec: Dict, push) -> None:
    name = clean_name(rec["name"])
    if not _name_ok(name):
        return
    _repair_split_range(rec)
    flag = rec.get("flag", "")
    fl = nfkc(flag).upper()
    flag = "H" if fl in ("H", "HH", "↑", "HIGH", "高") else "L" if fl in ("L", "LL", "↓", "LOW", "低") else ""
    push({"name": name, "raw_name": name, "value_text": rec["value_text"], "unit": rec.get("unit", ""),
          "ref_text": rec.get("ref_text", ""), "flag": flag})


def _looks_like_name(c: str, cat: LabCatalog) -> bool:
    return _name_ok(c) and parse_value(c)["kind"] == "text" and not cat.is_unit(c) and \
        _header_role(c) is None and parse_ref(c)["kind"] not in ("range", "upper", "lower", "qual_neg", "sex")


def _heuristic_cells(cells, cat: LabCatalog, push) -> None:
    cur = None
    out = []
    for raw in cells:
        c_name = _join_cell(raw, True)
        c = _join_cell(raw, False)
        if not c:
            continue
        pv = parse_value(c)
        if cur is None:
            if _looks_like_name(c_name, cat):
                cur = {"name": c_name, "value_text": "", "unit": "", "ref_text": "", "flag": ""}
            continue
        if not cur["value_text"]:
            if pv["kind"] != "text":
                cur["value_text"] = c
            elif _looks_like_name(c_name, cat):
                cur["name"] = c_name  # e.g. category cell before the real name
            continue
        if cat.is_unit(c) and not cur["unit"] and parse_ref(c)["kind"] in ("none", "unit_only") \
                and not c.startswith("("):
            cur["unit"] = c
        elif parse_ref(c)["kind"] != "none" and not cur["ref_text"]:
            cur["ref_text"] = c
        elif nfkc(c).upper() in ("H", "L", "HH", "LL", "↑", "↓"):
            cur["flag"] = c
        elif _looks_like_name(c_name, cat):
            out.append(cur)
            cur = {"name": c_name, "value_text": "", "unit": "", "ref_text": "", "flag": ""}
    if cur:
        out.append(cur)
    for rec in out:
        if rec["value_text"]:
            _push_table_rec(rec, push)


def extract_lab_values(text: str, tables: List[List[List[str]]] = None, catalog: Optional[LabCatalog] = None) -> List[Dict]:
    """Pull raw lab records from report text lines and tables.

    Each record: {name, raw_name, value (float | str), value_text, value_kind
    ('num'|'qual'), censored ('<','<=','>','>=',''), unit, ref_text, flag,
    source ('table'|'text'), report_sex}.  Records are NOT yet matched to the
    catalog; evaluate_findings() does that (and de-duplicates).
    """
    cat = _as_catalog(catalog)
    sex = detect_sex(text or "")
    found: List[Dict] = []
    seen = set()

    def make_push(src):
        def push(rec):
            pv = parse_value(rec["value_text"])
            if pv["kind"] == "text":
                return
            flag = rec.get("flag") or pv["flag"]
            unit = rec.get("unit") or pv.get("unit") or ""
            if pv["kind"] == "num":
                value, kind = pv["v"], "num"
            else:
                value, kind = pv["text"], "qual"
            key = (nkey(rec["name"]), str(value), src)
            if key in seen:
                return
            seen.add(key)
            out = {"name": rec["name"], "raw_name": rec.get("raw_name", rec["name"]), "value": value,
                   "value_text": pv["text"], "value_kind": kind, "censored": pv["op"], "unit": nfkc(unit),
                   "ref_text": nfkc(rec.get("ref_text", "")), "flag": flag, "source": src, "report_sex": sex}
            if rec.get("hint_key"):
                out["hint_key"] = rec["hint_key"]
            found.append(out)
        return push

    for table in tables or []:
        _parse_table(table, cat, make_push("table"))
    push_text = make_push("text")
    for line in (text or "").splitlines():
        _scan_line(line, cat, push_text)
    return found


# ─── Classification ──────────────────────────────────────────────────────────

def _cmp_spec(pv: Dict, sp: Optional[Dict], factor: float = 1.0) -> Optional[str]:
    """Status of a parsed value against ONE spec; None when the spec cannot judge it."""
    if not sp or sp["kind"] in ("none",):
        return None
    kind = pv["kind"]
    if kind == "num" or (kind == "pos" and pv.get("v") is not None and sp["kind"] in ("range", "upper", "lower")):
        x = pv["v"] * factor
        op = pv["op"] if kind == "num" else ""
        k = sp["kind"]
        if k == "qual_neg":
            if op in ("<", "<="):  # below the detection limit ("<0.1") is not a positive result
                return "normal"
            return "high" if x > EPS or op in (">", ">=") else "normal"
        if k == "unit_only":
            return None
        if op in ("<", "<="):
            if k == "range":
                if x <= sp["lo"] + EPS and sp["lo"] > 0:
                    return "low"
                return "normal" if x <= sp["hi"] + EPS else None
            if k == "upper":
                return "normal" if x <= sp["hi"] + EPS else None
            if k == "lower":
                return "low" if x <= sp["lo"] + EPS else None
        if op in (">", ">="):
            if k == "range":
                return "high" if x >= sp["hi"] - EPS else None
            if k == "upper":
                return "high" if x >= sp["hi"] - EPS else None
            if k == "lower":
                return "normal" if x >= sp["lo"] - EPS else None
        if k == "range":
            lo, hi = sp["lo"], sp["hi"]
            if x < lo - EPS or (sp.get("lo_strict") and x <= lo + EPS):
                return "low"
            if x > hi + EPS or (sp.get("hi_strict") and x >= hi - EPS):
                return "high"
            return "normal"
        if k == "upper":
            hi = sp["hi"]
            return "high" if (x >= hi - EPS if sp["strict"] else x > hi + EPS) else "normal"
        if k == "lower":
            lo = sp["lo"]
            return "low" if (x <= lo + EPS if sp["strict"] else x < lo - EPS) else "normal"
        return None
    if kind == "pos":
        return "positive" if sp["kind"] in ("qual_neg", "unit_only") or sp.get("qual_neg") else None
    if kind in ("neg", "trace"):
        if sp["kind"] in ("qual_neg", "unit_only"):
            return "negative"
        if sp["kind"] == "upper" or (sp["kind"] == "range" and sp["lo"] <= 0):
            return "negative"
        return None
    return None


def _combine_sex(pv, spM, spF, factor=1.0) -> Tuple[Optional[str], bool]:
    """Unknown sex: abnormal only if outside BOTH sex ranges."""
    a, b = _cmp_spec(pv, spM, factor), _cmp_spec(pv, spF, factor)
    if a == b:
        return a, True
    if a is None or b is None:
        return None, True
    if "normal" in (a, b):
        return "normal", True
    return None, True


def _eval_with_spec(pv, sp, sex, factor=1.0) -> Tuple[Optional[str], Optional[Dict], bool]:
    """-> (status, spec actually used, sex_unknown flag)."""
    if not sp:
        return None, None, False
    if sp["kind"] == "sex":
        if sex in ("M", "F"):
            return _cmp_spec(pv, sp[sex], factor), sp[sex], False
        st, _ = _combine_sex(pv, sp["M"], sp["F"], factor)
        return st, None, True
    return _cmp_spec(pv, sp, factor), sp, False


_DIR = {"high": "high", "low": "low", "positive": "high", "negative": "normal", "normal": "normal"}


def _direction(status: Optional[str]) -> str:
    return _DIR.get(status or "", "unknown")


def _kb_spec_for(item: Optional[LabItem], sex: Optional[str]) -> Tuple[Optional[Dict], str, bool]:
    """Reference spec honouring sex-specific ranges -> (spec, text, sex_unknown)."""
    if not item:
        return None, "", False
    if item.sex_spec:
        if sex in ("M", "F") and sex in item.sex_spec:
            return item.sex_spec[sex], item.sex_ref_text[sex], False
        if "M" in item.sex_spec and "F" in item.sex_spec:
            sp = {"kind": "sex", "M": item.sex_spec["M"], "F": item.sex_spec["F"], "unit": item.unit,
                  "text": f"男 {item.sex_ref_text['M']} / 女 {item.sex_ref_text['F']}"}
            return sp, sp["text"], True
    if item.kb_spec and item.kb_spec["kind"] != "none":
        return item.kb_spec, item.kb_ref_text, False
    if item.qual_expected == "negative":
        return {"kind": "qual_neg", "unit": "", "text": "陰性"}, "陰性", False
    return None, "", False


def _classify(rec: Dict, pv: Dict, item: Optional[LabItem], sex: Optional[str], use_kb: bool = True) -> Dict:
    """Core status computation for one record -> dict of finding fields.

    use_kb=False (LLM-mapped names): judge only against the printed range /
    printed flag, never against the reference range of a possibly wrong item."""
    rep_sp = parse_ref(rec.get("ref_text", "")) if rec.get("ref_text") else None
    if rep_sp and rep_sp["kind"] == "none":
        rep_sp = None
    # A unit-only printed reference only means "expected negative" for qualitative results.
    rep_usable = bool(rep_sp) and (rep_sp["kind"] != "unit_only" or pv["kind"] in ("pos", "neg", "trace"))
    # A bare "陰性"/"(-)" printed next to a NUMERIC index ("HBsAg 0.36 COI (陰性)") is usually the
    # lab's reading of that result, not a cut-off: judge the number against the item's numeric
    # reference range instead (items without one, e.g. dipsticks, keep "> 0 = positive").
    if rep_usable and rep_sp["kind"] == "qual_neg" and pv["kind"] == "num" and item is not None \
            and (item.kb_spec or {}).get("kind") in ("range", "upper", "lower"):
        rep_usable = False

    rep_status, rep_used, rep_sex_unknown = (None, None, False)
    if rep_usable:
        rep_status, rep_used, rep_sex_unknown = _eval_with_spec(pv, rep_sp, sex)

    # reference-range side (with unit conversion)
    kb_status, kb_used, kb_sex_unknown, unit_note = None, None, False, ""
    kb_sp, kb_text, _ = _kb_spec_for(item, sex) if use_kb else (None, "", False)
    qual_value = pv["kind"] in ("pos", "neg", "trace")
    if use_kb and item and qual_value and item.qual_expected == "negative":
        kb_status = _cmp_spec(pv, {"kind": "qual_neg"})
        kb_text = kb_text if (kb_sp and kb_sp["kind"] == "qual_neg") else "陰性"
    elif kb_sp and not (kb_sp["kind"] == "qual_neg" and not qual_value):
        # (a numeric result is never judged against a merely *implied* "expected negative")
        val_unit = rec.get("unit") or (rep_sp or {}).get("unit", "") or ""
        factor = item.unit_factor(val_unit) if (item and pv["kind"] in ("num", "pos")) else 1.0
        if factor is None:
            unit_note = f"單位 {val_unit} 與參考範圍單位 {item.unit} 不相容，未套用參考範圍"
        else:
            kb_status, kb_used, kb_sex_unknown = _eval_with_spec(pv, kb_sp, sex, factor)

    if rep_status is not None:
        status, source, used, sex_unknown = rep_status, "report", rep_used, rep_sex_unknown
    elif kb_status is not None:
        status, source, used, sex_unknown = kb_status, RANGE_SOURCE_REFERENCE, kb_used, kb_sex_unknown
    else:
        status, source, used, sex_unknown = "unknown", None, None, (rep_sex_unknown or kb_sex_unknown)
        fl = (rec.get("flag") or "").upper()
        if fl in ("H", "L") and pv["kind"] == "num":
            status, source = ("high" if fl == "H" else "low"), "report_flag"

    if item and item.protective and status in ("high", "positive"):
        status = "normal"
    if item and item.protective and kb_status in ("high", "positive"):
        kb_status = "normal"

    conflict, note = False, ""
    if rep_status is not None and kb_status is not None and _direction(rep_status) != _direction(kb_status):
        conflict = True
        note = f"報告參考值判定 {rep_status}，一般參考範圍（{kb_text}）判定 {kb_status}"
    if unit_note:
        note = (note + "；" if note else "") + unit_note

    bounds = _spec_bounds(used)
    ref_unit = ""
    if source == "report":
        ref_unit = (rep_sp or {}).get("unit", "") or rec.get("unit", "")
    elif source == RANGE_SOURCE_REFERENCE and item:
        ref_unit = item.unit
    return {
        "status": status, "direction": _direction(status), "range_source": source,
        "ref_low": bounds[0] if bounds else None, "ref_high": bounds[1] if bounds else None,
        "ref_unit": ref_unit,
        "report_range": _strip_parens(nfkc(rec.get("ref_text", ""))) or None,
        "kb_range": kb_text or None,
        "report_status": rep_status, "kb_status": kb_status,
        "range_conflict": conflict, "range_conflict_note": note,
        "sex_unknown": bool(sex_unknown),
    }


def classify_value(value: str, reference: str, sex: Optional[str] = None, name: str = "") -> str:
    """Direction (high/low/normal/unknown) of `value` against a printed `reference`.

    Used by eval/test_abnormal.py; `name` (optional) lets the catalog apply
    item knowledge (protective antibody, expected-negative qualitative item).
    """
    cat = load_catalog()
    item = cat.resolve(name) if name else None
    pv = parse_value(value)
    if pv["kind"] == "text":
        return "unknown"
    res = _classify({"ref_text": reference, "unit": ""}, pv, item, sex)
    return res["direction"]


# ─── LLM fallback for names the synonym table cannot resolve ─────────────────

_LLM_CACHE: Dict[str, Optional[str]] = {}
_ADMIN = re.compile(r"日期|年齡|電話|手機|姓名|地址|編號|身分|身份|生日|出生|病歷|頁|報告|受檢|性別|時間|醫師|"
                    r"病史|症狀|問診|習慣|理學|外觀|顏色|辨色|理想|"
                    r"\bdate\b|\bage\b|phone|\btel\b|\bname\b|\bid\b|address|ideal", re.I)


def make_llm_normalizer(base_url: Optional[str] = None, model: Optional[str] = None, timeout: int = 60,
                        num_ctx: Optional[int] = None) -> Callable[[List[str], LabCatalog], Dict[str, Optional[str]]]:
    """Return fn(names, catalog) -> {name: canonical_key|None}.

    ONE batched /api/chat call (JSON-schema `format` whose enum is the catalog
    keys + "UNKNOWN"); results are cached in-process. Only called for names
    the synonym table could not resolve.
    """
    import requests  # local import: abnormal.py stays importable without it

    try:
        import config as _cfg
        base_url = base_url or _cfg.OLLAMA_BASE_URL
        model = model or _cfg.LLM_MODEL
        num_ctx = num_ctx or _cfg.LLM_NUM_CTX
    except Exception:  # noqa: BLE001
        base_url = base_url or "http://127.0.0.1:11434"
        model = model or "qwen2.5:7b"
        num_ctx = num_ctx or 8192

    def fn(names: List[str], cat: LabCatalog) -> Dict[str, Optional[str]]:
        todo = [n for n in dict.fromkeys(names) if n not in _LLM_CACHE][:40]
        if todo:
            keys = [k for k in cat.items if not k.startswith("kb_")] + \
                   [k for k in cat.items if k.startswith("kb_")]
            enum = keys + ["UNKNOWN"]
            listing = "\n".join(f"{k}: {cat.items[k].zh} / {cat.items[k].en}" for k in keys)
            props = {f"n{i}": {"type": "string", "enum": enum} for i in range(len(todo))}
            schema = {"type": "object", "properties": props, "required": list(props)}
            user = ("Canonical health-checkup test items (key: 中文 / English):\n" + listing +
                    "\n\nMap each raw test name below to the key of the SAME test. Answer UNKNOWN unless you "
                    "are certain (administrative fields, dates, ages, IDs, questionnaire items and body-part "
                    "descriptions are UNKNOWN). Left/right and percentage/count variants must match exactly.\n" +
                    "\n".join(f"n{i}: {n}" for i, n in enumerate(todo)))
            try:
                resp = requests.post(f"{base_url}/api/chat", json={
                    "model": model, "stream": False, "format": schema,
                    "messages": [{"role": "system", "content": "You normalise lab test names. Output JSON only."},
                                 {"role": "user", "content": user}],
                    "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": 40 * len(todo) + 50},
                }, timeout=timeout)
                resp.raise_for_status()
                out = json.loads(resp.json()["message"]["content"])
                for i, n in enumerate(todo):
                    k = out.get(f"n{i}")
                    _LLM_CACHE[n] = k if (k in cat.items) else None
            except Exception:  # noqa: BLE001 — fallback must never break the pipeline
                for n in todo:
                    _LLM_CACHE.setdefault(n, None)
        return {n: _LLM_CACHE.get(n) for n in names}

    return fn


def _llm_candidate(rec: Dict) -> bool:
    """Only plausible lab rows: a numeric result, or a qualitative one printed
    with a reference; never administrative / questionnaire fields."""
    n = rec["name"]
    if not (2 <= len(n) <= 40) or _ADMIN.search(n) or not re.search(f"[A-Za-z{_CJK}]", n):
        return False
    return rec.get("value_kind") == "num" or bool(rec.get("ref_text"))


# ─── Evaluation ──────────────────────────────────────────────────────────────

def _value_sig(f: Dict):
    v = f.get("value")
    if isinstance(v, (int, float)):
        return ("n", round(float(v), 6))
    pv = parse_value(f.get("value_text", ""))
    if pv.get("v") is not None:
        return ("n", round(float(pv["v"]), 6))
    return ("q", (pv.get("grade") or pv["kind"]))


_SEX_VOTE_TOL = 0.25  # a printed bound must lie within 25% of the nearer sex bound


def _sex_vote(sp: Dict, spM: Optional[Dict], spF: Optional[Dict]) -> Optional[str]:
    """'M'/'F' when a printed range matches one sex's reference range better.

    Only bounds on which the two sexes differ are compared; each must be
    strictly nearer to the same sex and within tolerance (so a range printed
    in another unit, or a unisex range, never votes)."""
    if not spM or not spF:
        return None
    if _spec_equal(sp, spM) != _spec_equal(sp, spF):
        return "M" if _spec_equal(sp, spM) else "F"
    bp, bm, bf = _spec_bounds(sp), _spec_bounds(spM), _spec_bounds(spF)
    if not (bp and bm and bf):
        return None
    votes = set()
    for p, m, f in zip(bp, bm, bf):
        if p is None or m is None or f is None or abs(m - f) < 1e-9:
            continue
        dm, df = abs(p - m), abs(p - f)
        if min(dm, df) > _SEX_VOTE_TOL * max(abs(m), abs(f)) or abs(dm - df) < 1e-9:
            return None
        votes.add("M" if dm < df else "F")
    return votes.pop() if len(votes) == 1 else None


def infer_sex_from_ranges(records: List[Dict], cat: LabCatalog) -> Optional[str]:
    """Vote M/F from printed ranges that match one sex's reference range."""
    votes = {"M": 0, "F": 0}
    for rec in records:
        item = rec.get("_item")
        if not item or not item.sex_spec or not rec.get("ref_text"):
            continue
        sp = parse_ref(rec["ref_text"])
        if sp["kind"] not in ("range", "upper", "lower"):
            continue
        v = _sex_vote(sp, item.sex_spec.get("M"), item.sex_spec.get("F"))
        if v:
            votes[v] += 1
    if votes["M"] >= 2 and votes["F"] == 0:
        return "M"
    if votes["F"] >= 2 and votes["M"] == 0:
        return "F"
    return None


def evaluate_findings(values: List[Dict], kb=None, sex: Optional[str] = None,
                      llm_normalizer: Optional[Callable] = None, debug: Optional[Dict] = None) -> List[Dict]:
    """Match records to canonical items, classify, de-duplicate.

    sex: "M"/"F"/None. None -> use the report header (report_sex on the
    records), then sex-specific range votes, else unknown.
    llm_normalizer: optional fn(names, catalog) (see make_llm_normalizer).
    debug: optional dict filled with {unmatched, sex, sex_source, llm_mapped}.
    """
    cat = _as_catalog(kb)
    dbg = debug if debug is not None else {}
    dbg.setdefault("unmatched", [])
    dbg.setdefault("llm_mapped", {})

    recs = []
    for v in values or []:
        r = dict(v)
        r.setdefault("value_text", str(r.get("value", "")))
        r.setdefault("ref_text", "")
        r.setdefault("flag", "")
        r.setdefault("source", "text")
        r.setdefault("raw_name", r.get("name", ""))
        has_pct = "%" in (r.get("name", "") + r.get("unit", "") + r.get("ref_text", ""))
        item = cat.get(r["hint_key"]) if r.get("hint_key") else None
        item = item or cat.resolve(r.get("name", ""), has_pct=has_pct)
        r["_item"], r["_method"] = item, ("synonym" if item else None)
        recs.append(r)

    unresolved = [r for r in recs if r["_item"] is None]
    if unresolved and llm_normalizer:
        cands = [r for r in unresolved if _llm_candidate(r)]
        if cands:
            try:
                mapping = llm_normalizer([r["name"] for r in cands], cat) or {}
            except Exception:  # noqa: BLE001
                mapping = {}
            already = {r["_item"].key for r in recs if r["_item"] is not None}
            for r in cands:
                k = mapping.get(r["name"])
                it = cat.get(k) if k else None
                if it and r.get("unit") and it.unit_factor(r["unit"]) is None:
                    it = None  # unit contradicts the LLM's guess
                if it and it.key in already:
                    it = None  # the table already resolved this item under its real name
                if it:
                    r["_item"], r["_method"] = it, "llm"
                    dbg["llm_mapped"][r["name"]] = it.key
    for r in recs:
        if r["_item"] is None:
            dbg["unmatched"].append(r.get("name", ""))

    # sex
    sex_source = "param" if sex in ("M", "F") else None
    if sex not in ("M", "F"):
        sex = next((r.get("report_sex") for r in recs if r.get("report_sex") in ("M", "F")), None)
        sex_source = "report_header" if sex else None
    if sex not in ("M", "F"):
        sex = infer_sex_from_ranges([r for r in recs if r["_item"]], cat)
        sex_source = "range_inference" if sex else None
    dbg["sex"], dbg["sex_source"] = sex, sex_source

    findings = []
    for r in recs:
        item: Optional[LabItem] = r["_item"]
        if item is None:
            continue
        pv = parse_value(r["value_text"])
        if pv["kind"] == "text":
            continue
        via_llm = r["_method"] == "llm"
        res = _classify(r, pv, item, sex, use_kb=not via_llm)
        value = r.get("value")
        f = {
            "name": r.get("name", ""), "value": value, "unit": r.get("unit", ""),
            "direction": res["direction"], "ref_low": res["ref_low"], "ref_high": res["ref_high"],
            "ref_unit": res["ref_unit"], "matched_name": (r.get("name", "") if via_llm else item.kb_zh), "source": r.get("source", "text"),
            "canonical_key": item.key, "display_name": (r.get("name", "") if via_llm else item.display_name), "status": res["status"],
            "range_source": res["range_source"], "report_range": res["report_range"], "kb_range": res["kb_range"],
            "report_status": res["report_status"], "kb_status": res["kb_status"],
            "range_conflict": res["range_conflict"], "range_conflict_note": res["range_conflict_note"],
            "sex_used": sex, "sex_unknown": res["sex_unknown"], "raw_name": r.get("raw_name", ""),
            "value_text": r.get("value_text", ""), "censored": r.get("censored", pv["op"]),
            "report_flag": r.get("flag", "") or pv["flag"], "match_method": r["_method"],
            "duplicate_conflict": False,
        }
        findings.append(f)
    return _dedupe(findings)


def _dedupe(findings: List[Dict]) -> List[Dict]:
    """One finding per canonical item: table rows win over text lines; keep a
    second occurrence only if its value differs (both marked duplicate_conflict)."""
    by_key: Dict[str, List[Dict]] = {}
    order: List[str] = []
    for f in findings:
        k = f["canonical_key"]
        if k not in by_key:
            order.append(k)
        by_key.setdefault(k, []).append(f)
    out = []
    for k in order:
        group = by_key[k]
        group.sort(key=lambda f: 0 if f["source"] == "table" else 1)  # stable
        kept, sigs = [], []
        for f in group:
            sig = _value_sig(f)
            if sig in sigs:
                continue
            sigs.append(sig)
            kept.append(f)
        if len(kept) > 1:
            kept = _merge_numeric_and_qualitative(kept)
        if len(kept) > 1:
            for f in kept:
                f["duplicate_conflict"] = True
        out.extend(kept)
    return out


def _merge_numeric_and_qualitative(kept: List[Dict]) -> List[Dict]:
    """Serology is often printed twice: an index ('HBsAg results 0.38') and its
    interpretation ('HBsAg Negative'). One numeric + qualitative copies that do
    not contradict it -> one finding carrying `qualitative_result`."""
    nums = [f for f in kept if _value_sig(f)[0] == "n"]
    quals = [f for f in kept if _value_sig(f)[0] == "q"]
    if len(nums) != 1 or not quals:
        return kept
    num = nums[0]
    if all(q["direction"] in ("normal", "unknown", num["direction"]) or num["direction"] == "unknown"
           and q["direction"] == "normal" for q in quals) and             not any(q["direction"] in ("high", "low") and q["direction"] != num["direction"] for q in quals):
        num["qualitative_result"] = "；".join(str(q.get("value_text")) for q in quals)
        return [num]
    return kept


# ─── Presentation helpers (unchanged API) ────────────────────────────────────

def _fmt_value(f: Dict) -> str:
    v = f.get("value")
    if isinstance(v, float):
        s = f"{v:g}"
        if f.get("censored"):
            s = f["censored"] + s
        return s
    return str(f.get("value_text") or v)


def format_findings_for_llm(findings: List[Dict], only_abnormal: bool = True) -> str:
    """Render findings as a compact bulleted list for the LLM prompt."""
    rows = []
    for f in findings:
        if only_abnormal and f["direction"] not in ("high", "low"):
            continue
        ref = ""
        if f.get("report_range") and f.get("range_source") == "report":
            ref = f" (ref {f['report_range']})"
        elif f["ref_low"] is not None or f["ref_high"] is not None:
            lo = f["ref_low"] if f["ref_low"] is not None else "−∞"
            hi = f["ref_high"] if f["ref_high"] is not None else "+∞"
            ref = f" (ref {lo}~{hi}{f['ref_unit'] or ''})"
        elif f.get("kb_range"):
            ref = f" (ref {f['kb_range']})"
        status = f.get("status", f["direction"])
        arrow = {"high": "↑ HIGH", "low": "↓ LOW", "positive": "陽性 POSITIVE", "negative": "陰性",
                 "normal": "ok", "unknown": "?"}.get(status, "?")
        label = f.get("display_name") or f["name"]
        name = f["name"] if not f.get("display_name") or nkey(f["display_name"]) == nkey(f["name"]) \
            else f"{label}（{f['name']}）"
        extra = f"；注意：{f['range_conflict_note']}" if f.get("range_conflict") else ""
        rows.append(f"- {name} = {_fmt_value(f)}{(' ' + f['unit']) if f.get('unit') else ''} [{arrow}]{ref}{extra}")
    return "\n".join(rows)


def abnormal_only(findings: List[Dict]) -> List[Dict]:
    return [f for f in findings if f["direction"] in ("high", "low")]
