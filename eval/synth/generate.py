"""
generate.py — Deterministic generator for a FULLY SYNTHETIC evaluation set of
Traditional-Chinese Taiwanese health-checkup PDFs + gold ground truth.

No real patient data is read or used: values come from seeded random draws
around public reference ranges; names/IDs in the header are obviously fake.
The only other input is a table of "KB range" texts (one generic range per
catalogue item), recorded next to each report-printed range. It is read from
the internal item KB when that file exists and otherwise from the kb_range
fields of the committed ground_truth.json, so a public checkout regenerates
the same set byte-for-byte.

Single source of truth: every report is first built as a Python dict
(profile, items, statuses, conditions); the PDF and ground_truth.json are both
rendered from that dict.

Usage (from project root):
    .venv/Scripts/python eval/synth/generate.py                 # 60 reports, seed 20260925
    .venv/Scripts/python eval/synth/generate.py --n 60 --seed 20260925 --out eval/synth
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # eval/ (gt_rules for metric-key aliases)

from catalog import (ABNORMAL_SCENARIOS, CLIN, CONDITION_SYNONYMS, CORE,  # noqa: E402
                     EXTRA_ALIASES, FAKE_NAME_PREFIX, FAKE_NAME_STEM, ITEMS, OPTIONAL, PAIRS,
                     QUAL_STYLES, SCENARIO_EXPECTS, SCENARIOS, SECONDARY, SECTIONS)
from gt_rules import name_aliases, value_tokens  # noqa: E402

GEN_VERSION = "synth-v1"
DEFAULT_SEED = 20260925
KB_PATH = HERE.parent.parent / "data" / "checkitem_kb.json"
DEFAULT_FONT = "C:/Windows/Fonts/msjh.ttc"
DEFAULT_FONT_BOLD = "C:/Windows/Fonts/msjhbd.ttc"

LAYOUTS = ["standard", "twocol", "inline_flags", "fullwidth", "plain", "scanned"]
LAYOUT_SHARE = {"standard": .28, "twocol": .18, "inline_flags": .20, "fullwidth": .17, "plain": .12}

EPS = 1e-9


# ═══════════════════════════════ KB ranges ══════════════════════════════════

def load_kb_ranges():
    """{catalogue kb name: KB range text}."""
    if not KB_PATH.exists():
        return _kb_ranges_from_ground_truth(HERE / "ground_truth.json")
    data = json.loads(KB_PATH.read_text(encoding="utf-8"))
    out = {}
    for e in data:
        c = e.get("Content", "")
        m = re.search(r"【檢查項目】(.*)", c)
        if not m:
            continue
        r = re.search(r"【正常參考範圍】(.*)", c)
        out[m.group(1).strip()] = r.group(1).strip() if r else ""
    return out


def _kb_ranges_from_ground_truth(path):
    """Public fallback: the range texts recorded per item in ground_truth.json."""
    out = {}
    gt = json.loads(Path(path).read_text(encoding="utf-8"))
    for e in gt["samples"].values():
        for it in e["items"]:
            if it.get("kb_name") and it.get("kb_range") is not None:
                out.setdefault(it["kb_name"], it["kb_range"])
    return out


def parse_kb_range(s):
    """KB range text -> bounds tuple (lo, hi, lo_strict, hi_strict) or None."""
    if not s:
        return None
    t = unicodedata.normalize("NFKC", s)
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[~\-]\s*(-?\d+(?:\.\d+)?)", t)
    if m:
        return (float(m.group(1)), float(m.group(2)), False, False)
    m = re.match(r"^\s*(<=|≦|≤|<)\s*(\d+(?:\.\d+)?)", t)
    if m:
        return (None, float(m.group(2)), False, m.group(1) == "<")
    m = re.match(r"^\s*(>=|≧|≥|>)\s*(\d+(?:\.\d+)?)", t)
    if m:
        return (float(m.group(2)), None, m.group(1) == ">", False)
    return None


# ═══════════════════════════════ range helpers ══════════════════════════════

def spec_bounds(spec):
    """printed spec -> (lo, hi, lo_strict, hi_strict)."""
    if spec is None or spec == "QUAL":
        return None
    k = spec[0]
    if k == "range":
        return (float(spec[1]), float(spec[2]), False, False)
    if k == "upper":
        return (None, float(spec[1]), False, True)
    if k == "lower":
        return (float(spec[1]), None, True, False)
    raise ValueError(spec)


def status_of(b, v):
    if b is None:
        return "no_range"
    lo, hi, los, his = b
    if lo is not None and (v <= lo + EPS if los else v < lo - EPS):
        return "low"
    if hi is not None and (v >= hi - EPS if his else v > hi + EPS):
        return "high"
    return "normal"


def clin_bounds(key, sex):
    c = CLIN.get(key)
    if c is None:
        return None
    return c[sex] if isinstance(c, dict) else c


def sex_spec(spec, sex):
    return spec[sex] if isinstance(spec, dict) else spec


def spec_plain(spec):
    """half-width canonical text of a (sex-applicable) spec."""
    if spec is None:
        return ""
    if spec == "QUAL":
        return "QUAL"
    k = spec[0]
    if k == "range":
        return f"{spec[1]}-{spec[2]}"
    if k == "upper":
        return f"<{spec[1]}"
    return f">{spec[1]}"


def range_conflict(pb, kb):
    """Compare printed vs KB bounds side by side. A bound counts only when BOTH
    ranges state it (a lower bound of 0 counts as 'no lower bound'); a side
    stated by only one range is reported separately as `kb_only_bounds` /
    `printed_only_bounds`. Returns (conflict: bool, detail: str)."""
    def norm(b):
        lo, hi = b[0], b[1]
        if lo is not None and abs(lo) < EPS:
            lo = None
        return lo, hi
    (plo, phi), (klo, khi) = norm(pb), norm(kb)
    det = []
    if plo is not None and klo is not None and abs(plo - klo) > EPS:
        det.append(f"low bound {plo:g} vs KB {klo:g}")
    if phi is not None and khi is not None and abs(phi - khi) > EPS:
        det.append(f"high bound {phi:g} vs KB {khi:g}")
    extra = []
    if plo is None and klo is not None:
        extra.append("kb_only_low")
    if phi is None and khi is not None:
        extra.append("kb_only_high")
    if klo is None and plo is not None:
        extra.append("printed_only_low")
    if khi is None and phi is not None:
        extra.append("printed_only_high")
    return bool(det), "; ".join(det), extra


FW = str.maketrans({**{str(d): chr(0xFF10 + d) for d in range(10)},
                    ".": "．", "<": "＜", ">": "＞", "+": "＋", ":": "：", "~": "～", "-": "－"})


def to_fw(s):
    return s.translate(FW)


def fmt_num(v, dec):
    return f"{v:.{dec}f}"


# ═══════════════════════════════ sampling ═══════════════════════════════════

class Ctx:
    def __init__(self, rng, sex, age, printed, kb_bounds):
        self.rng, self.sex, self.age = rng, sex, age
        self.printed = printed        # key -> sex-applicable printed spec
        self.kb = kb_bounds           # key -> kb bounds or None

    def pb(self, key):
        return spec_bounds(self.printed.get(key))


def _typ(key, sex):
    t = ITEMS[key]["typ"]
    return t[sex] if isinstance(t, dict) else t


def _step(key):
    it = ITEMS[key]
    return it.get("quant") or 10 ** (-it["dec"])


def _round(key, v):
    it = ITEMS[key]
    q = it.get("quant")
    if q:
        v = round(round(v / q) * q, it["dec"])
    return round(v, it["dec"])


def all_bounds(key, ctx):
    pts = []
    for b in (ctx.pb(key), ctx.kb.get(key), clin_bounds(key, ctx.sex)):
        if b:
            pts += [x for x in b[:2] if x is not None]
    return pts


def nudge(key, v, ctx, direction):
    """Move a rounded value off any printed/KB/clinical bound (no ties in GT)."""
    st = _step(key)
    for _ in range(6):
        if not any(abs(v - p) < st / 2 for p in all_bounds(key, ctx)):
            break
        v = _round(key, v + (st if direction >= 0 else -st))
    return v


def normal_interval(key, ctx):
    lo, hi = ITEMS[key]["plaus"]
    for b in (ctx.pb(key), ctx.kb.get(key), clin_bounds(key, ctx.sex)):
        if not b:
            continue
        if b[0] is not None:
            lo = max(lo, b[0])
        if b[1] is not None:
            hi = min(hi, b[1])
    w = hi - lo
    if w <= 0:
        raise RuntimeError(f"empty normal interval for {key}")
    return lo + .12 * w, hi - .12 * w


def abnormal_interval(key, ctx, d, interval=None):
    pl, ph = ITEMS[key]["plaus"]
    pb, cb = ctx.pb(key), clin_bounds(key, ctx.sex)
    st = _step(key)
    if d == "high":
        cut = max(b[1] for b in (pb, cb) if b and b[1] is not None)
        lo, hi = interval if interval else (cut * 1.05, cut * 1.35)
        lo = max(lo, cut + st)
        if hi <= lo:
            hi = lo + max(cut * .25, 5 * st)
        return lo, hi
    cut = min(b[0] for b in (pb, cb) if b and b[0] is not None)
    lo, hi = interval if interval else (cut * .7, cut * .95)
    hi = min(hi, cut - st)
    lo = max(lo, pl)
    if hi <= lo:
        lo = hi - max(cut * .2, 5 * st)
    return lo, hi


def probe_zones(key, ctx):
    """Intervals where the report-printed status and the KB status disagree."""
    pb, kb = ctx.pb(key), ctx.kb.get(key)
    if not pb or not kb:
        return []
    pl, ph = ITEMS[key]["plaus"]
    pts = sorted({pl, ph, *[x for x in (pb[0], pb[1], kb[0], kb[1]) if x is not None and pl < x < ph]})
    zones = []
    st = _step(key)
    for a, b in zip(pts, pts[1:]):
        if b - a < 3 * st:
            continue
        mid = (a + b) / 2
        if status_of(pb, mid) != status_of(kb, mid):
            zones.append((a + st, b - st))
    return zones


def sample_value(key, want, ctx, interval=None):
    rng = ctx.rng
    if want == "normal":
        lo, hi = normal_interval(key, ctx)
        mode = min(max(_typ(key, ctx.sex), lo), hi)
        v = rng.triangular(lo, hi, mode)
        return nudge(key, _round(key, v), ctx, 1 if v < (lo + hi) / 2 else -1)
    if want in ("high", "low"):
        lo, hi = abnormal_interval(key, ctx, want, interval)
        v = rng.uniform(lo, hi)
        return nudge(key, _round(key, v), ctx, 1 if want == "high" else -1)
    if want == "probe":
        zones = probe_zones(key, ctx)
        a, b = zones[rng.randrange(len(zones))]
        v = rng.uniform(a, b)
        return nudge(key, _round(key, v), ctx, 1 if v < (a + b) / 2 else -1)
    if want == "any":
        lo, hi = interval if interval else normal_interval(key, ctx)
        v = rng.uniform(lo, hi)
        return _round(key, v)
    raise ValueError(want)


def meets(key, v, want, ctx, qual_neg=None):
    it = ITEMS[key]
    if it.get("qual"):
        pos = v != qual_neg
        return (want == "positive") == pos if want in ("positive", "normal") else True
    if want in ("any", "probe", None):
        return True
    st = status_of(ctx.pb(key), v)
    cb = clin_bounds(key, ctx.sex)
    cst = status_of(cb, v) if cb else None
    if want == "normal":
        kb = ctx.kb.get(key)
        return st in ("normal", "no_range") and (kb is None or status_of(kb, v) == "normal") and cst in (None, "normal")
    return st == want and cst in (None, want)


def ckd_epi_2021(cr, age, sex):
    k, a = (0.7, -0.241) if sex == "F" else (0.9, -0.302)
    r = cr / k
    e = 142 * min(r, 1) ** a * max(r, 1) ** -1.200 * 0.9938 ** age
    return e * (1.012 if sex == "F" else 1.0)


# ═══════════════════════════════ report plan ════════════════════════════════

def scenario_wants(scen, ctx, panel):
    """Return {key: (want, interval_or_grades)} for one scenario."""
    rng, sex = ctx.rng, ctx.sex
    w = {}
    if scen == "metabolic_syndrome":
        crit = ["waist", "bp", "glu", "tg", "hdl"]
        k = rng.choice([3, 3, 4, 4, 5])
        chosen = set(rng.sample(crit, k))
        w["waist"] = ("high", (91.5, 108) if sex == "M" else (81.5, 98)) if "waist" in chosen else ("normal", None)
        if "bp" in chosen:
            w["sbp"] = ("high", (132, 158))
            w["dbp"] = ("high", (86, 97)) if rng.random() < .6 else ("normal", None)
        w["glu"] = ("high", (101, 124)) if "glu" in chosen else ("normal", None)
        w["tg"] = ("high", (160, 420)) if "tg" in chosen else ("normal", None)
        w["hdl"] = ("low", (28, 39) if sex == "M" else (33, 49)) if "hdl" in chosen else ("normal", None)
        w["bmi"] = ("any", None)
        w["_crit"] = sorted(chosen)
    elif scen == "fatty_liver":
        w["alt"] = ("high", (58, 165))
        w["ast"] = ("high", (42, 98)) if rng.random() < .7 else ("normal", None)
        w["ggt"] = ("high", (70, 210)) if rng.random() < .6 else ("normal", None)
        w["bmi"] = ("high", (25.2, 31.5))
        w["waist"] = ("any", None)
        if rng.random() < .5:
            w["tg"] = ("high", (155, 320))
    elif scen == "ckd":
        w["cr"] = ("high", (1.55, 3.2) if sex == "M" else (1.25, 2.6))
        w["egfr"] = ("low", None)
        w["bun"] = ("high", (27, 55))
        w["u_pro"] = ("positive", ["+", "2+", "3+"])
        if rng.random() < .4:
            w["ua"] = ("high", (7.6, 10.2))
        if rng.random() < .35:
            w["hb"] = ("low", (10.0, 12.4) if sex == "M" else (9.2, 11.4))
            for d in ("hct", "rbc", "mch"):
                w[d] = ("any", None)
    elif scen == "anemia":
        w["hb"] = ("low", (9.0, 12.4) if sex == "M" else (8.0, 11.4))
        if rng.random() < .7:  # iron-deficiency pattern
            w["mcv"] = ("low", (65, 79))
            w["mchc"] = ("low", (29.0, 31.3))
            w["_ida"] = True
        for d in ("hct", "rbc", "mch"):
            w[d] = ("any", None)
    elif scen == "hyperuricemia":
        w["ua"] = ("high", (7.8, 10.8) if sex == "M" else (7.3, 9.6))
    elif scen == "hypothyroid":
        w["tsh"] = ("high", (5.6, 24))
        # alternate overt / subclinical across the set
        w["ft4"] = ("low", (0.45, 0.85)) if ctx.occ.get(scen, 0) % 2 == 1 else ("normal", None)
        if rng.random() < .4:
            w["ldl"] = ("high", (138, 185))
            w["tc"] = ("any", None)
    elif scen == "diabetes":
        if ctx.occ.get(scen, 0) % 2 == 0:  # alternate prediabetes / diabetes across the set
            w["glu"] = ("high", (101, 125))
            w["hba1c"] = ("high", (5.7, 6.4))
        else:
            w["glu"] = ("high", (130, 235))
            w["hba1c"] = ("high", (6.6, 9.8))
            if rng.random() < .35:
                w["u_pro"] = ("positive", ["+", "2+"])
    elif scen == "dyslipidemia":
        w["ldl"] = ("high", (140, 212))
        w["tc"] = ("any", None)
        if rng.random() < .4:
            w["tg"] = ("high", (155, 300))
        if rng.random() < .3:
            w["hdl"] = ("low", (30, 39) if sex == "M" else (36, 49))
    elif scen == "hypertension":
        w["sbp"] = ("high", (142, 172))
        w["dbp"] = ("high", (91, 106)) if rng.random() < .7 else ("any", (76, 84))
    elif scen == "tumor_marker":
        cands = [k for k in ("cea", "afp", "ca199", "psa", "ca125") if
                 ITEMS[k].get("sex") in (None, sex) and not (k == "psa" and ctx.age < 50)]
        k = rng.choice(cands)
        rng_int = {"cea": (5.6, 14), "afp": (10.5, 28), "ca199": (38, 90), "psa": (4.6, 9.5),
                   "ca125": (38, 80)}[k]
        w[k] = ("high", rng_int)
        w["_marker"] = k
    return w


def choose_panel(rng, sex, age, scenarios, wants):
    required = set()
    for s in scenarios:
        required |= set(SCENARIOS[s]["required"])
    required |= {k for k in wants if not k.startswith("_")}
    basic = rng.random() < .25
    panel = set(CORE)
    opt = []
    for k, p in OPTIONAL.items():
        if ITEMS[k].get("sex") not in (None, sex) or (k == "psa" and age < 40):
            continue
        opt.append(k)
        if rng.random() < (p * (.3 if basic else 1)):
            panel.add(k)
    for a, (b, p) in PAIRS.items():
        opt += [a, b]
        if rng.random() < p * (.3 if basic else 1):
            panel |= {a, b}
    panel |= required
    if "egfr" in panel:
        panel.add("cr")
    # clamp 20..45
    removable = sorted(k for k in panel if k in opt and k not in required)
    while len(panel) > 45 and removable:
        panel.discard(removable.pop(rng.randrange(len(removable))))
    addable = sorted(k for k in opt if k not in panel)
    while len(panel) < 20 and addable:
        panel.add(addable.pop(rng.randrange(len(addable))))
    order = [k for k in ITEMS if k in panel]
    return order, basic


# ═══════════════════════════════ value generation ═══════════════════════════

def W(wants, k):
    return wants.get(k, ("normal", None))


def gen_values(ctx, wants, panel, qual_neg):
    rng, sex, age = ctx.rng, ctx.sex, ctx.age
    vals = {}

    def ok(keys):
        return all(meets(k, vals[k], W(wants, k)[0], ctx, qual_neg) for k in keys if k in panel)

    def sv(k):
        want, arg = W(wants, k)
        return sample_value(k, want, ctx, arg)

    def loop(fn, keys, n=3000):
        for _ in range(n):
            fn()
            if ok(keys):
                return
        raise RuntimeError(f"could not satisfy {keys} wants={ {k: W(wants, k) for k in keys} }")

    # body -----------------------------------------------------------------
    a, b = (2.2, 33.0) if sex == "M" else (2.0, 30.0)

    def body():
        h = _round("height", min(max(rng.gauss(_typ("height", sex), 6 if sex == "M" else 5.5), 145), 192))
        if W(wants, "waist")[0] == "high":
            waist = sv("waist")
            bmi = min(max((waist - b) / a + rng.gauss(0, .8), 21.0), 38.0)
        else:
            bw = W(wants, "bmi")
            if bw[0] == "any":
                bmi = min(max(rng.gauss(24.2, 1.6), 20.5), 27.5)
            else:
                bmi = sv("bmi")
            waist = a * bmi + b + rng.gauss(0, 2.5)
        wt = _round("weight", bmi * (h / 100) ** 2)
        bmi = _round("bmi", wt / (h / 100) ** 2)
        vals.update(height=h, weight=wt, bmi=bmi, waist=nudge("waist", _round("waist", waist), ctx, 1))
        for _ in range(50):
            vals["sbp"], vals["dbp"] = sv("sbp"), sv("dbp")
            if vals["sbp"] - vals["dbp"] >= 25:
                break
        vals["pulse"] = sv("pulse")
    loop(body, ["height", "weight", "bmi", "waist", "sbp", "dbp", "pulse"])

    # CBC ------------------------------------------------------------------
    def cbc():
        vals["wbc"], vals["plt"] = sv("wbc"), sv("plt")
        hb, mcv, mchc = sv("hb"), sv("mcv"), sv("mchc")
        hct = _round("hct", hb / mchc * 100)
        rbc = _round("rbc", hct / mcv * 10)
        mch = _round("mch", hb / rbc * 10)
        vals.update(hb=hb, mcv=mcv, mchc=mchc, hct=hct, rbc=rbc, mch=mch)
    loop(cbc, ["wbc", "plt", "hb", "mcv", "mchc", "hct", "rbc", "mch"])

    # liver ----------------------------------------------------------------
    def liver():
        for k in ("ast", "alt", "ggt", "alp", "tbil", "alb", "tp"):
            vals[k] = sv(k)
        if vals["tp"] - vals["alb"] < 1.8:
            vals["tp"] = -1  # force retry
    for _ in range(3000):
        liver()
        if vals["tp"] > 0 and ok(["ast", "alt", "ggt", "alp", "tbil", "alb", "tp"]):
            break
    else:
        raise RuntimeError("liver")

    # kidney ---------------------------------------------------------------
    def kidney():
        for k in ("bun", "cr", "ua"):
            vals[k] = sv(k)
        vals["egfr"] = _round("egfr", ckd_epi_2021(vals["cr"], age, sex))
        vals["egfr"] = nudge("egfr", vals["egfr"], ctx, -1 if W(wants, "egfr")[0] == "low" else 1)
    loop(kidney, ["bun", "cr", "ua", "egfr"])

    # lipid ----------------------------------------------------------------
    def lipid():
        for k in ("hdl", "tg", "ldl"):
            vals[k] = sv(k)
        vals["tc"] = nudge("tc", _round("tc", vals["ldl"] + vals["hdl"] + vals["tg"] / 5), ctx, 1)
    loop(lipid, ["hdl", "tg", "ldl", "tc"])

    # glucose / thyroid / tumor ---------------------------------------------
    for k in ("glu", "hba1c", "tsh", "ft4", "afp", "cea", "ca199", "ca125", "ca153", "psa"):
        if ITEMS[k].get("sex") not in (None, sex):
            continue
        loop(lambda k=k: vals.__setitem__(k, sv(k)), [k])

    # urine ----------------------------------------------------------------
    for k in ("u_ph", "u_sg", "u_ubg"):
        loop(lambda k=k: vals.__setitem__(k, sv(k)), [k])
    for k in ("u_pro", "u_glu", "u_ob", "u_ket"):
        want, arg = W(wants, k)
        if k == "u_glu" and vals.get("glu", 0) >= 180:
            want, arg = "positive", (["+", "2+"] if vals["glu"] < 250 else ["2+", "3+"])
            wants["u_glu"] = (want, arg)
        if want == "positive":
            grades = list(arg or ["+"])
            if qual_neg == "陰性" and rng.random() < .25:
                grades.append("陽性")
            vals[k] = rng.choice(grades)
        else:
            vals[k] = qual_neg
    return vals


# ═══════════════════════════════ rules → conditions ═════════════════════════

def derive_conditions(items_by_key, sex):
    """Explicit rule set; operates only on items present in the report."""
    v = {k: it["v"] for k, it in items_by_key.items() if it["v"] is not None}
    st = {k: it["gold_status"] for k, it in items_by_key.items()}
    cond, risk, ev = [], [], {}

    def S(k):  # value exactly as displayed, for evidence strings
        return items_by_key[k]["value"] if k in items_by_key else "n/a"

    def add(lst, x, why=None):
        if x not in lst:
            lst.append(x)
        if why:
            ev.setdefault(x, why)

    # metabolic syndrome (國健署 2007): >=3 of 5
    crit = {}
    if "waist" in v:
        crit["waist"] = v["waist"] >= (90 if sex == "M" else 80)
    if "sbp" in v or "dbp" in v:
        crit["bp"] = v.get("sbp", 0) >= 130 or v.get("dbp", 0) >= 85
    if "glu" in v:
        crit["glu"] = v["glu"] >= 100
    if "tg" in v:
        crit["tg"] = v["tg"] >= 150
    if "hdl" in v:
        crit["hdl"] = v["hdl"] < (40 if sex == "M" else 50)
    met = [k for k, x in crit.items() if x]
    if len(met) >= 3:
        add(cond, "代謝症候群", f"國健署 criteria met: {','.join(met)} ({len(met)}/5)")
        add(risk, "心血管疾病")
        add(risk, "第二型糖尿病")
    # blood pressure
    if v.get("sbp", 0) >= 140 or v.get("dbp", 0) >= 90:
        add(cond, "高血壓", f"SBP {S('sbp')} / DBP {S('dbp')} >= 140/90")
        add(risk, "心血管疾病")
        add(risk, "腦中風")
    elif v.get("sbp", 0) >= 130 or v.get("dbp", 0) >= 85:
        add(cond, "血壓偏高", f"SBP {S('sbp')} / DBP {S('dbp')} in 130-139/85-89")
    # glycaemia
    g, a1c = v.get("glu"), v.get("hba1c")
    if (g is not None and g >= 126) or (a1c is not None and a1c >= 6.5):
        add(cond, "糖尿病", f"AC glucose {S('glu')} >=126 or HbA1c {S('hba1c')} >=6.5")
        add(risk, "糖尿病併發症")
        add(risk, "心血管疾病")
    elif (g is not None and g >= 100) or (a1c is not None and a1c >= 5.7):
        add(cond, "糖尿病前期", f"AC glucose {S('glu')} 100-125 or HbA1c {S('hba1c')} 5.7-6.4")
        add(risk, "第二型糖尿病")
    # lipids
    lip = []
    if v.get("tc", 0) >= 200:
        lip.append("TC>=200")
    if v.get("ldl", 0) >= 130:
        lip.append("LDL>=130")
    if v.get("tg", 0) >= 150:
        lip.append("TG>=150")
    if "hdl" in v and v["hdl"] < (40 if sex == "M" else 50):
        lip.append("HDL low")
    if lip:
        add(cond, "血脂異常", ", ".join(lip))
        add(risk, "動脈粥狀硬化")
        add(risk, "心血管疾病")
    # liver (printed-range status)
    if st.get("alt") == "high" or st.get("ast") == "high":
        add(cond, "肝功能異常", f"ALT {S('alt')} / AST {S('ast')} above printed range")
        metab = (v.get("bmi", 0) >= 24 or v.get("tg", 0) >= 150 or crit.get("waist"))
        add(risk, "脂肪肝" if metab else "肝臟疾病")
    # kidney
    if "egfr" in v and v["egfr"] < 60:
        add(cond, "慢性腎臟病", f"eGFR {S('egfr')} < 60")
        add(risk, "腎衰竭")
        add(risk, "心血管疾病")
    if st.get("u_pro") == "positive":
        add(cond, "蛋白尿", f"urine protein {items_by_key['u_pro']['value']}")
        add(risk, "腎衰竭" if "慢性腎臟病" in cond else "慢性腎臟病")
    # anaemia (WHO)
    if "hb" in v and v["hb"] < (13 if sex == "M" else 12):
        add(cond, "貧血", f"Hb {S('hb')} < {13 if sex == 'M' else 12} (WHO)")
        if "mcv" in v and v["mcv"] < 80:
            add(cond, "小球性貧血", f"MCV {S('mcv')} < 80")
            add(risk, "缺鐵性貧血")
    # uric acid
    if "ua" in v and v["ua"] > 7.0:
        add(cond, "高尿酸血症", f"UA {S('ua')} > 7.0")
        add(risk, "痛風")
    # thyroid (printed-range status)
    if st.get("tsh") == "high":
        if st.get("ft4") == "low":
            add(cond, "甲狀腺功能低下", f"TSH {S('tsh')} high and FT4 {S('ft4')} low")
        else:
            add(cond, "亞臨床甲狀腺功能低下", f"TSH {S('tsh')} high, FT4 {S('ft4')} not low")
            add(risk, "甲狀腺功能低下")
    # BMI (國健署)
    if "bmi" in v:
        if v["bmi"] >= 27:
            add(cond, "肥胖", f"BMI {S('bmi')} >= 27")
        elif v["bmi"] >= 24:
            add(cond, "過重", f"BMI {S('bmi')} 24-27")
        if v["bmi"] >= 24 and "代謝症候群" not in cond:
            add(risk, "代謝症候群")
    # tumour markers (printed-range status)
    tm = [k for k in ("afp", "cea", "ca199", "ca125", "ca153", "psa") if st.get(k) == "high"]
    if tm:
        add(cond, "腫瘤標記偏高", ", ".join(f"{ITEMS[k]['en']} {S(k)}" for k in tm))
    return cond, risk, ev


# ═══════════════════════════════ build one report ═══════════════════════════

def build_report(idx, sid, layout, scen_plan, seed, kb_text):
    rng = random.Random(f"{seed}:{GEN_VERSION}:{idx}")
    primary, secondary, occ = scen_plan
    scenarios = [primary] + ([secondary] if secondary else [])
    sd = SCENARIOS[primary]
    sex = "F" if rng.random() < sd["p_female"] else "M"
    band = rng.choice(sd["ages"])
    if secondary:
        common = [a for a in sd["ages"] if a in SCENARIOS[secondary]["ages"]]
        band = rng.choice(common or sd["ages"])
    lo = int(band[:2])
    age = rng.randint(lo, lo + 9)
    healthy = primary == "healthy"

    # layout features ----------------------------------------------------------
    feats = {"layout": layout, "image_only": layout == "scanned"}
    feats["base_layout"] = rng.choice(["standard", "twocol"]) if layout == "scanned" else layout
    feats["sex_ref_text"] = rng.random() < .45
    feats["fullwidth"] = layout == "fullwidth"
    feats["flag_style"] = {"inline_flags": rng.choice(["HL", "arrow", "star"]),
                           "twocol": rng.choice(["HL", "none"]),
                           "plain": "arrow_star"}.get(feats["base_layout"], "none")
    feats["judgement_col"] = rng.choice(["HL", "zh"]) if feats["base_layout"] in ("standard", "fullwidth") else None
    feats["range_sep"] = "～" if layout in ("fullwidth", "plain") else rng.choice(["-", "~"])
    qual_neg = rng.choice(QUAL_STYLES)
    feats["qual_style"] = qual_neg

    # printed ranges -------------------------------------------------------------
    alt_rate = rng.uniform(.10, .45)
    chosen_spec = {}
    for k, it in ITEMS.items():
        spec = it["ref"]
        if it["alts"] and rng.random() < alt_rate:
            spec = rng.choice(it["alts"])
        chosen_spec[k] = spec
    printed = {k: (sex_spec(s, sex) if s not in (None, "QUAL") else s) for k, s in chosen_spec.items()}
    kb_bounds = {}
    for k, it in ITEMS.items():
        if it["kb"] is None:
            kb_bounds[k] = None
            continue
        if it["kb"] not in kb_text:
            raise KeyError(f"KB name not found: {it['kb']}")
        kb_bounds[k] = parse_kb_range(kb_text[it["kb"]])
    ctx = Ctx(rng, sex, age, printed, kb_bounds)
    ctx.occ = occ

    # wants ----------------------------------------------------------------------
    wants, meta = {}, {}
    strong = ("high", "low", "positive")
    for s in sorted(scenarios, key=lambda x: x != "metabolic_syndrome"):
        for k, x in scenario_wants(s, ctx, None).items():
            if k.startswith("_"):
                meta[f"{s}{k}"] = x
            elif k not in wants or (x[0] in strong and wants[k][0] not in strong):
                wants[k] = x
    panel, basic = choose_panel(rng, sex, age, scenarios, wants)
    origin = {k: ("scenario" if k in wants and wants[k][0] in strong else
                  "derived" if k in wants and wants[k][0] == "any" else "normal") for k in panel}
    derived = {"weight", "hct", "rbc", "mch", "egfr", "tc"}
    if not healthy:
        for k in panel:
            if k in wants or k in derived:
                continue
            bg = ITEMS[k].get("bg")
            if bg and rng.random() < .05:
                d = rng.choice(bg)
                wants[k] = (d, ["+", "2+"] if d == "positive" else None)
                origin[k] = "background"
        for k in panel:
            if k in wants or k in derived or k in ("bmi", "waist") or ITEMS[k].get("qual"):
                continue
            if probe_zones(k, ctx) and rng.random() < .22:
                wants[k] = ("probe", None)
                origin[k] = "kb_probe"
    # derived items follow their (non-normal) primaries
    for prims, ders in ((("hb", "mcv", "mchc"), ("hct", "rbc", "mch")), (("cr",), ("egfr",)),
                        (("hdl", "tg", "ldl"), ("tc",)), (("bmi",), ("waist",))):
        if any(wants.get(p, ("normal",))[0] != "normal" for p in prims):
            for d in ders:
                if d not in wants:
                    wants[d] = ("any", None)
                    if d in origin:
                        origin[d] = "derived"

    vals = gen_values(ctx, wants, set(panel), qual_neg)
    if wants.get("u_glu", ("normal",))[0] == "positive" and origin.get("u_glu") == "normal":
        origin["u_glu"] = "derived"

    # items ------------------------------------------------------------------------
    items, by_key = [], {}
    for k in panel:
        it = ITEMS[k]
        v = vals[k]
        spec_all = chosen_spec[k]
        sp = printed[k]
        if it.get("qual"):
            value = v
            gold = "negative" if v == qual_neg else "positive"
            direction = "normal" if gold == "negative" else "high"
            vnum = None
            kb_status, conflict, st_conflict, cdet, cextra = None, None, None, "", []
            sex_rng = qual_neg
        else:
            value = fmt_num(v, it["dec"])
            vnum = float(value)
            pb = spec_bounds(sp)
            gold = status_of(pb, vnum)
            direction = gold if gold != "no_range" else "unknown"
            kb = kb_bounds[k]
            kb_status = status_of(kb, vnum) if kb else None
            if pb and kb:
                conflict, cdet, cextra = range_conflict(pb, kb)
                st_conflict = kb_status != gold
            else:
                conflict, st_conflict, cdet, cextra = None, None, "", []
            sex_rng = spec_plain(sp)
        rec = {
            "key": k, "canonical_name": it["zh"], "canonical_en": it["en"], "kb_name": it["kb"],
            "section": dict(SECTIONS)[it["sec"]],
            "printed_name": rng.choice(it["names"]),
            "value": value, "v": vnum, "unit": it["unit"],
            "sex_specific_range": isinstance(spec_all, dict),
            "sex_applicable_range": sex_rng,
            "reference": (f"{sex_rng} {it['unit']}".strip() if sex_rng else ""),
            "kb_range": kb_text.get(it["kb"], "") if it["kb"] else None,
            "kb_status": kb_status,
            "kb_range_conflict": conflict,
            "kb_conflict_detail": cdet,
            "kb_one_sided_bounds": cextra,
            "status_conflict": st_conflict,
            "gold_status": gold,
            "direction": direction,
            "origin": origin.get(k, "normal"),
            "_spec_all": spec_all,
        }
        items.append(rec)
        by_key[k] = rec

    cond, risk, ev = derive_conditions(by_key, sex)

    # consistency checks -------------------------------------------------------------
    if healthy:
        bad = [r["key"] for r in items if r["gold_status"] not in ("normal", "negative", "no_range")
               or r["status_conflict"]]
        assert not bad and not cond, (sid, bad, cond)
    for s in scenarios:
        exp = SCENARIO_EXPECTS.get(s, [])
        assert not exp or any(c in cond for c in exp), (sid, s, cond)

    return {
        "sample_id": sid, "idx": idx, "layout": layout, "features": feats,
        "profile": {"sex": sex, "age": age, "age_band": band},
        "scenarios": scenarios, "scenario_meta": meta, "basic_panel": basic,
        "items": items, "conditions": cond, "risks": risk, "condition_evidence": ev,
        "header": fake_header(idx, sex, age, rng),
    }


def fake_header(idx, sex, age, rng):
    names = [f"{p} {s}" for p in FAKE_NAME_PREFIX for s in FAKE_NAME_STEM]
    year = 2025
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    return {
        "name": names[(idx - 1) % len(names)],
        "id_no": f"A{idx:09d}",
        "chart_no": f"SYN-{idx:06d}",
        "birth_date": f"{year - age}-01-01",
        "exam_date": f"{year}-{month:02d}-{day:02d}",
        "phone": f"0900-000-{idx % 1000:03d}",
        "address": f"虛構市測試區範例路{idx}號",
        "sex_label": "男" if sex == "M" else "女",
        "age": age,
    }


# ═══════════════════════════════ printed strings ════════════════════════════

def printed_range_text(rec, rep):
    f = rep["features"]
    it = ITEMS[rec["key"]]
    spec = rec["_spec_all"]
    fw = f["fullwidth"]
    sep = f["range_sep"]

    def one(s):
        if s == "QUAL":
            return f["qual_style"]
        if s is None:
            return ""
        t = spec_plain(s)
        if s[0] == "range":
            t = t.replace("-", sep, 1) if not t.startswith("-") else t
        return to_fw(t) if fw else t

    if spec is None:
        return ""
    if isinstance(spec, dict):
        if f["sex_ref_text"]:
            if fw:
                return f"男：{one(spec['M'])}　女：{one(spec['F'])}"
            return f"男:{one(spec['M'])} 女:{one(spec['F'])}"
        return one(spec[rep["profile"]["sex"]])
    _ = it
    return one(spec)


def printed_value_text(rec, rep, style):
    f = rep["features"]
    v = rec["value"]
    if f["fullwidth"]:
        v = to_fw(v) if v not in ("Negative",) else v
    g = rec["gold_status"]
    up = g in ("high", "positive")
    down = g == "low"
    if style == "HL":
        return f"{v} {'H' if up else 'L' if down else ''}".rstrip()
    if style == "arrow":
        return f"{v} {'↑' if up else '↓' if down else ''}".rstrip()
    if style == "star":
        return f"{v}*" if (up or down) else v
    if style == "arrow_star":
        return f"{v}{'↑' if up else '↓' if down else ''}"
    return v


def judgement_text(rec, style):
    g = rec["gold_status"]
    if style == "HL":
        return {"high": "H", "low": "L", "positive": "H"}.get(g, "")
    return {"high": "偏高", "low": "偏低", "positive": "異常"}.get(g, "")


# ═══════════════════════════════ PDF rendering ══════════════════════════════

_FONT_OK = {}


def register_fonts(font, font_bold):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    if _FONT_OK:
        return
    pdfmetrics.registerFont(TTFont("SYN", font, subfontIndex=0))
    pdfmetrics.registerFont(TTFont("SYNB", font_bold or font, subfontIndex=0))
    _FONT_OK["ok"] = True


def render_text_pdf(rep, out, layout):
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    def P(t, st):
        return Paragraph(escape(str(t)), st)

    base = ParagraphStyle("b", fontName="SYN", fontSize=8.5, leading=11)
    small = ParagraphStyle("s", parent=base, fontSize=7.8, leading=10)
    h1 = ParagraphStyle("h1", fontName="SYNB", fontSize=15, leading=19, alignment=1)
    h2 = ParagraphStyle("h2", fontName="SYNB", fontSize=10.5, leading=13, spaceBefore=5, spaceAfter=2,
                        keepWithNext=1)
    f = rep["features"]
    hd = rep["header"]
    grid = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "SYN"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3e9f1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])
    story = [P("健康檢查報告", h1), P("虛構健康檢查中心（合成測試資料）", ParagraphStyle(
        "c", parent=base, alignment=1)), Spacer(1, 3 * mm)]
    hdr = [["姓名", hd["name"], "性別", hd["sex_label"], "年齡", str(hd["age"])],
           ["身分證字號", hd["id_no"], "出生日期", hd["birth_date"], "檢查日期", hd["exam_date"]],
           ["病歷號碼", hd["chart_no"], "聯絡電話", hd["phone"], "地址", hd["address"]]]
    ht = Table([[P(c, small) for c in row] for row in hdr],
               colWidths=[20 * mm, 32 * mm, 18 * mm, 30 * mm, 17 * mm, 58 * mm])
    ht.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f1f1")),
                            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f1f1f1")),
                            ("BACKGROUND", (4, 0), (4, -1), colors.HexColor("#f1f1f1"))]))
    story += [ht, Spacer(1, 3 * mm)]

    by_sec = {}
    for r in rep["items"]:
        by_sec.setdefault(r["section"], []).append(r)

    for sec, its in by_sec.items():
        story.append(P(f"【{sec}】", h2))
        if layout in ("standard", "fullwidth"):
            jst = f["judgement_col"]
            data = [["檢查項目", "結果", "單位", "參考值", "判定"]]
            for r in its:
                data.append([P(r["printed_name"], base), P(printed_value_text(r, rep, "none"), base),
                             P(r["unit"], base), P(printed_range_text(r, rep), base),
                             P(judgement_text(r, jst), base)])
            t = Table(data, colWidths=[58 * mm, 25 * mm, 27 * mm, 46 * mm, 19 * mm], repeatRows=1)
            t.setStyle(grid)
            story.append(t)
        elif layout == "inline_flags":
            data = [["項目", "結果", "單位", "參考範圍"]]
            for r in its:
                data.append([P(r["printed_name"], base), P(printed_value_text(r, rep, f["flag_style"]), base),
                             P(r["unit"], base), P(printed_range_text(r, rep), base)])
            t = Table(data, colWidths=[64 * mm, 30 * mm, 30 * mm, 51 * mm], repeatRows=1)
            t.setStyle(grid)
            story.append(t)
        elif layout == "twocol":
            half = (len(its) + 1) // 2

            def mk(sub):
                d = [["項目", "結果", "參考值"]]
                for r in sub:
                    ref = printed_range_text(r, rep)
                    ref = f"{ref} {r['unit']}".strip() if ref and r["unit"] else ref or r["unit"]
                    d.append([P(r["printed_name"], small), P(printed_value_text(r, rep, f["flag_style"]), small),
                              P(ref, small)])
                t = Table(d, colWidths=[38 * mm, 17 * mm, 32 * mm], repeatRows=1)
                ts = TableStyle(grid.getCommands() + [("FONTSIZE", (0, 0), (-1, -1), 7.8)])
                t.setStyle(ts)
                return t
            left = mk(its[:half])
            right = mk(its[half:]) if its[half:] else Spacer(1, 1)
            outer = Table([[left, right]], colWidths=[89 * mm, 89 * mm])
            outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
            story.append(outer)
        elif layout == "plain":
            mono = ParagraphStyle("m", parent=base, fontSize=8.8, leading=12.5)
            for r in its:
                g = r["gold_status"]
                star = "＊" if g in ("high", "low", "positive") else "　"
                val = printed_value_text(r, rep, "arrow_star")
                ref = printed_range_text(r, rep)
                unit = r["unit"]
                line = (escape(f"{star}{r['printed_name']}：") + "&nbsp;&nbsp;" + escape(val) +
                        ("&nbsp;" + escape(unit) if unit else "") +
                        ("&nbsp;&nbsp;&nbsp;&nbsp;" + escape(f"參考值 {ref}") if ref else ""))
                story.append(Paragraph(line, mono))
        else:
            raise ValueError(layout)

    story += [Spacer(1, 4 * mm), P("判定說明：H/↑ 高於參考值，L/↓ 低於參考值，＊ 異常。異常項目請依醫師建議追蹤。", small)]

    def footer(canv, doc):
        canv.saveState()
        canv.setFont("SYN", 7)
        canv.drawString(15 * mm, 7 * mm, f"SYNTHETIC TEST DATA — 合成資料，非真實個案   {rep['sample_id']}")
        canv.drawRightString(195 * mm, 7 * mm, f"第 {doc.page} 頁")
        canv.restoreState()

    doc = SimpleDocTemplate(str(out) if not isinstance(out, io.BytesIO) else out, pagesize=A4,
                            leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm, bottomMargin=14 * mm,
                            title="健康檢查報告 (synthetic)", author="synthetic-generator", subject=rep["sample_id"])
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def render_scanned_pdf(rep, out_path, seed):
    """Render the base layout, rasterise, rotate + add noise, embed as image-only PDF."""
    import numpy as np
    import pypdfium2 as pdfium
    from PIL import Image, ImageFilter
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    render_text_pdf(rep, buf, rep["features"]["base_layout"])
    pdf = pdfium.PdfDocument(buf.getvalue())
    nrng = np.random.default_rng(abs(hash_int(f"{seed}:{rep['sample_id']}:scan")))
    prng = random.Random(f"{seed}:{rep['sample_id']}:scan")
    dpi = 150
    angle = prng.choice([-1, 1]) * prng.uniform(0.6, 1.6)
    rep["features"].update({"scan_dpi": dpi, "scan_rotation_deg": round(angle, 2)})
    c = canvas.Canvas(str(out_path), pagesize=A4, invariant=1)
    c.setTitle("scanned report (synthetic)")
    for i in range(len(pdf)):
        page = pdf[i]
        img = page.render(scale=dpi / 72).to_pil().convert("L")
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=245)
        arr = np.asarray(img).astype(np.float32)
        arr = arr * 0.93 + 12 + nrng.normal(0, 9, arr.shape)
        speck = nrng.random(arr.shape) < 0.0015
        arr[speck] = nrng.integers(0, 90, speck.sum())
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
        jpg = io.BytesIO()
        img.save(jpg, format="JPEG", quality=72, optimize=False)  # embedded as-is (DCTDecode)
        jpg.seek(0)
        c.drawImage(ImageReader(jpg), 0, 0, width=A4[0], height=A4[1])
        c.showPage()
    c.save()
    rep["features"]["pages"] = len(pdf)
    pdf.close()


def hash_int(s):
    import hashlib
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


# ═══════════════════════════════ GT assembly ════════════════════════════════

def metric_tag(r):
    arrow = "↓" if r["gold_status"] == "low" else "↑"
    parts = [r["canonical_name"], r["value"]]
    if r["unit"]:
        parts.append(r["unit"])
    parts.append(arrow)
    return " ".join(parts)


def gt_entry(rep):
    items_out, metrics, mkeys, abn = [], [], {}, []
    for r in rep["items"]:
        rec = {"name": r["printed_name"]}  # run_eval.py-compatible alias of printed_name
        rec.update({k: v for k, v in r.items() if not k.startswith("_") and k != "v"})
        rec["printed_range"] = r["_printed_range"]
        rec["printed_value"] = r["_printed_value"]
        items_out.append(rec)
        if r["gold_status"] in ("high", "low", "positive"):
            tag = metric_tag(r)
            metrics.append(tag)
            al = name_aliases(r["printed_name"], r["canonical_name"], r["canonical_en"])
            for a in EXTRA_ALIASES.get(r["key"], []):
                if a.lower() not in al:
                    al.append(a.lower())
            mkeys[tag] = {"aliases": al, "values": value_tokens(r["value"]),
                          "numeric": r["v"] is not None}
            abn.append({"name": r["printed_name"], "canonical_name": r["canonical_name"], "value": r["value"],
                        "unit": r["unit"], "reference": r["reference"], "direction": r["direction"],
                        "gold_status": r["gold_status"]})
    syn = {t: CONDITION_SYNONYMS[t] for t in rep["conditions"] + rep["risks"] if t in CONDITION_SYNONYMS}
    f = dict(rep["features"])
    return {
        "sample_id": rep["sample_id"],
        "layout": rep["layout"],
        "layout_features": f,
        "profile": rep["profile"],
        "scenarios": rep["scenarios"],
        "scenario_meta": rep["scenario_meta"],
        "fake_pii": rep["header"],
        "label_quality": {"metrics": "gold (generator source of truth)",
                          "items": "gold (generator source of truth)",
                          "conditions": "gold (explicit rules on generated values)",
                          "risks": "gold (explicit rules on generated values)",
                          "lifestyle": "not labelled (judge by citation validity + LLM judge)",
                          "food": "not labelled", "exercise": "not labelled",
                          "supplements": "not labelled", "avoid": "not labelled"},
        "expected": {"conditions": rep["conditions"], "risks": rep["risks"], "metrics": metrics,
                     "lifestyle": [], "food": [], "exercise": [], "supplements": [], "avoid": []},
        "condition_evidence": rep["condition_evidence"],
        "synonyms": syn,
        "metric_keys": mkeys,
        "abnormal_findings": abn,
        "n_items": len(items_out),
        "n_abnormal": len(abn),
        "n_kb_range_conflicts": sum(1 for r in items_out if r["kb_range_conflict"]),
        "n_status_conflicts": sum(1 for r in items_out if r["status_conflict"]),
        "items": items_out,
    }


# ═══════════════════════════════ plan / main ════════════════════════════════

def plan(n, seed):
    rng = random.Random(f"{seed}:{GEN_VERSION}:plan")
    n_scan = 3 if n >= 30 else max(1, n // 10)
    rest = n - n_scan
    tot = sum(LAYOUT_SHARE.values())
    counts = {k: int(round(v / tot * rest)) for k, v in LAYOUT_SHARE.items()}
    diff = rest - sum(counts.values())
    counts["standard"] += diff
    layouts = ["scanned"] * n_scan + [k for k, c in counts.items() for _ in range(c)]
    rng.shuffle(layouts)
    n_healthy = max(1, round(n * .10))
    scen = ["healthy"] * n_healthy
    i = 0
    while len(scen) < n:
        scen.append(ABNORMAL_SCENARIOS[i % len(ABNORMAL_SCENARIOS)])
        i += 1
    rng.shuffle(scen)
    plans, seen = [], Counter()
    for s in scen:
        sec = None
        if s != "healthy" and rng.random() < .35:
            sec = rng.choice(SECONDARY[s])
        occ = {x: seen[x] for x in (s, sec) if x}  # occurrence index -> balanced sub-variants
        seen.update(occ.keys())
        plans.append((s, sec, occ))
    return layouts, plans


def pick_quick(entries, k=10):
    facets = []
    for e in entries:
        f = e["layout_features"]
        fs = {f"layout:{e['layout']}"}
        if e["layout"] == "inline_flags":
            fs.add(f"flag:{f['flag_style']}")
        if f["sex_ref_text"] and any(i["sex_specific_range"] for i in e["items"]):
            fs.add("sex_ref_text")
        if "healthy" in e["scenarios"]:
            fs.add("healthy")
        fs.add(f"sex:{e['profile']['sex']}")
        if e["n_status_conflicts"]:
            fs.add("status_conflict")
        if any(i["gold_status"] == "positive" for i in e["items"]):
            fs.add("qual_positive")
        for s in e["scenarios"]:
            fs.add(f"scen:{s}")
        facets.append((e["sample_id"], fs))
    must = {"layout:" + L for L in LAYOUTS} | {"flag:HL", "flag:arrow", "flag:star", "sex_ref_text",
                                                "healthy", "sex:M", "sex:F", "status_conflict", "qual_positive"}
    chosen, covered = [], set()
    while len(chosen) < k:
        best, gain = None, -1
        for sid, fs in facets:
            if sid in chosen:
                continue
            g = 10 * len((fs & must) - covered) + len(fs - covered)
            if g > gain:
                best, gain = sid, g
        chosen.append(best)
        covered |= dict(facets)[best]
    missing = sorted(must - covered)
    return sorted(chosen), sorted(covered), missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--font", default=DEFAULT_FONT, help="TTF/TTC with Traditional Chinese glyphs")
    ap.add_argument("--font-bold", default=DEFAULT_FONT_BOLD)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    from reportlab import rl_config
    rl_config.invariant = 1  # deterministic PDF bytes (no timestamps / random IDs)
    register_fonts(args.font, args.font_bold if Path(args.font_bold).exists() else None)

    out = Path(args.out)
    sdir = out / "samples"
    sdir.mkdir(parents=True, exist_ok=True)
    for old in sdir.glob("syn_*.pdf"):
        old.unlink()
    kb_text = load_kb_ranges()
    layouts, plans = plan(args.n, args.seed)
    entries = []
    for i in range(args.n):
        idx = i + 1
        sid = f"syn_{idx:03d}"
        rep = build_report(idx, sid, layouts[i], plans[i], args.seed, kb_text)
        for r in rep["items"]:
            r["_printed_range"] = printed_range_text(r, rep)
            st = rep["features"]["flag_style"]
            r["_printed_value"] = printed_value_text(r, rep, st if rep["features"]["base_layout"] != "standard"
                                                     and rep["features"]["base_layout"] != "fullwidth" else "none")
        pdf_path = sdir / f"{sid}.pdf"
        if rep["layout"] == "scanned":
            render_scanned_pdf(rep, pdf_path, args.seed)
        else:
            render_text_pdf(rep, pdf_path, rep["layout"])
        e = gt_entry(rep)
        e["file"] = f"{sid}.pdf"
        entries.append(e)
        if not args.quiet:
            print(f"{sid} {rep['layout']:12s} {rep['profile']['sex']} {rep['profile']['age']:2d} "
                  f"{'+'.join(rep['scenarios']):32s} items={e['n_items']:2d} abn={e['n_abnormal']:2d} "
                  f"cond={','.join(rep['conditions'])}")

    stats = compute_stats(entries)
    gt = {
        "_doc": ("FULLY SYNTHETIC evaluation set built by eval/synth/generate.py — do not hand-edit. "
                 "No real patient data. Items/metrics/conditions/risks are gold, derived from the same "
                 "generator state that rendered each PDF. Advice categories are intentionally empty."),
        "generator": {"version": GEN_VERSION, "seed": args.seed, "n": args.n,
                      "kb_source": "item KB range texts (recorded per item as kb_range)"},
        "stats": stats,
        "samples": {e["file"]: e for e in entries},
    }
    (out / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    ids, covered, missing = pick_quick(entries)
    (out / "quick.json").write_text(json.dumps(
        {"_doc": "Stable 10-sample subset covering every layout variant (greedy facet cover, deterministic).",
         "seed": args.seed, "ids": ids, "files": [f"{i}.pdf" for i in ids],
         "facets_covered": covered, "facets_missing": missing}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print("quick:", ids, "missing facets:", missing)


def compute_stats(entries):
    items = [i for e in entries for i in e["items"]]
    ab = [i for i in items if i["gold_status"] in ("high", "low", "positive")]
    scored = [i for i in items if i["gold_status"] != "no_range"]
    return {
        "n_reports": len(entries),
        "layouts": dict(Counter(e["layout"] for e in entries)),
        "primary_scenarios": dict(Counter(e["scenarios"][0] for e in entries)),
        "all_scenario_labels": dict(Counter(s for e in entries for s in e["scenarios"])),
        "sex": dict(Counter(e["profile"]["sex"] for e in entries)),
        "age_bands": dict(sorted(Counter(e["profile"]["age_band"] for e in entries).items())),
        "items_total": len(items),
        "items_per_report": {"min": min(e["n_items"] for e in entries),
                             "max": max(e["n_items"] for e in entries),
                             "mean": round(len(items) / len(entries), 1)},
        "gold_status": dict(Counter(i["gold_status"] for i in items)),
        "abnormal_prevalence_items": round(len(ab) / len(scored), 3),
        "reports_all_normal": sum(1 for e in entries if e["n_abnormal"] == 0),
        "item_origin": dict(Counter(i["origin"] for i in items)),
        "kb_range_conflict_items": sum(1 for i in items if i["kb_range_conflict"]),
        "status_conflict_items": sum(1 for i in items if i["status_conflict"]),
        "sex_ref_text_reports": sum(1 for e in entries if e["layout_features"]["sex_ref_text"]),
        "conditions": dict(Counter(c for e in entries for c in e["expected"]["conditions"]).most_common()),
        "risks": dict(Counter(c for e in entries for c in e["expected"]["risks"]).most_common()),
    }


if __name__ == "__main__":
    main()
