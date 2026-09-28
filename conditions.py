"""
conditions.py — Deterministic lab-finding -> condition / risk rules (R5).

Lab-derived conditions (dyslipidaemia, anaemia, metabolic syndrome, …) follow
published screening criteria, so they are computed by rules rather than asked
of the LLM. The LLM still handles conditions that are only *stated in text*
(imaging / physical-exam findings such as 脂肪肝, 膽囊瘜肉), and every rule tag
carries the evidence that triggered it.

Criteria (Taiwan adult screening conventions):
  * Metabolic syndrome (國健署): >=3 of waist M>=90/F>=80 cm, SBP>=130 or DBP>=85,
    fasting glucose >=100, TG >=150, HDL M<40/F<50.
  * Diabetes: FPG >=126 or HbA1c >=6.5; prediabetes FPG 100–125 or HbA1c 5.7–6.4.
  * Obesity (Taiwan BMI): >=27 obese, 24–27 overweight.
  * CKD: eGFR <60.
These are screening flags, never diagnoses; labels are phrased accordingly.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# Canonical lab keys each rule condition is derived from (mirrors derive() below).
# Used by tag-routed retrieval: a finding may also use passages tagged
# "cond:<label>" when that condition fired and the finding is one of its inputs.
CONDITION_KEYS: Dict[str, tuple] = {
    "血脂異常": ("ldl", "tg", "chol", "hdl"),
    "高血壓": ("sbp", "dbp"),
    "血壓偏高": ("sbp", "dbp"),
    "糖尿病": ("glucose_ac", "hba1c"),
    "糖尿病前期": ("glucose_ac", "hba1c"),
    "代謝症候群": ("waist", "sbp", "dbp", "glucose_ac", "tg", "hdl"),
    "肥胖": ("bmi",),
    "過重": ("bmi",),
    "高尿酸血症": ("uric_acid",),
    "肝功能異常": ("alt", "ast", "ggt"),
    "膽紅素偏高": ("bil_total",),
    "B型肝炎帶原": ("hbsag",),
    "慢性腎臟病": ("egfr",),
    "蛋白尿": ("urine_protein",),
    "尿糖陽性": ("urine_glucose",),
    "尿潛血": ("urine_ob",),
    "小球性貧血": ("hb", "mcv"),
    "貧血": ("hb",),
    "甲狀腺功能低下": ("tsh", "free_t4", "t4"),
    "亞臨床甲狀腺功能低下": ("tsh",),
    "甲狀腺功能亢進": ("tsh", "free_t4", "t4"),
    "腫瘤標記偏高": ("afp", "cea", "ca199", "ca125", "ca153", "psa", "cyfra", "nse"),
}


def _index(findings: List[Dict]) -> Dict[str, Dict]:
    return {f["canonical_key"]: f for f in findings if f.get("canonical_key")}


def _num(f: Optional[Dict]) -> Optional[float]:
    if not f:
        return None
    v = f.get("value")
    return float(v) if isinstance(v, (int, float)) else None


def _abn(f: Optional[Dict], *statuses: str) -> bool:
    return bool(f) and (f.get("status") or f.get("direction")) in statuses


def _ev(*fs: Optional[Dict]) -> str:
    parts = []
    for f in fs:
        if f:
            unit = f" {f['unit']}" if f.get("unit") else ""
            parts.append(f"{f.get('display_name') or f['name']} {f['value']}{unit}")
    return "、".join(parts)


def derive(findings: List[Dict], sex: Optional[str] = None) -> Dict[str, List[Dict]]:
    """Return {"conditions": [tag], "risks": [tag]}; tag = {text, src, conf, evidence}."""
    x = _index(findings)
    conds: List[Dict] = []
    risks: Dict[str, Dict] = {}

    def cond(text, evidence, conf=0.9):
        if evidence and all(c["text"] != text for c in conds):
            conds.append({"text": text, "src": "rule", "conf": conf, "evidence": evidence})

    def risk(text, because, conf=0.7):
        risks.setdefault(text, {"text": text, "src": "rule", "conf": conf, "evidence": because})

    ldl, tg, chol, hdl = x.get("ldl"), x.get("tg"), x.get("chol"), x.get("hdl")
    sbp, dbp, glu, a1c = x.get("sbp"), x.get("dbp"), x.get("glucose_ac"), x.get("hba1c")
    waist, bmi = x.get("waist"), x.get("bmi")
    female = sex == "F"

    # Lipids
    lipid_hits = [f for f, s in ((ldl, "high"), (tg, "high"), (chol, "high"), (hdl, "low")) if _abn(f, s)]
    if lipid_hits:
        cond("血脂異常", _ev(*lipid_hits))
        risk("心血管疾病", "血脂異常")
        risk("動脈粥狀硬化", "血脂異常")

    # Blood pressure
    s_v, d_v = _num(sbp), _num(dbp)
    if (s_v and s_v >= 140) or (d_v and d_v >= 90):
        cond("高血壓", _ev(sbp, dbp), 0.75)
    elif _abn(sbp, "high") or _abn(dbp, "high"):
        cond("血壓偏高", _ev(sbp, dbp), 0.8)
    if any(c["text"] in ("高血壓", "血壓偏高") for c in conds):
        risk("心血管疾病", "血壓偏高")
        risk("腦中風", "血壓偏高")

    # Glycaemia
    g_v, a_v = _num(glu), _num(a1c)
    if (g_v and g_v >= 126) or (a_v and a_v >= 6.5):
        cond("糖尿病", _ev(glu, a1c), 0.75)
        risk("糖尿病併發症", "血糖過高")
        risk("心血管疾病", "血糖過高")
    elif (g_v and g_v >= 100) or (a_v and a_v >= 5.7):
        cond("糖尿病前期", _ev(glu, a1c), 0.85)
        risk("第二型糖尿病", "血糖偏高")

    # Metabolic syndrome (needs values, not just flags)
    w_v, t_v, h_v = _num(waist), _num(tg), _num(hdl)
    ms = []  # one entry per criterion met: the finding(s) that meet it
    if w_v is not None and w_v >= (80 if female else 90):
        ms.append([waist])
    bp_hits = [f for f, v, cut in ((sbp, s_v, 130), (dbp, d_v, 85)) if v and v >= cut]
    if bp_hits:
        ms.append(bp_hits)
    if g_v is not None and g_v >= 100:
        ms.append([glu])
    if t_v is not None and t_v >= 150:
        ms.append([tg])
    if h_v is not None and h_v < (50 if female else 40):
        ms.append([hdl])
    ms_ev = _ev(*(f for crit in ms for f in crit))
    if len(ms) >= 3:
        cond("代謝症候群", f"符合 {len(ms)}/5 項：" + ms_ev, 0.9)
        risk("心血管疾病", "代謝症候群")
        risk("第二型糖尿病", "代謝症候群")
    elif len(ms) == 2:
        risk("代謝症候群", f"已符合 2/5 項：{ms_ev}", 0.6)

    # Body weight
    b_v = _num(bmi)
    if b_v is not None and b_v >= 27:
        cond("肥胖", _ev(bmi))
        risk("代謝症候群", "肥胖")
    elif b_v is not None and b_v >= 24:
        cond("過重", _ev(bmi), 0.85)

    # Uric acid
    if _abn(x.get("uric_acid"), "high"):
        cond("高尿酸血症", _ev(x["uric_acid"]))
        risk("痛風", "尿酸偏高")

    # Liver
    liver = [x.get(k) for k in ("alt", "ast", "ggt") if _abn(x.get(k), "high")]
    if liver:
        cond("肝功能異常", _ev(*liver), 0.85)
        risk("脂肪肝", "肝指數偏高", 0.55)
    if _abn(x.get("bil_total"), "high"):
        cond("膽紅素偏高", _ev(x["bil_total"]), 0.85)
    if _abn(x.get("hbsag"), "positive", "high"):  # numeric index above the cut-off reads as "high"
        cond("B型肝炎帶原", _ev(x["hbsag"]), 0.9)
        risk("肝臟疾病", "B型肝炎帶原")

    # Kidney
    e_v = _num(x.get("egfr"))
    if e_v is not None and e_v < 60:
        cond("慢性腎臟病", _ev(x["egfr"]), 0.8)
        risk("腎衰竭", "腎絲球過濾率下降", 0.6)
    elif _abn(x.get("creatinine"), "high"):
        risk("慢性腎臟病", "肌酸酐偏高", 0.6)
    if _abn(x.get("urine_protein"), "positive", "high"):
        cond("蛋白尿", _ev(x["urine_protein"]), 0.85)
        risk("慢性腎臟病", "蛋白尿", 0.6)
    if _abn(x.get("urine_glucose"), "positive", "high"):
        cond("尿糖陽性", _ev(x["urine_glucose"]), 0.85)
    if _abn(x.get("urine_ob"), "positive", "high"):
        cond("尿潛血", _ev(x["urine_ob"]), 0.8)

    # Blood count
    hb, mcv = x.get("hb"), x.get("mcv")
    if _abn(hb, "low"):
        if _abn(mcv, "low"):
            cond("小球性貧血", _ev(hb, mcv))
            risk("缺鐵性貧血", "小球性貧血", 0.6)
        cond("貧血", _ev(hb))

    # Thyroid
    tsh, ft4 = x.get("tsh"), x.get("free_t4") or x.get("t4")
    if _abn(tsh, "high"):
        if _abn(ft4, "low"):
            cond("甲狀腺功能低下", _ev(tsh, ft4), 0.8)
        else:
            cond("亞臨床甲狀腺功能低下", _ev(tsh), 0.75)
            risk("甲狀腺功能低下", "甲狀腺刺激素偏高", 0.6)
    elif _abn(tsh, "low"):
        cond("甲狀腺功能亢進", _ev(tsh, ft4), 0.7)

    # Tumour markers (screening flag only)
    markers = [x.get(k) for k in ("afp", "cea", "ca199", "ca125", "ca153", "psa", "cyfra", "nse")
               if _abn(x.get(k), "high", "positive")]
    if markers:
        cond("腫瘤標記偏高", _ev(*markers), 0.8)

    # Inflammation / homocysteine
    if _abn(x.get("hs_crp"), "high") or _abn(x.get("homocysteine"), "high"):
        risk("心血管疾病", "發炎指數或同半胱胺酸偏高")

    return {"conditions": conds, "risks": list(risks.values())}


def merge_tags(rule_tags: List[Dict], llm_tags: List[Dict], limit: int = 8) -> List[Dict]:
    """Rule tags first; LLM tags added only when not a near-duplicate of a rule tag."""
    out = list(rule_tags)
    seen = [t["text"] for t in out]
    for t in llm_tags:
        txt = t["text"]
        if any(txt in s or s in txt for s in seen):
            continue
        out.append(t)
        seen.append(txt)
    return out[:limit]
