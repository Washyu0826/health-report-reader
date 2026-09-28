"""Rule-based condition/risk derivation (conditions.py)."""

from conditions import derive, merge_tags


def f(key, value, status, unit=""):
    return {"canonical_key": key, "name": key, "display_name": key,
            "value": value, "status": status, "direction": status, "unit": unit}


def texts(tags):
    return [t["text"] for t in tags]


def test_all_normal_yields_nothing():
    out = derive([f("ldl", 100, "normal"), f("glucose_ac", 90, "normal")])
    assert out == {"conditions": [], "risks": []}


def test_dyslipidaemia_from_low_hdl_only():
    out = derive([f("hdl", 35, "low")], sex="M")
    assert "血脂異常" in texts(out["conditions"])
    assert "心血管疾病" in texts(out["risks"])


def test_metabolic_syndrome_needs_three_criteria_and_is_sex_aware():
    base = [f("tg", 180, "high"), f("glucose_ac", 105, "high"), f("waist", 85, "normal")]
    assert "代謝症候群" not in texts(derive(base, sex="M")["conditions"])   # waist 85 < 90 (M)
    assert "代謝症候群" in texts(derive(base, sex="F")["conditions"])       # waist 85 >= 80 (F)


def test_two_criteria_is_only_a_risk():
    out = derive([f("tg", 180, "high"), f("glucose_ac", 105, "high")], sex="M")
    assert "代謝症候群" not in texts(out["conditions"])
    assert "代謝症候群" in texts(out["risks"])


def test_diabetes_vs_prediabetes_thresholds():
    assert "糖尿病" in texts(derive([f("glucose_ac", 130, "high")])["conditions"])
    pre = texts(derive([f("hba1c", 6.0, "high")])["conditions"])
    assert "糖尿病前期" in pre and "糖尿病" not in pre


def test_bp_grading():
    assert "高血壓" in texts(derive([f("sbp", 150, "high")])["conditions"])
    assert "血壓偏高" in texts(derive([f("sbp", 132, "high")])["conditions"])


def test_microcytic_anaemia_adds_both_labels():
    out = texts(derive([f("hb", 10.5, "low"), f("mcv", 72, "low")], sex="F")["conditions"])
    assert "小球性貧血" in out and "貧血" in out


def test_qualitative_positive_and_tumour_marker():
    out = texts(derive([f("urine_protein", "2+", "positive"), f("cea", 8.2, "high")])["conditions"])
    assert "蛋白尿" in out and "腫瘤標記偏高" in out


def test_evidence_is_attached():
    c = derive([f("uric_acid", 8.1, "high", "mg/dL")])["conditions"][0]
    assert c["src"] == "rule" and "8.1" in c["evidence"]


def test_merge_prefers_rules_and_drops_near_duplicates():
    rules = [{"text": "血脂異常", "src": "rule", "conf": 0.9}]
    llm = [{"text": "血脂異常傾向", "src": "report", "conf": 0.8},
           {"text": "脂肪肝", "src": "report", "conf": 0.9}]
    assert texts(merge_tags(rules, llm)) == ["血脂異常", "脂肪肝"]


def test_metabolic_syndrome_evidence_cites_the_qualifying_bp_value():
    fs = [f("sbp", 120, "normal", "mmHg"), f("dbp", 88, "high", "mmHg"),
          f("glucose_ac", 105, "high"), f("tg", 180, "high")]
    ms = next(c for c in derive(fs, sex="M")["conditions"] if c["text"] == "代謝症候群")
    assert "dbp 88" in ms["evidence"] and "sbp" not in ms["evidence"]


def test_numeric_hbsag_above_cutoff_flags_carrier():
    assert "B型肝炎帶原" in texts(derive([f("hbsag", 250.3, "high")])["conditions"])
