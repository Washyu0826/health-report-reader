"""
catalog.py — Item catalog, scenarios and clinical rules for the SYNTHETIC
checkup-report generator (eval/synth/generate.py).

Everything here is hand-written from general, public clinical knowledge
(Taiwanese checkup panels, 國健署 metabolic-syndrome criteria, CKD-EPI 2021,
WHO anaemia cut-offs). No value, name or row is taken from any real report.

Reference-range spec notation (strings keep the printed decimals):
    ("range", lo, hi)   printed "lo-hi"   inclusive band
    ("upper", hi)       printed "<hi"     v >= hi  -> high
    ("lower", lo)       printed ">lo"     v <= lo  -> low
    {"M": spec, "F": spec}                sex-specific range
"""
from __future__ import annotations


def R(lo, hi):
    return ("range", lo, hi)


def U(hi):
    return ("upper", hi)


def L(lo):
    return ("lower", lo)


def SX(m, f):
    return {"M": m, "F": f}


SECTIONS = [
    ("body", "一般體格檢查"),
    ("cbc", "血液常規檢查"),
    ("liver", "肝膽功能檢查"),
    ("kidney", "腎臟功能檢查"),
    ("lipid", "血脂肪檢查"),
    ("glucose", "血糖檢查"),
    ("thyroid", "甲狀腺功能檢查"),
    ("tumor", "腫瘤標記檢查"),
    ("urine", "尿液常規檢查"),
]

# key: zh (canonical display), en, kb (exact item name in the item KB, the key of
# its generic "KB range" text; or None), sec, names (printed-name variants), unit, dec, quant (rounding step),
# ref (default printed range), alts (lab-specific variants), plaus (physiological
# sampling domain), typ (typical value, may be per sex), sex (eligibility),
# bg (directions allowed for random "background" abnormalities), qual.
ITEMS = {
    # ── body ────────────────────────────────────────────────────────────────
    "height": dict(zh="身高", en="Height", kb="身高（Body height）", sec="body",
                   names=["身高", "身高 Height"], unit="cm", dec=1, ref=None, alts=[],
                   plaus=(140, 195), typ={"M": 171, "F": 159}),
    "weight": dict(zh="體重", en="Weight", kb="體重（Body weight）", sec="body",
                   names=["體重", "體重 Weight"], unit="kg", dec=1, ref=None, alts=[],
                   plaus=(35, 140), typ={"M": 70, "F": 57}),
    "bmi": dict(zh="身體質量指數", en="BMI", kb="身體質量指數BMI（BMI）", sec="body",
                names=["BMI 身體質量指數", "身體質量指數(BMI)", "BMI"], unit="kg/m2", dec=1,
                ref=R("18.5", "24.0"), alts=[R("18.5", "23.9")], plaus=(16, 40), typ=21.8),
    "waist": dict(zh="腰圍", en="Waist", kb="腰圍（Waist girth）", sec="body",
                  names=["腰圍", "腰圍 Waist"], unit="cm", dec=1,
                  ref=SX(U("90"), U("80")), alts=[], plaus=(58, 125), typ={"M": 82, "F": 72}),
    "sbp": dict(zh="收縮壓", en="SBP", kb="收縮壓（Systolic pressure）", sec="body",
                names=["收縮壓", "血壓(收縮壓) SBP", "SBP 收縮壓"], unit="mmHg", dec=0,
                ref=U("130"), alts=[R("90", "139"), R("90", "130")], plaus=(92, 190), typ=116),
    "dbp": dict(zh="舒張壓", en="DBP", kb="舒張壓（Diastolic pressure）", sec="body",
                names=["舒張壓", "血壓(舒張壓) DBP", "DBP 舒張壓"], unit="mmHg", dec=0,
                ref=U("85"), alts=[R("60", "89"), R("60", "85")], plaus=(55, 115), typ=74),
    "pulse": dict(zh="脈搏", en="Pulse", kb="脈搏（Pulse rate）", sec="body",
                  names=["脈搏", "心跳 Pulse"], unit="次/分", dec=0,
                  ref=R("60", "100"), alts=[R("50", "100")], plaus=(48, 130), typ=74, bg=["high"]),
    # ── CBC ─────────────────────────────────────────────────────────────────
    "wbc": dict(zh="白血球", en="WBC", kb="白血球總數（WBC）", sec="cbc",
                names=["白血球 WBC", "WBC 白血球計數", "白血球數"], unit="10^3/uL", dec=1,
                ref=R("3.8", "10.0"), alts=[R("4.0", "10.0"), R("3.5", "10.5")],
                plaus=(2.0, 20.0), typ=6.2, bg=["high", "low"]),
    "rbc": dict(zh="紅血球", en="RBC", kb="紅血球總數（RBC）", sec="cbc",
                names=["紅血球 RBC", "RBC 紅血球計數", "紅血球數"], unit="10^6/uL", dec=2,
                ref=R("4.0", "5.52"), alts=[SX(R("4.5", "5.9"), R("4.0", "5.2"))],
                plaus=(2.5, 7.0), typ={"M": 5.0, "F": 4.5}),
    "hb": dict(zh="血色素", en="Hemoglobin", kb="血色素Hb（Hemoglobin）", sec="cbc",
               names=["血色素 Hb", "Hemoglobin 血紅素", "血紅素 Hgb"], unit="g/dL", dec=1,
               ref=SX(R("13", "18"), R("12", "16")), alts=[SX(R("13.5", "17.5"), R("12.0", "15.5"))],
               plaus=(6.5, 19.5), typ={"M": 15.0, "F": 13.6}),
    "hct": dict(zh="血球容積比", en="Hct", kb="紅血球容積比（Hct）", sec="cbc",
                names=["血球容積比 Hct", "Hematocrit 血比容", "HCT 血容比"], unit="%", dec=1,
                ref=SX(R("40", "54"), R("36", "48")), alts=[SX(R("41", "53"), R("36", "46"))],
                plaus=(20, 60), typ={"M": 45, "F": 41}),
    "mcv": dict(zh="平均紅血球容積", en="MCV", kb="平均血球容積MCV（MCV）", sec="cbc",
                names=["平均紅血球容積 MCV", "MCV"], unit="fL", dec=1,
                ref=R("81", "100"), alts=[R("80", "100")], plaus=(60, 110), typ=89),
    "mch": dict(zh="平均紅血球血紅素量", en="MCH", kb="平均血紅素MCH（MCH）", sec="cbc",
                names=["平均紅血球血紅素量 MCH", "MCH"], unit="pg", dec=1,
                ref=R("27", "32"), alts=[R("27", "33")], plaus=(18, 38), typ=30),
    "mchc": dict(zh="平均紅血球血紅素濃度", en="MCHC", kb="平均血色素濃度MCHC（MCHC）", sec="cbc",
                 names=["平均紅血球血紅素濃度 MCHC", "MCHC"], unit="g/dL", dec=1,
                 ref=R("32", "36"), alts=[R("31.5", "36.0")], plaus=(28, 37.5), typ=34),
    "plt": dict(zh="血小板", en="Platelet", kb="血小板（Platelet）", sec="cbc",
                names=["血小板 PLT", "Platelet 血小板計數", "血小板數"], unit="10^3/uL", dec=0,
                ref=R("140", "450"), alts=[R("150", "400")], plaus=(60, 700), typ=250,
                bg=["high", "low"]),
    # ── liver ───────────────────────────────────────────────────────────────
    "ast": dict(zh="天門冬胺酸轉胺酶", en="AST", kb="草酸轉氨基SGOT（AST(GOT)）", sec="liver",
                names=["AST(GOT) 天門冬胺酸轉胺酶", "SGOT 麩草酸轉胺酶", "GOT"], unit="U/L", dec=0,
                ref=R("13", "39"), alts=[U("40"), R("8", "38")], plaus=(9, 300), typ=24),
    "alt": dict(zh="丙胺酸轉胺酶", en="ALT", kb="丙氨酸轉氨SGPT（ALT(GPT)）", sec="liver",
                names=["ALT(GPT) 丙胺酸轉胺酶", "SGPT 麩丙酮酸轉胺酶", "GPT"], unit="U/L", dec=0,
                ref=R("7", "52"), alts=[U("41"), R("0", "40")], plaus=(6, 400), typ=24),
    "ggt": dict(zh="丙麩胺轉移酶", en="GGT", kb="麩胺轉酸酶GT（r-GT）", sec="liver",
                names=["γ-GT 丙麩胺轉移酶", "r-GT", "GGT 伽瑪麩胺酸轉移酶"], unit="U/L", dec=0,
                ref=R("9", "64"), alts=[SX(R("8", "61"), R("5", "36")), U("73")], plaus=(6, 500), typ=24),
    "alp": dict(zh="鹼性磷酸酶", en="ALP", kb="鹼性磷酸酶（ALP）", sec="liver",
                names=["ALK-P 鹼性磷酸酶", "ALP"], unit="U/L", dec=0,
                ref=R("34", "104"), alts=[R("38", "126")], plaus=(25, 300), typ=68, bg=["high"]),
    "tbil": dict(zh="總膽紅素", en="T-Bil", kb="總膽紅素（Bilirubin）", sec="liver",
                 names=["總膽紅素 T-Bil", "T-Bilirubin 總膽紅素"], unit="mg/dL", dec=1,
                 ref=R("0.3", "1.0"), alts=[R("0.2", "1.2")], plaus=(0.2, 4.0), typ=0.7, bg=["high"]),
    "alb": dict(zh="白蛋白", en="Albumin", kb="白蛋白（Albumin）", sec="liver",
                names=["白蛋白 Albumin", "ALB 白蛋白"], unit="g/dL", dec=1,
                ref=R("3.5", "5.7"), alts=[R("3.5", "5.2")], plaus=(2.5, 5.8), typ=4.5, bg=["low"]),
    "tp": dict(zh="總蛋白", en="Total Protein", kb="總蛋白TP（Total Protein）", sec="liver",
               names=["總蛋白 T-Protein", "TP 總蛋白"], unit="g/dL", dec=1,
               ref=R("6.4", "8.9"), alts=[R("6.0", "8.3")], plaus=(5.0, 9.5), typ=7.4),
    # ── kidney ──────────────────────────────────────────────────────────────
    "bun": dict(zh="尿素氮", en="BUN", kb="血中尿素氮BUN（BUN）", sec="kidney",
                names=["血中尿素氮 BUN", "BUN 尿素氮"], unit="mg/dL", dec=0,
                ref=R("7", "25"), alts=[R("6", "20"), R("8", "23")], plaus=(5, 90), typ=14,
                bg=["high"]),
    "cr": dict(zh="肌酸酐", en="Creatinine", kb="肌酸酐（Creatinine）", sec="kidney",
               names=["肌酸酐 Creatinine", "CRE 肌酸酐", "Creatinine"], unit="mg/dL", dec=2,
               ref=SX(R("0.70", "1.30"), R("0.50", "1.00")), alts=[SX(R("0.60", "1.20"), R("0.50", "0.90"))],
               plaus=(0.4, 6.0), typ={"M": 0.95, "F": 0.75}),
    "egfr": dict(zh="腎絲球過濾率", en="eGFR",
                 kb="腎絲球過濾率（Estimated Glomerular filtration rate(eGFR)）", sec="kidney",
                 names=["腎絲球過濾率 eGFR", "eGFR 估算腎絲球過濾率"], unit="mL/min/1.73m2", dec=1,
                 ref=L("60"), alts=[], plaus=(5, 150), typ=95),
    "ua": dict(zh="尿酸", en="Uric Acid", kb="尿酸（Uric acid）", sec="kidney",
               names=["尿酸 Uric Acid", "UA 尿酸"], unit="mg/dL", dec=1,
               ref=SX(R("4.4", "7.6"), R("2.3", "6.6")), alts=[SX(R("3.5", "7.2"), R("2.6", "6.0")), R("3.0", "7.0")],
               plaus=(2.0, 13.0), typ={"M": 6.0, "F": 5.0}),
    # ── lipid ───────────────────────────────────────────────────────────────
    "tc": dict(zh="總膽固醇", en="Total Cholesterol", kb="總膽固醇CHOL（Cholesterol, total）", sec="lipid",
               names=["總膽固醇 T-Cholesterol", "Cholesterol 總膽固醇", "T-CHO 總膽固醇"], unit="mg/dL", dec=0,
               ref=U("200"), alts=[R("120", "200")], plaus=(100, 380), typ=180),
    "tg": dict(zh="三酸甘油酯", en="Triglyceride", kb="三酸甘油脂（Triglyceride）", sec="lipid",
               names=["三酸甘油酯 TG", "Triglyceride 三酸甘油脂", "TG 中性脂肪"], unit="mg/dL", dec=0,
               ref=U("150"), alts=[R("35", "150"), R("0", "149")], plaus=(35, 900), typ=95),
    "hdl": dict(zh="高密度脂蛋白膽固醇", en="HDL-C", kb="高密度脂蛋白HDL（HDL-C）", sec="lipid",
                names=["高密度脂蛋白膽固醇 HDL-C", "HDL 好膽固醇", "HDL-Cholesterol"], unit="mg/dL", dec=0,
                ref=SX(L("40"), L("50")), alts=[L("40"), L("35")], plaus=(22, 110), typ={"M": 50, "F": 62}),
    "ldl": dict(zh="低密度脂蛋白膽固醇", en="LDL-C", kb="低密度脂蛋白LDL（LDL-C）", sec="lipid",
                names=["低密度脂蛋白膽固醇 LDL-C", "LDL 壞膽固醇", "LDL-Cholesterol"], unit="mg/dL", dec=0,
                ref=U("130"), alts=[U("100"), R("0", "129")], plaus=(40, 260), typ=105),
    # ── glucose ─────────────────────────────────────────────────────────────
    "glu": dict(zh="飯前血糖", en="Glucose AC", kb="飯前血糖（Glucose, AC）", sec="glucose",
                names=["飯前血糖 Glucose AC", "空腹血糖 FBS", "AC Glucose 飯前血糖"], unit="mg/dL", dec=0,
                ref=R("70", "100"), alts=[R("70", "99"), R("74", "106")], plaus=(60, 400), typ=90),
    "hba1c": dict(zh="醣化血色素", en="HbA1c", kb="醣化血色素HbA1c（HbA1c）", sec="glucose",
                  names=["醣化血色素 HbA1c", "HbA1c 糖化血色素"], unit="%", dec=1,
                  ref=R("4.0", "6.0"), alts=[R("4.0", "5.6"), R("4.3", "5.8")], plaus=(4.0, 13.0), typ=5.3),
    # ── thyroid ─────────────────────────────────────────────────────────────
    "tsh": dict(zh="甲狀腺刺激素", en="TSH", kb="甲狀腺刺激素TSH（Thyroid-stimulating hormone ( TSH )）",
                sec="thyroid", names=["甲狀腺刺激素 TSH", "TSH"], unit="uIU/mL", dec=2,
                ref=R("0.27", "4.20"), alts=[R("0.35", "4.94"), R("0.40", "4.00")], plaus=(0.05, 40), typ=1.8),
    "ft4": dict(zh="游離甲狀腺素", en="Free T4", kb="游離四碘甲狀腺素（Free-T4）", sec="thyroid",
                names=["游離甲狀腺素 Free T4", "FT4 游離四碘甲狀腺素"], unit="ng/dL", dec=2,
                ref=R("0.93", "1.70"), alts=[R("0.70", "1.48"), R("0.89", "1.76")], plaus=(0.3, 3.0), typ=1.2),
    # ── tumor markers ───────────────────────────────────────────────────────
    "afp": dict(zh="甲型胎兒蛋白", en="AFP", kb="胎兒蛋白AFP（a-Fetoprotein）", sec="tumor",
                names=["甲型胎兒蛋白 AFP", "AFP 胎兒蛋白"], unit="ng/mL", dec=1,
                ref=R("0", "9.0"), alts=[U("8.8")], plaus=(0.5, 400), typ=3.0),
    "cea": dict(zh="癌胚抗原", en="CEA", kb="癌胚胎抗原CEA（CEA）", sec="tumor",
                names=["癌胚抗原 CEA", "CEA"], unit="ng/mL", dec=1,
                ref=R("0", "5.0"), alts=[U("5.0"), U("3.4")], plaus=(0.3, 60), typ=1.8),
    "ca199": dict(zh="CA19-9", en="CA19-9", kb="CA199胰臟癌（CA19-9）", sec="tumor",
                  names=["CA19-9 消化道腫瘤標記", "CA 19-9"], unit="U/mL", dec=1,
                  ref=U("27"), alts=[U("37")], plaus=(1, 500), typ=9.0),
    "ca125": dict(zh="CA-125", en="CA-125", kb="CA125卵巢癌（CA-125）", sec="tumor",
                  names=["CA-125 卵巢腫瘤標記", "CA125"], unit="U/mL", dec=1,
                  ref=U("35"), alts=[], plaus=(2, 300), typ=12.0, sex="F"),
    "ca153": dict(zh="CA15-3", en="CA15-3", kb="CA-153乳房癌（CA-153）", sec="tumor",
                  names=["CA15-3 乳房腫瘤標記", "CA 15-3"], unit="U/mL", dec=1,
                  ref=U("25"), alts=[U("31.3")], plaus=(2, 200), typ=10.0, sex="F"),
    "psa": dict(zh="攝護腺特異抗原", en="PSA", kb="攝護腺特異抗原PSA（PSA）", sec="tumor",
                names=["攝護腺特異抗原 PSA", "PSA"], unit="ng/mL", dec=2,
                ref=R("0", "4.0"), alts=[U("4.0")], plaus=(0.1, 50), typ=1.0, sex="M"),
    # ── urine ───────────────────────────────────────────────────────────────
    "u_ph": dict(zh="尿液酸鹼值", en="Urine pH", kb="尿液酸鹼值（pH）", sec="urine",
                 names=["尿酸鹼值 pH", "pH 酸鹼度"], unit="", dec=1, quant=0.5,
                 ref=R("4.5", "8.0"), alts=[R("5.0", "8.0")], plaus=(4.5, 9.0), typ=6.0),
    "u_sg": dict(zh="尿比重", en="Specific Gravity", kb="尿液比重（Specific gravity）", sec="urine",
                 names=["尿比重 S.G.", "Specific Gravity 比重"], unit="", dec=3, quant=0.005,
                 ref=R("1.002", "1.025"), alts=[R("1.003", "1.030"), R("1.005", "1.030")],
                 plaus=(1.000, 1.040), typ=1.015),
    "u_ubg": dict(zh="尿膽素原", en="Urobilinogen", kb="尿膽素原（Urobilinogen）", sec="urine",
                  names=["尿膽素原 URO", "Urobilinogen 尿膽素原"], unit="EU/dL", dec=1,
                  ref=R("0.1", "1.0"), alts=[R("0.2", "1.0")], plaus=(0.1, 4.0), typ=0.3),
    "u_pro": dict(zh="尿蛋白", en="Urine Protein", kb=None, sec="urine", qual=True,
                  names=["尿蛋白 Protein", "Protein 尿蛋白質"], unit="", dec=0, ref="QUAL", alts=[]),
    "u_glu": dict(zh="尿糖", en="Urine Glucose", kb=None, sec="urine", qual=True,
                  names=["尿糖 Glucose", "Urine Glucose 尿糖"], unit="", dec=0, ref="QUAL", alts=[]),
    "u_ob": dict(zh="尿潛血", en="Occult Blood", kb=None, sec="urine", qual=True,
                 names=["尿潛血 OB", "Occult Blood 潛血"], unit="", dec=0, ref="QUAL", alts=[],
                 bg=["positive"]),
    "u_ket": dict(zh="尿酮體", en="Ketone", kb=None, sec="urine", qual=True,
                  names=["尿酮體 Ketone", "Ketone 酮體"], unit="", dec=0, ref="QUAL", alts=[]),
}

# Extra match aliases for metric_keys (lower-cased on use)
EXTRA_ALIASES = {
    "ast": ["got", "sgot", "ast"], "alt": ["gpt", "sgpt", "alt"], "ggt": ["ggt", "r-gt", "γ-gt", "gamma-gt"],
    "tc": ["總膽固醇", "cholesterol", "t-cho"], "tg": ["三酸甘油脂", "三酸甘油酯", "tg", "triglyceride"],
    "hdl": ["hdl", "hdl-c", "高密度"], "ldl": ["ldl", "ldl-c", "低密度"],
    "glu": ["血糖", "glucose", "ac sugar", "fbs"], "hba1c": ["hba1c", "a1c", "醣化血色素", "糖化血色素"],
    "hb": ["hb", "hgb", "hemoglobin", "血色素", "血紅素"], "cr": ["creatinine", "cre", "肌酸酐"],
    "egfr": ["egfr", "腎絲球過濾率"], "ua": ["uric acid", "尿酸"], "waist": ["腰圍", "waist"],
    "sbp": ["收縮壓", "sbp", "血壓"], "dbp": ["舒張壓", "dbp", "血壓"], "bmi": ["bmi", "身體質量指數"],
    "tsh": ["tsh", "甲狀腺刺激素"], "ft4": ["ft4", "free t4", "游離甲狀腺素"],
    "u_pro": ["尿蛋白", "protein", "蛋白尿"], "u_glu": ["尿糖", "urine glucose"],
    "u_ob": ["潛血", "occult blood", "尿潛血"], "u_ket": ["酮體", "ketone"],
}

QUAL_STYLES = ["陰性", "-", "Negative"]

# Clinical (guideline) cut-offs, independent of the printed range.
# (lo, hi, lo_strict, hi_strict): value < lo (<= if lo_strict) -> low;
# value > hi (>= if hi_strict) -> high.
CLIN = {
    "waist": {"M": (None, 90, False, True), "F": (None, 80, False, True)},
    "sbp": (None, 130, False, True),
    "dbp": (None, 85, False, True),
    "glu": (70, 100, False, True),
    "hba1c": (None, 5.7, False, True),
    "tg": (None, 150, False, True),
    "tc": (None, 200, False, True),
    "ldl": (None, 130, False, True),
    "hdl": {"M": (40, None, False, False), "F": (50, None, False, False)},
    "ua": (None, 7.0, False, False),
    "bmi": (18.5, 24.0, False, True),
    "egfr": (60, None, False, False),
    "hb": {"M": (13, None, False, False), "F": (12, None, False, False)},
}

# Panels
CORE = ["height", "weight", "bmi", "waist", "sbp", "dbp", "wbc", "rbc", "hb", "plt",
        "ast", "alt", "cr", "egfr", "ua", "tc", "tg", "hdl", "ldl", "glu",
        "u_ph", "u_sg", "u_pro", "u_glu", "u_ob"]
OPTIONAL = {"pulse": .5, "hct": .8, "mcv": .8, "mch": .7, "mchc": .7, "ggt": .6, "alp": .5,
            "tbil": .5, "alb": .4, "tp": .4, "bun": .8, "hba1c": .5, "afp": .45, "cea": .45,
            "ca199": .3, "psa": .5, "ca125": .35, "ca153": .3, "u_ket": .5, "u_ubg": .5}
PAIRS = {"tsh": ("ft4", .35)}  # optional pair (both or neither)

AGE_BANDS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]

# Scenario definitions. `required` items are forced into the panel.
# `p_female` is the probability the patient is female; `ages` the allowed bands.
SCENARIOS = {
    "healthy": dict(required=[], p_female=.5, ages=AGE_BANDS[:3]),
    "metabolic_syndrome": dict(required=["waist", "sbp", "dbp", "glu", "tg", "hdl", "bmi"],
                               p_female=.4, ages=AGE_BANDS[2:]),
    "fatty_liver": dict(required=["ast", "alt", "ggt", "tg", "bmi"], p_female=.3, ages=AGE_BANDS[1:5]),
    "ckd": dict(required=["bun", "cr", "egfr", "u_pro", "ua", "hb"], p_female=.4, ages=AGE_BANDS[3:]),
    "anemia": dict(required=["hb", "hct", "mcv", "mch", "mchc", "rbc"], p_female=.7, ages=AGE_BANDS),
    "hyperuricemia": dict(required=["ua", "cr"], p_female=.25, ages=AGE_BANDS[1:]),
    "hypothyroid": dict(required=["tsh", "ft4", "tc", "ldl"], p_female=.8, ages=AGE_BANDS[1:]),
    "diabetes": dict(required=["glu", "hba1c", "u_glu"], p_female=.45, ages=AGE_BANDS[2:]),
    "dyslipidemia": dict(required=["tc", "tg", "hdl", "ldl"], p_female=.45, ages=AGE_BANDS[1:]),
    "hypertension": dict(required=["sbp", "dbp"], p_female=.4, ages=AGE_BANDS[3:]),
    "tumor_marker": dict(required=["afp", "cea"], p_female=.5, ages=AGE_BANDS[3:]),
}
ABNORMAL_SCENARIOS = [s for s in SCENARIOS if s != "healthy"]
# second scenario compatibility (secondary drawn from this list for a primary)
SECONDARY = {
    "metabolic_syndrome": ["fatty_liver", "hyperuricemia", "dyslipidemia", "diabetes"],
    "fatty_liver": ["hyperuricemia", "dyslipidemia", "metabolic_syndrome"],
    "ckd": ["anemia", "hypertension", "hyperuricemia"],
    "anemia": ["hypothyroid"],
    "hyperuricemia": ["hypertension", "fatty_liver"],
    "hypothyroid": ["dyslipidemia", "anemia"],
    "diabetes": ["hypertension", "dyslipidemia"],
    "dyslipidemia": ["hypertension", "hyperuricemia"],
    "hypertension": ["dyslipidemia", "hyperuricemia"],
    "tumor_marker": ["hypertension"],
}
# the condition each scenario must produce (checked after generation)
SCENARIO_EXPECTS = {
    "metabolic_syndrome": ["代謝症候群"],
    "fatty_liver": ["肝功能異常"],
    "ckd": ["慢性腎臟病"],
    "anemia": ["貧血"],
    "hyperuricemia": ["高尿酸血症"],
    "hypothyroid": ["甲狀腺功能低下", "亞臨床甲狀腺功能低下"],  # any of
    "diabetes": ["糖尿病", "糖尿病前期"],  # any of
    "dyslipidemia": ["血脂異常"],
    "hypertension": ["高血壓"],
    "tumor_marker": ["腫瘤標記偏高"],
}

CONDITION_SYNONYMS = {
    "代謝症候群": ["metabolic syndrome", "代謝症"],
    "高血壓": ["hypertension", "血壓過高"],
    "血壓偏高": ["高血壓前期", "elevated blood pressure", "血壓偏高"],
    "糖尿病": ["diabetes", "血糖過高"],
    "糖尿病前期": ["prediabetes", "空腹血糖偏高", "葡萄糖耐受不良", "血糖偏高"],
    "血脂異常": ["高血脂", "dyslipidemia", "hyperlipidemia", "高膽固醇"],
    "肝功能異常": ["肝指數偏高", "轉胺酶偏高", "肝功能指數異常"],
    "慢性腎臟病": ["腎功能異常", "ckd", "腎功能不全", "腎功能下降"],
    "蛋白尿": ["proteinuria", "尿蛋白"],
    "貧血": ["anemia"],
    "小球性貧血": ["microcytic anemia", "小細胞性貧血"],
    "高尿酸血症": ["尿酸過高", "hyperuricemia", "尿酸偏高"],
    "甲狀腺功能低下": ["hypothyroidism", "甲狀腺低下"],
    "亞臨床甲狀腺功能低下": ["subclinical hypothyroidism", "甲狀腺功能低下"],
    "過重": ["overweight", "體重過重"],
    "肥胖": ["obesity"],
    "腫瘤標記偏高": ["腫瘤指數偏高", "tumor marker"],
    "心血管疾病": ["心臟病", "cardiovascular", "冠心病", "心血管"],
    "第二型糖尿病": ["糖尿病", "type 2 diabetes"],
    "腦中風": ["中風", "stroke"],
    "動脈粥狀硬化": ["動脈硬化", "atherosclerosis"],
    "脂肪肝": ["fatty liver", "nafld", "mafld"],
    "肝臟疾病": ["肝病", "肝炎", "liver disease"],
    "腎衰竭": ["末期腎臟病", "腎功能惡化", "洗腎"],
    "痛風": ["gout"],
    "缺鐵性貧血": ["iron deficiency", "缺鐵"],
    "糖尿病併發症": ["糖尿病腎病變", "視網膜病變"],
}

FAKE_NAME_PREFIX = ["測試員", "範例人", "虛擬者", "模擬者", "假名者", "示範員"]
FAKE_NAME_STEM = list("甲乙丙丁戊己庚辛壬癸")
