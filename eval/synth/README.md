# Synthetic checkup-report evaluation set

A **fully synthetic** evaluation set for the health-report tagger: 60
Traditional-Chinese, Taiwanese-style health-checkup PDFs, each with gold
ground truth. It is safe to publish.

* **No real patient data.** Every value is drawn from a seeded random generator
  centred on public reference ranges. Nothing is read from real reports.
* **Fake identity headers only.** Names look like `測試員 甲`, IDs like
  `A000000001`, chart numbers like `SYN-000001`, addresses like
  `虛構市測試區範例路1號` and phone numbers like `0900-000-001`. Every page
  footer says `SYNTHETIC TEST DATA`.
* **One extra input.** A generic knowledge-base (KB) reference range per
  catalogue item. Each item records its KB range (`kb_range`) next to the range
  printed on the report; in a public checkout `generate.py` reads these texts
  back from `ground_truth.json`, so regeneration is still byte-identical. (The
  app itself judges values against `data/reference_ranges.json`, not these.)
* **Deterministic output.** The same `--n` and `--seed` give byte-identical PDFs
  and JSON (reportlab `invariant` mode, string-seeded RNGs).

## Files

| File | Purpose |
|---|---|
| `generate.py` | Generator CLI: `--n` (default 60), `--seed` (default 20260925), `--out` (default `eval/synth`), `--font` (a TTF/TTC with CJK glyphs; default Microsoft JhengHei). |
| `catalog.py` | Item catalogue (47 analytes), lab-specific alternative ranges, clinical cut-offs, scenarios and condition synonyms. |
| `samples/syn_NNN.pdf` | The rendered reports. |
| `ground_truth.json` | Gold labels. Its top-level `samples` map is the layout `eval/run_eval.py` reads. |
| `quick.json` | A stable 10-sample subset that covers every layout variant. |
| `verify.py` | Checks the dataset: text layers, image-only PDFs, cross-check with `eval/gt_rules.py`, and determinism. |
| `check_no_real_data.py` | Leak check: a brand denylist plus hashed row comparison against the private source CSV (skipped when that CSV is absent). |

```bash
.venv/Scripts/python eval/synth/generate.py            # build (≈15 s)
.venv/Scripts/python eval/synth/verify.py              # verify (regenerates to a temp dir)
.venv/Scripts/python eval/synth/check_no_real_data.py  # leak check
# run the tagger harness on it
.venv/Scripts/python eval/run_eval.py --samples eval/synth/samples --gt eval/synth/ground_truth.json
```

## Generator

For each report, the generator does the following:

1. **Plan.** It assigns a layout and a primary scenario (plus an optional,
   compatible secondary scenario, p = 0.35) from a seeded shuffle. Healthy
   reports make up about 10% of the set.
2. **Profile.** It draws sex and age band from the scenario's priors. For
   example, hypothyroid is 80% female, CKD is age 50+, and fatty liver is 70%
   male.
3. **Panel.** It selects 20–45 items. The core panel is body measurements
   (height, weight, BMI, waist, BP), complete blood count (CBC), AST/ALT,
   creatinine/eGFR/uric acid, the four lipids, fasting glucose and the urine
   dipstick. It then adds optional items (MCV/MCH/MCHC/Hct, GGT, ALP, bilirubin,
   albumin, total protein, BUN, HbA1c, TSH+FT4, AFP, CEA, CA19-9, sex-specific
   PSA/CA-125/CA15-3, urine ketone and urobilinogen) and the items each scenario
   requires. About 25% of reports use a slimmer "basic" panel.
4. **Printed ranges.** Each report has its own rate (10–45%) of swapping an
   item's default range for a realistic lab-specific alternative, for example
   ALT `<41` or `0-40` instead of the KB's `7-52`, or HbA1c `4.0-5.6`.
   Sex-specific items (Hb, Hct, creatinine, uric acid, HDL, waist) print the
   range for the patient's sex. With the `sex_ref_text` feature they print both
   sexes instead, e.g. `男:13-18 女:12-16`.
5. **Values.** Each item gets a *want*: `normal`, `high`, `low`, `positive`,
   `any` or `probe`. Values are then sampled, with correlated physiology:
   * weight = BMI × height², and waist depends on BMI (sex-specific slope);
   * Hct = Hb / MCHC, RBC = Hct / MCV, MCH = Hb / RBC, so the CBC stays
     internally consistent;
   * TC = LDL + HDL + TG/5 (Friedewald);
   * eGFR comes from creatinine, age and sex (CKD-EPI 2021);
   * urine glucose is positive when fasting glucose is 180 or higher.

   A `normal` value must be normal under the printed range, the KB range *and*
   the clinical cut-off. An abnormal want must be abnormal under the printed
   range and the clinical cut-off. The generator uses per-group rejection
   sampling. It also moves values off every range boundary, so the gold labels
   never depend on whether a bound is inclusive.
6. **Background noise** (non-healthy reports only):
   * About 5% of the items that don't drive a condition get a mild abnormality
     (WBC, PLT, ALP, bilirubin, albumin, BUN, pulse, urine occult blood).
   * About 22% of items whose printed range differs from the KB range become
     **KB probes**. Their value is sampled inside the gap where the printed-range
     status and the KB status disagree.
7. **Rules.** Conditions and risks come from the final values through explicit
   rules (see below). The generator asserts that each scenario produces its
   expected condition and that healthy reports are entirely normal.
8. **Rendering.** The same report dict is rendered to PDF and written to
   `ground_truth.json`.

### Scenarios

| Scenario | What is generated |
|---|---|
| `healthy` | Every item normal under the printed range, the KB range and the clinical cut-offs. No KB probes, no background noise. |
| `metabolic_syndrome` | 3–5 of the 5 國健署 criteria are abnormal (waist, BP, glucose, TG, HDL); BMI rises with waist. |
| `fatty_liver` | ALT high; AST (p 0.7) and GGT (p 0.6) high; BMI 25–31; TG high (p 0.5). |
| `ckd` | Creatinine high, so eGFR < 60 (age 50+); BUN high; urine protein `+`/`2+`/`3+`. Hyperuricaemia (p 0.4) and renal anaemia (p 0.35) are optional. |
| `anemia` | Hb below the sex-specific cut-off. Iron-deficiency pattern with low MCV and low MCHC (p 0.7). |
| `hyperuricemia` | Uric acid above 7.0 and above the printed upper bound. |
| `hypothyroid` | TSH high; alternates between overt (FT4 low) and subclinical. LDL high (p 0.4). |
| `diabetes` | Alternates between prediabetes (glucose 101–125, HbA1c 5.7–6.4) and diabetes (glucose ≥ 130, HbA1c ≥ 6.6, urine glucose `+` when glucose ≥ 180, urine protein p 0.35). |
| `dyslipidemia` | LDL 140–212 (so TC is usually high); TG high (p 0.4); HDL low (p 0.3). |
| `hypertension` | SBP 142–172; DBP ≥ 91 (p 0.7). |
| `tumor_marker` | One of CEA, AFP, CA19-9, PSA (men 50+) or CA-125 (women) mildly raised. |

### Layout variants (`layout_features` in the ground truth)

| Layout | What it stresses |
|---|---|
| `standard` | A single grid table per section: 檢查項目 / 結果 / 單位 / 參考值 / 判定. The 判定 column uses `H`/`L` or `偏高`/`偏低`/`異常`. |
| `twocol` | Two item tables side by side per section. The reference cell includes the unit. Values optionally carry `H`/`L`. |
| `inline_flags` | The flag is inside the value cell: `165 H`, `165 ↑` or `165*` (`flag_style` = `HL`, `arrow` or `star`). |
| `fullwidth` | Full-width digits and symbols in values and ranges: `０．８５`, `１８．５～２４．０`, `＜２００`, `＞６０`, `男：…　女：…`. |
| `plain` | No grid. One line per item, such as `＊SBP 收縮壓： 156↑ mmHg 參考值 90～139`. |
| `scanned` | Image-only (3 PDFs). A `standard` or `twocol` page is rasterised at 150 dpi, rotated 0.6–1.6°, given Gaussian noise, specks and a slight blur, then embedded as an image. There is **no text layer**; use it for the OCR round. |

Every layout also has:

* one-sided ranges (`<200`, `>40`, `>60`);
* qualitative urine results (`陰性` / `-` / `Negative`, `+`, `2+`, `3+`, `陽性`);
* section headers (`【血脂肪檢查】` …);
* the fake identity header block (for de-identification tests).

`sex_ref_text` appears on about half the reports, across all layouts.

## Ground-truth schema (`ground_truth.json`)

```text
{
  "_doc", "generator": {version, seed, n, kb_source}, "stats": {...},
  "samples": {
    "syn_001.pdf": {
      sample_id, file, layout,
      layout_features: {layout, image_only, base_layout, sex_ref_text, fullwidth,
                        flag_style, judgement_col, range_sep, qual_style,
                        [scan_dpi, scan_rotation_deg, pages]},
      profile: {sex: "M"|"F", age, age_band},
      scenarios: [primary, (secondary)], scenario_meta,
      fake_pii: {name, id_no, chart_no, birth_date, exam_date, phone, address, ...},
      expected: {conditions: [...], risks: [...], metrics: ["尿酸 6.8 mg/dL ↑", ...],
                 lifestyle: [], food: [], exercise: [], supplements: [], avoid: []},
      condition_evidence: {condition: "rule + values that fired it"},
      synonyms: {tag: [accepted synonyms]},
      metric_keys: {metric tag: {aliases, values, numeric}},   # run_eval.py name+value matcher
      abnormal_findings: [{name, canonical_name, value, unit, reference, direction, gold_status}],
      n_items, n_abnormal, n_kb_range_conflicts, n_status_conflicts,
      label_quality: {...},
      items: [{
        name,                  # = printed_name (run_eval.py compatibility)
        key, canonical_name, canonical_en, kb_name, section,
        printed_name, printed_value,     # exactly as rendered (flags, full-width)
        value, unit,                     # canonical half-width value string
        printed_range,                   # exactly as rendered (e.g. "男:13-18 女:12-16", "＜２００")
        sex_specific_range, sex_applicable_range, reference,
        kb_range, kb_status, kb_range_conflict, kb_conflict_detail, kb_one_sided_bounds,
        status_conflict,                 # printed-range status != KB status
        gold_status,   # high | low | normal | positive | negative | no_range
        direction,     # high | low | normal | unknown  (positive -> high; run_eval.py)
        origin         # normal | scenario | derived | background | kb_probe
      }]
    }
  }
}
```

**Gold status.** Status is judged against the **report-printed range for the
patient's sex**, which is what a reader of the report sees:

* `lo-hi` is inclusive;
* `<x` means a value of x or more is high;
* `>x` means a value of x or less is low.

These rules agree with `eval/gt_rules.classify` on 100% of items (checked by
`verify.py`).

**KB conflicts.** `kb_range_conflict` is true when the printed range and the KB
range both state a bound on the same side and the bounds differ. A lower bound
of 0 counts as "no bound". A side that only one of the two ranges states is
listed in `kb_one_sided_bounds` and is not counted as a conflict (for example,
printed `<130` against KB `90～130`). `status_conflict` is true when the value's
status differs between the two ranges. Healthy reports never have status
conflicts.

**Conditions and risks.** These are exact rules on the generated values. Items
missing from the panel are not evaluated.

| Condition | Rule | Risks added |
|---|---|---|
| 代謝症候群 | ≥ 3 of: waist ≥ 90 (M) / ≥ 80 (F); SBP ≥ 130 or DBP ≥ 85; glucose ≥ 100; TG ≥ 150; HDL < 40 (M) / < 50 (F) (國健署) | 心血管疾病, 第二型糖尿病 |
| 高血壓 | SBP ≥ 140 or DBP ≥ 90 | 心血管疾病, 腦中風 |
| 血壓偏高 | Otherwise, SBP ≥ 130 or DBP ≥ 85 | — |
| 糖尿病 | Glucose ≥ 126 or HbA1c ≥ 6.5 | 糖尿病併發症, 心血管疾病 |
| 糖尿病前期 | Otherwise, glucose ≥ 100 or HbA1c ≥ 5.7 | 第二型糖尿病 |
| 血脂異常 | TC ≥ 200, LDL ≥ 130, TG ≥ 150 or low HDL | 動脈粥狀硬化, 心血管疾病 |
| 肝功能異常 | ALT or AST above the printed range | 脂肪肝 if BMI ≥ 24, TG ≥ 150 or waist criterion met; otherwise 肝臟疾病 |
| 慢性腎臟病 | eGFR < 60 | 腎衰竭, 心血管疾病 |
| 蛋白尿 | Urine protein positive | 腎衰竭 (with CKD); otherwise 慢性腎臟病 |
| 貧血 | Hb < 13 (M) / < 12 (F) (WHO) | — |
| 小球性貧血 | Anaemia and MCV < 80 | 缺鐵性貧血 |
| 高尿酸血症 | Uric acid > 7.0 | 痛風 |
| 甲狀腺功能低下 | TSH high and FT4 low (printed ranges) | — |
| 亞臨床甲狀腺功能低下 | TSH high and FT4 not low | 甲狀腺功能低下 |
| 過重 | 24 ≤ BMI < 27 | 代謝症候群 (if not already a condition) |
| 肥胖 | BMI ≥ 27 | 代謝症候群 (if not already a condition) |
| 腫瘤標記偏高 | Any tumour marker above the printed range | — |

**Metrics.** Every item whose gold status is high, low or positive becomes a
metric tag, formatted `canonical name + value + unit + ↑/↓`. Positive
qualitative results take `↑`.

**Advice categories.** `lifestyle`, `food`, `exercise`, `supplements` and
`avoid` are deliberately empty. Grade them with citation validity and an LLM
judge.

## Quick subset

`quick.json` lists 10 sample ids, chosen by a deterministic greedy cover. Together
they include:

* all six layouts;
* all three inline flag styles;
* `sex_ref_text`;
* a healthy report;
* both sexes;
* a printed-vs-KB status conflict;
* a qualitative positive;
* every scenario.

## Verification (current build)

* **Text layer.** All 57 text PDFs contain every item value in their pdfplumber
  text layer (NFKC-normalised), plus the fake name, ID and date. The 3 scanned
  PDFs have 0 text characters.
* **Rule agreement.** Gold statuses agree with `eval/gt_rules.classify` on
  100% of scored items.
* **Determinism.** Regenerating the set gives byte-identical output.
* **Leak check** (`check_no_real_data.py`):
  * no denylisted brand string appears anywhere under `eval/synth/`;
  * no non-generic (name, value, reference) row equals a row of the private
    source table;
  * (name, value) collisions are at chance level, measured against a
    shifted-value null model (they involve low-cardinality values such as
    integer blood pressures);
  * no synthetic report shares more than 20% of its pairs with any real report
    (the observed maximum is about 4%).
