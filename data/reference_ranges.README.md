# reference_ranges.json

Public, redistributable adult reference ranges for the canonical lab items defined in
`lab_synonyms.json`. Every value was compiled from publicly available sources (listed below)
and written for this project; nothing here is copied from any proprietary dataset.

## What the numbers mean

- These are **indicative screening intervals** for non-pregnant adults. They are meant to help
  the demo flag values that look out of range. They are not diagnostic thresholds.
- Lab reference intervals depend on the instrument, reagent, method and population, so they
  differ between hospitals. Where no national standard exists, we used the interval
  published by National Taiwan University Hospital (NTUH) and say so in `note`.
- **The reference range printed on the user's own report always takes precedence.** The app
  falls back to this file only when the report shows no range.
- For items where a Taiwanese guideline sets a public-health cut-off (blood pressure, fasting
  glucose, HbA1c, lipids, waist, BMI), we used that cut-off rather than a lab interval.

## Schema

`items[]` entries:

| field | meaning |
|---|---|
| `key`, `zh`, `en`, `unit` | canonical id, display names and unit (same as `lab_synonyms.json`) |
| `low`, `high` | bounds in `unit`; `null` means one-sided. Bounds are inclusive by default |
| `low_exclusive` / `high_exclusive` | if `true`, the bound value itself is already abnormal (e.g. glucose `high: 100` with `high_exclusive`, so 100 counts as elevated) |
| `sex_specific` | `{"M": [lo, hi], "F": [lo, hi]}`; use it instead of `low`/`high` when sex is known |
| `qualitative_normal` | expected qualitative result (e.g. `陰性`) for dipstick, serology and screening tests |
| `source` | short citation and URL |
| `note` | caveats, guideline staging and alternative lab intervals |

`skipped[]` lists canonical keys with no standard range (height, body-composition device
outputs, visual acuity, intraocular pressure, audiometry, computed risk scores), each with a
reason.

## Main sources

- Health Promotion Administration (國民健康署), Ministry of Health and Welfare, Taiwan:
  adult metabolic syndrome criteria, the healthy body-weight standard (BMI), obesity measures
  (waist, body fat), and the colorectal cancer (FIT) screening page — https://www.hpa.gov.tw
- Taiwan Society of Cardiology / Taiwan Hypertension Society, 2022 Hypertension Guideline
  (hypertension defined as ≥130/80 mmHg).
- Diabetes Association of the Republic of China (Taiwan), Clinical Practice Guideline, and
  ADA Standards of Care (prediabetes 100–125 mg/dL or HbA1c 5.7–6.4%).
- National Taiwan University Hospital health-education page "檢驗參考值" —
  https://health.ntuh.gov.tw/health/hrc_v3/DataFiles/kensa.htm
- KDIGO CKD guideline (eGFR categories) — https://kdigo.org
- WHO guideline on haemoglobin cut-offs for anaemia (2024).
- WHO T-score criteria for osteoporosis.
- MedlinePlus Medical Encyclopedia (U.S. National Library of Medicine): pulse, body
  temperature, creatinine, CBC, differential, amylase, magnesium, CRP, rheumatoid factor,
  T3, FSH, LH, estradiol, progesterone, prolactin, hCG.
- CDC hepatitis B and C serology; NCI tumor-marker, HPV and PSA fact sheets; Medscape (CA 15-3).

Some items use manufacturer cut-offs that many Taiwanese labs share (CYFRA 21-1, NSE), or
intervals that depend on the analyzer (MPV, PCT, PDW). Their `note` field marks lower
confidence.

## Disclaimer

This file is for education and software demonstration only. It is not medical advice and it
does not replace a clinician's judgement or the laboratory's own reference intervals. A value
outside these ranges does not mean the person is ill, and a value inside them does not rule
out disease. Anyone with questions about their results should talk to a qualified healthcare
professional.
