# data/loinc_map.json — LOINC codes for the canonical lab items

This material contains content from LOINC® (http://loinc.org). LOINC is copyright © Regenstrief Institute, Inc.
and the Logical Observation Identifiers Names and Codes (LOINC) Committee and is available at no cost under the
license at http://loinc.org/license. LOINC® is a registered United States trademark of Regenstrief Institute, Inc.

> **For interoperability demos only.** This mapping has not been reviewed by a
> laboratory or terminology professional. It **must be reviewed (against the
> performing lab's methods, specimens and units) before any clinical use.**

## What it is

One entry per canonical item key in `data/lab_synonyms.json` (all 143 keys).
`fhir_export.py` reads it to put a LOINC `Coding` on each FHIR `Observation`.

```json
{"version": 1, "source": "...", "items": {
  "glucose_ac": {"loinc": "1558-6",
                 "display": "Fasting glucose [Mass/volume] in Serum or Plasma",
                 "ucum": "mg/dL", "confidence": "high",
                 "source_url": "https://loinc.org/1558-6", "note": "..."}}}
```

| field | meaning |
|---|---|
| `loinc` | LOINC code, or `null` when the item is not mapped (reason in `note`) |
| `display` | LOINC long common name (LOINC 2.82), copied verbatim from the terminology server |
| `ucum` | UCUM code for the item's **canonical unit in `lab_synonyms.json`** (e.g. `10*3/uL`, `mm[Hg]`, `[arb'U]/mL`); `null` = no unit / not asserted |
| `confidence` | `high` / `medium` for mapped items; `null` for unmapped ones |
| `source_url` | `https://loinc.org/<code>` (`null` when unmapped) |
| `note` | why this code (or why none); alternatives for other methods |

## How codes were chosen

The target is the test **as measured in a routine adult health check** in
Taiwan: serum/plasma chemistry, whole-blood automated CBC, urine dipstick and
sediment microscopy, bedside vital signs.

* **high**: the LOINC component, property, specimen and scale clearly match the
  item as it is reported.
* **medium**: a real choice exists (method, scale or site) and the most common
  health-check variant was picked; the note names the alternative code(s).
* **null**: no code is safe — the item is not a standard measurement
  (device indices, free-text lists), or LOINC codes are assay-specific and the
  report does not identify the assay. **A wrong code is worse than none.**

`fhir_export.py` adds one runtime safeguard: when a report prints a unit whose
property contradicts the mapped code (e.g. glucose in mmol/L against the
mass-concentration code 1558-6) the LOINC coding is dropped and a note is added.

## How codes were verified

loinc.org pages are behind a bot challenge (HTTP 403 for scripted fetches), so
every code was checked against two machine-readable copies of LOINC instead
(2026-09-30):

1. **HL7 tx.fhir.org** — `CodeSystem/$lookup?system=http://loinc.org&code=…`
   (LOINC **2.82**): long common name, `STATUS`, example UCUM units.
2. **NLM Clinical Tables LOINC API** — component, property and method, and
   keyword searches to find candidates.

Every mapped code is **ACTIVE** in LOINC 2.82 (the build script refused any
other status); `display` is the server's long common name. Codes mentioned only
as alternatives in notes were looked up the same way. `tests/test_fhir.py`
re-checks the format (`^\d{1,7}-\d$`) and the LOINC mod-10 check digit offline.

Two obvious first choices turned out to be **DISCOURAGED** in 2.82 and were
replaced: 33914-3 (eGFR MDRD) → 77147-7, and 2532-0 (LDH, no method) → 14804-9.

## Counts

| | items |
|---|---|
| total keys | 143 |
| mapped | **122** |
| high | 99 |
| medium | 23 |
| null | 21 |

## Medium-confidence items

| key | code | reason |
|---|---|---|
| body_fat_pct | 77233-5 | Assumes BIA scale; 41982-0 if measured another way (e.g. DXA). |
| waist | 8280-0 | LOINC term is "at umbilicus"; Taiwan HPA measures midway rib–iliac crest. |
| ideal_weight | 50064-5 | Calculated value; formula differs by provider. |
| muscle_mass | 73964-9 | "Calculated" muscle mass; device algorithms differ. |
| pdw | 51631-0 | Chosen for % units; analysers reporting fL need 32207-3. |
| egfr | 77147-7 | Equation rarely printed; MDRD assumed (CKD-EPI: 62238-1 / 98979-8). |
| ldl | 2089-1 | Method-neutral; direct (18262-6) vs Friedewald (13457-7) unknown. |
| homa_ir | 47214-2 | LOINC "Homeostasis model assessment" does not separate HOMA-IR from HOMA-%B. |
| hbsag | 5196-1 | Qualitative presence code carrying the printed COI index; quantitative IU/mL is 63557-3. |
| hbeag | 13954-3 | Qualitative presence code; printed index of the same assay. |
| anti_hcv | 13955-0 | Qualitative presence code carrying the printed COI index. |
| h_pylori_ubt | 29891-9 | Qualitative result; the numeric DOB code 29892-7 is DISCOURAGED. |
| fobt | 58453-2 | Quantitative FIT (ng/mL); qualitative-only reports → 29771-3. |
| beta_hcg | 21198-7 | Labelled β-hCG; many assays measure total hCG (19080-1). |
| ldh | 14804-9 | Method-less 2532-0 is DISCOURAGED; IFCC lactate→pyruvate assumed (P→L: 14805-6). |
| urobilinogen | 32727-0 | Matches EU/dL units but has no method; dipstick codes are 20405-7 / 5818-0. |
| urine_protein | 20454-5 | Ordinal dipstick (−/±/1+); numeric mg/dL → 5804-0. |
| urine_glucose | 25428-4 | Ordinal dipstick; numeric mg/dL → 5792-7. |
| urine_ketone | 2514-8 | Ordinal dipstick; numeric mg/dL → 5797-6. |
| urine_wbc | 5799-2 | Dipstick "WBC" pad = leukocyte esterase (same code as urine_le). |
| urine_epi_sed | 105113-5 | Generic epithelial cells; cell type and method not reported. |
| va_naked_right | 98499-7 | Decimal acuity notation; distance/chart unspecified, no UCUM unit asserted. |
| va_naked_left | 98498-9 | Same as right eye. |

## Null (unmapped) items

| key | reason |
|---|---|
| bone_mass | BIA-scale bone-mass estimate; no LOINC term found. |
| metabolic_age | Proprietary device index. |
| bmr | Device-estimated BMR; LOINC only has resting-metabolic-rate terms with a different definition. |
| visceral_fat | Proprietary 1–59 rating; LOINC 73707-2 is visceral fat *area* (imaging). |
| fat_right_arm | Segmental BIA fat %; no segment-specific term. |
| fat_right_leg | Segmental BIA fat %; no segment-specific term. |
| fat_left_arm | Segmental BIA fat %; no segment-specific term. |
| fat_left_leg | Segmental BIA fat %; no segment-specific term. |
| fat_trunk | Segmental BIA fat %; no segment-specific term. |
| body_water | Reported as kg or % (101683-1 vs 101684-9); catalog unit unknown. |
| risk_factor | "動脈硬化指數" has lab-specific definitions (TC/HDL, LDL/HDL, (TC−HDL)/HDL). |
| ebv_iga | VCA-IgA vs EBNA1-IgA and titre/index/qualitative vary by kit. |
| hpv_rlu | Hybrid Capture signal ratio (RLU/cutoff); no LOINC term found. |
| hpv_result | Assay-specific codes (30167-1 HC2 vs 82675-0 NAA); assay not identified. |
| hpv_highrisk | Free-text genotype list. |
| urine_crystal_sed | LOINC crystal terms are type-specific; "unidentified crystals" means something else. |
| bmd | Health-check BMD is usually calcaneal ultrasound; LOINC T-score terms found are DXA. |
| va_corr_right | "矯正視力" is presenting acuity with own glasses; LOINC only has "best corrected". |
| va_corr_left | Same as right eye. |
| cvd_10y_risk | Local risk models / outcomes; LOINC risk terms are model-specific. |
| framingham_score | Point total, not a probability; LOINC Framingham terms are model-specific risks. |

## Updating

Add or change an entry only after looking the code up on a LOINC source
(loinc.org, tx.fhir.org `$lookup`, or the LOINC release files) and checking its
status is ACTIVE. Keep `display` identical to the long common name, keep
`ucum` in line with the item's canonical unit in `lab_synonyms.json`, and
record the reason in `note`. Run `pytest -q tests/test_fhir.py`.
