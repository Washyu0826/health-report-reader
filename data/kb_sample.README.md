# data/kb_sample.json — structured public health-education passages

Two files, same format:

* `data/kb_sample.json` — **the default retrieval corpus: 57 passages.**
* `data/kb_sample_extended.json` — 152 passages: the same 57 plus 95 more, written to give every common
  finding a `meaning` / `diet` / `lifestyle` / `followup` passage.

The default is the smaller set because it measured better (eval/HISTORY.md, R9). Advice judged directly
relevant was 43% vs 35% on the synthetic quick set and 35% vs 16% on the internal set, with the v4 prompt.
Per-finding tag routing over the extended set did not close the gap. Why is not isolated yet; one
hypothesis is that more loosely related passages reach the limited context budget.

Traditional Chinese (Taiwan wording), written for this project. **Every passage is original writing**:
it paraphrases general, public clinical guidance and cites the public page it is based on (Taiwan Health
Promotion Administration / 國民健康署, Taiwanese medical-society guidance, MedlinePlus, NHS, WHO, NCI, KDIGO,
NIH ODS). No text is copied from those pages or from any third-party corpus.

The passages are conservative health education, not medical advice: no drug names with
doses, no supplement dosing; medication and supplement decisions are always referred to a
physician or pharmacist, and each topic says when to see a doctor.

## Schema

A JSON list of objects:

| field | type | meaning |
|---|---|---|
| `Title` | str | short title (unique) |
| `Content` | str | passage body, 150–400 characters |
| `TagName` | str | comma-separated Chinese topic tags (used in the chunk prefix) |
| `kind` | str | `meaning` (what it means + causes) · `diet` · `lifestyle` (exercise / habits) · `followup` (re-testing, when to see a doctor) |
| `applies_to` | list[str] | exactly which findings the passage is for (see below) |
| `Source` | str | https URL of the public page the passage is based on |

`applies_to` tokens:

* `"<canonical_key>:<direction>"` — `canonical_key` is a `key` from `data/lab_synonyms.json`,
  `direction` is `high`, `low` or `positive` (qualitative items), e.g. `"ldl:high"`,
  `"egfr:low"`, `"urine_protein:positive"`.
* `"cond:<label>"` — a rule-derived condition label emitted by `conditions.derive()`,
  e.g. `"cond:代謝症候群"`, `"cond:血脂異常"`. `conditions.CONDITION_KEYS` maps each label to
  the lab keys it is derived from.

`kb_index.prepare_chunks()` stores `applies_to` comma-joined in each chunk's metadata
(Chroma metadata must be scalars). Articles without the field (any other corpus) get `""`
and simply never route.

## Coverage and validation

    .venv/Scripts/python tools/kb_coverage.py            # checks data/kb_sample_extended.json

validates the schema (fields, lengths, https sources, known keys / condition labels, kinds)
and prints, per `applies_to` token, how many passages of each kind exist. Every abnormal
key:direction in the synthetic ground truth (`eval/synth`) and the common checkup findings
(lipids, glucose/HbA1c, blood pressure, uric acid, liver enzymes, ALP, bilirubin, albumin,
kidney/eGFR/BUN, urine protein/blood/glucose, CBC incl. anaemia indices, WBC and platelets,
thyroid, tumour markers, BMI/waist/body fat, hs-CRP, homocysteine, Na/K/Ca, bone density)
has at least one passage of each of the four kinds in the extended set. `tests/test_kb_routing.py`
enforces this, and pins the default file to the measured 57-passage subset.

## Tag routing

`Retriever(..., route_by_tags=True)` (eval: `eval/run_eval.py --route-by-tags`) searches
each abnormal finding only among the passages whose `applies_to` matches it
(`<key>:<status>`, plus `cond:<label>` for rule conditions derived from that key), ranked by
the usual retrieval score, and falls back to the unfiltered search when no passage is
tagged for the finding. See the module docstring of `retrieval.py`.
