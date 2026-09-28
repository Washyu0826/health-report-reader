# Eval harness

Offline regression suite. Without it, every prompt / threshold / model swap is guesswork.
Everything here runs on public files only: the fully synthetic report set in `synth/`,
`data/reference_ranges.json` (reference ranges + fact sheets) and `data/kb_sample.json`
(health-education passages). The round-by-round history is in [HISTORY.md](HISTORY.md).

## Files

| file | role |
|---|---|
| `synth/` | 60 fully synthetic checkup PDFs (six layouts, 3 image-only) with gold labels, plus the deterministic generator, a verifier and a leak check. See [synth/README.md](synth/README.md). |
| `gt_rules.py` | Pipeline-independent rules: reference-range parsing, high/low/normal, metric aliases. It does **not** import `abnormal.py`, so it can cross-check it. |
| `run_eval.py` | Runs the pipeline over every sample. Reports tag P/R/F1 per category, micro/macro F1, abnormal-detection P/R/F1 and direction accuracy, JSON parse-failure rate, latency and Ollama call counts. |
| `test_abnormal.py` | 65 unit cases for reference parsing, extraction and end-to-end flagging (R/E/P groups). Runs standalone or under pytest. |
| `retrieval_bench.py` | Retrieval ablation (embeddings, BM25, hybrid RRF, reranker) with LLM-judged pooled relevance. |

## Run (from the project root)

```bash
# no model needed
python eval/test_abnormal.py                        # abnormal-flagging regression set
python eval/run_eval.py --no-llm                    # abnormal detection on the 60 synthetic reports
                                                    # (image-only PDFs are skipped: OCR needs Ollama)
python eval/synth/verify.py                         # dataset checks + byte-identical regeneration
python eval/synth/check_no_real_data.py             # leak check

# full pipeline (Ollama running with the models from config.py)
python eval/run_eval.py --tag RX --out eval/results_RX.json
python eval/run_eval.py --mode hybrid --rerank      # hybrid retrieval + cross-encoder reranker
python eval/retrieval_bench.py --systems all        # retrieval ablation
```

Defaults: `--samples eval/synth/samples --gt eval/synth/ground_truth.json`. Point them at any other
set that uses the same ground-truth layout (a top-level `samples` map keyed by file name).

Useful flags:

- `--only syn_00,syn_01`: filter samples by name prefix.
- `--mode dense|bm25|hybrid`, `--rerank`, `--embed <model>`: retrieval variants.
- `--no-verify`, `--no-rules`: ablate claim verification or rule-based conditions.
- `--ranges <file>`: reference-range source (default `data/reference_ranges.json`).
- `--kb <file>`: passages to index (default `data/kb_sample.json`).
- `--route-by-tags`: search each abnormal finding among the passages whose `applies_to` matches it, falling back to the normal search (see `data/kb_sample.README.md`).
- `--rescore eval/results_RX.json --out ...`: re-score saved predictions after a matcher change, with no model calls.
- `--diag-num-ctx 16384`: diagnostic only; injects `options.num_ctx` into `/api/chat`.

`--web` sends canonical item names (never values or report text) to a search engine. It is off by default.

## Matching rules

A predicted tag matches an expected tag if (case-insensitive, whitespace-normalized)
`predicted == expected`, or one is a substring of the other, or it matches that tag's
synonyms. For `metrics`, it also matches when the prediction contains one of the item's
name aliases **and** its value (numeric equality; `3+`/`陽性` for qualitative).
The substring rule is generous: a very short prediction ("運動") can match a longer
synonym. Watch precision: recall is easy to gain by being verbose.

Abnormal detection is scored against the gold `items` of each sample. The `v2` score also
matches qualitative (`2+`, `Negative`) and censored (`<0.01`) values by their printed text.
