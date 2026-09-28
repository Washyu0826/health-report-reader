"""
run_eval.py — Run the tagger over every PDF of an evaluation set and score each
category's tags against its ground truth.

Usage (from project root):
    .venv/Scripts/python eval/run_eval.py                         # synthetic set, full pipeline
    .venv/Scripts/python eval/run_eval.py --mode hybrid --rerank  # retrieval variants
    .venv/Scripts/python eval/run_eval.py --route-by-tags         # applies_to-routed retrieval
    .venv/Scripts/python eval/run_eval.py --no-llm                # abnormal detection only (no Ollama)
    .venv/Scripts/python eval/run_eval.py --graph                 # LangGraph workflow, rewrite unsupported tags
    .venv/Scripts/python eval/run_eval.py --graph --no-rewrite    # same graph, drop unsupported (ablation control)
    .venv/Scripts/python eval/run_eval.py --samples DIR --gt GT.json --only syn_00 --tag RX

Defaults: --samples eval/synth/samples, --gt eval/synth/ground_truth.json.

Output: per-file + category-aggregated precision / recall / F1, global
micro/macro F1, abnormal-detection accuracy (pipeline's abnormal.py vs the
rule-derived gold abnormal list), JSON parse-failure rate and per-report
latency + number of Ollama calls (instrumented by wrapping requests.post —
no app code is modified). JSON dump at --out (default eval/results.json).

--no-llm runs extraction + abnormal detection + fact-sheet lookup only: no LLM,
no embeddings, no OCR. Image-only (scanned) PDFs then have no text and are
reported as skipped.

Matching (tags): case-insensitive substring match either way, OR a hit
against the sample's `synonyms` map. For `metrics`, an extra name+value
matcher is used when ground truth has `metric_keys`: the predicted tag must
contain one of the item's name aliases AND the item's value (numeric equality
for numbers, grade/陽性 for qualitative) — e.g. "低密度脂蛋白膽固醇 309 mg/dL ↑"
matches expected "LDL 低密度脂蛋白 309".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

# Make the project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

import config  # noqa: E402
from llm import tag_texts  # noqa: E402
from pipeline import analyze_pdf  # noqa: E402
from abnormal import load_checkitem_lookup  # noqa: E402
from gt_rules import name_aliases, parse_value  # noqa: E402

CATEGORIES = ["conditions", "risks", "metrics", "lifestyle",
              "food", "exercise", "supplements", "avoid"]
SILVER_CATEGORIES = [c for c in CATEGORIES if c != "metrics"]


# ─── Instrumentation (wrap requests.post; app code untouched) ───────────────

class CallLog:
    def __init__(self):
        self.reset()

    def reset(self):
        self.embed_calls = 0
        self.embed_sec = 0.0
        self.chat_calls = 0
        self.chat_sec = 0.0
        self.other_calls = 0
        self.last_chat = {}


CALLS = CallLog()
_orig_post = requests.post


DIAG = {"num_ctx": None}


def _instrumented_post(url, *a, **kw):
    if DIAG["num_ctx"] and str(url).endswith("/api/chat") and isinstance(kw.get("json"), dict):
        kw["json"] = {**kw["json"], "options": {**(kw["json"].get("options") or {}), "num_ctx": DIAG["num_ctx"]}}
    t0 = time.time()
    resp = _orig_post(url, *a, **kw)
    dt = time.time() - t0
    u = str(url)
    if u.endswith("/api/embeddings") or u.endswith("/api/embed"):
        CALLS.embed_calls += 1
        CALLS.embed_sec += dt
    elif u.endswith("/api/chat") or u.endswith("/api/generate"):
        CALLS.chat_calls += 1
        CALLS.chat_sec += dt
        try:
            j = resp.json()
            payload = kw.get("json") or {}
            msgs = payload.get("messages") or []
            CALLS.last_chat = {
                "raw": (j.get("message") or {}).get("content", j.get("response", "")),
                "prompt_eval_count": j.get("prompt_eval_count"),
                "eval_count": j.get("eval_count"),
                "total_duration_s": round((j.get("total_duration") or 0) / 1e9, 2),
                "prompt_chars": sum(len(m.get("content", "")) for m in msgs),
                "options": payload.get("options"),
            }
        except Exception:  # noqa: BLE001
            pass
    else:
        CALLS.other_calls += 1
    return resp


requests.post = _instrumented_post  # llm.py / embeddings.py / verify.py call requests.post


# ─── Matching ────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return " ".join(s.lower().replace("，", ",").replace("、", " ").split())


def _match(predicted: str, expected: str, synonyms_for_expected) -> bool:
    p, e = _norm(predicted), _norm(expected)
    if not p:
        return False
    if p == e or p in e or e in p:
        return True
    for syn in synonyms_for_expected or []:
        s = _norm(syn)
        if p == s or p in s or s in p:
            return True
    return False


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _metric_match(predicted: str, key: dict) -> bool:
    p = _norm(predicted)
    if not any(a in p for a in key.get("aliases", [])):
        return False
    if key.get("numeric"):
        try:
            targets = [float(v) for v in key["values"]]
        except ValueError:
            return False
        nums = [float(x) for x in _NUM_RE.findall(p.replace(",", ""))]
        return any(abs(n - t) < 1e-6 for n in nums for t in targets)
    return any(v.lower() in p for v in key.get("values", []))


def score_category(predicted, expected, synonyms_map, metric_keys=None):
    """Return (tp, fp, fn, matched_pairs)."""
    if not predicted and not expected:
        return 0, 0, 0, []
    metric_keys = metric_keys or {}
    matched_e = set()
    matched_p = set()
    pairs = []
    for pi, p in enumerate(predicted):
        for ei, e in enumerate(expected):
            if ei in matched_e:
                continue
            ok = _match(p, e, synonyms_map.get(e, []))
            if not ok and e in metric_keys:
                ok = _metric_match(p, metric_keys[e])
            if ok:
                matched_e.add(ei)
                matched_p.add(pi)
                pairs.append((p, e))
                break
    tp = len(matched_p)
    fp = len(predicted) - tp
    fn = len(expected) - len(matched_e)
    return tp, fp, fn, pairs


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ─── Abnormal-detection scoring (pipeline abnormal.py vs rule-derived gold) ──

def _gt_num(value: str):
    pv = parse_value(value)
    return pv["v"] if pv["kind"] == "num" and not pv.get("censored") else None


def _is_plain_number(f: dict) -> bool:
    """R0/R1-comparable findings: a plain numeric value (no '<x' censoring, no
    qualitative grade). R2+ findings may carry str values ('3+', 'Negative')."""
    v = f.get("value")
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not f.get("censored")


def _name_matches(f: dict, item: dict, aliases) -> bool:
    fn = f["name"].lower().strip()
    iname = item["name"].lower()
    return (len(fn) >= 2 and (fn in iname or iname in fn)) or any(a in fn for a in aliases)


def _finding_matches_item(f: dict, item: dict, aliases) -> bool:
    v = _gt_num(item["value"])
    if v is None or not _is_plain_number(f) or abs(f["value"] - v) > 1e-6:
        return False
    return _name_matches(f, item, aliases)


def _qual_sig(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKC", str(s or "")).lower().replace(" ", "")


def _finding_matches_item_v2(f: dict, item: dict, aliases) -> bool:
    """v2 (R2+): additionally matches censored ('<0.01') and qualitative
    ('>=1000(3+)', '2+', 'Negative') values by their printed text."""
    if _finding_matches_item(f, item, aliases):
        return True
    if _gt_num(item["value"]) is not None and _is_plain_number(f):
        return False
    gt_txt = _qual_sig(item["value"])
    ft = _qual_sig(f.get("value_text") or f.get("value"))
    if f.get("censored") and isinstance(f.get("value"), (int, float)):
        ft_alt = _qual_sig(f"{f['censored']}{f['value']:g}")
    else:
        ft_alt = ft
    ok = bool(gt_txt) and (gt_txt == ft or gt_txt == ft_alt or (len(ft) >= 2 and ft in gt_txt)
                           or (len(gt_txt) >= 2 and gt_txt in ft))
    return ok and _name_matches(f, item, aliases)


def score_abnormal(findings: list, gt_items: list, v2: bool = False) -> dict:
    """Detection P/R/F1 over abnormal items + direction accuracy over all
    gold-classifiable items (high/low/normal). An item never extracted counts
    as wrong for direction accuracy.

    v2=False (default, R0/R1-comparable): only plain-numeric findings are
    scored — qualitative/censored findings (which R0/R1 could not produce and
    the old matcher cannot pair with gold) are ignored on the prediction side.
    v2=True: qualitative and censored values are matched by printed text too."""
    match = _finding_matches_item_v2 if v2 else _finding_matches_item
    if not v2:
        findings = [f for f in findings if _is_plain_number(f)]
    gold_abn = [i for i in gt_items if i["direction"] in ("high", "low")]
    gold_cls = [i for i in gt_items if i["direction"] in ("high", "low", "normal")]
    alias_cache = {id(i): name_aliases(i["name"]) for i in gold_cls}
    pred_abn = [f for f in findings if f["direction"] in ("high", "low")]

    # item-level direction accuracy
    correct, extracted, qual_items = 0, 0, 0
    per_item = []
    for it in gold_cls:
        if _gt_num(it["value"]) is None:
            qual_items += 1
        cands = [f for f in findings if match(f, it, alias_cache[id(it)])]
        got = cands[0]["direction"] if cands else "not_extracted"
        if cands:
            extracted += 1
        ok = got == it["direction"]
        correct += ok
        if it["direction"] != "normal" or (cands and not ok):
            per_item.append({"item": f"{it['name']}={it['value']}", "gold": it["direction"], "pipeline": got})

    # detection P/R
    tp = 0
    used = set()
    dup = set()
    for it in gold_abn:
        hit = False
        for k, f in enumerate(pred_abn):
            if k in used:
                continue
            if match(f, it, alias_cache[id(it)]) and f["direction"] == it["direction"]:
                if not hit:
                    used.add(k)
                    tp += 1
                    hit = True
                else:  # same gold item found again (text + table copy) -> duplicate, not FP
                    dup.add(k)
    # R2 fix: a finding first seen as a "duplicate" of gold item A may later be
    # the true positive of gold item B (e.g. left/right hearing with the same
    # value); it must not be subtracted twice (that produced negative FP).
    dup -= used
    fp = len(pred_abn) - tp - len(dup)
    fn = len(gold_abn) - tp
    p, r, f1 = prf(tp, fp, fn)
    fps = [f"{f['name']}={f['value']} {f['direction']} (ref {f['ref_low']}~{f['ref_high']}, kb='{f['matched_name']}')"
           for k, f in enumerate(pred_abn) if k not in used and k not in dup]
    return {
        "gold_abnormal": len(gold_abn), "pred_abnormal": len(pred_abn),
        "tp": tp, "fp": fp, "fn": fn, "duplicates": len(dup),
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
        "gold_classifiable_items": len(gold_cls), "qualitative_items": qual_items,
        "items_extracted": extracted, "direction_correct": correct,
        "direction_accuracy": round(correct / len(gold_cls), 3) if gold_cls else None,
        "false_positive_findings": fps,
        "abnormal_item_detail": per_item,
    }


# ─── One sample ──────────────────────────────────────────────────────────────

RETRIEVER = {"obj": None}


def build_retriever(args):
    """Report-driven retriever over the KB index (built/reused in args.db).

    --no-llm: fact sheets only (no vector index, no embeddings)."""
    from kb_index import build_kb_index, load_articles, prepare_chunks
    from retrieval import BM25Index, CheckitemFacts, Reranker, Retriever
    facts = CheckitemFacts(args.ranges)
    if args.no_llm:
        print("retriever: fact sheets only (--no-llm)")
        return Retriever(None, facts=facts)
    col = None
    if args.mode != "bm25":
        col, stats = build_kb_index(json_path=args.kb, embed_model=args.embed, db_path=args.db,
                                    base_url=args.ollama)
        if col is None:
            raise SystemExit(f"KB index build failed: {stats}")
    bm25 = None
    if args.mode in ("bm25", "hybrid"):
        chunks, _ = prepare_chunks(load_articles(args.kb))
        bm25 = BM25Index(chunks, user_terms=list(facts.by_name))
    print(f"retriever: v2 embed={args.embed} mode={args.mode} rerank={args.rerank} "
          f"route_by_tags={args.route_by_tags}")
    # --route-by-tags: the pipeline calls retrieve(abnormals) without conditions, so the
    # Retriever derives the rule conditions itself (conditions.derive, sex unknown).
    return Retriever(col, embed_model=args.embed, base_url=args.ollama, facts=facts,
                     mode=args.mode, bm25=bm25, reranker=Reranker() if args.rerank else None,
                     route_by_tags=args.route_by_tags)


def run_one(pdf_path: Path, spec: dict, args, ci_lookup) -> dict:
    expected = spec.get("expected", {})
    synonyms_map = spec.get("synonyms", {})
    metric_keys = spec.get("metric_keys", {})

    web = None
    if args.web:
        from web_fallback import WebFallback
        web = WebFallback()
    kw = dict(checkitem_lookup=ci_lookup, retriever=RETRIEVER["obj"], verify=not args.no_verify,
              use_rules=not args.no_rules, model=args.model, base_url=args.ollama,
              ocr=not args.no_llm, web=web)
    if args.graph:
        # LangGraph generate -> verify -> rewrite workflow (graph_workflow.py);
        # --no-rewrite = same graph with the baseline drop-unsupported behaviour.
        from graph_workflow import run_graph
        pr = run_graph(pdf_path.read_bytes(), rewrite=not args.no_rewrite, **kw)
    else:
        pr = analyze_pdf(pdf_path.read_bytes(), run_llm=not args.no_llm, **kw)
    if pr.error_stage == "extract":
        etype = "skipped_no_text" if args.no_llm else "pdf"
        return {"file": pdf_path.name, "error": pr.error, "error_type": etype}
    text, tables, findings, chunks = pr.text, pr.tables, pr.findings, pr.chunks
    abnormals, rag_debug, timing = pr.abnormals, pr.rag_debug, pr.timing
    result, error = pr.tags, pr.error
    abn_score = score_abnormal(findings, spec.get("items", []))
    abn_score_v2 = score_abnormal(findings, spec.get("items", []), v2=True)
    base = {
        "file": pdf_path.name,
        "text_chars": len(text),
        "n_tables": len(tables),
        "n_findings_extracted": len(findings),
        "abnormal_findings": [f"{f['name']}={f['value']} {f['direction']}" for f in abnormals],
        "findings": findings,
        "abnormal_score": abn_score,
        "abnormal_score_v2": abn_score_v2,
        "abnormal_debug": getattr(pr, "abnormal_debug", {}),
        "rag": {**rag_debug, "n_chunks": len(chunks), "n_cited_ids": len(pr.cited_ids)},
        "timing": timing,
        "llm_call": dict(CALLS.last_chat),
        "verification": (pr.tags or {}).get("_verification"),
        "rag_queries": (rag_debug or {}).get("queries"),
        "rewrite_log": getattr(pr, "rewrite_log", None),
        "node_trace": getattr(pr, "node_trace", None),
    }
    if args.no_llm:
        return {**base, "llm": "skipped (--no-llm)"}
    if error:
        etype = "json_parse" if "parse" in error.lower() else "llm_call"
        return {**base, "error": error, "error_type": etype}

    per_cat = {}
    for k in CATEGORIES:
        predicted_texts = tag_texts(result.get(k, []))
        exp = expected.get(k, [])
        tp, fp, fn, pairs = score_category(predicted_texts, exp, synonyms_map,
                                           metric_keys if k == "metrics" else None)
        p, r, f = prf(tp, fp, fn)
        per_cat[k] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "predicted": predicted_texts,
            "predicted_src": [t.get("src") for t in result.get(k, [])],
            "expected": exp,
            "matched": pairs,
        }
    return {**base, "summary_text": result.get("summary", ""), "per_category": per_cat}


def aggregate(results):
    agg = {k: {"tp": 0, "fp": 0, "fn": 0} for k in CATEGORIES}
    for r in results:
        if "per_category" not in r:
            continue
        for k, v in r["per_category"].items():
            agg[k]["tp"] += v["tp"]
            agg[k]["fp"] += v["fp"]
            agg[k]["fn"] += v["fn"]
    summary = {}
    for k, v in agg.items():
        p, r, f = prf(v["tp"], v["fp"], v["fn"])
        summary[k] = {**v, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
    return summary


def global_scores(summary, results, n_attempted):
    def pooled(cats):
        tp = sum(summary[c]["tp"] for c in cats)
        fp = sum(summary[c]["fp"] for c in cats)
        fn = sum(summary[c]["fn"] for c in cats)
        p, r, f = prf(tp, fp, fn)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}

    active = [c for c in CATEGORIES if summary[c]["tp"] + summary[c]["fp"] + summary[c]["fn"] > 0]
    macro = sum(summary[c]["f1"] for c in active) / len(active) if active else 0.0
    def _abn(key):
        ab_ = [r[key] for r in results if key in r]
        tp_, fp_, fn_ = (sum(a[k] for a in ab_) for k in ("tp", "fp", "fn"))
        p_, r_, f_ = prf(tp_, fp_, fn_)
        n_ = sum(a["gold_classifiable_items"] for a in ab_)
        return {"tp": tp_, "fp": fp_, "fn": fn_, "duplicates": sum(a.get("duplicates", 0) for a in ab_),
                "precision": round(p_, 3), "recall": round(r_, 3), "f1": round(f_, 3),
                "direction_accuracy": round(sum(a["direction_correct"] for a in ab_) / n_, 3) if n_ else None,
                "items_extracted_rate": round(sum(a["items_extracted"] for a in ab_) / n_, 3) if n_ else None}

    ab = [r["abnormal_score"] for r in results if "abnormal_score" in r]
    a_tp = sum(a["tp"] for a in ab)
    a_fp = sum(a["fp"] for a in ab)
    a_fn = sum(a["fn"] for a in ab)
    a_dup = sum(a.get("duplicates", 0) for a in ab)
    ap, ar, af = prf(a_tp, a_fp, a_fn)
    dir_ok = sum(a["direction_correct"] for a in ab)
    dir_n = sum(a["gold_classifiable_items"] for a in ab)
    extracted = sum(a["items_extracted"] for a in ab)
    parse_fail = sum(1 for r in results if r.get("error_type") in ("json_parse", "json_schema"))
    other_fail = sum(1 for r in results if r.get("error") and r.get("error_type") not in ("json_parse", "json_schema"))
    by_type = {}
    for r in results:
        if r.get("error"):
            by_type[r.get("error_type")] = by_type.get(r.get("error_type"), 0) + 1
    lat = [r["elapsed_sec"] for r in results if "elapsed_sec" in r]
    return {
        "micro_all": pooled(CATEGORIES),
        "micro_silver": pooled(SILVER_CATEGORIES),
        "macro_f1_categories": round(macro, 3),
        "macro_over": active,
        "abnormal_detection": {"tp": a_tp, "fp": a_fp, "fn": a_fn, "duplicates": a_dup, "precision": round(ap, 3),
                               "recall": round(ar, 3), "f1": round(af, 3)},
        "abnormal_detection_v2": _abn("abnormal_score_v2"),
        "abnormal_direction_accuracy": round(dir_ok / dir_n, 3) if dir_n else None,
        "abnormal_items_extracted_rate": round(extracted / dir_n, 3) if dir_n else None,
        "n_samples": n_attempted,
        "json_parse_failures": parse_fail,
        "json_parse_failure_rate": round(parse_fail / n_attempted, 3) if n_attempted else None,
        "other_failures": other_fail,
        "failures_by_type": by_type,
        "llm_success_rate": round(sum(1 for r in results if "per_category" in r) / n_attempted, 3) if n_attempted else None,
        "latency_sec_mean": round(sum(lat) / len(lat), 1) if lat else None,
        "latency_sec_max": max(lat) if lat else None,
        "latency_sec_total": round(sum(lat), 1),
    }


def rescore(path: str, gt: dict, out: str):
    """Re-score a saved results file against the CURRENT ground truth without
    calling any model (for GT/matcher edits)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    specs = gt["samples"]
    for r in data["results"]:
        spec = specs.get(r["file"])
        if not spec:
            continue
        if "findings" in r:
            r["abnormal_score"] = score_abnormal(r["findings"], spec.get("items", []))
            r["abnormal_score_v2"] = score_abnormal(r["findings"], spec.get("items", []), v2=True)
        if "per_category" in r:
            for k, v in r["per_category"].items():
                exp = spec["expected"].get(k, [])
                tp, fp, fn, pairs = score_category(v["predicted"], exp, spec.get("synonyms", {}),
                                                   spec.get("metric_keys", {}) if k == "metrics" else None)
                p, rr, f = prf(tp, fp, fn)
                v.update({"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(rr, 3),
                          "f1": round(f, 3), "expected": exp, "matched": pairs})
    summary = aggregate(data["results"])
    data["summary"] = summary
    data["global"] = global_scores(summary, data["results"], len(data["results"]))
    data["rescored_from"] = path
    Path(out).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data["global"], ensure_ascii=False, indent=1))
    for k, v in summary.items():
        print(f"  {k:12s} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f}  tp={v['tp']} fp={v['fp']} fn={v['fn']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--ollama", default=config.OLLAMA_BASE_URL)
    ap.add_argument("--embed", default=config.EMBED_MODEL)
    ap.add_argument("--db", default="eval/_chroma")
    ap.add_argument("--kb", default=config.KB_JSON_PATH, help="passages to index (Title/Content/TagName/Source)")
    ap.add_argument("--ranges", "--checkitem", dest="ranges", default=config.REFERENCE_RANGES_PATH,
                    help="reference-range source for abnormal detection + fact sheets")
    ap.add_argument("--samples", default="eval/synth/samples")
    ap.add_argument("--gt", default="eval/synth/ground_truth.json")
    ap.add_argument("--web", action="store_true",
                    help="web fallback (sends canonical item names to a search engine)")
    ap.add_argument("--no-llm", action="store_true",
                    help="abnormal detection only: no LLM, embeddings or OCR (scanned PDFs are skipped)")
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="dense")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--route-by-tags", action="store_true",
                    help="tag-routed retrieval: search each finding among KB passages whose applies_to "
                         "matches it (see retrieval.py); needs a KB with applies_to, e.g. data/kb_sample.json")
    ap.add_argument("--no-verify", action="store_true", help="skip the claim-verification layer")
    ap.add_argument("--no-rules", action="store_true", help="skip rule-based conditions/risks")
    ap.add_argument("--graph", action="store_true",
                    help="run through graph_workflow.run_graph (LangGraph; rewrites unsupported tags)")
    ap.add_argument("--no-rewrite", action="store_true",
                    help="with --graph: drop unsupported tags instead of rewriting (ablation control)")
    ap.add_argument("--only", default="", help="comma-separated sample-name prefixes")
    ap.add_argument("--tag", default="", help="free-text run label stored in the output")
    ap.add_argument("--diag-num-ctx", type=int, default=0,
                    help="DIAGNOSTIC: inject options.num_ctx into /api/chat at runtime (app default = Ollama's 4096)")
    ap.add_argument("--rescore", default="", help="re-score a saved results JSON (no model calls)")
    ap.add_argument("--out", default="eval/results.json")
    args = ap.parse_args()
    if args.no_rewrite and not args.graph:
        ap.error("--no-rewrite requires --graph")
    if args.graph and args.no_llm:
        ap.error("--graph always runs the LLM; drop --no-llm")
    if args.rescore:
        rescore(args.rescore, json.loads(Path(args.gt).read_text(encoding="utf-8")), args.out)
        return
    if args.diag_num_ctx:
        DIAG["num_ctx"] = args.diag_num_ctx
        print(f"NOTE: DIAGNOSTIC num_ctx={args.diag_num_ctx} injected into /api/chat")

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    samples_dir = Path(args.samples)
    samples_dir.mkdir(parents=True, exist_ok=True)

    sample_specs = gt.get("samples", {})
    if not sample_specs:
        print("No samples in ground_truth.json. Add entries under 'samples'.")
        return
    if args.only:
        pref = [p.strip() for p in args.only.split(",") if p.strip()]
        sample_specs = {k: v for k, v in sample_specs.items() if any(k.startswith(p) for p in pref)}

    ci_lookup = load_checkitem_lookup(args.ranges)
    RETRIEVER["obj"] = build_retriever(args)
    print(f"catalog: {len(ci_lookup)} items from {args.ranges}  kb={args.kb}  db={args.db}  "
          f"model={'(none: --no-llm)' if args.no_llm else args.model}")

    results = []
    t_all = time.time()
    for fname, spec in sample_specs.items():
        pdf_path = samples_dir / fname
        if not pdf_path.exists():
            print(f"!! missing: {pdf_path} — skipping")
            continue
        print(f"-> {fname}", flush=True)
        CALLS.reset()
        t0 = time.time()
        try:
            r = run_one(pdf_path, spec, args, ci_lookup)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            # R0: llm._normalize_tags() assumes a dict; a top-level JSON *array*
            # from the model raises AttributeError -> count as schema failure.
            etype = "json_schema" if "object has no attribute 'get'" in msg else "exception"
            r = {"file": fname, "error": msg, "error_type": etype,
                 "traceback": traceback.format_exc(), "llm_call": dict(CALLS.last_chat)}
        r["elapsed_sec"] = round(time.time() - t0, 1)
        r["calls"] = {"embed": CALLS.embed_calls, "embed_sec": round(CALLS.embed_sec, 2),
                      "chat": CALLS.chat_calls, "chat_sec": round(CALLS.chat_sec, 2),
                      "other": CALLS.other_calls}
        lc = r.get("llm_call", {})
        print(f"   {r['elapsed_sec']}s  embed={CALLS.embed_calls} chat={CALLS.chat_calls}  "
              f"prompt_chars={lc.get('prompt_chars')} prompt_tokens={lc.get('prompt_eval_count')} "
              f"out_tokens={lc.get('eval_count')}")
        if "abnormal_score" in r:
            a = r["abnormal_score"]
            print(f"   abnormal-detect P={a['precision']:.2f} R={a['recall']:.2f} "
                  f"(gold {a['gold_abnormal']}, pred {a['pred_abnormal']}, tp {a['tp']})  "
                  f"dir-acc={a['direction_accuracy']}")
        if "error" in r:
            print(f"   ERROR[{r.get('error_type')}]: {r['error'][:300]}")
        elif "per_category" in r:
            for k, v in r["per_category"].items():
                if v["expected"] or v["predicted"]:
                    print(f"   {k:12s} P={v['precision']:.2f} R={v['recall']:.2f} "
                          f"F1={v['f1']:.2f}  ({v['tp']}/{v['fp']+v['tp']} pred, "
                          f"{v['tp']}/{v['fn']+v['tp']} exp)")
        results.append(r)

    summary = aggregate(results)
    glob = global_scores(summary, results, len(results))
    print("\n=== AGGREGATE (per category, pooled over samples) ===")
    for k, v in summary.items():
        if v["tp"] + v["fp"] + v["fn"] == 0:
            continue
        print(f"  {k:12s} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f} "
              f"  tp={v['tp']} fp={v['fp']} fn={v['fn']}")
    print("\n=== GLOBAL ===")
    print(f"  micro-F1 (all 8 cats)   = {glob['micro_all']['f1']:.3f}  "
          f"(P={glob['micro_all']['precision']:.3f} R={glob['micro_all']['recall']:.3f})")
    print(f"  micro-F1 (silver 7 cats)= {glob['micro_silver']['f1']:.3f}")
    print(f"  macro-F1 (categories)   = {glob['macro_f1_categories']:.3f}")
    ad = glob["abnormal_detection"]
    print(f"  abnormal detection      P={ad['precision']:.3f} R={ad['recall']:.3f} F1={ad['f1']:.3f} "
          f"(tp={ad['tp']} fp={ad['fp']} fn={ad['fn']} dup={ad.get('duplicates')})")
    print(f"  item direction accuracy = {glob['abnormal_direction_accuracy']}  "
          f"(extracted rate {glob['abnormal_items_extracted_rate']})")
    a2 = glob["abnormal_detection_v2"]
    print(f"  abnormal detection v2   P={a2['precision']:.3f} R={a2['recall']:.3f} F1={a2['f1']:.3f} "
          f"(tp={a2['tp']} fp={a2['fp']} fn={a2['fn']} dup={a2['duplicates']})  "
          f"dir-acc={a2['direction_accuracy']} extracted={a2['items_extracted_rate']}  "
          f"[v2 also matches qualitative/censored values]")
    print(f"  JSON parse/schema fails = {glob['json_parse_failures']}/{glob['n_samples']}  "
          f"other failures = {glob['other_failures']}  by type = {glob['failures_by_type']}")
    print(f"  latency mean/max/total  = {glob['latency_sec_mean']}s / {glob['latency_sec_max']}s / "
          f"{glob['latency_sec_total']}s   (wall {time.time() - t_all:.0f}s)")

    # Grounding: how many tags cite a KB passage that actually supports them.
    ver = [r["verification"] for r in results if r.get("verification")]
    if ver:
        tot = {k: sum(v.get(k) or 0 for v in ver) for k in
               ("kb_checked", "kb_lexical_ok", "kb_llm_ok", "kb_dropped", "kb_unverified",
                "report_checked", "report_downgraded", "inferred_capped")}
        ok = tot["kb_lexical_ok"] + tot["kb_llm_ok"]
        glob["grounding"] = {**tot, "citation_support_rate":
                             round(ok / tot["kb_checked"], 3) if tot["kb_checked"] else None}
        print(f"  grounding               = kb citations {tot['kb_checked']} "
              f"(supported {ok}: lexical {tot['kb_lexical_ok']} + llm {tot['kb_llm_ok']}; "
              f"dropped {tot['kb_dropped']}; unverified {tot['kb_unverified']})  "
              f"report-claims downgraded {tot['report_downgraded']}/{tot['report_checked']}  "
              f"inferred capped {tot['inferred_capped']}")
    rw = [v["rewrite"] for v in ver if v.get("rewrite")]
    if rw:
        keys = ("unsupported_initial", "revised", "dropped_by_rewriter", "kept",
                "dropped_after_reverify", "unverified_after_reverify")
        glob["rewrite"] = {"enabled": rw[0].get("enabled"), **{k: sum(x.get(k) or 0 for x in rw) for k in keys}}
        g = glob["rewrite"]
        print(f"  rewrite                 = unsupported {g['unsupported_initial']} -> revised {g['revised']} "
              f"(kept {g['kept']}, dropped after re-verify {g['dropped_after_reverify']}); "
              f"dropped by rewriter {g['dropped_by_rewriter']}  [enabled={g['enabled']}]")
    if args.graph:
        per_node: dict = {}
        for r in results:
            for k, v in (r.get("timing") or {}).items():
                per_node.setdefault(k, []).append(v)
        glob["node_timing_mean_s"] = {k: round(sum(v) / len(v), 3) for k, v in per_node.items()}
        print(f"  node timing mean (s)    = {glob['node_timing_mean_s']}")
    lat = sorted(r["elapsed_sec"] for r in results if "elapsed_sec" in r)
    if lat:
        glob["latency_sec_p50"] = lat[len(lat) // 2]
        print(f"  latency p50             = {glob['latency_sec_p50']}s")

    Path(args.out).write_text(
        json.dumps({"tag": args.tag, "args": vars(args), "global": glob, "summary": summary,
                    "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
