"""Latency work (R9): v4 LLM schema (advice + text-stated conditions only),
prompt trimming, streaming helpers, batched/cached query embeddings and the
report-level cache. No network, no LLM: requests.post is faked."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import embeddings  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402
from report_cache import ReportCache, cache_key, config_fingerprint  # noqa: E402

PDF = ROOT / "eval" / "synth" / "samples" / "syn_047.pdf"


# ─── schema ──────────────────────────────────────────────────────────────────

def test_v4_schema_has_no_lab_conditions_or_risks():
    s = llm.build_output_schema(["report", "inferred", "kb:a"], rules_cover_labs=True)
    props = s["properties"]
    assert "conditions" not in props and "risks" not in props
    assert set(s["required"]) == {"summary", "text_conditions", *llm.ADVICE_KEYS}
    assert props["summary"]["maxLength"] == llm.SUMMARY_MAX_CHARS
    cond = props["text_conditions"]["items"]
    assert set(cond["properties"]) == {"text"}                        # no src, no conf
    tag = props["food"]["items"]
    assert set(tag["properties"]) == {"text", "src"}                  # conf is not generated
    assert tag["properties"]["src"]["enum"] == ["report", "inferred", "kb:a"]
    assert all(props[k]["maxItems"] == llm.MAX_TAGS_PER_CATEGORY == 4 for k in llm.ADVICE_KEYS)


def test_legacy_schema_keeps_conditions_and_risks():
    props = llm.build_output_schema(["report"], rules_cover_labs=False)["properties"]
    assert "conditions" in props and "risks" in props and "text_conditions" not in props


# ─── analyze_report with a fake Ollama ───────────────────────────────────────

class FakeResp:
    def __init__(self, content, stats=None, lines=None):
        self._content, self._stats, self._lines = content, stats or {}, lines

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": json.dumps(self._content, ensure_ascii=False)},
                "done_reason": "stop", **self._stats}

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


V4 = {"summary": "血脂偏高，並有輕度脂肪肝。",
      "text_conditions": [{"text": "輕度脂肪肝"}, {"text": "三酸甘油脂 156"}, {"text": "胃潰瘍"}],
      "lifestyle": [{"text": "規律運動", "src": "kb:c1"}], "food": [{"text": "少吃油炸", "src": "inferred"},
                                                              {"text": "多吃蔬菜", "src": "inferred"}],
      "exercise": [], "supplements": [], "avoid": [{"text": "不在清單", "src": "kb:zzz"}]}
TEXT = "三酸甘油脂 TG 156 ↑ mg/dL <150\n【腹部超音波】輕度脂肪肝。\n醫師建議：減重。"
FINDINGS = [{"name": "三酸甘油脂 TG", "raw_name": "三酸甘油脂 TG", "display_name": "三酸甘油脂", "value": 156.0,
             "value_text": "156", "unit": "mg/dL", "report_range": "<150", "direction": "high", "status": "high"}]


def test_analyze_report_v4_maps_text_conditions_and_filters(monkeypatch):
    sent = {}

    def post(url, json=None, timeout=None, **kw):
        sent.update(json)
        return FakeResp(V4, {"prompt_eval_count": 900, "eval_count": 120, "eval_duration": 3e9,
                             "prompt_eval_duration": 4e8, "load_duration": 1e7})
    monkeypatch.setattr(llm.requests, "post", post)
    tags, err = llm.analyze_report(TEXT, context="[c1] x", findings_text="- TG 156", chunk_ids=["c1"],
                                   findings=FINDINGS, rules_cover_labs=True)
    assert err == ""
    # text-stated only: "三酸甘油脂 156" names a lab item, "胃潰瘍" is not in the report text
    assert [c["text"] for c in tags["conditions"]] == ["輕度脂肪肝"]
    assert tags["conditions"][0]["src"] == "report" and tags["risks"] == []
    assert tags["avoid"] == []                                       # uncited id rejected
    assert [t["conf"] for t in tags["food"]] == [0.6, 0.58]            # prior by src + position
    assert tags["lifestyle"][0]["conf"] == 0.8
    assert tags["metrics"] and tags["_llm"]["out_tokens"] == 120 and tags["_llm"]["decode_s"] == 3.0
    # options identical to every other qwen call (no reload); keep_alive set
    assert sent["options"]["num_ctx"] == llm.config.LLM_NUM_CTX and sent["keep_alive"]
    user = sent["messages"][1]["content"]
    assert "TG 156 ↑" not in user                                    # lab row stripped from report text
    assert "【腹部超音波】輕度脂肪肝" in user and "文字結論段落" in user   # narrative kept + highlighted
    assert sent["messages"][0]["content"] == llm.SYSTEM_PROMPT


def test_v4_tags_merge_with_rules():
    rules = {"conditions": [{"text": "血脂異常", "src": "rule", "conf": 0.9, "evidence": "TG 156"}],
             "risks": [{"text": "心血管疾病", "src": "rule", "conf": 0.7, "evidence": "血脂異常"}]}
    tags = {"conditions": [{"text": "輕度脂肪肝", "src": "report", "conf": 0.9},
                           {"text": "血脂異常", "src": "report", "conf": 0.9}], "risks": []}
    out = pipeline.merge_rule_tags(tags, rules)
    assert [c["text"] for c in out["conditions"]] == ["血脂異常", "輕度脂肪肝"]
    assert out["conditions"][0]["src"] == "rule" and [r["text"] for r in out["risks"]] == ["心血管疾病"]


def test_streaming_matches_non_streaming(monkeypatch):
    raw = json.dumps(V4, ensure_ascii=False)
    pieces = [raw[i:i + 7] for i in range(0, len(raw), 7)]
    lines = [json.dumps({"message": {"content": p}, "done": False}).encode() for p in pieces]
    lines.append(json.dumps({"done": True, "done_reason": "stop", "eval_count": 50}).encode())
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(None, lines=lines)
                        if k.get("stream") else FakeResp(V4))
    seen = []
    streamed, err = llm.analyze_report(TEXT, chunk_ids=["c1"], findings=FINDINGS, rules_cover_labs=True,
                                       on_text=seen.append)
    plain, _ = llm.analyze_report(TEXT, chunk_ids=["c1"], findings=FINDINGS, rules_cover_labs=True)
    assert err == "" and len(seen) == len(pieces) and seen[-1] == raw
    strip = lambda d: {k: v for k, v in d.items() if k != "_llm"}  # noqa: E731
    assert strip(streamed) == strip(plain) and streamed["_llm"]["out_tokens"] == 50


def test_partial_tags_reads_completed_objects_only():
    raw = ('{"summary": "血脂偏高", "text_conditions": [{"text": "脂肪肝"}], '
           '"lifestyle": [{"text": "規律運動", "src": "kb:c1"}, {"text": "早')
    p = llm.partial_tags(raw)
    assert p["summary"] == "血脂偏高" and p["_summary_done"]
    assert [t["text"] for t in p["conditions"]] == ["脂肪肝"]
    assert [t["text"] for t in p["lifestyle"]] == ["規律運動"]      # the unfinished tag is skipped
    assert llm.partial_tags('{"summ') == {} and llm.partial_tags("") == {}


def test_strip_lab_lines_keeps_narrative():
    text = ("項目 結果 單位 參考範圍\n三酸甘油脂 TG 156 ↑ mg/dL <150\n"
            "醫師建議：三酸甘油脂 156 偏高，請減少精緻糖並三個月後複檢\n【腹部超音波】輕度脂肪肝")
    out, n = llm.strip_lab_lines(text, FINDINGS)
    assert n == 2 and "TG 156 ↑" not in out
    assert "請減少精緻糖" in out and "輕度脂肪肝" in out
    assert llm.strip_lab_lines(text, []) == (text, 0)


# ─── batched / cached query embeddings ───────────────────────────────────────

def test_query_embeddings_are_batched_and_cached(monkeypatch):
    calls = []

    def fake_batch(inputs, model, base_url, retries=1):
        calls.append(list(inputs))
        return [[float(len(s))] for s in inputs], ""
    monkeypatch.setattr(embeddings, "_embed_batch", fake_batch)
    embeddings.clear_query_cache()
    n, err = embeddings.prefetch_queries(["q1", "q2", "q1"], "m", "http://x")
    assert (n, err) == (2, "") and len(calls) == 1 and len(calls[0]) == 2
    vec, err = embeddings.embed_query("q2", "m", "http://x")
    assert err == "" and len(calls) == 1                               # served from the cache
    assert embeddings.prefetch_queries(["q1", "q2"], "m", "http://x") == (0, "") and len(calls) == 1
    embeddings.embed_query("q3", "m", "http://x")
    assert len(calls) == 2                                             # miss -> single call, then cached
    embeddings.embed_query("q3", "m", "http://x")
    assert len(calls) == 2
    embeddings.clear_query_cache()


def test_pipeline_prefetches_all_finding_queries_in_one_call(monkeypatch):
    got = []
    monkeypatch.setattr(pipeline, "prefetch_queries", lambda qs, m, b, p: got.append(qs) or (len(qs), ""))

    class R:
        col, mode, embed_model, base_url, use_prefix = object(), "dense", "m", "http://x", True
    abn = [{"display_name": f"item{i}", "direction": "high"} for i in range(10)]
    pipeline.prefetch_query_embeddings(R(), abn)
    assert len(got) == 1 and len(got[0]) == pipeline.MAX_QUERY_FINDINGS
    pipeline.prefetch_query_embeddings(R(), [])
    assert got[-1] == [pipeline.GENERAL_QUERY]
    pipeline.prefetch_query_embeddings(pipeline.default_retriever(), abn)   # no dense index -> no call
    assert len(got) == 2


# ─── report cache ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not PDF.exists(), reason="synthetic sample PDF missing")
def test_report_cache_hit_miss_and_no_report_text(tmp_path):
    cache = ReportCache(str(tmp_path), enabled=True)
    pdf = PDF.read_bytes()
    r = pipeline.analyze_pdf(pdf, run_llm=False, ocr=False)
    r.tags = {"summary": "s", "food": [{"text": "少吃油炸", "src": "inferred", "conf": 0.6}]}
    fp = config_fingerprint(sex=None)
    key = cache_key(pdf, fp)
    assert cache.get(key) is None                                      # miss
    assert cache.put(key, r)
    hit, _ = cache.get(key)                                            # hit
    assert hit.findings == r.findings and hit.tags == r.tags and hit.rules == r.rules
    assert hit.text == "" and hit.tables == []                         # PII-bearing text not stored
    stored = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert "A000000047" not in stored and "假名者" not in stored        # ID / name printed on the report
    assert cache.get(cache_key(pdf, config_fingerprint(sex="F"))) is None   # other options -> miss
    assert cache.get(cache_key(pdf + b" ", fp)) is None                     # other bytes -> miss
    r.error = "boom"
    assert not cache.put(cache_key(pdf, "x"), r)                       # errors are never cached
    assert ReportCache(str(tmp_path), enabled=False).get(key) is None
    assert cache.clear() == 1 and cache.get(key) is None


@pytest.mark.skipif(not PDF.exists(), reason="synthetic sample PDF missing")
def test_ui_serves_second_run_from_cache(tmp_path, monkeypatch):
    import ui
    res = ui.build_resources(stub=True)
    res.cache = ReportCache(str(tmp_path), enabled=True)
    db = str(tmp_path / "runs.sqlite3")
    partials = []
    a1 = ui.run_analysis(PDF.read_bytes(), res, runs_db=db, on_partial=partials.append)
    assert not a1.cached and partials and partials[0].preliminary
    assert partials[0].tags["conditions"] and partials[0].tags["metrics"]      # rules shown early
    assert a1.result.timing["first_result_s"] <= a1.result.timing["total_s"]
    monkeypatch.setattr(ui, "_analyze_with_stages", lambda *a, **k: pytest.fail("cache miss"))
    a2 = ui.run_analysis(PDF.read_bytes(), res, runs_db=db)
    assert a2.cached and a2.tags == a1.tags and a2.result.findings == a1.result.findings
    assert a2.report_date == a1.report_date
    assert "快取" in ui.render_perf(a2, res)
