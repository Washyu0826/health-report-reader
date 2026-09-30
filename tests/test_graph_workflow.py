"""LangGraph generate -> verify -> rewrite workflow (graph_workflow.py) and the
batched support judge (verify.llm_support_judge). No network, no LLM: the
generate / judge / rewrite callables and requests.post are faked."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("langgraph")

import verify  # noqa: E402
from graph_workflow import MAX_REWRITES, mermaid, run_graph  # noqa: E402
from pipeline import analyze_pdf  # noqa: E402
from verify import llm_support_judge, verify_tags  # noqa: E402

PDF = ROOT / "eval" / "synth" / "samples" / "syn_002.pdf"
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="synthetic sample PDF missing")

# Latin nonsense so the lexical check never passes and every KB tag reaches the judge.
GOOD, BAD = "QXGOOD", "QXBAD"


def fake_generate(advice):
    """advice: {category: [text, ...]} -> analyze_report-compatible callable citing the first chunk."""
    def gen(**kw):
        ids = kw["chunk_ids"]
        assert ids, "fact-sheet retriever should return at least one chunk"
        out = {"summary": "s", "conditions": [], "risks": [], "metrics": [],
               "lifestyle": [], "food": [], "exercise": [], "supplements": [], "avoid": []}
        for k, texts in advice.items():
            out[k] = [{"text": t, "src": f"kb:{ids[0]}", "conf": 0.8} for t in texts]
        return out, ""
    return gen


def judge(pairs):
    return [GOOD in p["tag"] for p in pairs]


class Rewriter:
    """Fake rewriter: returns `reply(item)` per item and records calls."""
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, items, context_text, findings_text, allowed_ids):
        self.calls.append(items)
        return [{"i": i, **self.reply(it, allowed_ids)} for i, it in enumerate(items)]


def run(advice, rewriter=None, **kw):
    kw.setdefault("rewrite", True)  # the rewrite loop is opt-in (config.VERIFY_REWRITE=0)
    return run_graph(PDF.read_bytes(), ocr=False, generate_fn=fake_generate(advice), judge=judge,
                     rewrite_fn=rewriter or Rewriter(lambda it, ids: {"action": "drop"}), **kw)


def texts(r, k):
    return [t["text"] for t in r.tags.get(k, [])]


def test_no_unsupported_means_no_rewrite():
    def boom(*a):
        raise AssertionError("rewrite must not be called")
    r = run({"food": [GOOD + "1"], "exercise": [GOOD + "2"]}, rewriter=boom)
    assert r.error == ""
    assert texts(r, "food") == [GOOD + "1"] and texts(r, "exercise") == [GOOD + "2"]
    assert "rewrite" not in r.node_trace and r.rewrite_log == []
    assert r.tags["_verification"]["kb_dropped"] == 0
    assert r.tags["_verification"]["rewrite"]["rounds"] == 0


def test_unsupported_is_rewritten_verified_and_kept_with_history():
    rw = Rewriter(lambda it, ids: {"action": "revise", "text": GOOD + "fixed", "src": f"kb:{ids[-1]}"})
    r = run({"food": [BAD + "1", GOOD + "keep"]}, rewriter=rw)
    assert len(rw.calls) == 1 and rw.calls[0][0]["category"] == "food"
    assert texts(r, "food") == [GOOD + "fixed", GOOD + "keep"]      # replaced in place
    fixed = r.tags["food"][0]
    assert fixed["revised_from"] == BAD + "1" and fixed["verdict"] == "llm_supported"
    assert not any(k.startswith("_") for k in fixed)
    (entry,) = r.rewrite_log
    assert entry["original"]["text"] == BAD + "1" and entry["revised"]["text"] == GOOD + "fixed"
    assert entry["outcome"] == "kept" and entry["verdict_after"] == "llm_supported"
    st = r.tags["_verification"]
    assert st["kb_dropped"] == 0
    assert st["rewrite"] == {"enabled": True, "rounds": 1, "unsupported_initial": 1, "revised": 1,
                             "dropped_by_rewriter": 0, "kept": 1, "dropped_after_reverify": 0,
                             "unverified_after_reverify": 0}
    assert r.node_trace.count("verify") == 2


def test_still_unsupported_after_rewrite_is_dropped():
    rw = Rewriter(lambda it, ids: {"action": "revise", "text": BAD + "again", "src": f"kb:{ids[0]}"})
    r = run({"avoid": [BAD + "1"]}, rewriter=rw)
    assert texts(r, "avoid") == []
    assert r.rewrite_log[0]["outcome"] == "dropped"
    assert r.tags["_verification"]["kb_dropped"] == 1
    assert r.tags["_verification"]["rewrite"]["dropped_after_reverify"] == 1


def test_rewriter_drop_and_invalid_revision_are_dropped_and_logged():
    replies = iter([{"action": "drop", "text": "", "src": "x"},
                    {"action": "revise", "text": GOOD, "src": "kb:not-retrieved"}])
    rw = Rewriter(lambda it, ids: next(replies))
    r = run({"lifestyle": [BAD + "1", BAD + "2"]}, rewriter=rw)
    assert texts(r, "lifestyle") == []
    assert [e["action"] for e in r.rewrite_log] == ["drop", "invalid_revision"]
    assert r.tags["_verification"]["kb_dropped"] == 2


def test_only_advice_is_rewritten_and_long_revisions_rejected():
    def boom(*a):
        raise AssertionError("risks are not rewritten")
    r = run({"risks": [BAD + "risk"]}, rewriter=boom)
    assert BAD + "risk" not in texts(r, "risks") and r.rewrite_log == []
    assert r.tags["_verification"]["kb_dropped"] == 1

    rw = Rewriter(lambda it, ids: {"action": "revise", "text": GOOD + "x" * 20, "src": f"kb:{ids[0]}"})
    r2 = run({"food": [BAD + "1"]}, rewriter=rw)
    assert texts(r2, "food") == [] and r2.rewrite_log[0]["action"] == "invalid_revision"


def test_max_rewrites_respected():
    always_bad = Rewriter(lambda it, ids: {"action": "revise", "text": BAD + "x", "src": f"kb:{ids[0]}"})
    r = run({"food": [BAD + "1"]}, rewriter=always_bad)
    assert MAX_REWRITES == 1 and len(always_bad.calls) == 1
    assert r.node_trace.count("rewrite") == 1 and texts(r, "food") == []

    twice = Rewriter(lambda it, ids: {"action": "revise", "text": BAD + "y", "src": f"kb:{ids[0]}"})
    r2 = run({"food": [BAD + "1"]}, rewriter=twice, max_rewrites=2)
    assert len(twice.calls) == 2 and r2.tags["_verification"]["rewrite"]["rounds"] == 2
    assert [e["outcome"] for e in r2.rewrite_log] == ["superseded", "dropped"]


def test_no_rewrite_matches_drop_baseline():
    r = run({"food": [BAD + "1", GOOD + "2"]}, rewrite=False)
    assert texts(r, "food") == [GOOD + "2"] and r.rewrite_log == []
    assert r.tags["_verification"]["kb_dropped"] == 1
    assert "rewrite" not in r.node_trace


def test_rewrite_is_off_by_default(monkeypatch):
    import config
    import graph_workflow
    monkeypatch.setattr(config, "VERIFY_REWRITE", False)

    def boom(*a):
        raise AssertionError("rewrite must be opt-in")
    r = run_graph(PDF.read_bytes(), ocr=False, generate_fn=fake_generate({"food": [BAD + "1", GOOD + "2"]}),
                  judge=judge, rewrite_fn=boom)
    assert texts(r, "food") == [GOOD + "2"] and "rewrite" not in r.node_trace
    assert r.tags["_verification"]["rewrite"]["enabled"] is False
    assert graph_workflow.GraphDeps(checkitem_lookup={}, retriever=None).rewrite is config.VERIFY_REWRITE


def test_generate_is_told_rules_cover_labs():
    seen = {}

    def gen(**kw):
        seen.update(kw)
        return fake_generate({"food": [GOOD]})(**kw)
    run_graph(PDF.read_bytes(), ocr=False, generate_fn=gen, judge=judge)
    assert seen["rules_cover_labs"] is True
    run_graph(PDF.read_bytes(), ocr=False, generate_fn=gen, judge=judge, use_rules=False)
    assert seen["rules_cover_labs"] is False


def test_timings_and_trace_present():
    rw = Rewriter(lambda it, ids: {"action": "revise", "text": GOOD, "src": f"kb:{ids[0]}"})
    r = run({"food": [BAD + "1"]}, rewriter=rw)
    for key in ("extract_s", "abnormal_s", "rules_s", "retrieve_s", "llm_s", "merge_s",
                "verify_s", "rewrite_s", "finalize_s"):
        assert key in r.timing and r.timing[key] >= 0
    assert r.node_trace[0] == "extract" and r.node_trace[-1] == "finalize"
    assert set(r.node_trace[2:4]) == {"derive_rules", "retrieve"}      # parallel superstep


def test_result_shape_matches_pipeline():
    base = analyze_pdf(PDF.read_bytes(), run_llm=False, ocr=False)
    r = run({"food": [GOOD]})
    assert r.findings == base.findings and [c["id"] for c in r.chunks] == [c["id"] for c in base.chunks]
    assert r.sex == base.sex and r.text == base.text


def test_on_update_streams_node_updates():
    seen = []
    run({"food": [GOOD]}, on_update=lambda node, upd: seen.append(node))
    assert seen[0] == "extract" and seen[-1] == "finalize" and "generate" in seen


def test_generate_error_short_circuits():
    r = run_graph(PDF.read_bytes(), ocr=False, generate_fn=lambda **kw: ({}, "boom"), judge=judge,
                  rewrite_fn=lambda *a: [])
    assert r.error == "boom" and r.error_stage == "llm" and r.tags == {}
    assert "verify" not in r.node_trace


def test_mermaid_contains_loop():
    m = mermaid()
    assert "rewrite --> verify" in m and "derive_rules" in m


# ─── verify_tags option ──────────────────────────────────────────────────────

CHUNKS = [{"id": "c1", "text": "多吃蔬菜水果，減少紅肉"}]


def _tags():
    return {"summary": "", "food": [{"text": "QXBAD", "src": "kb:c1", "conf": 0.9},
                                    {"text": "多吃蔬菜", "src": "kb:c1", "conf": 0.9}]}


def test_verify_tags_default_still_drops():
    out = verify_tags(_tags(), CHUNKS, "", [], judge=lambda p: [False] * len(p))
    assert [t["text"] for t in out["food"]] == ["多吃蔬菜"]
    assert out["_verification"]["kb_dropped"] == 1 and "kb_unsupported" not in out["_verification"]


def test_verify_tags_can_keep_unsupported_marked():
    out = verify_tags(_tags(), CHUNKS, "", [], judge=lambda p: [False] * len(p), drop_unsupported=False)
    assert [t["verdict"] for t in out["food"]] == ["unsupported", "lexical"]
    assert out["_verification"]["kb_dropped"] == 0 and out["_verification"]["kb_unsupported"] == 1


# ─── llm_support_judge robustness (fake Ollama) ──────────────────────────────

class FakeResp:
    def __init__(self, status=200, content=None):
        self.status_code, self._content = status, content

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return {"message": {"content": json.dumps(self._content)}, "done_reason": "stop"}


def _n_pairs(payload):
    return payload["format"]["properties"]["results"]["maxItems"]


def test_judge_batches_and_maps_by_index(monkeypatch):
    calls = []

    def post(url, json=None, timeout=None):
        n = _n_pairs(json)
        calls.append(n)
        # answer in REVERSE order: index mapping, not position, must be used
        return FakeResp(content={"results": [{"i": i, "supported": i % 2 == 0} for i in reversed(range(n))]})

    monkeypatch.setattr(verify.requests, "post", post)
    pairs = [{"tag": f"t{i}", "passage": "段" * 500} for i in range(30)]
    out = llm_support_judge()(pairs)
    assert len(out) == 30 and None not in out
    assert max(calls) <= verify.JUDGE_BATCH and len(calls) >= 4
    # each batch restarts local indices at 0 -> verdict parity follows the local index
    starts = [sum(calls[:b]) for b in range(len(calls))]
    for s, n in zip(starts, calls):
        assert out[s:s + n] == [i % 2 == 0 for i in range(n)]


def test_judge_retries_missing_and_http_errors(monkeypatch):
    replies = iter([
        FakeResp(500),                                                     # runner crash
        FakeResp(content={"results": [{"i": 0, "supported": True}]}),      # retry: only #0 answered
        FakeResp(content={"results": [{"i": 0, "supported": False}]}),     # batch 2 first try
    ])
    sizes = []

    def post(url, json=None, timeout=None):
        sizes.append(_n_pairs(json))
        return next(replies)

    monkeypatch.setattr(verify.requests, "post", post)
    out = llm_support_judge(batch_size=2)([{"tag": "a", "passage": "p"}, {"tag": "b", "passage": "p"},
                                           {"tag": "c", "passage": "p"}])
    # batch1 [a,b]: 500 -> retry answers a only -> b stays None (retries exhausted); batch2 [c]: False
    assert out == [True, None, False]
    assert sizes == [2, 2, 1]


def test_judge_retry_asks_only_for_missing(monkeypatch):
    replies = iter([FakeResp(content={"results": [{"i": 1, "supported": True}]}),
                    FakeResp(content={"results": [{"i": 0, "supported": False}]})])
    sizes = []

    def post(url, json=None, timeout=None):
        sizes.append(_n_pairs(json))
        return next(replies)

    monkeypatch.setattr(verify.requests, "post", post)
    out = llm_support_judge()([{"tag": "a", "passage": "p"}, {"tag": "b", "passage": "p"}])
    assert out == [False, True] and sizes == [2, 1]


def test_echoed_revision_is_not_reverified():
    rw = Rewriter(lambda it, ids: {"action": "revise", "text": it["text"], "src": it["src"]})
    r = run({"food": [BAD + "1"]}, rewriter=rw)
    assert texts(r, "food") == [] and r.rewrite_log[0]["action"] == "unchanged"
    assert r.node_trace.count("verify") == 2      # loop still closes; nothing left to re-check


# ─── single orchestration path: pipeline.analyze_pdf runs this graph ─────────

def test_analyze_pdf_runs_the_graph():
    r = analyze_pdf(PDF.read_bytes(), run_llm=False, ocr=False)
    assert r.node_trace[0] == "extract" and r.node_trace[-1] == "finalize"
    assert r.findings and r.chunks
    assert r.tags == {} and r.cited_ids == [] and "llm_s" not in r.timing   # generate was a no-op


def test_on_stage_order_and_one_result_object():
    seen, objs = [], []

    def on_stage(stage, r):
        seen.append((stage, len(r.findings), bool(r.rules)))
        objs.append(r)
        if stage == "findings":
            r.timing["hook_mark"] = 1.0   # the UI records timing["first_result_s"] this way
    r = run_graph(PDF.read_bytes(), ocr=False, generate_fn=fake_generate({"food": [GOOD]}), judge=judge,
                  on_stage=on_stage)
    assert [s for s, *_ in seen] == ["extract", "findings", "retrieve", "llm", "verify"]
    assert seen[1][1] > 0 and seen[1][2]              # findings + rule tags ready at "findings"
    assert all(o is r for o in objs) and r.timing["hook_mark"] == 1.0


def test_llm_text_is_streamed_through_the_graph():
    got = []

    def gen(**kw):
        kw["on_text"]('{"summary": "a')
        kw["on_text"]('{"summary": "ab"')
        return fake_generate({"food": [GOOD]})(**kw)
    run_graph(PDF.read_bytes(), ocr=False, generate_fn=gen, judge=judge, on_llm_text=got.append)
    assert got == ['{"summary": "a', '{"summary": "ab"']


def test_no_stream_callback_means_no_on_text():
    seen = {}

    def gen(**kw):
        seen.update(kw)
        return fake_generate({"food": [GOOD]})(**kw)
    run_graph(PDF.read_bytes(), ocr=False, generate_fn=gen, judge=judge)
    assert "on_text" not in seen


def test_langsmith_tracing_is_forced_off(monkeypatch):
    """A LangSmith trace would carry the report text off the machine, so run_graph turns
    tracing off even when the environment turns it on. (Measured: with LANGSMITH_TRACING=true
    and no guard, LangGraph POSTs the run to the tracing endpoint.)"""
    lu = pytest.importorskip("langsmith.utils")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_fake")
    lu.get_env_var.cache_clear()
    try:
        assert lu.tracing_is_enabled()                # control: the environment asks for tracing
        seen = []

        def gen(**kw):
            seen.append(lu.tracing_is_enabled())
            return fake_generate({"food": [GOOD]})(**kw)
        run_graph(PDF.read_bytes(), ocr=False, generate_fn=gen, judge=judge)
        assert seen == [False]
    finally:
        monkeypatch.undo()
        lu.get_env_var.cache_clear()


# ─── nurse review (LangGraph interrupt) ──────────────────────────────────────

NORMAL_PDF = ROOT / "eval" / "synth" / "samples" / "syn_001.pdf"   # all normal: no rule tags


def test_review_off_by_default_never_pauses():
    r = run_graph(PDF.read_bytes(), ocr=False, generate_fn=fake_generate({"food": [GOOD]}), judge=judge)
    assert r.pending_review == {} and r.review == {} and r.tags


def test_review_pauses_before_the_llm_and_resumes_without_removed_items():
    import graph_workflow
    calls = []

    def gen(**kw):
        calls.append(kw["findings_text"])
        out, err = fake_generate({"food": [GOOD]})(**kw)
        out["conditions"] = [{"text": "過重", "src": "report", "conf": 0.9}]  # LLM repeats a removed one
        return out, err
    r = run_graph(PDF.read_bytes(), ocr=False, generate_fn=gen, judge=judge, review=True)
    p = r.pending_review
    assert p["token"] and not calls and r.tags == {}                    # paused before generate
    assert "過重" in [c["text"] for c in p["conditions"]] and p["conditions"][0]["evidence"]
    assert r.findings and r.rules["conditions"]                          # findings/rules already shown

    r2 = graph_workflow.resume_review(p["token"], {"remove": {"conditions": ["過重", "not-a-tag"],
                                                             "risks": ["痛風"]}})
    assert r2 is r and r2.pending_review == {}
    assert "過重" not in texts(r2, "conditions") and "痛風" not in texts(r2, "risks")
    assert "血脂異常" in texts(r2, "conditions") and texts(r2, "food") == [GOOD]
    assert r2.review["removed"] == {"conditions": ["過重"], "risks": ["痛風"]}   # unknown text ignored
    assert "過重" not in calls[0]                                        # the LLM never saw it
    assert token_is_released(p["token"])


def token_is_released(token):
    import graph_workflow
    return token not in graph_workflow._PAUSED


def test_review_with_nothing_flagged_does_not_pause():
    def gen(**kw):  # an all-normal report retrieves no chunks, so cite nothing
        return {"summary": "s", "conditions": [], "risks": [], "metrics": [], "lifestyle": [],
                "food": [], "exercise": [], "supplements": [], "avoid": []}, ""
    r = run_graph(NORMAL_PDF.read_bytes(), ocr=False, generate_fn=gen, judge=judge, review=True)
    assert r.pending_review == {} and r.tags["summary"] == "s"


def test_resume_streams_the_remaining_stages():
    import graph_workflow
    first, second, text = [], [], []
    r = run_graph(PDF.read_bytes(), ocr=False, judge=judge, review=True,
                  generate_fn=fake_generate({"food": [GOOD]}), on_stage=lambda s, _: first.append(s))

    def gen(**kw):
        kw["on_text"]('{"summary": "x"')
        return fake_generate({"food": [GOOD]})(**kw)
    graph_workflow._PAUSED[r.pending_review["token"]].deps.generate_fn = gen
    graph_workflow.resume_review(r.pending_review["token"], {"remove": {}},
                                 on_stage=lambda s, _: second.append(s), on_llm_text=text.append)
    assert first == ["extract", "findings", "retrieve"] and second == ["llm", "verify"]
    assert text == ['{"summary": "x"']


def test_resume_unknown_token_and_discard():
    import graph_workflow
    with pytest.raises(KeyError):
        graph_workflow.resume_review("nope", {})
    r = run_graph(PDF.read_bytes(), ocr=False, generate_fn=fake_generate({}), judge=judge, review=True)
    assert graph_workflow.discard_review(r.pending_review["token"]) is True
    assert token_is_released(r.pending_review["token"])
