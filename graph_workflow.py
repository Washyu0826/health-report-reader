"""
graph_workflow.py — LangGraph "generate -> verify -> rewrite" workflow (R6).

Same stages as pipeline.analyze_pdf (the functions are shared, not copied),
but orchestrated as a LangGraph StateGraph with one extra loop:

    extract -> detect_abnormal -> {derive_rules || retrieve} -> generate
            -> merge_rules -> verify --(unsupported & round < MAX)--> rewrite -> verify
                                     \\-> finalize

Why: the linear pipeline DROPS advice tags whose cited passage does not
support them (measured: 39/143 KB-cited tags on 12 internal reports, with a
bias against biomarker-fact-sheet citations). Here unsupported tags go to ONE
batched, schema-constrained rewrite call that may (a) revise the tag so a
passage the model was shown supports it, (b) re-cite a better retrieved
passage, or (c) drop it. Revised tags are verified again; whatever is still
unsupported after MAX_REWRITES rounds is dropped. Every decision is logged
(`rewrite_log`) for the UI and the eval.

The LLM-facing callables (generate / judge / rewrite) are injectable so the
graph is testable without Ollama (tests/test_graph_workflow.py).

    from graph_workflow import run_graph
    r = run_graph(pdf_bytes, retriever=...)                 # PipelineResult + rewrite_log
    r = run_graph(pdf_bytes, retriever=..., rewrite=True)   # + rewrite loop (off by default:
                                                            #   config.VERIFY_REWRITE=0)
    python graph_workflow.py --mermaid docs/graph.md        # export the diagram
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import config
from abnormal import abnormal_only, load_checkitem_lookup
from llm import _chat, analyze_report, llm_options
from pipeline import (
    NO_TEXT_ERROR, PipelineResult, default_retriever, detect_stage, extract_stage, findings_prompt,
    merge_rule_tags, retrieve_stage, rules_stage,
)
from retrieval import select_context
from verify import ADVICE_KEYS, ALL_KEYS, llm_support_judge, verify_tags

MAX_REWRITES = 1
# Revised tags must stay tag-sized: in the smoke run the rewriter otherwise copied
# whole passage sentences, which passes the lexical check trivially.
REWRITE_MAX_CHARS = 16
# Only advice is rewritten; an unsupported KB-cited condition/risk is dropped as before
# (the smoke run turned a risk tag into advice when allowed to rewrite it).
REWRITE_KEYS = ADVICE_KEYS

GenerateFn = Callable[..., Tuple[Dict, str]]                    # llm.analyze_report signature
JudgeFn = Callable[[List[Dict]], List[Optional[bool]]]           # verify.llm_support_judge()
# rewrite_fn(items, context_text, findings_text, allowed_ids) -> [{i, action, text?, src?}]
RewriteFn = Callable[[List[Dict], str, str, List[str]], List[Dict]]


# ─── State ───────────────────────────────────────────────────────────────────

def _sum_timings(old: Optional[Dict[str, float]], new: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Reducer: parallel branches write their own keys; a node that runs twice
    (verify) accumulates its time."""
    out = dict(old or {})
    for k, v in (new or {}).items():
        out[k] = round(out.get(k, 0.0) + v, 3)
    return out


class GraphState(TypedDict, total=False):
    pdf_bytes: bytes
    text: str
    tables: List
    used_ocr: bool
    findings: List[Dict]
    abnormal_debug: Dict
    sex: Optional[str]
    chunks: List[Dict]
    rag_debug: Dict
    cited_ids: List[str]
    context_text: str
    rules: Dict[str, List[Dict]]
    tags: Dict
    verification: Dict          # first-pass verify_tags stats (+ "rewrite" block at finalize)
    rewrite_round: int
    rewrite_log: List[Dict]
    error: str
    error_stage: str
    timing: Annotated[Dict[str, float], _sum_timings]
    errors: Annotated[List[str], operator.add]
    trace: Annotated[List[str], operator.add]


@dataclass
class GraphDeps:
    """Everything the nodes need that is not per-report state."""
    checkitem_lookup: Any
    retriever: Any
    model: str = config.LLM_MODEL
    base_url: str = config.OLLAMA_BASE_URL
    sex: Optional[str] = None
    verify: bool = True
    use_rules: bool = True
    ocr: bool = True
    llm_normalize: bool = False
    web: Any = None
    rewrite: bool = config.VERIFY_REWRITE
    max_rewrites: int = MAX_REWRITES
    generate_fn: GenerateFn = analyze_report
    judge: Optional[JudgeFn] = None
    rewrite_fn: Optional[RewriteFn] = None


# ─── Rewrite call ────────────────────────────────────────────────────────────
# Transport choice: the same requests + Ollama `format` JSON-schema path as
# llm.py (llm._chat), not ChatOllama.with_structured_output. Reasons:
#   * the `src` enum is built per request from the retrieved ids — a plain dict
#     schema is the most direct way to express that grammar constraint;
#   * identical options (num_ctx = config.LLM_NUM_CTX) keep Ollama from
#     reloading the model between generate / judge / rewrite;
#   * eval/run_eval.py counts and times chat calls by wrapping requests.post;
#     ChatOllama goes through the ollama/httpx client and would be invisible
#     to that instrumentation.

REWRITE_SYSTEM = """你是健康建議標籤的修訂員。下列建議標籤經查核「引用的參考資料段落並不支持該建議」。
對每個標籤擇一：
- action="revise"：改寫成 2–12 字的短標籤（不是句子，不要照抄整句段落），且必須能在 src 所指的段落中找到明確依據（段落明確提到或直接蘊含）。可改引用其他更合適的段落（src 只能使用 schema 允許的 id）。原標籤＋原引用已被判定不支持，不可原封不動照抄。修訂後仍須針對這位受檢者的異常項目，並保持原類別（lifestyle/food/exercise/supplements/avoid）。
- action="drop"：若沒有任何段落支持與此受檢者異常相關的同類建議，就刪除。寧缺勿濫，不可捏造。
所有文字使用繁體中文（台灣用語）。只輸出符合 schema 的 JSON；action="drop" 時 text 請填空字串、src 任填一個允許值。"""


def build_rewrite_schema(n: int, allowed_src: List[str]) -> dict:
    return {"type": "object", "properties": {"items": {
        "type": "array", "minItems": n, "maxItems": n,
        "items": {"type": "object", "properties": {
            "i": {"type": "integer", "enum": list(range(n))},
            "action": {"type": "string", "enum": ["revise", "drop"]},
            "text": {"type": "string"},
            "src": {"type": "string", "enum": allowed_src}},
            "required": ["i", "action", "text", "src"]}}},
        "required": ["items"]}


def build_rewrite_message(items: List[Dict], context_text: str, findings_text: str) -> str:
    rows = "\n".join(f"[{i}] 類別={it['category']}　原標籤：{it['text']}　原引用：{it['src']}"
                     for i, it in enumerate(items))
    return (f"# 受檢者的異常檢驗值\n{findings_text or '（無異常值）'}\n\n---\n"
            f"# 參考資料片段（每段開頭 [id]，引用時寫 kb:<id>）\n\n{context_text}\n\n---\n"
            f"# 待修訂標籤（共 {len(items)} 個，編號 0–{len(items) - 1}）\n{rows}\n\n請輸出 JSON。")


def ollama_rewriter(model: str = config.LLM_MODEL, base_url: str = config.OLLAMA_BASE_URL,
                    retries: int = 1) -> RewriteFn:
    """ONE batched, schema-constrained rewrite call (+1 retry on failure).
    Returns [] when no usable answer -> the caller drops every item (= baseline)."""
    import json

    def rewrite(items: List[Dict], context_text: str, findings_text: str,
                allowed_ids: List[str]) -> List[Dict]:
        allowed = [f"kb:{cid}" for cid in allowed_ids]
        if not items or not allowed:
            return []
        payload = {
            "model": model, "stream": False,
            "format": build_rewrite_schema(len(items), allowed),
            "messages": [{"role": "system", "content": REWRITE_SYSTEM},
                         {"role": "user", "content": build_rewrite_message(items, context_text, findings_text)}],
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": llm_options(num_predict=96 + 64 * len(items)),
        }
        for _attempt in range(retries + 1):
            data, err = _chat(payload, base_url)
            if err or data.get("done_reason") == "length":
                continue
            try:
                return json.loads((data.get("message") or {}).get("content", "")).get("items", [])
            except (ValueError, AttributeError):
                continue
        return []
    return rewrite


# ─── Nodes ───────────────────────────────────────────────────────────────────

def _timed(name: str, key: str, fn: Callable[[GraphState], Dict]) -> Callable[[GraphState], Dict]:
    def node(state: GraphState) -> Dict:
        t = time.time()
        out = fn(state) or {}
        out["timing"] = {key: round(time.time() - t, 3)}
        out["trace"] = [name]
        return out
    node.__name__ = name
    return node


def _strip_internal(tag: Dict) -> Dict:
    return {k: v for k, v in tag.items() if not k.startswith("_")}


def build_graph(deps: GraphDeps):
    judge = deps.judge or (llm_support_judge(deps.model, deps.base_url) if deps.verify else None)
    rewrite_fn = deps.rewrite_fn or ollama_rewriter(deps.model, deps.base_url)

    def extract(state: GraphState) -> Dict:
        content, used_ocr, err = extract_stage(state["pdf_bytes"], ocr=deps.ocr, base_url=deps.base_url)
        if err:
            return {"error": err, "error_stage": "extract", "errors": [err]}
        out = {"text": content["text"], "tables": content["tables"], "used_ocr": used_ocr}
        if not content["text"].strip():
            out.update(error=NO_TEXT_ERROR, error_stage="extract", errors=[NO_TEXT_ERROR])
        return out

    def detect_abnormal(state: GraphState) -> Dict:
        findings, dbg = detect_stage(state["text"], state["tables"], deps.checkitem_lookup, sex=deps.sex,
                                     llm_normalize=deps.llm_normalize, model=deps.model,
                                     base_url=deps.base_url)
        return {"findings": findings, "abnormal_debug": dbg, "sex": dbg.get("sex")}

    def derive_rules(state: GraphState) -> Dict:
        return {"rules": rules_stage(state["findings"], state.get("sex"), deps.use_rules)}

    def retrieve(state: GraphState) -> Dict:
        chunks, rag_debug = retrieve_stage(deps.retriever, abnormal_only(state["findings"]), web=deps.web)
        context_text, cited_ids = select_context(chunks)
        return {"chunks": chunks, "rag_debug": rag_debug, "context_text": context_text,
                "cited_ids": cited_ids}

    def generate(state: GraphState) -> Dict:
        tags, err = deps.generate_fn(
            text=state["text"], context=state["context_text"],
            findings_text=findings_prompt(state["findings"], state["rules"]),
            model=deps.model, base_url=deps.base_url, chunk_ids=state["cited_ids"],
            findings=state["findings"], rules_cover_labs=deps.use_rules)
        if err:
            return {"tags": tags or {}, "error": err, "error_stage": "llm", "errors": [err]}
        return {"tags": tags}

    def merge_rules(state: GraphState) -> Dict:
        return {"tags": merge_rule_tags(dict(state["tags"]), state["rules"])}

    def verify(state: GraphState) -> Dict:
        rnd = state.get("rewrite_round", 0)
        args = (state["chunks"], state["text"], state["findings"])
        if rnd == 0:
            tags = verify_tags(state["tags"], *args, judge=judge, drop_unsupported=False)
            return {"tags": tags, "verification": tags.get("_verification", {})}
        # Re-verify only the tags revised in the last round; everything else keeps its verdict.
        tags = {k: list(v) if k in ALL_KEYS else v for k, v in state["tags"].items()}
        sub = {k: [t for t in tags.get(k, []) if t.get("verdict") == "revised"] for k in ALL_KEYS}
        checked = verify_tags(sub, *args, judge=judge, drop_unsupported=False)
        new = {t["_rw"]: t for k in ALL_KEYS for t in checked.get(k, [])}
        log = [dict(e) for e in state.get("rewrite_log", [])]
        for k in ALL_KEYS:
            tags[k] = [new.get(t["_rw"], t) if t.get("verdict") == "revised" else t for t in tags[k]]
        for idx, t in new.items():
            log[idx]["verdict_after"] = t.get("verdict")
        return {"tags": tags, "rewrite_log": log}

    def rewrite(state: GraphState) -> Dict:
        rnd = state.get("rewrite_round", 0) + 1
        tags = {k: list(v) if k in ALL_KEYS else v for k, v in state["tags"].items()}
        log = [dict(e) for e in state.get("rewrite_log", [])]
        targets = [(k, pos) for k in REWRITE_KEYS for pos, t in enumerate(tags[k])
                   if t.get("verdict") == "unsupported"]
        items = [{"category": k, "text": tags[k][pos]["text"], "src": tags[k][pos]["src"]}
                 for k, pos in targets]
        answers: Dict[int, Dict] = {}
        for a in rewrite_fn(items, state.get("context_text", ""),
                            findings_prompt(state["findings"], state["rules"]), state.get("cited_ids", [])):
            i = a.get("i") if isinstance(a, dict) else None
            if isinstance(i, int) and 0 <= i < len(items) and i not in answers:
                answers[i] = a
        allowed = {f"kb:{cid}" for cid in state.get("cited_ids", [])}
        for i, (k, pos) in enumerate(targets):
            t = tags[k][pos]
            a = answers.get(i, {"action": "no_answer"})
            text = (a.get("text") or "").strip()
            entry = {"round": rnd, "category": k, "original": {"text": t["text"], "src": t["src"]},
                     "action": a.get("action"), "revised": None, "verdict_after": None}
            if "_rw" in t:  # a revision that failed re-verification and is being revised again
                log[t["_rw"]]["outcome"] = "superseded"
            others = {x["text"] for x in tags[k] if x is not None and x is not t}
            echo = text == t["text"] and a.get("src") == t["src"]   # same tag, same rejected passage
            if a.get("action") == "revise" and echo:
                entry["action"] = "unchanged"
            if a.get("action") == "revise" and not echo and 0 < len(text) <= REWRITE_MAX_CHARS \
                    and a.get("src") in allowed and text not in others:
                entry["revised"] = {"text": text, "src": a["src"]}
                tags[k][pos] = {"text": text, "src": a["src"], "conf": t.get("conf", 0.5),
                                "verdict": "revised", "revised_from": t.get("revised_from", t["text"]),
                                "_rw": len(log)}
            else:
                if entry["action"] == "revise":
                    entry["action"] = "invalid_revision"
                entry["outcome"] = "dropped"
                tags[k][pos] = None
            log.append(entry)
        for k in ALL_KEYS:
            tags[k] = [t for t in tags[k] if t is not None]
        return {"tags": tags, "rewrite_log": log, "rewrite_round": rnd}

    def finalize(state: GraphState) -> Dict:
        if state.get("error") or "tags" not in state or "_verification" not in state.get("tags", {}):
            return {}
        tags = {k: list(v) if k in ALL_KEYS else v for k, v in state["tags"].items()}
        log = [dict(e) for e in state.get("rewrite_log", [])]
        final_dropped = 0
        for k in ALL_KEYS:
            keep = []
            for t in tags[k]:
                if t.get("verdict") == "unsupported":
                    final_dropped += 1
                    if "_rw" in t:
                        log[t["_rw"]]["outcome"] = "dropped"
                    continue
                if "_rw" in t:
                    log[t["_rw"]]["outcome"] = "kept"
                keep.append(_strip_internal(t))
            tags[k] = keep
        stats = dict(state.get("verification") or tags["_verification"])
        rewrite_dropped = sum(e["action"] != "revise" for e in log)
        stats["kb_dropped"] = final_dropped + rewrite_dropped
        stats["rewrite"] = {
            "enabled": deps.rewrite, "rounds": state.get("rewrite_round", 0),
            "unsupported_initial": stats.get("kb_unsupported", 0),
            "revised": sum(e["action"] == "revise" for e in log),
            "dropped_by_rewriter": rewrite_dropped,
            "kept": sum(e.get("outcome") == "kept" for e in log),
            "dropped_after_reverify": sum(e["action"] == "revise" and e.get("outcome") == "dropped"
                                          for e in log),
            "unverified_after_reverify": sum(e.get("verdict_after") == "unverified" for e in log),
        }
        tags["_verification"] = stats
        return {"tags": tags, "rewrite_log": log, "verification": stats}

    # ── routing ──
    def after_extract(state: GraphState) -> str:
        return "finalize" if state.get("error") else "detect_abnormal"

    def after_generate(state: GraphState) -> str:
        return "finalize" if state.get("error") else "merge_rules"

    def after_merge(state: GraphState) -> str:
        return "verify" if deps.verify else "finalize"

    def after_verify(state: GraphState) -> str:
        unsupported = any(t.get("verdict") == "unsupported"
                          for k in REWRITE_KEYS for t in state["tags"].get(k, []))
        if deps.rewrite and unsupported and state.get("rewrite_round", 0) < deps.max_rewrites:
            return "rewrite"
        return "finalize"

    g = StateGraph(GraphState)
    for name, key, fn in [("extract", "extract_s", extract), ("detect_abnormal", "abnormal_s", detect_abnormal),
                          ("derive_rules", "rules_s", derive_rules), ("retrieve", "retrieve_s", retrieve),
                          ("generate", "llm_s", generate), ("merge_rules", "merge_s", merge_rules),
                          ("verify", "verify_s", verify), ("rewrite", "rewrite_s", rewrite),
                          ("finalize", "finalize_s", finalize)]:
        g.add_node(name, _timed(name, key, fn))
    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", after_extract, ["detect_abnormal", "finalize"])
    g.add_edge("detect_abnormal", "derive_rules")       # fan-out: both branches run in one superstep
    g.add_edge("detect_abnormal", "retrieve")
    g.add_edge(["derive_rules", "retrieve"], "generate")  # fan-in: waits for both
    g.add_conditional_edges("generate", after_generate, ["merge_rules", "finalize"])
    g.add_conditional_edges("merge_rules", after_merge, ["verify", "finalize"])
    g.add_conditional_edges("verify", after_verify, ["rewrite", "finalize"])
    g.add_edge("rewrite", "verify")
    g.add_edge("finalize", END)
    return g.compile()


# ─── Entry point ─────────────────────────────────────────────────────────────

@dataclass
class GraphResult(PipelineResult):
    rewrite_log: List[Dict] = field(default_factory=list)
    node_trace: List[str] = field(default_factory=list)


def run_graph(
    pdf_bytes: bytes,
    checkitem_lookup=None,
    model: str = config.LLM_MODEL,
    base_url: str = config.OLLAMA_BASE_URL,
    sex: Optional[str] = None,
    retriever=None,
    verify: bool = True,
    use_rules: bool = True,
    ocr: bool = True,
    llm_normalize: bool = False,
    web=None,
    rewrite: Optional[bool] = None,
    max_rewrites: int = MAX_REWRITES,
    generate_fn: Optional[GenerateFn] = None,
    judge: Optional[JudgeFn] = None,
    rewrite_fn: Optional[RewriteFn] = None,
    on_update: Optional[Callable[[str, Dict], None]] = None,
) -> GraphResult:
    """Graph version of pipeline.analyze_pdf (same arguments, same result shape,
    plus `rewrite_log` and `node_trace`). rewrite=None -> config.VERIFY_REWRITE
    (default OFF: drop unsupported advice, the linear pipeline's behaviour);
    rewrite=True enables the generate -> verify -> rewrite loop.
    on_update(node, update) is called as each node finishes (UI progress)."""
    deps = GraphDeps(
        checkitem_lookup=checkitem_lookup if checkitem_lookup is not None
        else load_checkitem_lookup(config.REFERENCE_RANGES_PATH),
        retriever=retriever if retriever is not None else default_retriever(),
        model=model, base_url=base_url, sex=sex, verify=verify, use_rules=use_rules, ocr=ocr,
        llm_normalize=llm_normalize, web=web,
        rewrite=config.VERIFY_REWRITE if rewrite is None else rewrite, max_rewrites=max_rewrites,
        generate_fn=generate_fn or analyze_report, judge=judge, rewrite_fn=rewrite_fn)
    graph = build_graph(deps)
    # supersteps: extract, detect, rules||retrieve, generate, merge, verify, 2 per rewrite round, finalize
    limit = 12 + 2 * max(0, max_rewrites)
    state: Dict = {}
    for mode, chunk in graph.stream({"pdf_bytes": pdf_bytes, "rewrite_round": 0, "rewrite_log": []},
                                    {"recursion_limit": limit}, stream_mode=["updates", "values"]):
        if mode == "values":
            state = chunk
        elif on_update:
            for node, update in chunk.items():
                on_update(node, update or {})

    r = GraphResult()
    r.text, r.tables = state.get("text", ""), state.get("tables", [])
    r.used_ocr = state.get("used_ocr", False)
    r.findings, r.abnormal_debug = state.get("findings", []), state.get("abnormal_debug", {})
    r.rules = state.get("rules", {})
    r.sex = state.get("sex")
    r.chunks, r.rag_debug = state.get("chunks", []), state.get("rag_debug", {})
    r.cited_ids = state.get("cited_ids", [])
    r.tags = state.get("tags", {}) if not state.get("error") or state.get("error_stage") != "llm" else {}
    r.error, r.error_stage = state.get("error", ""), state.get("error_stage", "")
    r.timing = dict(state.get("timing", {}))
    r.rewrite_log = state.get("rewrite_log", [])
    r.node_trace = state.get("trace", [])
    return r


def mermaid(deps: Optional[GraphDeps] = None) -> str:
    """Mermaid source of the compiled graph (no model needed to draw it)."""
    deps = deps or GraphDeps(checkitem_lookup={}, retriever=None, judge=lambda p: [],
                             rewrite_fn=lambda *a: [])
    return build_graph(deps).get_graph().draw_mermaid()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--mermaid", default="docs/graph.md", help="write the diagram to this Markdown file")
    a = ap.parse_args()
    Path(a.mermaid).parent.mkdir(parents=True, exist_ok=True)
    Path(a.mermaid).write_bytes((
        "# Tagging workflow (LangGraph)\n\n"
        "Generated by `python graph_workflow.py --mermaid docs/graph.md` from the compiled graph.\n"
        "Dashed edges are conditional. `derive_rules` and `retrieve` run in the same superstep;\n"
        "`verify -> rewrite -> verify` runs at most `MAX_REWRITES` (1) times, then anything still\n"
        "unsupported is dropped in `finalize`. The rewrite loop is **off by default**\n"
        "(`config.VERIFY_REWRITE=0`; enable with `VERIFY_REWRITE=1` or `run_graph(rewrite=True)`): on the\n"
        "synthetic quick set it rescued 3 of 59 unsupported tags (+3 pts advice relevance) for ~7-9 s per\n"
        "affected report. With it off, `verify` routes straight to `finalize`, which drops unsupported tags.\n\n"
        f"```mermaid\n{mermaid()}```\n").encode("utf-8"))  # LF on every OS
    print(f"wrote {a.mermaid}")
