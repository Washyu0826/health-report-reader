"""
pipeline.py — Analysis stages and the stable entry point, analyze_pdf().

    PDF bytes -> extract text/tables -> rule-based abnormal findings
              -> report-driven retrieval (+ optional web fallback)
              -> schema-constrained LLM tagging -> claim verification

The stage functions live here; graph_workflow.py wires them into a LangGraph
StateGraph, and analyze_pdf() runs that graph (UI, eval and integrations all go
through it). Each stage is timed; the result carries everything the UI needs to
render and the eval needs to score. With run_llm=False no model is needed
(extraction, abnormal detection, rules and retrieval only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import config
from abnormal import (
    abnormal_only, evaluate_findings, extract_lab_values, format_findings_for_llm, make_llm_normalizer,
)
import conditions as cond_rules
from embeddings import prefetch_queries
from ocr import is_scanned, ocr_pdf
from pdf_utils import extract_pdf_content
from retrieval import GENERAL_QUERY, CheckitemFacts, Retriever, finding_query

_DEFAULT_FACTS: Optional[CheckitemFacts] = None


def default_retriever() -> Retriever:
    """Fact-sheet-only retriever (no vector index, no embeddings, no network)."""
    global _DEFAULT_FACTS
    if _DEFAULT_FACTS is None:
        _DEFAULT_FACTS = CheckitemFacts(config.REFERENCE_RANGES_PATH)
    return Retriever(None, facts=_DEFAULT_FACTS)


def add_web_chunks(chunks: List[Dict], abnormal_findings: List[Dict], web) -> Tuple[List[Dict], int]:
    """Insert at most one web result per finding right after the fact sheets,
    so the context budget in select_context() cannot silently drop them.
    `web` is a web_fallback.WebFallback (only canonical names leave the machine)."""
    web_chunks, seen = [], set()
    for c in web.search(abnormal_findings):
        if c.get("for_finding") not in seen:
            seen.add(c.get("for_finding"))
            web_chunks.append(c)
    facts = [c for c in chunks if c.get("source") == "checkitem"]
    rest = [c for c in chunks if c.get("source") != "checkitem"]
    return facts + web_chunks + rest, len(web_chunks)


# ─── Stages (shared by analyze_pdf and graph_workflow.py) ────────────────────

NO_TEXT_ERROR = "No text layer found (scanned PDF?)"


def extract_stage(pdf_bytes: bytes, ocr: bool = True, base_url: str = config.OLLAMA_BASE_URL
                  ) -> Tuple[Dict, bool, str]:
    """Return (content {text, tables}, used_ocr, error)."""
    content = extract_pdf_content(pdf_bytes)
    if ocr and is_scanned(content):
        try:
            return ocr_pdf(pdf_bytes, base_url=base_url), True, ""
        except Exception as e:  # noqa: BLE001
            return content, False, f"OCR failed: {e}"
    return content, False, ""


def detect_stage(text: str, tables: List, checkitem_lookup, sex: Optional[str] = None,
                 llm_normalize: bool = False, model: str = config.LLM_MODEL,
                 base_url: str = config.OLLAMA_BASE_URL) -> Tuple[List[Dict], Dict]:
    """Rule-based lab extraction + range evaluation; returns (findings, abnormal_debug)."""
    debug: Dict = {}
    normalizer = make_llm_normalizer(base_url=base_url, model=model) if llm_normalize else None
    findings = evaluate_findings(extract_lab_values(text, tables), checkitem_lookup,
                                 sex=sex, llm_normalizer=normalizer, debug=debug)
    return findings, debug


MAX_QUERY_FINDINGS = 8  # = Retriever.retrieve(max_findings) default


def prefetch_query_embeddings(retriever, abnormals: List[Dict]) -> str:
    """Embed all of this report's retrieval queries in ONE /api/embed call
    (Retriever.dense_search would otherwise make one round trip per finding);
    Retriever picks the vectors up from embeddings' query cache. No-op for
    retrievers without a dense index. Returns an error string ('' = ok)."""
    if getattr(retriever, "col", None) is None or getattr(retriever, "mode", "dense") == "bm25"             or not hasattr(retriever, "embed_model"):
        return ""
    queries = [finding_query(f) for f in abnormals[:MAX_QUERY_FINDINGS]] or [GENERAL_QUERY]
    _, err = prefetch_queries(queries, retriever.embed_model, retriever.base_url,
                              getattr(retriever, "use_prefix", True))
    return err  # on error Retriever falls back to per-query calls (and records its own errors)


def retrieve_stage(retriever, abnormals: List[Dict], web=None) -> Tuple[List[Dict], Dict]:
    prefetch_query_embeddings(retriever, abnormals)
    chunks, rag_debug = retriever.retrieve(abnormals)
    if web is not None and abnormals:
        chunks, rag_debug["web_hits"] = add_web_chunks(chunks, abnormals, web)
    return chunks, rag_debug


def rules_stage(findings: List[Dict], sex: Optional[str], use_rules: bool = True) -> Dict[str, List[Dict]]:
    return cond_rules.derive(findings, sex) if use_rules else {"conditions": [], "risks": []}


def findings_prompt(findings: List[Dict], rules: Dict[str, List[Dict]]) -> str:
    """Abnormal findings (+ rule-derived conditions) as the LLM prompt section."""
    text = format_findings_for_llm(findings, only_abnormal=True)
    if rules["conditions"]:
        text += "\n\n# 規則判定的狀況（已確定，建議請針對這些狀況）\n" + "\n".join(
            f"- {c['text']}（{c['evidence']}）" for c in rules["conditions"])
    return text


def merge_rule_tags(tags: Dict, rules: Dict[str, List[Dict]]) -> Dict:
    """Rule conditions/risks first, then non-duplicate LLM ones (mutates + returns tags)."""
    for key in ("conditions", "risks"):
        tags[key] = cond_rules.merge_tags(rules[key], tags.get(key, []))
    return tags


@dataclass
class PipelineResult:
    text: str = ""
    tables: List = field(default_factory=list)
    findings: List[Dict] = field(default_factory=list)
    chunks: List[Dict] = field(default_factory=list)
    cited_ids: List[str] = field(default_factory=list)
    rag_debug: Dict = field(default_factory=dict)
    tags: Dict = field(default_factory=dict)
    error: str = ""
    error_stage: str = ""
    timing: Dict[str, float] = field(default_factory=dict)
    # R2: sex used for range selection + abnormal-detection debug info
    # ({unmatched: [...names dropped...], sex, sex_source, llm_mapped}).
    sex: Optional[str] = None
    abnormal_debug: Dict = field(default_factory=dict)
    used_ocr: bool = False
    # rule-derived {"conditions", "risks"} (conditions.py); merged into tags after the LLM
    rules: Dict = field(default_factory=dict)

    @property
    def abnormals(self) -> List[Dict]:
        return abnormal_only(self.findings)


def analyze_pdf(
    pdf_bytes: bytes,
    checkitem_lookup=None,
    model: str = config.LLM_MODEL,
    base_url: str = config.OLLAMA_BASE_URL,
    run_llm: bool = True,
    sex: Optional[str] = None,
    retriever=None,
    verify: bool = True,
    use_rules: bool = True,
    ocr: bool = True,
    llm_normalize: bool = False,
    web=None,
    on_stage: Optional[Callable[[str, "PipelineResult"], None]] = None,
    on_llm_text: Optional[Callable[[str], None]] = None,
    review: bool = False,
) -> PipelineResult:
    """checkitem_lookup: lab catalog (abnormal.load_checkitem_lookup); None =
    the public reference ranges (config.REFERENCE_RANGES_PATH).
    retriever: retrieval.Retriever (or anything with .retrieve(findings));
    None = fact sheets only.
    sex: "M"/"F"/None — explicit patient sex; None = read it from the report
    (性別 header or sex-specific printed ranges), else treat as unknown.
    llm_normalize: batched LLM fallback for lab names the synonym table cannot
    resolve. Off by default: in testing it produced confident wrong mappings
    (e.g. AST -> GGT); enable only with a name validation set to measure it.
    web: optional web_fallback.WebFallback; adds at most one web passage per
    abnormal finding (off by default — nothing leaves the machine).
    on_stage(stage, partial_result): progress hook, stage in
      "extract" (starting), "findings" (findings + r.rules ready: the first
      useful result, ~0.1-1 s), "retrieve" (starting), "llm" (starting),
      "verify" (starting). The partial result must be treated as read-only.
    on_llm_text(raw): streams the LLM's JSON as it is generated
      (llm.partial_tags turns it into displayable tags).
    review: pause for a nurse review of the rule conditions/risks before the
      LLM (see graph_workflow.run_graph / resume_review).

    Orchestration is the LangGraph workflow (graph_workflow.run_graph); this
    function is its stable entry point. The rewrite loop follows
    config.VERIFY_REWRITE (off by default: unsupported advice is dropped)."""
    from graph_workflow import run_graph  # local: graph_workflow imports the stages above
    return run_graph(pdf_bytes, checkitem_lookup=checkitem_lookup, model=model, base_url=base_url, sex=sex,
                     retriever=retriever, verify=verify, use_rules=use_rules, ocr=ocr,
                     llm_normalize=llm_normalize, web=web, run_llm=run_llm,
                     on_stage=on_stage, on_llm_text=on_llm_text, review=review)
