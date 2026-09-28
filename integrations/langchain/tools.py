"""
LangChain tools — deterministic wrappers around existing functions.

analyze_health_report(pdf_path, sex=None, run_llm=True)
    pipeline.analyze_pdf() on a local PDF, summarised as JSON-serialisable data.
    run_llm=False: extraction + rule-based abnormal detection + rule-derived
    conditions/risks only (no model call at all).
lookup_reference_range(item, sex=None)
    abnormal.load_catalog() name resolution (lab_synonyms.json) + the public
    range and citation from reference_ranges.json. No LLM.

Tools return dicts; LangChain serialises them to JSON in the ToolMessage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

import conditions as cond_rules
from abnormal import load_catalog
from pipeline import PipelineResult, analyze_pdf
from retrieval import source_url
from verify import ADVICE_KEYS

_SEX = {"M": "M", "MALE": "M", "男": "M", "F": "F", "FEMALE": "F", "女": "F"}
_FINDING_KEYS = ("canonical_key", "display_name", "value", "unit", "status", "range_source")


def _finding_summary(f: Dict) -> Dict[str, Any]:
    out = {k: f.get(k) for k in _FINDING_KEYS}
    out["reference_range"] = f.get("report_range") if f.get("range_source") == "report" else f.get("kb_range")
    return out


def summarize_result(r: PipelineResult, run_llm: bool = True) -> Dict[str, Any]:
    """PipelineResult -> compact JSON-serialisable summary (no raw report text)."""
    urls = {c["id"]: c.get("url", "") for c in r.chunks}
    if run_llm and r.tags:
        conds, risks = r.tags.get("conditions", []), r.tags.get("risks", [])
    else:  # the same deterministic rules the pipeline merges into LLM tags
        rules = cond_rules.derive(r.findings, r.sex)
        conds, risks = rules["conditions"], rules["risks"]

    def tag(t: Dict) -> Dict[str, Any]:
        src = t.get("src", "")
        out = {"text": t.get("text"), "src": src, "conf": t.get("conf")}
        if t.get("evidence"):
            out["evidence"] = t["evidence"]
        if t.get("verdict"):
            out["verdict"] = t["verdict"]
        if src.startswith("kb:"):
            out["source_url"] = urls.get(src[3:], "")
        return out

    return {
        "error": r.error or None,
        "error_stage": r.error_stage or None,
        "sex": r.sex,
        "n_items_parsed": len(r.findings),
        "abnormal_findings": [_finding_summary(f) for f in r.abnormals],
        "conditions": [tag(t) for t in conds],
        "risks": [tag(t) for t in risks],
        "summary": r.tags.get("summary") if r.tags else None,
        "advice": {k: [tag(t) for t in r.tags.get(k, [])] for k in ADVICE_KEYS} if r.tags else {},
        "sources": [{"id": c["id"], "source": c.get("source"), "url": c.get("url", "")} for c in r.chunks],
        "verification": r.tags.get("_verification") if r.tags else None,
        "timing": r.timing,
    }


@tool(parse_docstring=True)
def analyze_health_report(pdf_path: str, sex: Optional[str] = None, run_llm: bool = True) -> Dict[str, Any]:
    """Analyze a health-check report PDF: abnormal lab findings, conditions (with evidence),
    risks, cited lifestyle/food/exercise advice, and citation-verification stats.

    Args:
        pdf_path: Path to a local PDF health-check report.
        sex: Patient sex "M" or "F"; omit to read it from the report.
        run_llm: False = rule-based findings/conditions only (fast, no LLM, no advice).
    """
    p = Path(pdf_path)
    if p.suffix.lower() != ".pdf" or not p.is_file():
        return {"error": f"not a readable PDF file: {pdf_path}"}
    r = analyze_pdf(p.read_bytes(), run_llm=run_llm, sex=_SEX.get((sex or "").strip().upper()))
    return summarize_result(r, run_llm=run_llm)


@tool(parse_docstring=True)
def lookup_reference_range(item: str, sex: Optional[str] = None) -> Dict[str, Any]:
    """Look up a lab test by any common name (e.g. "LDL", "飯前血糖", "ALT") and return its
    canonical item, public adult reference range, unit and citation. No LLM involved.

    Args:
        item: Lab test name as printed on a report, in Chinese or English.
        sex: Optional "M" or "F" to select a sex-specific range.
    """
    it = load_catalog().resolve(item)
    if it is None:
        return {"query": item, "found": False,
                "note": "unknown or ambiguous lab name (the resolver never guesses)"}
    sx = _SEX.get((sex or "").strip().upper())
    by_sex: Dict[str, str] = dict(it.sex_ref_text)
    rng = by_sex.get(sx) or it.kb_ref_text or None
    return {
        "query": item,
        "found": True,
        "canonical_key": it.key,
        "name_zh": it.zh,
        "name_en": it.en,
        "unit": it.unit,
        "reference_range": rng,
        "reference_range_by_sex": by_sex,
        "note": it.ref_note or None,
        "source": it.ref_source or None,
        "source_url": source_url(it.ref_source) or None,
        "caveat": "Adult screening reference; the range printed on the report takes precedence.",
    }


TOOLS: List = [analyze_health_report, lookup_reference_range]
