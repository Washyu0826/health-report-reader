"""Privacy guarantees: no egress by default; web fallback sends only canonical names."""

import socket
from pathlib import Path

import pytest

from web_fallback import WebFallback, build_query

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "eval" / "synth" / "samples"
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture
def egress_guard(monkeypatch):
    """Fail on any socket connection to a non-loopback address."""
    attempts = []
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in LOOPBACK:
            attempts.append(host)
            raise AssertionError(f"outbound connection attempted to {host}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    real_gai = socket.getaddrinfo

    def guarded_gai(host, *a, **kw):
        if host not in LOOPBACK:
            attempts.append(host)
            raise AssertionError(f"DNS lookup attempted for {host}")
        return real_gai(host, *a, **kw)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_gai)
    return attempts


def finding(**kw):
    base = {"canonical_key": "sbp", "display_name": "收縮壓", "name": "收縮壓",
            "value": 150, "unit": "mmHg", "status": "high"}
    return {**base, **kw}


def test_query_contains_only_canonical_name_and_direction():
    q = build_query(finding(name="陳大文收縮壓"))
    assert q == "收縮壓 偏高 衛教"
    assert "150" not in q and "陳大文" not in q and "mmHg" not in q


def test_unmatched_or_normal_findings_are_never_sent():
    assert build_query(finding(canonical_key=None)) is None
    assert build_query(finding(status="normal")) is None
    assert build_query(finding(display_name="王小明 0912345678")) == "收縮壓 偏高 衛教"  # display_name ignored
    assert build_query(finding(canonical_key="not_a_key")) is None


def test_web_fallback_sends_only_safe_queries_and_uses_cache(tmp_path):
    sent = []

    def fake_search(q):
        sent.append(q)
        return [{"url": "https://example.org", "title": "t", "snippet": "x" * 40}]

    wf = WebFallback(cache_path=tmp_path / "c.json", search_fn=fake_search)
    wf.search([finding(), finding(canonical_key=None, name="林美麗 地址")])
    assert sent == ["收縮壓 偏高 衛教"]
    WebFallback(cache_path=tmp_path / "c.json", search_fn=fake_search).search([finding()])
    assert sent == ["收縮壓 偏高 衛教"], "second run must be served from the on-disk cache"


def test_pipeline_makes_no_outbound_connections_by_default(egress_guard):
    from abnormal import load_checkitem_lookup
    from pipeline import analyze_pdf
    from retrieval import CheckitemFacts, Retriever

    pdf = next(p for p in sorted(SYNTH.glob("syn_*.pdf")) if p.stat().st_size < 200_000)
    ranges = str(ROOT / "data" / "reference_ranges.json")
    lookup = load_checkitem_lookup(ranges)
    retriever = Retriever(None, facts=CheckitemFacts(ranges))
    r = analyze_pdf(pdf.read_bytes(), lookup, retriever=retriever, run_llm=False)
    assert r.findings, "sanity: the pipeline actually ran"
    assert egress_guard == []


def test_pipeline_defaults_make_no_outbound_connections(egress_guard):
    """No catalog / retriever passed: public reference ranges + fact sheets only."""
    from pipeline import analyze_pdf

    pdf = SYNTH / "syn_048.pdf"                                  # has several abnormal items
    r = analyze_pdf(pdf.read_bytes(), run_llm=False)
    assert r.findings and r.abnormals
    assert r.chunks and all(c["source"] == "checkitem" and c["id"].startswith("ref_") for c in r.chunks)
    assert egress_guard == []


def test_pipeline_web_fallback_adds_one_passage_per_finding_after_fact_sheets(egress_guard, tmp_path):
    from pipeline import analyze_pdf

    sent = []

    def fake_search(q):
        sent.append(q)
        return [{"url": "https://example.org/a", "title": "a", "snippet": "x" * 40},
                {"url": "https://example.org/b", "title": "b", "snippet": "y" * 40}]

    web = WebFallback(cache_path=tmp_path / "c.json", search_fn=fake_search, max_findings=2)
    pdf = SYNTH / "syn_048.pdf"                                  # has several abnormal items
    r = analyze_pdf(pdf.read_bytes(), run_llm=False, web=web)
    assert len(r.abnormals) >= 2
    kinds = [c["source"] for c in r.chunks]
    web_idx = [i for i, k in enumerate(kinds) if k == "web"]
    assert len(web_idx) == 2 == r.rag_debug["web_hits"]          # one per finding, not two
    assert all(k == "checkitem" for k in kinds[:web_idx[0]])     # right after the fact sheets
    assert sent == web.sent_queries and all(q.endswith("衛教") for q in sent)
    assert egress_guard == []
