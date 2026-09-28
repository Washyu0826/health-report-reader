"""LangChain adapter (integrations/langchain): no network, no LLM, no embeddings."""

import json
import socket
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402

from integrations.langchain import (  # noqa: E402
    FindingsRetriever, HealthReportRetriever, analyze_health_report, documents_to_context,
    lookup_reference_range,
)
from retrieval import BM25Index, CheckitemFacts, Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "eval" / "synth" / "samples"

CHUNKS = [
    {"id": "kb_0_0", "url": "https://example.org/ldl",
     "text": "低密度脂蛋白膽固醇 LDL 偏高 會增加 心血管疾病 風險，應減少 飽和脂肪 並 增加 膳食纖維。"},
    {"id": "kb_1_0", "url": "https://example.org/bp",
     "text": "血壓 偏高 時 應 減少 鈉 攝取，規律 有氧運動，並 定期 量測 血壓。"},
    {"id": "kb_2_0",
     "text": "尿酸 偏高 應 少喝 含糖飲料 與 酒精，多喝水。"},
]


@pytest.fixture
def no_network(monkeypatch):
    def refuse(self, address):
        raise AssertionError(f"network access attempted: {address}")
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(scope="module")
def core():
    facts = CheckitemFacts()
    bm25 = BM25Index(CHUNKS, user_terms=list(facts.by_name))
    return Retriever(None, facts=facts, mode="bm25", bm25=bm25)


def test_retriever_returns_documents_with_metadata(core, no_network):
    docs = HealthReportRetriever(retriever=core, k=2).invoke("LDL 偏高 飲食")
    assert docs and all(isinstance(d, Document) for d in docs)
    top = docs[0]
    assert top.metadata["id"] == "kb_0_0" and top.id == "kb_0_0"
    assert top.metadata["source"] == "kb"
    assert top.metadata["url"] == "https://example.org/ldl"
    assert top.metadata["score"] > 0
    assert {"id", "source", "url", "score", "for_finding"} <= set(top.metadata)
    assert len(docs) <= 2


def test_documents_render_like_core_context(core):
    docs = HealthReportRetriever(retriever=core, k=1).invoke("血壓")
    assert documents_to_context(docs).startswith("[kb_1_0] (source=kb)")


def test_findings_retriever_puts_fact_sheet_first(core, no_network):
    f = {"canonical_key": "ldl", "display_name": "低密度脂蛋白膽固醇", "name": "LDL",
         "status": "high", "direction": "high"}
    r = HealthReportRetriever.from_findings(core, [f])
    assert isinstance(r, FindingsRetriever)
    docs = r.invoke("ignored")
    assert docs[0].metadata["id"] == "ref_ldl" and docs[0].metadata["source"] == "checkitem"
    assert docs[0].metadata["url"].startswith("https://")
    kb = [d for d in docs if d.metadata["source"] == "kb"]
    assert kb and kb[0].metadata["for_finding"] == "低密度脂蛋白膽固醇"
    assert r.last_debug["fact_hits"] == 1


@pytest.mark.parametrize("name,key", [("LDL", "ldl"), ("飯前血糖", "glucose_ac")])
def test_lookup_reference_range(name, key):
    out = lookup_reference_range.invoke({"item": name})
    assert out["found"] and out["canonical_key"] == key
    assert out["reference_range"] and out["source"] and out["source_url"].startswith("https://")


def test_lookup_unknown_name_does_not_guess():
    assert lookup_reference_range.invoke({"item": "AST/ALT"})["found"] is False


def test_tool_call_yields_json_tool_message():
    msg = lookup_reference_range.invoke(
        {"type": "tool_call", "id": "call_1", "name": "lookup_reference_range", "args": {"item": "LDL"}})
    assert isinstance(msg, ToolMessage) and msg.tool_call_id == "call_1"
    assert json.loads(msg.content)["canonical_key"] == "ldl"


def test_analyze_health_report_without_llm(no_network):
    out = analyze_health_report.invoke({"pdf_path": str(SYNTH / "syn_011.pdf"), "run_llm": False})
    json.dumps(out, ensure_ascii=False)  # JSON-serialisable
    assert out["error"] is None
    keys = {f["canonical_key"] for f in out["abnormal_findings"]}
    assert {"ldl", "chol", "hdl"} <= keys
    ldl = next(f for f in out["abnormal_findings"] if f["canonical_key"] == "ldl")
    assert ldl["status"] == "high" and ldl["reference_range"]
    assert any(c["text"] == "血脂異常" and c["evidence"] for c in out["conditions"])
    assert out["advice"] == {} and out["verification"] is None
    assert any(s["id"] == "ref_ldl" for s in out["sources"])


def test_analyze_rejects_non_pdf(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hi", encoding="utf-8")
    assert "error" in analyze_health_report.invoke({"pdf_path": str(p), "run_llm": False})
