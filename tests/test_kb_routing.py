"""Structured public KB (applies_to; data/kb_sample_extended.json and its default subset
data/kb_sample.json) + tag-routed retrieval.

No network, no LLM, no embeddings: dense search is exercised with a fake
Chroma-like collection and a stubbed query embedding.
"""

import socket
from pathlib import Path

import pytest

import retrieval
from conditions import CONDITION_KEYS, derive
from kb_index import INDEX_VERSION, prepare_chunks
from retrieval import BM25Index, CheckitemFacts, Retriever
from tools.kb_coverage import (
    KINDS, MAX_CHARS, MIN_CHARS, canonical_keys, condition_labels, coverage, load_kb, missing_kinds,
    required_findings, synth_abnormal, validate,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(self, address):
        raise AssertionError(f"network access attempted: {address}")
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(scope="module")
def kb():
    return load_kb()


# ── KB schema / validity ─────────────────────────────────────────────────────

def test_kb_schema_is_valid(kb):
    assert len(kb) >= 150
    assert validate(kb, canonical_keys(), condition_labels()) == []


def test_kb_lengths_sources_and_kinds(kb):
    for p in kb:
        assert MIN_CHARS <= len(p["Content"]) <= MAX_CHARS, p["Title"]
        assert p["Source"].startswith("https://"), p["Title"]
        assert p["kind"] in KINDS
        assert p["TagName"].strip() and p["applies_to"]


def test_kb_has_no_publisher_terms(kb):
    terms_file = ROOT / "data" / "brand_terms.txt"
    if not terms_file.exists():
        pytest.skip("no private brand-term list in this checkout")
    terms = [t.strip().lower() for t in terms_file.read_text(encoding="utf-8").splitlines() if t.strip()]
    for p in kb:
        blob = " ".join(str(v) for v in p.values()).lower()
        assert not any(t in blob for t in terms), p["Title"]


def test_condition_keys_cover_every_rule_label():
    labels = condition_labels()
    assert labels, "no cond(...) labels parsed from conditions.py"
    assert labels == set(CONDITION_KEYS)
    keys = canonical_keys()
    assert all(k in keys for ks in CONDITION_KEYS.values() for k in ks)


def test_default_kb_is_the_measured_subset_of_the_extended_corpus(kb):
    """data/kb_sample.json = the 57 passages measured best (see data/kb_sample.README.md):
    same text as the extended corpus, so routing metadata stays available."""
    default = load_kb(ROOT / "data" / "kb_sample.json")
    by_title = {p["Title"]: p for p in kb}
    assert len(default) == 57 and all(p == by_title[p["Title"]] for p in default)
    assert validate(default, canonical_keys(), condition_labels()) == []


# ── coverage ─────────────────────────────────────────────────────────────────

def test_every_synthetic_abnormal_direction_has_a_passage(kb):
    cov = coverage(kb)
    synth = synth_abnormal()
    assert synth, "synthetic ground truth has no abnormal items"
    missing = sorted(s for s in synth if not sum(cov.get(s, {}).values()))
    assert missing == []


def test_required_findings_have_all_four_kinds(kb):
    assert missing_kinds(coverage(kb), required_findings()) == {}


# ── indexing carries applies_to ──────────────────────────────────────────────

def test_prepare_chunks_keeps_applies_to_as_scalar(kb):
    chunks, _ = prepare_chunks(kb)
    assert chunks and all(isinstance(c["applies_to"], str) for c in chunks)
    by_title = {c["title"]: c for c in chunks}
    first = kb[0]
    assert by_title[first["Title"]]["applies_to"] == ",".join(first["applies_to"])
    untagged, _ = prepare_chunks([{"Title": "x", "Content": "無標記的段落。" * 20, "TagName": "a"}])
    assert untagged[0]["applies_to"] == ""
    assert INDEX_VERSION == "r3-v4"


# ── routing: in-memory fixtures ──────────────────────────────────────────────

LDL = {"canonical_key": "ldl", "display_name": "低密度脂蛋白膽固醇", "name": "LDL", "value": 165,
       "unit": "mg/dL", "status": "high", "direction": "high"}
CHOL = {"canonical_key": "chol", "display_name": "總膽固醇", "name": "CHOL", "value": 245,
        "unit": "mg/dL", "status": "high", "direction": "high"}
UA = {"canonical_key": "uric_acid", "display_name": "尿酸", "name": "UA", "value": 8.4,
      "unit": "mg/dL", "status": "high", "direction": "high"}
WAIST = {"canonical_key": "waist", "display_name": "腰圍", "name": "腰圍", "value": 96,
         "unit": "cm", "status": "high", "direction": "high"}
TG = {"canonical_key": "tg", "display_name": "三酸甘油脂", "name": "TG", "value": 210,
      "unit": "mg/dL", "status": "high", "direction": "high"}
GLU = {"canonical_key": "glucose_ac", "display_name": "飯前血糖", "name": "AC", "value": 112,
       "unit": "mg/dL", "status": "high", "direction": "high"}

CHUNKS = [
    # generic chunk that matches every finding query very well but is tagged for nothing relevant
    {"id": "gen", "applies_to": "",
     "text": "低密度脂蛋白膽固醇 總膽固醇 尿酸 三酸甘油脂 偏高 原因 風險 飲食 運動 生活習慣 改善 "
             "偏高 原因 風險 飲食 運動 生活習慣 改善"},
    {"id": "ldl_diet", "applies_to": "ldl:high,chol:high,cond:血脂異常",
     "text": "低密度脂蛋白膽固醇 偏高 飲食 減少 飽和脂肪"},
    {"id": "ldl_ex", "applies_to": "ldl:high,chol:high",
     "text": "低密度脂蛋白膽固醇 偏高 運動 快走"},
    {"id": "chol_only", "applies_to": "chol:high",
     "text": "總膽固醇 偏高 原因"},
    {"id": "ms_cond", "applies_to": "cond:代謝症候群",
     "text": "代謝症候群 三酸甘油脂 偏高 腰圍 改善 運動"},
    {"id": "hb_low", "applies_to": "hb:low",
     "text": "血色素 偏低 貧血 鐵質 飲食"},
]
TERMS = ["低密度脂蛋白膽固醇", "總膽固醇", "三酸甘油脂", "代謝症候群", "飽和脂肪", "生活習慣"]


def _bm25_retriever(route: bool) -> Retriever:
    return Retriever(None, facts=CheckitemFacts(), mode="bm25",
                     bm25=BM25Index(CHUNKS, user_terms=TERMS), route_by_tags=route)


def _kb_ids(chunks):
    return [c["id"] for c in chunks if c["source"] == "kb"]


def test_unrouted_search_prefers_the_generic_chunk():
    r = _bm25_retriever(False)
    assert r.search(retrieval.finding_query(LDL), 2)[0]["id"] == "gen"


def test_routing_off_is_identical_to_default():
    plain = Retriever(None, facts=CheckitemFacts(), mode="bm25", bm25=BM25Index(CHUNKS, user_terms=TERMS))
    off = _bm25_retriever(False)
    a, da = plain.retrieve([LDL, UA])
    b, db = off.retrieve([LDL, UA])
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert "routed" not in db and da.keys() == db.keys()


def test_routed_results_only_contain_matching_applies_to():
    r = _bm25_retriever(True)
    chunks, debug = r.retrieve([LDL])
    ids = _kb_ids(chunks)
    assert ids and set(ids) <= {"ldl_diet", "ldl_ex"}
    assert all("ldl:high" in c["applies_to"].split(",") for c in chunks if c["source"] == "kb")
    assert debug["routed"] == 1 and debug["route_fallback"] == 0


def test_routing_falls_back_to_unfiltered_search_when_no_passage_is_tagged():
    r = _bm25_retriever(True)
    chunks, debug = r.retrieve([UA])  # nothing tagged uric_acid:high
    assert _kb_ids(chunks)[0] == "gen"
    assert debug["route_fallback"] == 1 and debug["routed"] == 0


def test_routing_prefers_passages_not_used_by_an_earlier_finding():
    r = _bm25_retriever(True)
    chunks, _ = r.retrieve([LDL, CHOL], per_finding_k=2)
    ids = _kb_ids(chunks)
    assert set(ids) == {"ldl_diet", "ldl_ex", "chol_only"}  # CHOL adds its own passage, no duplicates
    assert len(ids) == len(set(ids))


def test_condition_labels_extend_the_routed_subset():
    r = _bm25_retriever(True)
    chunks, debug = r.retrieve([TG], conditions=["代謝症候群"])
    assert _kb_ids(chunks) == ["ms_cond"]
    # a condition the finding does not feed (貧血 is derived from hb) adds nothing
    assert Retriever.finding_tokens(TG, ["代謝症候群", "貧血"]) == ["tg:high", "cond:代謝症候群"]
    # conditions may also be passed as rule tags
    chunks2, _ = r.retrieve([TG], conditions=[{"text": "代謝症候群", "src": "rule"}])
    assert _kb_ids(chunks2) == ["ms_cond"]


def test_conditions_are_derived_when_not_passed():
    findings = [WAIST, TG, GLU]
    assert "代謝症候群" in [c["text"] for c in derive(findings)["conditions"]]
    r = _bm25_retriever(True)
    _, debug = r.retrieve(findings)
    assert "代謝症候群" in debug["route_conditions"]


def test_positive_status_uses_the_positive_token():
    f = {"canonical_key": "urine_ob", "status": "positive", "direction": "high"}
    assert Retriever.finding_tokens(f) == ["urine_ob:positive", "urine_ob:high"]


# ── routing: fake dense collection ───────────────────────────────────────────

class FakeCollection:
    """Minimal Chroma-like collection: fixed ranking, honours where={"citation_id": {"$in": ids}}."""

    def __init__(self, chunks, order):
        self.chunks = {c["id"]: c for c in chunks}
        self.order = order
        self.calls = []

    def count(self):
        return len(self.chunks)

    def get(self, include=None):
        ids = list(self.chunks)
        return {"ids": ids, "metadatas": [{"citation_id": i, "applies_to": self.chunks[i]["applies_to"]}
                                          for i in ids]}

    def query(self, query_embeddings, n_results, include=None, where=None):
        self.calls.append(where)
        allowed = set(where["citation_id"]["$in"]) if where else None
        ids = [i for i in self.order if allowed is None or i in allowed][:n_results]
        metas = [{"citation_id": i, "applies_to": self.chunks[i]["applies_to"]} for i in ids]
        return {"ids": [ids], "documents": [[self.chunks[i]["text"] for i in ids]],
                "distances": [[0.1 * (n + 1) for n in range(len(ids))]], "metadatas": [metas]}


@pytest.fixture
def dense(monkeypatch):
    monkeypatch.setattr(retrieval, "embed_query", lambda *a, **k: ([0.0, 1.0], None))
    order = ["gen", "hb_low", "chol_only", "ldl_ex", "ms_cond", "ldl_diet"]

    def make(route):
        col = FakeCollection(CHUNKS, order)
        return col, Retriever(col, facts=CheckitemFacts(), mode="dense", route_by_tags=route)
    return make


def test_dense_routing_filters_with_where_in(dense):
    col, r = dense(True)
    chunks, _ = r.retrieve([LDL])
    assert _kb_ids(chunks) == ["ldl_ex", "ldl_diet"]  # ranked by the (fake) dense score within the subset
    assert col.calls[-1] == {"citation_id": {"$in": ["ldl_diet", "ldl_ex"]}}


def test_dense_routing_fallback_and_default_do_not_filter(dense):
    col, r = dense(True)
    chunks, debug = r.retrieve([UA])
    assert _kb_ids(chunks) == ["gen", "hb_low"] and col.calls[-1] is None
    assert debug["route_fallback"] == 1
    col2, off = dense(False)
    chunks2, _ = off.retrieve([LDL])
    assert _kb_ids(chunks2) == ["gen", "hb_low"] and col2.calls == [None]


def test_untagged_corpus_never_routes(dense, monkeypatch):
    untagged = [{**c, "applies_to": ""} for c in CHUNKS]
    monkeypatch.setattr(retrieval, "embed_query", lambda *a, **k: ([0.0, 1.0], None))
    col = FakeCollection(untagged, [c["id"] for c in untagged])
    r = Retriever(col, facts=CheckitemFacts(), mode="dense", route_by_tags=True)
    chunks, debug = r.retrieve([LDL, TG])
    assert r.routes() == {} and debug["routed"] == 0 and debug["route_conditions"] == []
    assert all(w is None for w in col.calls)
