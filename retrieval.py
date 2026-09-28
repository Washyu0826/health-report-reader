"""
retrieval.py — Report-driven retrieval (R3).

R0 ran 8 fixed category queries that ignored the report, and their hits filled
the context budget before any patient-specific chunk. Now:

  1. Each abnormal finding gets its own query ("LDL 偏高 原因 飲食 運動 …").
  2. Biomarker fact sheets are looked up directly by canonical key / name (no
     vector search). Default source: data/reference_ranges.json, rendered as
     one sheet per item (name, public reference range, caveat, citation).
  3. Context is assembled round-robin across findings (fact sheets first, then
     each finding's best KB chunk, then second-best, …) so every abnormality
     gets a fair share of the budget.

`Retriever.search()` is the single extension point for dense / BM25 / hybrid /
reranked retrieval (R4). `select_context()` renders the assembled chunks into
the LLM prompt under the context budget.

Tag routing (`Retriever(route_by_tags=True)`, off by default)
-------------------------------------------------------------
KB passages may declare `applies_to` ("ldl:high", "cond:代謝症候群", …; see
data/kb_sample.README.md). kb_index stores it comma-joined in each chunk's
metadata. With routing on, each abnormal finding is searched only among the
chunks whose applies_to contains "<canonical_key>:<status>" — or
"cond:<label>" for a rule condition that fired and is derived from that key
(conditions.CONDITION_KEYS) — ranked by the normal search score inside that
subset. Chunks already picked for an earlier finding are skipped when the
subset has others, so each finding brings new passages. A finding with no
tagged chunk falls back to the normal unfiltered search, so an untagged corpus
behaves exactly as before.

Design: the token -> chunk-id map is built once in memory (from the BM25
chunks, or from the collection's metadatas via `col.get()`), because Chroma
metadata filters cannot do substring/`$contains` matching on a string field.
The subset search is then exact rather than over-fetch-and-filter: dense uses
`where={"citation_id": {"$in": ids}}` (a scalar-field filter Chroma supports),
BM25 scores the whole corpus and keeps the subset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import config
from embeddings import embed_query

_URL = re.compile(r"https?://[^\s）)」，,]+")
_HERE = Path(__file__).resolve().parent
_DIR_WORD = {"high": "偏高", "low": "偏低", "positive": "陽性", "abnormal": "異常"}
GENERAL_QUERY = "健康檢查 正常 維持 均衡飲食 規律運動 睡眠 保健"
_CI_NAME = re.compile(r"【檢查項目】([^（(\n]+)(?:[（(]([^）)]+)[)）])?")


def source_url(source: str) -> str:
    """First http(s) URL in a citation string ('' when none)."""
    m = _URL.search(source or "")
    return m.group(0) if m else ""


def finding_query(f: Dict) -> str:
    name = f.get("display_name") or f.get("matched_name") or f.get("name", "")
    word = _DIR_WORD.get(f.get("status") or f.get("direction"), "異常")
    return f"{name} {word} 原因 風險 飲食 運動 生活習慣 改善"


FACT_REMINDER = ("參考範圍會因檢驗單位、儀器、方法與族群而不同，請以報告印製的參考值為準；"
                 "結果超出範圍時，請諮詢醫療專業人員進一步評估。")


def reference_fact_sheet(entry: Dict) -> str:
    """One biomarker fact sheet from a reference_ranges.json item."""
    from abnormal import reference_entry_specs  # local: keeps import order light
    general, by_sex = reference_entry_specs(entry)
    name = entry.get("zh") or entry.get("key", "")
    if entry.get("en") and entry.get("en") != name:
        name += f"（{entry['en']}）"
    ranges = []
    if by_sex:
        ranges += [f"{'男性' if sx == 'M' else '女性'} {sp['text']}" for sx, sp in sorted(by_sex.items(),
                                                                                    key=lambda kv: kv[0] != "M")]
    elif general:
        ranges.append(general["text"])
    if entry.get("qualitative_normal"):
        ranges.append(f"定性結果以「{entry['qualitative_normal']}」為正常")
    lines = [f"【檢查項目】{name}"]
    if ranges:
        lines.append(f"【一般參考範圍】{'；'.join(ranges)}（成人篩檢參考，非診斷標準）")
    if entry.get("note"):
        lines.append(f"【說明】{entry['note']}")
    lines.append(f"【提醒】{FACT_REMINDER}")
    if entry.get("source"):
        lines.append(f"【資料來源】{entry['source']}")
    return "\n".join(lines)


class CheckitemFacts:
    """Direct name -> biomarker fact-sheet lookup (no embeddings).

    path: data/reference_ranges.json (default, public; ids `ref_<key>`) or,
    for internal use, a legacy KB list file with 【檢查項目】 entries (ids
    `ci_<i>`). The format is detected from the content.
    """

    def __init__(self, path: str = config.REFERENCE_RANGES_PATH):
        self.by_name: Dict[str, Tuple[str, str]] = {}
        self.urls: Dict[str, str] = {}
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = _HERE / p
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            self._load_reference(data["items"])
        else:
            self._load_legacy(data)

    @classmethod
    def from_reference_ranges(cls, path: str = config.REFERENCE_RANGES_PATH) -> "CheckitemFacts":
        return cls(path)

    def _load_reference(self, items: List[Dict]) -> None:
        for e in items:
            key = e.get("key")
            if not key:
                continue
            cid = f"ref_{key}"
            text = reference_fact_sheet(e)
            self.urls[cid] = source_url(e.get("source", ""))
            for n in (key, e.get("zh"), e.get("en")):
                if n and n.strip():
                    self.by_name.setdefault(n.strip().lower(), (cid, text))

    def _load_legacy(self, data: List[Dict]) -> None:
        for i, entry in enumerate(data or []):
            content = entry.get("Content") or ""
            m = _CI_NAME.search(content)
            names = [n.strip() for n in (m.groups() if m else ()) if n and n.strip()]
            if entry.get("canonical_key"):
                names.append(entry["canonical_key"])
            for n in names:
                self.by_name.setdefault(n.lower(), (f"ci_{i}", content))

    def lookup(self, f: Dict) -> Optional[Dict]:
        for key in (f.get("canonical_key"), f.get("display_name"), f.get("matched_name")):
            if key and key.lower() in self.by_name:
                cid, text = self.by_name[key.lower()]
                hit = {"id": cid, "text": text, "score": 0.0, "source": "checkitem"}
                if self.urls.get(cid):
                    hit["url"] = self.urls[cid]
                return hit
        return None


_STOP = set("的 了 是 在 和 與 及 或 也 就 都 而 被 把 讓 對 會 能 可 可以 要 有 沒有 這 那 一 個 些 等 很 更 最 如果 因為 所以 但 但是 還 並 其 其中 之 為 於 以 從 到 中 上 下 時 後 前 我 你 他 她 我們 大家 自己 什麼 怎麼 如何 可能 需要 應該 進行 以及 透過 原因 改善".split())


class BM25Index:
    """Chinese BM25 over the same chunks/ids as the vector index.

    Tokenisation: jieba with a user dictionary of lab-item names and KB topic
    tags, so "三酸甘油酯" or "γ-GT" stay single tokens.
    """

    def __init__(self, chunks: List[Dict], user_terms: List[str]):
        import bm25s
        import jieba
        jieba.setLogLevel(60)
        for t in user_terms:
            if len(t) >= 2:
                jieba.add_word(t, freq=20000)
        self._jieba = jieba
        self.chunks = chunks
        self.bm25 = bm25s.BM25()
        self.bm25.index([self.tokenize(c["text"]) for c in chunks], show_progress=False)

    def tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in self._jieba.lcut(text)
                if w.strip() and w not in _STOP and not re.fullmatch(r"[\W_]+", w)]

    def search(self, query: str, k: int, ids: Optional[Set[str]] = None) -> List[Dict]:
        """Top-k chunks; `ids` (tag routing) restricts the result to those chunk ids."""
        toks = self.tokenize(query)
        if not toks or k <= 0:
            return []
        n = len(self.chunks) if ids is not None else min(k, len(self.chunks))
        docs, scores = self.bm25.retrieve([toks], k=n, show_progress=False)
        out = []
        for i, s in zip(docs[0], scores[0]):
            if s <= 0:
                continue
            c = self.chunks[i]
            if ids is not None and c["id"] not in ids:
                continue
            if len(out) >= k:
                break
            hit = {"id": c["id"], "text": c["text"], "score": float(s), "source": "kb"}
            if c.get("url"):
                hit["url"] = c["url"]
            if c.get("applies_to"):
                hit["applies_to"] = c["applies_to"]
            out.append(hit)
        return out


def rrf_fuse(rankings: List[List[Dict]], k: int, c: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion: score(d) = sum 1/(c + rank)."""
    score: Dict[str, float] = {}
    by_id: Dict[str, Dict] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking):
            score[h["id"]] = score.get(h["id"], 0.0) + 1.0 / (c + rank + 1)
            by_id.setdefault(h["id"], h)
    ids = sorted(score, key=score.get, reverse=True)[:k]
    return [{**by_id[i], "score": round(score[i], 5)} for i in ids]


class Reranker:
    """Cross-encoder reranker (sentence-transformers). GPU fp16 when available."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None,
                 max_length: int = 512):
        import torch
        from sentence_transformers import CrossEncoder
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"torch_dtype": torch.float16} if device == "cuda" else {}
        self.model = CrossEncoder(model_name, device=device, max_length=max_length,
                                  model_kwargs=kw)
        self.device = device

    def rerank(self, query: str, hits: List[Dict], k: int) -> List[Dict]:
        if not hits:
            return []
        scores = self.model.predict([(query, h["text"]) for h in hits], batch_size=16,
                                    show_progress_bar=False)
        ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
        return [{**h, "score": round(float(s), 4), "rerank": True} for h, s in ranked[:k]]


class Retriever:
    """mode: "dense" | "bm25" | "hybrid"; optional cross-encoder `reranker`."""

    def __init__(self, kb_collection, embed_model: str = config.EMBED_MODEL,
                 base_url: str = config.OLLAMA_BASE_URL,
                 facts: Optional[CheckitemFacts] = None, use_prefix: bool = True,
                 mode: str = "dense", bm25: Optional[BM25Index] = None,
                 reranker: Optional[Reranker] = None, candidates: int = 20,
                 route_by_tags: bool = False):
        self.col = kb_collection
        self.use_prefix = use_prefix
        self.embed_model = embed_model
        self.base_url = base_url
        self.facts = facts or CheckitemFacts()
        self.mode, self.bm25, self.reranker, self.candidates = mode, bm25, reranker, candidates
        self.route_by_tags = route_by_tags
        self._routes: Optional[Dict[str, Set[str]]] = None
        self.errors: List[str] = []

    # ── extension point ──
    def search(self, query: str, k: int, ids: Optional[Set[str]] = None) -> List[Dict]:
        """Top-k chunks for `query`; `ids` (tag routing) restricts the candidates."""
        n = self.candidates if self.reranker else k
        if self.mode == "bm25":
            hits = self.bm25.search(query, n, ids=ids)
        elif self.mode == "hybrid":
            hits = rrf_fuse([self.dense_search(query, n, ids=ids), self.bm25.search(query, n, ids=ids)], n)
        else:
            hits = self.dense_search(query, n, ids=ids)
        if self.reranker:
            hits = self.reranker.rerank(query, hits, k)
        return hits[:k]

    def dense_search(self, query: str, k: int, ids: Optional[Set[str]] = None) -> List[Dict]:
        if self.col is None or k <= 0 or (ids is not None and not ids):
            return []
        vec, err = embed_query(query, self.embed_model, self.base_url, self.use_prefix)
        if err or vec is None:
            self.errors.append(err)
            return []
        kw = {}
        n = min(k, self.col.count())
        if ids is not None:
            kw["where"] = {"citation_id": {"$in": sorted(ids)}}
            n = min(n, len(ids))
        res = self.col.query(query_embeddings=[vec], n_results=n,
                             include=["documents", "distances", "metadatas"], **kw)
        out = []
        for i, d, s, m in zip(res["ids"][0], res["documents"][0], res["distances"][0], res["metadatas"][0]):
            hit = {"id": (m or {}).get("citation_id", i), "text": d, "score": float(s), "source": "kb"}
            if (m or {}).get("url"):
                hit["url"] = m["url"]
            if (m or {}).get("applies_to"):
                hit["applies_to"] = m["applies_to"]
            out.append(hit)
        return out

    # ── tag routing ──
    def routes(self) -> Dict[str, Set[str]]:
        """applies_to token -> chunk ids (built once; empty for an untagged corpus)."""
        if self._routes is None:
            pairs: List[Tuple[str, str]] = []
            if self.bm25 is not None:
                pairs = [(c["id"], c.get("applies_to") or "") for c in self.bm25.chunks]
            elif self.col is not None:
                got = self.col.get(include=["metadatas"])
                pairs = [((m or {}).get("citation_id", i), (m or {}).get("applies_to") or "")
                         for i, m in zip(got["ids"], got["metadatas"])]
            routes: Dict[str, Set[str]] = {}
            for cid, applies in pairs:
                for tok in (t.strip() for t in applies.split(",")):
                    if tok:
                        routes.setdefault(tok, set()).add(cid)
            self._routes = routes
        return self._routes

    @staticmethod
    def finding_tokens(f: Dict, conditions: Iterable[str] = ()) -> List[str]:
        """applies_to tokens a finding may be routed to."""
        from conditions import CONDITION_KEYS
        key = f.get("canonical_key")
        if not key:
            return []
        toks = []
        for st in (f.get("status"), f.get("direction")):
            if st in ("high", "low", "positive") and f"{key}:{st}" not in toks:
                toks.append(f"{key}:{st}")
        toks += [f"cond:{c}" for c in conditions if key in CONDITION_KEYS.get(c, ())]
        return toks

    @staticmethod
    def rule_conditions(findings: List[Dict]) -> List[str]:
        """Rule condition labels (conditions.derive) used for routing when the caller passes none.
        Sex is unknown here, so sex-specific cut-offs (waist, HDL) use the male values."""
        import conditions as cond_rules
        return [c["text"] for c in cond_rules.derive(findings)["conditions"]]

    # ── report-driven assembly ──
    def retrieve(self, abnormal_findings: List[Dict], per_finding_k: int = 2,
                 max_findings: int = 8, general_k: int = 3,
                 conditions: Optional[List] = None) -> Tuple[List[Dict], Dict]:
        """conditions: rule-condition labels (str or {"text": ...}) for tag routing;
        None -> derived from the findings with conditions.derive(). Ignored when
        route_by_tags is off."""
        debug = {"queries": [], "fact_hits": 0, "kb_hits": 0, "errors": self.errors}
        findings = abnormal_findings[:max_findings]
        routes: Dict[str, Set[str]] = {}
        conds: List[str] = []
        if self.route_by_tags:
            routes = self.routes()
            if conditions is None:
                conds = self.rule_conditions(abnormal_findings) if routes else []
            else:
                conds = [c.get("text", "") if isinstance(c, dict) else str(c) for c in conditions]
            debug.update(routed=0, route_fallback=0, route_conditions=conds)
        used: Set[str] = set()
        facts: List[Dict] = []
        ranked: List[List[Dict]] = []
        if not findings:
            q = GENERAL_QUERY
            debug["queries"].append(q)
            ranked.append(self.search(q, general_k))
        for f in findings:
            fact = self.facts.lookup(f)
            if fact:
                facts.append(fact)
            q = finding_query(f)
            debug["queries"].append(q)
            subset: Optional[Set[str]] = None
            if routes:
                subset = set().union(*(routes.get(t, set()) for t in self.finding_tokens(f, conds)))
                if subset:
                    debug["routed"] += 1
                    subset = (subset - used) or subset  # prefer passages not picked for an earlier finding
                else:
                    debug["route_fallback"] += 1
                    subset = None
            hits = self.search(q, per_finding_k, ids=subset)
            used.update(h["id"] for h in hits)
            for h in hits:
                h["for_finding"] = f.get("display_name") or f.get("name")
            ranked.append(hits)

        out, seen = [], set()

        def add(c):
            if c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)

        for c in facts:
            add(c)
            debug["fact_hits"] += 1
        for rank in range(max((len(r) for r in ranked), default=0)):
            for hits in ranked:
                if rank < len(hits):
                    before = len(out)
                    add(hits[rank])
                    debug["kb_hits"] += len(out) - before
        return out, debug


def select_context(chunks: List[Dict], max_chars: int = config.CONTEXT_MAX_CHARS) -> Tuple[str, List[str]]:
    """Render chunks that fit the budget; return (context_text, included_ids).

    included_ids is exactly what the LLM sees, so it can be used as the
    allowed-citation enum.
    """
    lines, ids = [], []
    used = 0
    for c in chunks:
        block = f"[{c['id']}] (source={c['source']})\n{c['text']}\n"
        if used + len(block) > max_chars:
            break
        lines.append(block)
        ids.append(c["id"])
        used += len(block)
    return "\n---\n".join(lines), ids
