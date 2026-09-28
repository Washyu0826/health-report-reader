"""
retrieval_bench.py — Offline retrieval benchmark with LLM-judged pooled relevance.

Method (TREC-style pooling):
  1. 40 curated abnormal-finding queries (built exactly like production queries).
  2. Every system returns its top-10; the union per query is the judging pool.
  3. A local LLM grades each (query, passage) pair 0/1/2; judgments are cached,
     so adding a system later only judges its new passages.
  4. Metrics per system: nDCG@5 (graded), P@5 and MRR (grade>=1), Recall@10
     against the pooled relevant set, and off-topic rate in the top-5.

Usage:
  .venv/Scripts/python eval/retrieval_bench.py --systems all --out eval/retrieval_R3.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from kb_index import build_kb_index, collection_name, is_offtopic, load_articles  # noqa: E402
from retrieval import Retriever, finding_query  # noqa: E402

QUERIES = [
    ("LDL膽固醇", "high"), ("總膽固醇", "high"), ("三酸甘油酯", "high"), ("HDL膽固醇", "low"),
    ("空腹血糖", "high"), ("糖化血色素", "high"), ("尿酸", "high"), ("GPT(ALT)", "high"),
    ("GOT(AST)", "high"), ("γ-GT", "high"), ("總膽紅素", "high"), ("肌酸酐", "high"),
    ("腎絲球過濾率eGFR", "low"), ("尿素氮", "high"), ("尿蛋白", "positive"), ("尿糖", "positive"),
    ("尿潛血", "positive"), ("血紅素", "low"), ("白血球", "high"), ("血小板", "low"),
    ("收縮壓", "high"), ("舒張壓", "high"), ("BMI", "high"), ("腰圍", "high"),
    ("體脂肪率", "high"), ("甲狀腺刺激素TSH", "high"), ("甲狀腺刺激素TSH", "low"),
    ("胎兒蛋白AFP", "high"), ("癌胚抗原CEA", "high"), ("鈣", "low"), ("鉀", "high"),
    ("鐵蛋白", "low"), ("維生素D", "low"), ("骨密度", "low"), ("高敏感C反應蛋白", "high"),
    ("同半胱胺酸", "high"), ("白蛋白", "low"), ("紅血球", "high"), ("平均紅血球體積MCV", "low"),
    ("游離甲狀腺素Free T4", "low"),
]
_WORD = {"high": "偏高", "low": "偏低", "positive": "陽性"}

JUDGE_SYSTEM = (
    "你是醫療資訊檢索的評分員。給定一個健檢異常項目與一段衛教文字，判斷這段文字對"
    "「該項目異常的受檢者」有多大幫助（原因、風險、飲食、運動、生活建議）。"
    "2=直接相關且有具體資訊；1=部分相關或僅間接提及；0=無關。只輸出 JSON。"
)
JUDGE_SCHEMA = {"type": "object", "properties": {"grade": {"type": "integer", "enum": [0, 1, 2]}},
                "required": ["grade"]}

SYSTEMS = {
    # R3 embedding A/B (dense only)
    "nomic_plain": dict(model="nomic-embed-text", ctx=False, filt=False, pfx=False),
    "nomic": dict(model="nomic-embed-text"),
    "bge-m3": dict(model="bge-m3"),
    "qwen3-emb": dict(model="qwen3-embedding:0.6b"),
    # R4 ablation on top of the chosen embedding (set via --r4-embed)
    "bm25": dict(mode="bm25"),
    "hybrid": dict(mode="hybrid"),
    "dense+rerank": dict(rerank=True),
    "hybrid+rerank": dict(mode="hybrid", rerank=True),
}
R3_SYSTEMS = ["nomic_plain", "nomic", "bge-m3", "qwen3-emb"]


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


class Judge:
    def __init__(self, path: Path, model: str, base_url: str):
        self.path, self.model, self.base_url = path, model, base_url
        self.cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.calls = 0

    def grade(self, item: str, word: str, passage: str) -> int:
        body = passage.split("\n", 1)[-1] if passage.startswith("【主題】") else passage
        key = _h(f"{item}|{word}|{body[:600]}")
        if key in self.cache:
            return self.cache[key]
        payload = {
            "model": self.model, "stream": False, "format": JUDGE_SCHEMA,
            "options": {"temperature": 0, "num_ctx": 4096, "seed": 7},
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": f"異常項目：{item} {word}\n\n衛教文字：\n{body[:900]}"}],
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        g = int(json.loads(r.json()["message"]["content"])["grade"])
        self.cache[key] = g
        self.calls += 1
        if self.calls % 25 == 0:
            self.save()
        return g

    def save(self):
        self.path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")


def ndcg(grades, k=5):
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))
    ideal = sorted(grades, reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


_RERANKER = None
_BM25 = {}


def make_retriever(name: str, db: str, base_url: str, r4_embed: str):
    """Build the Retriever for a system spec (index builds are cached)."""
    global _RERANKER
    from kb_index import prepare_chunks
    from retrieval import BM25Index, CheckitemFacts, Reranker

    spec = {"model": r4_embed, "ctx": True, "filt": True, "pfx": True,
            "mode": "dense", "rerank": False, **SYSTEMS[name]}
    col = None
    if spec["mode"] != "bm25":
        col, stats = build_kb_index(embed_model=spec["model"], db_path=db, base_url=base_url,
                                    contextual=spec["ctx"], filter_offtopic=spec["filt"],
                                    use_prefix=spec["pfx"],
                                    name=collection_name(spec['model'], f"bench{int(spec['ctx'])}{int(spec['filt'])}{int(spec['pfx'])}"))
        if col is None:
            raise RuntimeError(f"index build failed for {name}: {stats}")
    bm25 = None
    if spec["mode"] in ("bm25", "hybrid"):
        key = (spec["ctx"], spec["filt"])
        if key not in _BM25:
            chunks, _ = prepare_chunks(load_articles(config.KB_JSON_PATH), spec["ctx"], spec["filt"])
            facts = CheckitemFacts()
            _BM25[key] = BM25Index(chunks, user_terms=list(facts.by_name) + [q for q, _ in QUERIES])
        bm25 = _BM25[key]
    reranker = None
    if spec["rerank"]:
        _RERANKER = _RERANKER or Reranker()
        reranker = _RERANKER
    return Retriever(col, embed_model=spec["model"], base_url=base_url, use_prefix=spec["pfx"],
                     mode=spec["mode"], bm25=bm25, reranker=reranker)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="all")
    ap.add_argument("--db", default="eval/_chroma_bench")
    ap.add_argument("--ollama", default=config.OLLAMA_BASE_URL)
    ap.add_argument("--judge-model", default="qwen2.5:7b")
    ap.add_argument("--judgments", default="eval/retrieval_judgments.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--r4-embed", default="bge-m3", help="embedding model under the R4 ablation")
    ap.add_argument("--out", default="eval/retrieval_results.json")
    args = ap.parse_args()

    names = {"all": list(SYSTEMS), "r3": R3_SYSTEMS}.get(args.systems) or args.systems.split(",")
    articles = load_articles(config.KB_JSON_PATH)
    offtopic_idx = {i for i, a in enumerate(articles) if is_offtopic(a)}

    runs = {}
    for n in names:
        t = time.time()
        r = make_retriever(n, args.db, args.ollama, args.r4_embed)
        print(f"[{n}] index ready ({time.time() - t:.0f}s)", flush=True)
        t = time.time()
        runs[n] = [r.search(finding_query({"display_name": item, "status": st}), args.k)
                   for item, st in QUERIES]
        runs[n + "__latency_ms"] = round((time.time() - t) / len(QUERIES) * 1000, 1)

    judge = Judge(Path(args.judgments), args.judge_model, args.ollama)
    report = {}
    pooled_rel = [set() for _ in QUERIES]
    graded = {}
    for n in names:
        graded[n] = []
        for qi, hits in enumerate(runs[n]):
            item, st = QUERIES[qi]
            gs = []
            for h in hits:
                g = judge.grade(item, _WORD[st], h["text"])
                gs.append(g)
                if g >= 1:
                    pooled_rel[qi].add(_h(h["text"].split("\n", 1)[-1][:600]))
            graded[n].append(gs)
        print(f"[{n}] judged (cache size {len(judge.cache)})", flush=True)
    judge.save()

    for n in names:
        nd, p5, mrr, rec, off = [], [], [], [], []
        for qi, hits in enumerate(runs[n]):
            gs = graded[n][qi]
            nd.append(ndcg(gs, 5))
            p5.append(sum(g >= 1 for g in gs[:5]) / 5)
            mrr.append(next((1 / (i + 1) for i, g in enumerate(gs) if g >= 1), 0.0))
            rel_ids = {_h(h["text"].split("\n", 1)[-1][:600]) for h, g in zip(hits, gs) if g >= 1}
            rec.append(len(rel_ids) / len(pooled_rel[qi]) if pooled_rel[qi] else 0.0)
            off.append(sum(int(h["id"].split("_")[1]) in offtopic_idx
                           for h in hits[:5] if h["id"].startswith("kb_")) / 5)
        mean = lambda xs: round(sum(xs) / len(xs), 3)  # noqa: E731
        report[n] = {"nDCG@5": mean(nd), "P@5": mean(p5), "MRR": mean(mrr),
                     "Recall@10(pooled)": mean(rec), "offtopic@5": mean(off),
                     "query_latency_ms": runs[n + "__latency_ms"]}

    print(f"\n{'system':<16}{'nDCG@5':>8}{'P@5':>7}{'MRR':>7}{'R@10':>7}{'off@5':>7}{'ms/q':>8}")
    for n, m in report.items():
        print(f"{n:<16}{m['nDCG@5']:>8}{m['P@5']:>7}{m['MRR']:>7}{m['Recall@10(pooled)']:>7}"
              f"{m['offtopic@5']:>7}{m['query_latency_ms']:>8}")
    Path(args.out).write_text(json.dumps({"queries": len(QUERIES), "judge": args.judge_model,
                                          "systems": report}, ensure_ascii=False, indent=2),
                              encoding="utf-8")


if __name__ == "__main__":
    main()
