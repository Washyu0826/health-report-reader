"""
embeddings.py — Batched Ollama embeddings with model-specific query/document prefixes.

Asymmetric retrieval models expect different framing for queries and passages;
omitting it (as R0 did for nomic-embed-text) silently degrades recall.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Tuple

import requests

import config

EMBED_INPUT_CAP = 1500  # chars per input

_QWEN3_INSTRUCT = (
    "Instruct: Given a health-check lab finding, retrieve health-education passages "
    "about its causes, risks, diet, exercise or lifestyle advice\nQuery: "
)

# model-name prefix -> (query_prefix, document_prefix)
_PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "qwen3-embedding": (_QWEN3_INSTRUCT, ""),
    "bge-m3": ("", ""),
}


def prefixes(model: str) -> Tuple[str, str]:
    base = model.split(":")[0].split("/")[-1]
    return _PREFIXES.get(base, ("", ""))


def embed(texts: List[str], model: str, base_url: str = config.OLLAMA_BASE_URL,
          kind: str = "document", batch: int = 32,
          use_prefix: bool = True) -> Tuple[List[List[float]], str]:
    """Embed texts via /api/embed (batched). kind: "query" | "document"."""
    q_pre, d_pre = prefixes(model) if use_prefix else ("", "")
    pre = q_pre if kind == "query" else d_pre
    out: List[List[float]] = []
    for i in range(0, len(texts), batch):
        inputs = [pre + t[:EMBED_INPUT_CAP] for t in texts[i:i + batch]]
        vecs, err = _embed_batch(inputs, model, base_url)
        if err:
            return [], err
        out.extend(vecs)
    return out, ""


def _embed_batch(inputs: List[str], model: str, base_url: str,
                 retries: int = 1) -> Tuple[List[List[float]], str]:
    """POST one batch; retry once, then bisect so a single bad input cannot
    fail a whole index build (it is embedded with a shortened copy instead)."""
    err = ""
    for _ in range(retries + 1):
        try:
            resp = requests.post(f"{base_url}/api/embed",
                                 json={"model": model, "input": inputs, "truncate": True,
                                       "keep_alive": config.OLLAMA_KEEP_ALIVE},
                                 timeout=300)
            resp.raise_for_status()
            vecs = resp.json().get("embeddings") or []
            if len(vecs) == len(inputs):
                return vecs, ""
            err = f"Embedding count mismatch ({len(vecs)} != {len(inputs)})"
        except requests.exceptions.ConnectionError:
            return [], f"Cannot connect to Ollama at {base_url}"
        except Exception as e:  # noqa: BLE001
            err = f"Embedding error ({model}): {e}"
    if len(inputs) > 1:
        mid = len(inputs) // 2
        left, e1 = _embed_batch(inputs[:mid], model, base_url, 0)
        right, e2 = _embed_batch(inputs[mid:], model, base_url, 0)
        return (left + right, "") if not (e1 or e2) else ([], e1 or e2)
    if len(inputs[0]) > 200:  # last resort for a single pathological input
        return _embed_batch([inputs[0][:500]], model, base_url, 0)
    return [], err


# ─── Query-embedding cache ──────────────────────────────────────────────────
# Retrieval embeds one query per abnormal finding, and the query strings repeat
# across reports ("LDL 偏高 原因 風險 …"). Each /api/embed round trip costs
# ~0.1 s even with the model resident, so queries are (a) cached in-process and
# (b) pre-embedded in ONE batched call per report (prefetch_queries, called by
# pipeline.retrieve_stage before Retriever.retrieve).
QUERY_CACHE_SIZE = 1024
_QCACHE: "OrderedDict[Tuple[str, str, str, bool], List[float]]" = OrderedDict()


def _qkey(text: str, model: str, base_url: str, use_prefix: bool) -> Tuple[str, str, str, bool]:
    return (text, model, base_url, bool(use_prefix))


def _qput(key, vec: List[float]) -> None:
    _QCACHE[key] = vec
    _QCACHE.move_to_end(key)
    while len(_QCACHE) > QUERY_CACHE_SIZE:
        _QCACHE.popitem(last=False)


def prefetch_queries(texts: List[str], model: str, base_url: str = config.OLLAMA_BASE_URL,
                     use_prefix: bool = True) -> Tuple[int, str]:
    """Embed every not-yet-cached query in one batched call. Returns (n_embedded, error)."""
    todo = list(dict.fromkeys(t for t in texts if _qkey(t, model, base_url, use_prefix) not in _QCACHE))
    if not todo:
        return 0, ""
    vecs, err = embed(todo, model, base_url, kind="query", use_prefix=use_prefix)
    if err:
        return 0, err
    for t, v in zip(todo, vecs):
        _qput(_qkey(t, model, base_url, use_prefix), v)
    return len(todo), ""


def clear_query_cache() -> None:
    _QCACHE.clear()


def embed_query(text: str, model: str, base_url: str = config.OLLAMA_BASE_URL,
                use_prefix: bool = True):
    key = _qkey(text, model, base_url, use_prefix)
    hit: Optional[List[float]] = _QCACHE.get(key)
    if hit is not None:
        _QCACHE.move_to_end(key)
        return hit, ""
    vecs, err = embed([text], model, base_url, kind="query", use_prefix=use_prefix)
    if vecs:
        _qput(key, vecs[0])
    return (vecs[0] if vecs else None), err
