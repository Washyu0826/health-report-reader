"""
kb_index.py — Knowledge-base preparation and vector indexing (R3).

Improvements over the R0 loader:
  * cleaning: strips publisher bylines 【…／…】 and boilerplate lines
  * off-topic filter: drops finance/insurance/etc. articles by tag denylist
  * contextual chunks: each chunk is prefixed with its article title (when the
    article has one) and topic tags, so a chunk that says "每天快走30分鐘" still
    carries "高血壓, 運動" semantics
  * article records: {Title?, Content, TagName, Source?}; Source (a citation
    URL) is kept in chunk metadata so the UI can link to it
  * one collection per embedding model (A/B indices coexist; switching models
    can never silently query vectors from another model)
  * completion marker written only after the whole index is built, so a
    crashed/partial build is detected and rebuilt
  * batched /api/embed with model-specific document prefixes
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import config
from embeddings import embed

INDEX_VERSION = "r3-v4"  # v3: title prefix + source URL in metadata; v4: applies_to in metadata
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
STATE_FILE = "kb_index_state.json"

# Articles whose tags mark them as non-medical content.
OFFTOPIC_TAGS = {
    "理財", "保險", "理賠", "儲蓄", "招財", "勞保年金", "退休", "退休金", "投資",
    "股票", "基金", "房貸", "命理", "星座", "風水", "開運", "彩妝", "美妝",
    "旅遊", "捐贈", "生活便利貼", "清潔", "收納", "家事",
}
# Generic site-section tags that carry no topical meaning in the chunk prefix.
GENERIC_TAGS = {
    "保健新聞", "養生保健", "醫師觀點", "健康小撇步", "生活智慧", "生活知識影片",
    "養生專欄", "其他身心健康", "其他健康食材", "生活知識",
}

_BYLINE = re.compile(r"【[^】]{0,40}[／/][^】]{0,60}】")
_BOILERPLATE = re.compile(r"^(延伸閱讀|本文(?:摘自|經)|（?圖片來源|責任編輯|資料來源).*$", re.M)
_URL = re.compile(r"https?://\S+|\b(?:bit\.ly|reurl\.cc|lihi\d?\.cc)/\S+")
# promotional sentences: book-store plugs, fan-page calls, magazine excerpt credits
_PROMO = re.compile(r"[^。！？\n]*(?:網路書店|熱賣中|免運|粉絲團|點我看更多|此為《[^》]*》[^。\n]*部分內容)[^。！？\n]*[。！？～]?")


def _brand_terms() -> List[str]:
    """Publisher names to neutralise. Optional: kept in an untracked data file
    next to the KB (never in code); absent -> nothing to neutralise."""
    p = Path(config.KB_JSON_PATH).with_name("brand_terms.txt")
    if not p.is_absolute() and not p.exists():
        p = Path(__file__).resolve().parent / p
    try:
        return [t.strip() for t in p.read_text(encoding="utf-8").splitlines() if t.strip()]
    except OSError:
        return []


_BRANDS = _brand_terms()


def clean_article(text: str) -> str:
    text = _BYLINE.sub("", text or "")
    text = _BOILERPLATE.sub("", text)
    text = _URL.sub("", text)
    text = _PROMO.sub("", text)
    for b in _BRANDS:
        text = re.sub(rf"【[^】]*{re.escape(b)}[^】]*】", "", text)
        text = text.replace(f"《{b}》", "本刊").replace(b, "本刊")
    text = text.replace("　", " ").replace("\xa0", " ")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def article_tags(article: Dict) -> List[str]:
    return [t.strip() for t in (article.get("TagName") or "").split(",") if t.strip()]


def is_offtopic(article: Dict) -> bool:
    return any(t in OFFTOPIC_TAGS for t in article_tags(article))


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sentence-aware sliding window."""
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        for sep in ["\n\n", "。\n", "。", "！", "？", "\n"]:
            idx = text.rfind(sep, start + size // 2, end)
            if idx > 0:
                end = idx + len(sep)
                break
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 40]


_URL_IN = re.compile(r"https?://\S+")


def article_url(article: Dict) -> str:
    """Citation URL of an article (its Source field), '' when none."""
    m = _URL_IN.search(article.get("Source") or "")
    return m.group(0) if m else ""


def prepare_chunks(articles: List[Dict], contextual: bool = True,
                   filter_offtopic: bool = True) -> Tuple[List[Dict], Dict]:
    """Return (chunks, stats). chunk = {id, text, article_idx, tags, title, url, applies_to}.

    applies_to: the article's `applies_to` list (e.g. "ldl:high", "cond:代謝症候群")
    comma-joined, "" when the article has none (Chroma metadata must be scalars)."""
    stats = {"articles": len(articles), "offtopic_dropped": 0, "empty_dropped": 0}
    out: List[Dict] = []
    for ai, art in enumerate(articles):
        if filter_offtopic and is_offtopic(art):
            stats["offtopic_dropped"] += 1
            continue
        body = clean_article(art.get("Content") or "")
        if len(body) < 80:
            stats["empty_dropped"] += 1
            continue
        tags = [t for t in article_tags(art) if t not in GENERIC_TAGS][:6]
        title = clean_article(art.get("Title") or "").replace("\n", " ").strip()
        prefix = ""
        if contextual:
            prefix = (f"【標題】{title}\n" if title else "") + (f"【主題】{'、'.join(tags)}\n" if tags else "")
        url = article_url(art)
        applies = ",".join(a.strip() for a in (art.get("applies_to") or []) if a and a.strip())
        for ci, ch in enumerate(chunk_text(body)):
            out.append({"id": f"kb_{ai}_{ci}", "text": prefix + ch, "article_idx": ai,
                        "tags": ",".join(tags), "title": title, "url": url, "applies_to": applies})
    stats["chunks"] = len(out)
    return out, stats


def load_articles(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collection_name(embed_model: str, kind: str = "health_kb") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", embed_model.lower()).strip("-")
    return f"{kind}__{slug}"[:60]


def _fingerprint(json_path: str, embed_model: str, contextual: bool, filt: bool) -> str:
    h = hashlib.sha256(Path(json_path).read_bytes())
    h.update(f"|{INDEX_VERSION}|{embed_model}|ctx={contextual}|filt={filt}"
             f"|size={CHUNK_SIZE}|ovlp={CHUNK_OVERLAP}".encode())
    return h.hexdigest()[:24]


def _read_state(db_path: str) -> Dict:
    p = Path(db_path) / STATE_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _write_state(db_path: str, state: Dict) -> None:
    Path(db_path).mkdir(parents=True, exist_ok=True)
    (Path(db_path) / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _client(db_path: str):
    import chromadb
    Path(db_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def build_kb_index(
    json_path: str = config.KB_JSON_PATH,
    embed_model: str = config.EMBED_MODEL,
    db_path: str = config.CHROMA_DB_PATH,
    base_url: str = config.OLLAMA_BASE_URL,
    contextual: bool = True,
    filter_offtopic: bool = True,
    name: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    use_prefix: bool = True,
) -> Tuple[Optional[object], Dict]:
    """Build (or reuse) the KB collection for `embed_model`. Returns (collection, stats)."""
    name = name or collection_name(embed_model)
    fp = _fingerprint(json_path, embed_model, contextual, filter_offtopic) + ("" if use_prefix else "-nopfx")
    client = _client(db_path)
    state = _read_state(db_path)
    existing = {c.name if hasattr(c, "name") else c for c in client.list_collections()}

    if name in existing and state.get(name, {}).get("fingerprint") == fp:
        col = client.get_collection(name)
        return col, {**state[name].get("stats", {}), "reused": True}
    if name in existing:
        client.delete_collection(name)  # stale or partial build

    chunks, stats = prepare_chunks(load_articles(json_path), contextual, filter_offtopic)
    col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    B = 64
    for i in range(0, len(chunks), B):
        batch = chunks[i:i + B]
        vecs, err = embed([c["text"] for c in batch], embed_model, base_url, kind="document",
                          use_prefix=use_prefix)
        if err:
            stats["error"] = err
            return None, stats  # no completion marker -> rebuilt next time
        col.add(ids=[c["id"] for c in batch], embeddings=vecs,
                documents=[c["text"] for c in batch],
                metadatas=[{"citation_id": c["id"], "article_idx": c["article_idx"],
                            "tags": c["tags"], "title": c["title"], "url": c["url"],
                            "applies_to": c["applies_to"]} for c in batch])
        if progress:
            progress(min(i + B, len(chunks)), len(chunks))
    state[name] = {"fingerprint": fp, "embed_model": embed_model, "stats": stats}
    _write_state(db_path, state)
    return col, stats


def get_kb_index(embed_model: str = config.EMBED_MODEL,
                 db_path: str = config.CHROMA_DB_PATH, name: Optional[str] = None):
    """Return the completed collection for `embed_model`, or None."""
    name = name or collection_name(embed_model)
    if name not in _read_state(db_path):
        return None
    try:
        return _client(db_path).get_collection(name)
    except Exception:  # noqa: BLE001
        return None
