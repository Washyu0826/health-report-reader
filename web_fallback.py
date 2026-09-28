"""
web_fallback.py — Optional, privacy-preserving web search for thin KB coverage (R6).

OFF by default. When enabled, the only thing that ever leaves the machine is a
query built from a *canonical* lab-item name and a direction word, e.g.
"LDL膽固醇 偏高 衛教". Never sent: values, units, raw extracted names (which
could contain a patient's name), report text, or unmatched items.

`build_query()` is the single choke point for outbound text and is unit-tested.
Results are cached on disk under a stable SHA-256 key and the cache is read
before any network call.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

_DIR = {"high": "偏高", "low": "偏低", "positive": "陽性"}
_SAFE = re.compile(r"^[\w一-鿿\s\-/().%γαβ]+$")
DEFAULT_CACHE = Path("chroma_db") / "web_cache.json"


_CANONICAL: Optional[Dict[str, str]] = None


def canonical_names(path: str = "data/lab_synonyms.json") -> Dict[str, str]:
    """canonical_key -> curated zh display name (the only names allowed outbound)."""
    global _CANONICAL
    if _CANONICAL is None:
        p = Path(__file__).resolve().parent / path
        data = json.loads(p.read_text(encoding="utf-8"))
        _CANONICAL = {it["key"]: it.get("zh", "") for it in data["items"]}
    return _CANONICAL


def build_query(finding: Dict) -> Optional[str]:
    """Return the outbound query for a finding, or None if it must not be sent.

    The name is looked up in the curated synonym table by canonical_key; no
    text that came from the report (raw name, display name, value) is used.
    """
    name = canonical_names().get(finding.get("canonical_key") or "", "").strip()
    word = _DIR.get(finding.get("status") or finding.get("direction"))
    if not word or not name or not _SAFE.match(name) or len(name) > 30:
        return None
    return f"{name} {word} 衛教"


def _ddg(query: str, max_results: int = 3, timeout: float = 8.0) -> List[Dict]:
    resp = requests.get("https://html.duckduckgo.com/html/", params={"q": query, "kl": "tw-tzh"},
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    if resp.status_code != 200:
        return []
    out = []
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
                         r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.S):
        snippet = re.sub(r"<[^>]+>", "", m.group(3)).strip()
        if len(snippet) >= 30:
            out.append({"url": m.group(1), "title": re.sub(r"<[^>]+>", "", m.group(2)).strip(),
                        "snippet": snippet})
        if len(out) >= max_results:
            break
    return out


class WebFallback:
    def __init__(self, cache_path: Path = DEFAULT_CACHE,
                 search_fn: Callable[[str], List[Dict]] = _ddg, max_findings: int = 3):
        self.cache_path = Path(cache_path)
        self.search_fn = search_fn
        self.max_findings = max_findings
        try:
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            self.cache = {}
        self.sent_queries: List[str] = []  # audit trail, shown in the UI

    def search(self, findings: List[Dict]) -> List[Dict]:
        chunks = []
        for f in findings[: self.max_findings]:
            q = build_query(f)
            if not q:
                continue
            key = hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]
            if key not in self.cache:
                try:
                    self.cache[key] = {"query": q, "results": self.search_fn(q)}
                except Exception:  # noqa: BLE001
                    continue
                self.sent_queries.append(q)
            for i, r in enumerate(self.cache[key]["results"]):
                chunks.append({"id": f"web_{key}_{i}", "text": f"{r['title']}\n{r['snippet']}",
                               "url": r.get("url", ""), "score": 0.0, "source": "web",
                               "for_finding": f.get("display_name")})
        self._save()
        return chunks

    def _save(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
