"""
report_cache.py — Report-level result cache for the UI.

Key = sha256(PDF bytes) + a fingerprint of everything that can change the
result: runtime config (models, num_ctx, prompt/context budgets, retrieval
mode, reranker), the prompt version, the data files (reference ranges, lab
synonyms, KB passages) and the source of the analysis modules. Any change
invalidates old entries automatically (they are simply never looked up again).

Privacy: the cached JSON holds the analysis result only — findings, tags,
retrieved KB passages, timings. The extracted report text and tables (which
carry the name / ID number printed on the report) are NOT stored, nor are the
raw unmatched lines of the abnormal-detection debug info or the file name; the
key is a one-way hash of the PDF bytes. Disable with REPORT_CACHE=0;
`ReportCache.clear()` deletes every entry.

The eval harness never uses this module (every eval run is a cold analysis).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import config

CACHE_VERSION = 1
_HERE = Path(__file__).resolve().parent
# Modules whose code determines the result.
_SOURCES = ["pipeline.py", "llm.py", "abnormal.py", "conditions.py", "verify.py", "retrieval.py",
            "embeddings.py", "pdf_utils.py", "ocr.py", "kb_index.py"]
_NOT_CACHED = ("text", "tables")  # carry PII printed on the report
_DEBUG_KEYS = ("sex", "sex_source", "llm_mapped")
_FILE_HASH: Dict[Tuple[str, float, int], str] = {}


def _file_hash(path: str) -> str:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = _HERE / p
    try:
        st = p.stat()
    except OSError:
        return "missing"
    key = (str(p), st.st_mtime, st.st_size)
    if key not in _FILE_HASH:
        _FILE_HASH[key] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return _FILE_HASH[key]


def config_fingerprint(**extra) -> str:
    """Hash of every setting / file / module that can change an analysis result.
    `extra`: per-request options (sex, retrieval mode, reranker, stub, …)."""
    from llm import PROMPT_VERSION  # local: keep this module import-light
    parts = {
        "v": CACHE_VERSION, "prompt": PROMPT_VERSION,
        "cfg": {k: getattr(config, k) for k in (
            "LLM_MODEL", "EMBED_MODEL", "LLM_NUM_CTX", "LLM_NUM_PREDICT", "LLM_TEMPERATURE",
            "REPORT_MAX_CHARS", "CONTEXT_MAX_CHARS", "RETRIEVAL_MODE", "USE_RERANKER", "RERANKER_MODEL",
            "KB_JSON_PATH", "REFERENCE_RANGES_PATH")},
        "data": {p: _file_hash(p) for p in (config.KB_JSON_PATH, config.REFERENCE_RANGES_PATH,
                                            "data/lab_synonyms.json")},
        "code": {s: _file_hash(str(_HERE / s)) for s in _SOURCES},
        "extra": extra,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def cache_key(pdf_bytes: bytes, fingerprint: str) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:32] + "_" + fingerprint


class ReportCache:
    def __init__(self, directory: str = config.REPORT_CACHE_DIR, enabled: bool = config.REPORT_CACHE):
        self.dir = Path(directory)
        self.enabled = enabled

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str):
        """Return (PipelineResult, meta) or None. A corrupt entry counts as a miss."""
        if not self.enabled:
            return None
        p = self._path(key)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        from pipeline import PipelineResult
        names = {f.name for f in dataclasses.fields(PipelineResult)}
        try:
            r = PipelineResult(**{k: v for k, v in d.get("result", {}).items() if k in names})
        except TypeError:
            return None
        return r, d.get("meta", {})

    def put(self, key: str, result, meta: Optional[Dict] = None) -> bool:
        """Store a successful result (errors are never cached). Atomic write."""
        if not self.enabled or result.error:
            return False
        data = {k: v for k, v in dataclasses.asdict(result).items() if k not in _NOT_CACHED}
        # abnormal_debug["unmatched"] lists raw unparsed lines (may include the name / ID header)
        data["abnormal_debug"] = {k: v for k, v in (data.get("abnormal_debug") or {}).items()
                                  if k in _DEBUG_KEYS}
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"result": data, "meta": meta or {}}, f, ensure_ascii=False)
            os.replace(tmp, self._path(key))
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False
        return True

    def clear(self) -> int:
        n = 0
        if self.dir.exists():
            for p in self.dir.glob("*.json"):
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
        return n
