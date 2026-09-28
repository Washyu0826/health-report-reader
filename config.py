"""
config.py — Single source of truth for runtime settings (env-overridable).
"""

import os

# 127.0.0.1, not localhost: on Windows "localhost" can resolve to ::1 first and
# stall ~2-3 s per request before falling back to IPv4.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")  # R3 benchmark winner

# Ollama silently keeps only the *tail* of the prompt when it exceeds num_ctx
# (default 4096 on <24 GB GPUs), so it must be set explicitly.
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "12288"))
# Cap on generated tokens (a safety net, not a speed knob: hitting it is a failure).
# The v4 schema (advice + text-stated conditions, <=4 tags per category, capped
# tag/summary lengths) needs ~200-450 tokens; the worst case is < 1200.
LLM_NUM_PREDICT = int(os.environ.get("LLM_NUM_PREDICT", "1200"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
LLM_TIMEOUT_S = int(os.environ.get("LLM_TIMEOUT_S", "240"))
# How long Ollama keeps a model resident after a call. The default (5 min)
# unloads the 7B model between UI sessions, and the next report pays ~9 s of
# load time. Changing keep_alive never triggers a reload (num_ctx does).
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

# Prompt budgets (characters). CJK ~1 token/char, so these keep the prompt
# comfortably inside LLM_NUM_CTX together with the system prompt + output.
REPORT_MAX_CHARS = int(os.environ.get("REPORT_MAX_CHARS", "6000"))
CONTEXT_MAX_CHARS = int(os.environ.get("CONTEXT_MAX_CHARS", "3500"))

# Health-education passages indexed for retrieval (Title/Content/TagName/Source).
KB_JSON_PATH = os.environ.get("KB_JSON_PATH", "data/kb_sample.json")
# Public adult reference ranges (with citations) used when a report prints no
# range; also the source of the per-item fact sheets. A range printed on the
# report always takes precedence.
REFERENCE_RANGES_PATH = os.environ.get("REFERENCE_RANGES_PATH", "data/reference_ranges.json")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")

# Retrieval (R7 UI). The final values are set after the retrieval benchmark.
# Defaults follow the R4 ablation: hybrid+rerank scored best on retrieval (nDCG) but
# lowered end-to-end advice relevance (26% -> 17%), so dense-only is the default.
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "dense")  # dense | bm25 | hybrid
USE_RERANKER = os.environ.get("USE_RERANKER", "0") not in ("0", "false", "False", "")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# LangGraph rewrite loop (graph_workflow.py). Off by default: on the synthetic
# quick set it rescued 3 of 59 unsupported advice tags (+3 pts relevance) for
# ~7-9 s extra per report that has any. Enable with VERIFY_REWRITE=1 or
# run_graph(rewrite=True).
VERIFY_REWRITE = os.environ.get("VERIFY_REWRITE", "0") not in ("0", "false", "False", "")

# Report-level result cache (UI only; eval always bypasses it).
REPORT_CACHE = os.environ.get("REPORT_CACHE", "1") not in ("0", "false", "False", "")
REPORT_CACHE_DIR = os.environ.get("REPORT_CACHE_DIR", os.path.join(CHROMA_DB_PATH, "cache"))
