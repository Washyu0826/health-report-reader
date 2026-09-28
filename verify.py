"""
verify.py — Claim-level verification of generated tags (R5).

Every tag carries a source claim ("report", "kb:<id>", "rule", "inferred").
This layer checks that claim, cheapest test first:

  report   -> the tag's key term must appear in the report text or findings;
              otherwise it is downgraded to "inferred".
  kb:<id>  -> lexical support: enough of the tag's character bigrams occur in
              the cited passage. Tags that fail go to small batched LLM calls
              ("does this passage support this advice?"); unsupported tags
              are dropped.
  inferred -> capped at MAX_INFERRED_SHARE of all tags (lowest confidence
              dropped first).
  rule     -> produced deterministically with evidence; always kept.

The result records per-tag verdicts so the UI can show them and the eval can
report citation validity.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional

import requests

import config

ADVICE_KEYS = ["lifestyle", "food", "exercise", "supplements", "avoid"]
ALL_KEYS = ["conditions", "risks", *ADVICE_KEYS]
LEXICAL_SUPPORT = 0.5
MAX_INFERRED_SHARE = 0.25

_STRIP = re.compile(r"(建議|避免|減少|增加|多吃|少吃|攝取|控制|維持|適量|每天|每週|規律|注意|定期|追蹤|風險|疑似|偏高|偏低|異常)")


def _core(text: str) -> str:
    core = _STRIP.sub("", re.sub(r"[\s，、。()（）:：/]+", "", text))
    return core or text


def _bigrams(s: str) -> set:
    s = re.sub(r"\W+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())


def lexical_support(tag: str, passage: str) -> float:
    grams = _bigrams(_core(tag))
    if not grams:
        return 0.0
    body = re.sub(r"\W+", "", passage)
    return sum(g in body for g in grams) / len(grams)


_JUDGE_SYSTEM = (
    "你是醫療內容查核員。對每一組「建議標籤＋引用段落」，判斷段落內容是否支持該建議"
    "（段落明確提到或直接蘊含此建議即為支持）。只輸出 JSON。"
)


# Judge batching (R6 fix). The original single call put every pending pair in
# one prompt with num_ctx=8192; a report with 30 pending pairs (~16k CJK chars)
# overflowed it. Ollama then either silently kept only the last ~num_ctx/2
# tokens (system prompt and first pairs lost -> degenerate all-true/all-false
# answers) or the runner crashed (HTTP 500) -> every pair "unverified".
# Now: small batches under a char budget, the same num_ctx as the generator
# (no model reload), bounded num_predict, answers keyed by an explicit index
# (not array position), and one retry for whatever is still missing.
JUDGE_BATCH = 8
JUDGE_PASSAGE_CHARS = 500
JUDGE_MAX_PROMPT_CHARS = 5000


def _judge_batches(pairs: List[Dict], size: int = JUDGE_BATCH,
                   max_chars: int = JUDGE_MAX_PROMPT_CHARS) -> List[List[int]]:
    """Split pair indices into batches of <= size pairs and <= max_chars of prompt."""
    batches, cur, used = [], [], 0
    for i, p in enumerate(pairs):
        cost = len(p["tag"]) + min(len(p["passage"]), JUDGE_PASSAGE_CHARS) + 20
        if cur and (len(cur) >= size or used + cost > max_chars):
            batches.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += cost
    if cur:
        batches.append(cur)
    return batches


def llm_support_judge(model: str = config.LLM_MODEL, base_url: str = config.OLLAMA_BASE_URL,
                      batch_size: int = JUDGE_BATCH, retries: int = 1
                      ) -> Callable[[List[Dict]], List[Optional[bool]]]:
    """Return judge(pairs) -> [True/False/None] aligned with `pairs`.
    None = no usable answer after retries (the tag is kept, marked unverified)."""

    def ask(sub: List[Dict]) -> Dict[int, bool]:
        n = len(sub)
        schema = {"type": "object", "properties": {"results": {
            "type": "array", "minItems": n, "maxItems": n,
            "items": {"type": "object", "properties": {
                "i": {"type": "integer", "enum": list(range(n))},
                "supported": {"type": "boolean"}}, "required": ["i", "supported"]}}},
            "required": ["results"]}
        body = "\n\n".join(f"[{i}] 標籤：{p['tag']}\n段落：{p['passage'][:JUDGE_PASSAGE_CHARS]}"
                           for i, p in enumerate(sub))
        # num_ctx = the generator's (no reload); ~14 output tokens per pair, so
        # 40 + 16n is ample headroom (a cap, not a target: hitting it = retry).
        payload = {"model": model, "stream": False, "format": schema,
                   "keep_alive": config.OLLAMA_KEEP_ALIVE,
                   "options": {"temperature": 0, "num_ctx": config.LLM_NUM_CTX, "seed": 11,
                               "num_predict": 40 + 16 * n},
                   "messages": [{"role": "system", "content": _JUDGE_SYSTEM},
                                {"role": "user", "content":
                                 f"共 {n} 組（編號 0–{n - 1}），每組輸出 {{\"i\": 編號, \"supported\": true/false}}：\n\n{body}"}]}
        try:
            r = requests.post(f"{base_url}/api/chat", json=payload, timeout=config.LLM_TIMEOUT_S)
            r.raise_for_status()
            data = r.json()
            if data.get("done_reason") == "length":
                return {}
            out = {}
            for item in json.loads(data["message"]["content"]).get("results", []):
                i = item.get("i")
                if isinstance(i, int) and 0 <= i < n and i not in out \
                        and isinstance(item.get("supported"), bool):
                    out[i] = item["supported"]
            return out
        except Exception:  # noqa: BLE001 — transport/parse failure -> retried, then None
            return {}

    def judge(pairs: List[Dict]) -> List[Optional[bool]]:
        verdicts: List[Optional[bool]] = [None] * len(pairs)
        for batch in _judge_batches(pairs, batch_size):
            todo = batch
            for _attempt in range(retries + 1):
                got = ask([pairs[j] for j in todo])
                for local, ok in got.items():
                    verdicts[todo[local]] = ok
                todo = [j for j in todo if verdicts[j] is None]
                if not todo:
                    break
        return verdicts
    return judge


def verify_tags(tags: Dict, chunks: List[Dict], report_text: str, findings: List[Dict],
                judge: Optional[Callable[[List[Dict]], List[Optional[bool]]]] = None,
                drop_unsupported: bool = True) -> Dict:
    """Return a new tags dict with verified tags and a `_verification` summary.

    drop_unsupported=False keeps tags the judge rejected, marked
    verdict="unsupported" (counted in stats["kb_unsupported"], not in
    kb_dropped), so a caller can try to rewrite them (graph_workflow.py).
    They are excluded from the inferred-share cap, exactly as when dropped."""
    by_id = {c["id"]: c["text"] for c in chunks}
    evidence_text = report_text + "\n" + "\n".join(
        f"{f.get('display_name', '')} {f.get('name', '')}" for f in findings)
    out = {k: v for k, v in tags.items() if k not in ALL_KEYS}
    stats = {"report_checked": 0, "report_downgraded": 0, "kb_checked": 0,
             "kb_lexical_ok": 0, "kb_llm_ok": 0, "kb_dropped": 0, "kb_unverified": 0,
             "inferred_capped": 0}
    pending = []  # (key, idx) needing LLM check

    for key in ALL_KEYS:
        kept = []
        for t in tags.get(key, []):
            t = dict(t)
            src = t.get("src", "")
            if src == "report":
                stats["report_checked"] += 1
                core = _core(t["text"])
                if core not in evidence_text and lexical_support(t["text"], evidence_text) < 0.75:
                    t.update(src="inferred", conf=round(t["conf"] * 0.6, 2), verdict="not_in_report")
                    stats["report_downgraded"] += 1
                else:
                    t["verdict"] = "in_report"
            elif src.startswith("kb:"):
                stats["kb_checked"] += 1
                passage = by_id.get(src[3:], "")
                if lexical_support(t["text"], passage) >= LEXICAL_SUPPORT:
                    t["verdict"] = "lexical"
                    stats["kb_lexical_ok"] += 1
                else:
                    t["verdict"] = "pending"
                    pending.append((key, len(kept), passage))
            kept.append(t)
        out[key] = kept

    if pending and judge:
        verdicts = judge([{"tag": out[k][i]["text"], "passage": p} for k, i, p in pending])
        for (k, i, _), ok in zip(pending, verdicts):
            if ok is None:
                out[k][i]["verdict"] = "unverified"
                stats["kb_unverified"] += 1
            elif ok:
                out[k][i]["verdict"] = "llm_supported"
                stats["kb_llm_ok"] += 1
            else:
                out[k][i]["verdict"] = "unsupported"
    if not drop_unsupported:
        stats["kb_unsupported"] = sum(t.get("verdict") == "unsupported" for k in ALL_KEYS for t in out[k])
    for k in ALL_KEYS:
        if drop_unsupported:
            before = len(out[k])
            out[k] = [t for t in out[k] if t.get("verdict") != "unsupported"]
            stats["kb_dropped"] += before - len(out[k])
        if not judge:
            for t in out[k]:
                if t.get("verdict") == "pending":
                    t["verdict"] = "unverified"
                    stats["kb_unverified"] += 1

    # Cap inferred share across all categories.
    flat = [(k, t) for k in ALL_KEYS for t in out[k] if t.get("verdict") != "unsupported"]
    inferred = sorted([(k, t) for k, t in flat if t.get("src") == "inferred"], key=lambda x: x[1]["conf"])
    allowed = int(MAX_INFERRED_SHARE * max(len(flat), 1))
    for k, t in inferred[:max(0, len(inferred) - allowed)]:
        out[k].remove(t)
        stats["inferred_capped"] += 1

    cited = stats["kb_checked"]
    stats["citation_support_rate"] = round(
        (stats["kb_lexical_ok"] + stats["kb_llm_ok"]) / cited, 3) if cited else None
    out["_verification"] = stats
    return out
