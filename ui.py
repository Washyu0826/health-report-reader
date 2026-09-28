"""
ui.py — Gradio front end for the health-report tagger (R7).

    python ui.py                 # full pipeline (Ollama LLM + embeddings + KB index)
    python ui.py --demo-stub     # no LLM / no embeddings / no network:
                                 # real extraction + abnormal detection + fact-sheet
                                 # lookup, tags filled from a canned example
    python ui.py --host 127.0.0.1 --port 7860

Design notes
  * Heavy resources (lab catalog, fact sheets, KB vector index, BM25, reranker)
    are built once at start-up, never per request.
  * Everything that came from the PDF or the LLM is passed through
    `html.escape` before it is placed in HTML.
  * The Ollama URL comes only from config / env; the UI cannot change it.
  * Web search is off by default; when enabled, only curated canonical item
    names leave the machine (web_fallback.build_query) and every outbound query
    is listed in an audit panel.
  * Trends store only an opaque (salted) alias hash, the report date and
    canonical findings — no file names, raw text, summary or tags.
  * Latency: findings, the rule-derived conditions/risks and the metric tags
    are shown as soon as they exist (~0.1-1 s, "time to first result"); the
    LLM's JSON is streamed and its advice tags appear as they are generated;
    the final, verified tags replace them. A report analysed before (same PDF
    bytes + same config/data/code fingerprint) is served from the report
    cache (report_cache.py; REPORT_CACHE=0 disables it) — the cache never
    stores the report text, tables or file name.
"""

from __future__ import annotations

import argparse
import copy
import csv
import html
import inspect
import io
import json
import logging
import os
import queue
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import config
import conditions as cond_rules
import storage
from abnormal import load_checkitem_lookup
from llm import metrics_from_findings, partial_tags, warmup
from pipeline import PipelineResult, analyze_pdf
from report_cache import ReportCache, cache_key, config_fingerprint
from retrieval import CheckitemFacts, Retriever
from verify import verify_tags
from web_fallback import WebFallback, build_query

log = logging.getLogger("ui")

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "eval" / "synth" / "samples"
RUNS_DB = str(Path(config.CHROMA_DB_PATH) / "runs.sqlite3")
MAX_PDF_BYTES = 20 * 1024 * 1024

TITLE = "健檢報告標籤器 Health Report Tagger"
DISCLAIMER = "本工具僅供健康資訊參考，不能取代醫師診斷。"

# Curated synthetic samples (fully synthetic, safe to ship).
SAMPLES = [
    ("syn_048 · 腎功能下降＋貧血（含參考值衝突）", "syn_048.pdf"),
    ("syn_012 · 血脂異常（含參考值衝突）", "syn_012.pdf"),
    ("syn_002 · 代謝症候群", "syn_002.pdf"),
    ("syn_007 · 脂肪肝指標", "syn_007.pdf"),
    ("syn_001 · 各項正常", "syn_001.pdf"),
]

CATEGORIES = [
    ("conditions", "健康狀況", "cond"),
    ("risks", "潛在風險", "risk"),
    ("metrics", "關鍵指標", "metric"),
    ("lifestyle", "生活型態", "life"),
    ("food", "飲食建議", "food"),
    ("exercise", "運動建議", "exer"),
    ("supplements", "營養補充", "supp"),
    ("avoid", "應避免", "avoid"),
]
CATEGORY_LABEL = {k: v for k, v, _ in CATEGORIES}

STATUS_ZH = {"high": ("偏高", "↑"), "low": ("偏低", "↓"), "positive": ("陽性", "+"),
             "negative": ("陰性", ""), "normal": ("正常", ""), "unknown": ("無法判定", "?")}
RANGE_SOURCE_ZH = {"report": "報告", "reference": "一般參考範圍", "kb": "一般參考範圍",  # "kb": older alias
                   "report_flag": "報告標記"}
VERDICT = {  # verdict -> (icon, css, 說明)
    "in_report": ("✔", "ok", "報告中可找到依據"),
    "lexical": ("✔", "ok", "引用段落字詞吻合"),
    "llm_supported": ("✔", "ok", "查核模型確認引用段落支持此建議"),
    "not_in_report": ("↓", "warn", "報告中找不到依據，已降為推論"),
    "unverified": ("?", "muted", "未能查核（查核服務無回應或未啟用）"),
    "pending": ("?", "muted", "尚未查核"),
}
SEX_CHOICES = ["自動（從報告判斷）", "男", "女"]
SEX_ZH = {"M": "男", "F": "女"}
STAGES = [("extract", "擷取文字"), ("abnormal", "異常判定"), ("retrieve", "檢索參考資料"),
          ("llm", "LLM 標籤生成"), ("verify", "引用查核")]


# ─── Resources (built once at start-up) ──────────────────────────────────────

@dataclass
class Resources:
    stub: bool
    lookup: object
    facts: CheckitemFacts
    retriever: Retriever
    status: Dict[str, str] = field(default_factory=dict)
    cache: Optional[ReportCache] = None


def build_resources(stub: bool = False) -> Resources:
    t0 = time.time()
    lookup = load_checkitem_lookup(config.REFERENCE_RANGES_PATH)
    facts = CheckitemFacts(config.REFERENCE_RANGES_PATH)
    status = {"模式": "示範模式（不呼叫 LLM／embedding／網路）" if stub else "完整模式",
              "檢驗項目目錄": f"{len(lookup)} 項", "項目說明卡": f"{len(facts.by_name)} 個名稱"}
    if stub:
        # item fact sheets only: no vector index, no embeddings, no BM25.
        retriever = Retriever(None, facts=facts)
        status["檢索"] = "僅項目說明卡（示範模式）"
        log.info("resources ready (stub) in %.1fs", time.time() - t0)
        return Resources(True, lookup, facts, retriever, status)

    from kb_index import build_kb_index, get_kb_index, load_articles, prepare_chunks
    from retrieval import BM25Index, Reranker

    mode = config.RETRIEVAL_MODE
    col = None
    if mode != "bm25":
        col = get_kb_index(embed_model=config.EMBED_MODEL, db_path=config.CHROMA_DB_PATH)
        if col is None:
            log.info("KB index for %s missing -> building (embeds the KB once)", config.EMBED_MODEL)
            col, stats = build_kb_index(embed_model=config.EMBED_MODEL, db_path=config.CHROMA_DB_PATH,
                                        base_url=config.OLLAMA_BASE_URL)
            if col is None:
                log.warning("KB index build failed (%s); falling back to BM25", stats.get("error"))
                mode = "bm25"
    bm25 = None
    if mode in ("bm25", "hybrid"):
        log.info("building BM25 index (jieba tokenisation, ~25 s) ...")
        t = time.time()
        chunks, _ = prepare_chunks(load_articles(config.KB_JSON_PATH))
        bm25 = BM25Index(chunks, user_terms=list(facts.by_name))
        log.info("BM25 ready: %d chunks in %.1fs", len(chunks), time.time() - t)
    reranker = None
    if config.USE_RERANKER:
        try:
            reranker = Reranker(config.RERANKER_MODEL)
            log.info("reranker %s on %s", config.RERANKER_MODEL, reranker.device)
        except Exception as e:  # noqa: BLE001 — optional component
            log.warning("reranker unavailable: %s", e)
    retriever = Retriever(col, embed_model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL,
                          facts=facts, mode=mode, bm25=bm25, reranker=reranker)
    status.update({"檢索": mode, "Embedding 模型": config.EMBED_MODEL,
                   "向量索引": f"{col.count()} 段" if col is not None else "無",
                   "BM25": f"{len(bm25.chunks)} 段" if bm25 else "無",
                   "Reranker": config.RERANKER_MODEL if reranker else "未啟用",
                   "LLM": config.LLM_MODEL,
                   "結果快取": config.REPORT_CACHE_DIR if config.REPORT_CACHE else "停用"})
    # Load the LLM now (same num_ctx as every later call) instead of on the first report (~9 s).
    threading.Thread(target=lambda: log.info("LLM warm-up: %s", warmup() or "ok"), daemon=True).start()
    log.info("resources ready in %.1fs", time.time() - t0)
    return Resources(False, lookup, facts, retriever, status, cache=ReportCache())


# ─── Web fallback wiring ─────────────────────────────────────────────────────

def _stub_search(query: str) -> List[Dict]:
    """Demo-mode stand-in for the web search: no network."""
    return [{"url": "https://example.org/health-education",
             "title": f"（示範）{query}",
             "snippet": "示範模式不會連網。實際模式下，這裡會是以標準化項目名稱搜尋到的衛教網頁摘要，"
                        "並附上來源網址供查證。"}]


class _MemoryWeb(WebFallback):
    """WebFallback with an in-memory cache only (demo mode / tests)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.cache = {}

    def _save(self):
        pass


# ─── Stub tags (demo mode) ───────────────────────────────────────────────────

_CANNED = {  # rule condition -> [(category, text)]
    "血脂異常": [("food", "高纖蔬果"), ("avoid", "油炸食物"), ("exercise", "中等強度有氧運動")],
    "貧血": [("food", "富含鐵質食物"), ("supplements", "鐵劑需經醫師評估")],
    "小球性貧血": [("food", "紅肉與深綠色蔬菜")],
    "慢性腎臟病": [("avoid", "高鹽飲食"), ("lifestyle", "定期追蹤腎功能")],
    "肝功能異常": [("avoid", "飲酒"), ("lifestyle", "控制體重")],
    "糖尿病前期": [("food", "全穀類取代精緻澱粉"), ("avoid", "含糖飲料")],
    "糖尿病": [("food", "全穀類取代精緻澱粉"), ("avoid", "含糖飲料")],
    "血壓偏高": [("avoid", "高鈉加工食品"), ("lifestyle", "居家量測血壓")],
    "高血壓": [("avoid", "高鈉加工食品"), ("lifestyle", "居家量測血壓")],
    "高尿酸血症": [("avoid", "高普林食物"), ("lifestyle", "多喝水")],
    "代謝症候群": [("exercise", "每週150分鐘運動"), ("lifestyle", "減少腰圍")],
    "肥胖": [("exercise", "每週150分鐘運動")],
    "過重": [("lifestyle", "控制體重")],
}
_GENERAL = [("lifestyle", "規律作息"), ("exercise", "規律運動"), ("food", "均衡飲食")]


def stub_tags(r: PipelineResult) -> Dict:
    """Deterministic example tags so every panel renders without an LLM.

    conditions/risks come from the real rule engine and metrics from the real
    findings; advice tags are canned but cite real fact-sheet chunks and are
    passed through the real verifier (without the LLM judge).
    """
    rules = cond_rules.derive(r.findings, r.sex)
    tags: Dict[str, List[Dict]] = {k: [] for k, _, _ in CATEGORIES}
    tags["conditions"], tags["risks"] = rules["conditions"], rules["risks"]
    tags["metrics"] = metrics_from_findings(r.findings)
    facts = [c for c in r.chunks if c.get("source") == "checkitem"]
    abn = r.abnormals
    picks = []
    for c in rules["conditions"]:
        picks += _CANNED.get(c["text"], [])
    if not picks:
        picks = list(_GENERAL)
    seen = set()
    for i, (cat, text) in enumerate(picks):
        if (cat, text) in seen:
            continue
        seen.add((cat, text))
        src = f"kb:{facts[i % len(facts)]['id']}" if facts and i % 2 == 0 else "inferred"
        tags[cat].append({"text": text, "src": src, "conf": 0.8 if src != "inferred" else 0.6})
    if facts:
        # a tag whose words do appear in the cited fact sheet -> "lexical" verdict
        tags["lifestyle"].append({"text": "諮詢醫療專業人員", "src": f"kb:{facts[0]['id']}", "conf": 0.9})
    if abn:
        name = abn[0].get("display_name") or abn[0]["name"]
        tags["lifestyle"].append({"text": f"追蹤{name}", "src": "report", "conf": 0.9})
        tags["risks"].append({"text": "需與醫師討論追蹤頻率", "src": "report", "conf": 0.5})
    names = "、".join((f.get("display_name") or f["name"]) for f in abn[:5])
    tags["summary"] = (
        f"報告共擷取 {len(r.findings)} 項檢驗值，其中 {len(abn)} 項超出參考範圍"
        + (f"（{names}{'等' if len(abn) > 5 else ''}）" if abn else "")
        + "。【示範模式】此摘要與建議標籤為範例內容，並非 LLM 生成。")
    t = time.time()
    out = verify_tags(tags, r.chunks, r.text, r.findings, judge=None)
    r.timing["llm_s"] = 0.0
    r.timing["verify_s"] = round(time.time() - t, 3)
    return out


# ─── Analysis (UI-agnostic; used by the Gradio handler and the tests) ────────

_DATE_RX = re.compile(r"(?:檢查|受檢|採檢|體檢|報告|檢驗)日期[\s:：]*"
                      r"(\d{4}|\d{2,3})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})")


def report_date_from_text(text: str) -> Optional[str]:
    """Exam date printed on the report (never the date of birth), ISO format."""
    m = _DATE_RX.search(text or "")
    if not m:
        return None
    y, mo, d = (int(x) for x in m.groups())
    if y < 1000:  # ROC (民國) year
        y += 1911
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def _parse_sex(choice: Optional[str]) -> Optional[str]:
    return {"男": "M", "女": "F"}.get((choice or "").strip())


@dataclass
class Analysis:
    result: PipelineResult
    tags: Dict
    fact_for: Dict[str, str]           # fact-sheet chunk id -> finding display name
    web_audit: Dict                    # {"enabled", "sent", "cached", "blocked"}
    report_date: Optional[str]
    stub: bool
    trend_msg: str = ""
    patient_id: str = ""
    cached: bool = False               # served from the report cache
    preliminary: bool = False          # partial view while the LLM is still running


def preliminary_tags(r: PipelineResult, llm_raw: str = "") -> Dict:
    """Tags shown before the LLM finishes: rule conditions/risks + metrics
    (final already), plus whatever advice the streamed LLM output contains so
    far (marked unverified until verification runs)."""
    rules = r.rules or cond_rules.derive(r.findings, r.sex)
    tags: Dict = {"conditions": list(rules.get("conditions", [])), "risks": list(rules.get("risks", [])),
                  "metrics": metrics_from_findings(r.findings)}
    part = partial_tags(llm_raw) if llm_raw else {}
    if part.get("conditions"):
        tags["conditions"] = cond_rules.merge_tags(tags["conditions"], part["conditions"])
    for k in ("lifestyle", "food", "exercise", "supplements", "avoid"):
        tags[k] = [dict(t, verdict="pending") for t in part.get(k, [])]
    tags["summary"] = part.get("summary") or ""
    return tags


def _fingerprint(res: Resources, sex: Optional[str]) -> str:
    rt = res.retriever
    return config_fingerprint(sex=sex, stub=res.stub, mode=getattr(rt, "mode", None),
                              reranker=bool(getattr(rt, "reranker", None)),
                              kb=getattr(rt, "col", None) is not None)


def run_analysis(pdf_bytes: bytes, res: Resources, sex_choice: str = "", alias: str = "",
                 report_date: str = "", allow_web: bool = False,
                 on_stage: Optional[Callable[[str], None]] = None,
                 runs_db: str = RUNS_DB,
                 on_partial: Optional[Callable[["Analysis"], None]] = None,
                 use_cache: bool = True) -> Analysis:
    """on_stage(stage): stage names from STAGES as each one starts.
    on_partial(analysis): preliminary Analysis (preliminary=True) — first when
    findings + rule tags are ready, then as streamed LLM advice arrives.
    use_cache: look up / store the result in res.cache (never with web search)."""
    on_stage = on_stage or (lambda s: None)
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError(f"檔案過大（上限 {MAX_PDF_BYTES // 1024 // 1024} MB）")
    sex = _parse_sex(sex_choice)
    t_all = time.time()
    cache = res.cache if (use_cache and not allow_web) else None
    key = cache_key(pdf_bytes, _fingerprint(res, sex)) if cache else ""
    hit = cache.get(key) if cache else None
    if hit:
        r, meta = hit
        r.timing = {**r.timing, "total_s": round(time.time() - t_all, 3), "first_result_s": 0.0,
                    "original_total_s": r.timing.get("total_s", 0.0)}
        a = _finish(r, r.tags, res, allow_web, None, alias, report_date, meta.get("report_date_auto"), runs_db)
        a.cached = True
        return a

    retriever = copy.copy(res.retriever)
    retriever.errors = []
    web = None
    if allow_web:  # the pipeline adds one web passage per abnormal finding
        web = (_MemoryWeb(search_fn=_stub_search) if res.stub else WebFallback())

    r = _analyze_with_stages(pdf_bytes, res, retriever, sex, on_stage, web, on_partial, t_all)
    tags = r.tags
    if res.stub and not r.error:
        on_stage("verify")
        tags = r.tags = stub_tags(r)
    r.timing["total_s"] = round(time.time() - t_all, 3)
    auto_date = report_date_from_text(r.text)
    if cache:
        cache.put(key, r, {"report_date_auto": auto_date})
    return _finish(r, tags, res, allow_web, web, alias, report_date, auto_date, runs_db)


def _finish(r: PipelineResult, tags: Dict, res: Resources, allow_web: bool, web, alias: str,
            report_date: str, auto_date: Optional[str], runs_db: str) -> Analysis:

    fact_for = {}
    for f in r.abnormals:
        fact = res.facts.lookup(f)
        if fact:
            fact_for.setdefault(fact["id"], f.get("display_name") or f.get("name"))

    audit = {"enabled": bool(allow_web), "sent": [], "cached": [], "blocked": 0}
    if web is not None:
        planned = [build_query(f) for f in r.abnormals[: web.max_findings]]
        audit["sent"] = list(web.sent_queries)
        audit["cached"] = [q for q in planned if q and q not in web.sent_queries]
        audit["blocked"] = sum(1 for q in planned if not q)
        audit["demo"] = res.stub

    rdate = (report_date or "").strip() or auto_date or None
    a = Analysis(r, tags or {}, fact_for, audit, rdate, res.stub)
    if (alias or "").strip() and r.findings:
        a.patient_id = storage.hash_patient_label(alias, db_path=runs_db)
        storage.record_trend_point(a.patient_id, rdate or date.today().isoformat(), r.findings,
                                   db_path=runs_db)
        a.trend_msg = "已儲存本次檢驗值到此代號的趨勢紀錄（僅代號雜湊、報告日期與標準化檢驗值）。"
    return a


PARTIAL_EVERY_S = 0.4  # throttle for streamed-advice UI updates


def _analyze_with_stages(pdf_bytes, res, retriever, sex, on_stage, web=None, on_partial=None,
                         t0: Optional[float] = None) -> PipelineResult:
    """Call pipeline.analyze_pdf, relaying its stage hooks to the UI; emits a
    preliminary Analysis when findings + rules are ready and while the LLM streams."""
    t0 = t0 or time.time()
    stage_map = {"extract": "extract", "findings": "abnormal", "retrieve": "retrieve",
                 "llm": "llm", "verify": "verify"}
    box: Dict = {"r": None, "last": 0.0}

    def partial(r: PipelineResult, raw: str = "") -> None:
        if on_partial is None:
            return
        a = Analysis(r, preliminary_tags(r, raw), {}, {"enabled": False, "sent": [], "cached": [],
                                                       "blocked": 0}, None, res.stub, preliminary=True)
        on_partial(a)

    def hook(stage: str, r: PipelineResult) -> None:
        box["r"] = r
        if stage in stage_map:
            on_stage(stage_map[stage])
        if stage == "findings":
            r.timing["first_result_s"] = round(time.time() - t0, 3)
            partial(r)

    def on_text(raw: str) -> None:
        now = time.time()
        if box["r"] is not None and now - box["last"] >= PARTIAL_EVERY_S:
            box["last"] = now
            partial(box["r"], raw)

    return analyze_pdf(pdf_bytes, res.lookup, retriever=retriever, run_llm=not res.stub,
                       ocr=not res.stub, sex=sex, model=config.LLM_MODEL,
                       base_url=config.OLLAMA_BASE_URL, web=web, on_stage=hook,
                       on_llm_text=on_text if on_partial else None)


# ─── Rendering (every dynamic string is html.escape'd) ───────────────────────

def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _fmt_num(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return "" if v is None else str(v)


def _src_badge(src: str) -> Tuple[str, str]:
    if src == "report":
        return "報告", "b-report"
    if src == "rule":
        return "規則", "b-rule"
    if src.startswith("kb:web_"):
        return "網路", "b-web"
    if src.startswith("kb:"):
        return "知識庫", "b-kb"
    return "推論", "b-inferred"


def render_notices(a: Analysis) -> str:
    r = a.result
    parts = []
    if a.stub:
        parts.append('<div class="note note-info">🧪 <b>示範模式</b>：擷取、異常判定、規則標籤與項目說明卡為真實運算；'
                     '摘要與建議標籤為範例內容（未呼叫 LLM）。</div>')
    if r.error:
        parts.append(f'<div class="note note-err">⚠ 分析未完成（{esc(r.error_stage)}）：{esc(r.error)}</div>')
    if r.sex is None and any(f.get("sex_unknown") for f in r.findings):
        n = sum(1 for f in r.findings if f.get("sex_unknown"))
        parts.append(f'<div class="note note-warn">⚧ 報告中找不到性別，而有 {n} 個項目的參考值因性別而異。'
                     '請在左側「性別」選擇後重新分析，以取得正確判定。</div>')
    if r.used_ocr:
        parts.append('<div class="note note-info">🔎 此 PDF 沒有文字層，已使用 OCR 辨識。</div>')
    if a.trend_msg:
        parts.append(f'<div class="note note-ok">📈 {esc(a.trend_msg)}</div>')
    return "".join(parts)


def render_overview(a: Analysis) -> str:
    r = a.result
    abn = r.abnormals
    conflicts = sum(1 for f in r.findings if f.get("range_conflict"))
    sex = SEX_ZH.get(r.sex, "未知")
    src = {"param": "使用者指定", "report_header": "報告表頭", "range_inference": "由參考值推斷"}.get(
        r.abnormal_debug.get("sex_source"), "—")
    cards = [("檢驗項目", len(r.findings)), ("異常", len(abn)), ("參考值衝突", conflicts),
             ("參考資料", len(r.chunks)), ("性別", f"{sex}<small>（{esc(src)}）</small>"),
             ("報告日期", esc(a.report_date or "—"))]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="kpi-v{" kpi-sm" if k == "報告日期" else ""}">{v}</div>'
        f'<div class="kpi-l">{esc(k)}</div></div>'
        for k, v in cards) + "</div>"


def render_tags(a: Analysis) -> str:
    tags = a.tags or {}
    by_id = {c["id"]: c for c in a.result.chunks}
    out = []
    summary = tags.get("summary")
    if a.preliminary:
        out.append('<div class="note note-info">⏳ 檢驗值、規則判定的狀況／風險與關鍵指標已完成；'
                   'LLM 建議標籤生成中（標示 ? 者為尚未查核的暫定標籤）。</div>')
    if summary:
        out.append(f'<div class="summary"><div class="summary-h">摘要</div>{esc(summary)}</div>')
    elif a.preliminary:
        out.append('<div class="summary muted">（摘要生成中…）</div>')
    else:
        out.append('<div class="summary muted">（沒有摘要）</div>')
    out.append('<div class="legend">來源：<span class="badge b-report">報告</span>'
               '<span class="badge b-rule">規則</span><span class="badge b-kb">知識庫</span>'
               '<span class="badge b-web">網路</span><span class="badge b-inferred">推論</span>'
               '　查核：<span class="v ok">✔</span>有依據 <span class="v warn">↓</span>已降級 '
               '<span class="v muted">?</span>未查核　<span class="muted">點擊標籤可展開依據</span></div>')
    for key, label, css in CATEGORIES:
        items = tags.get(key) or []
        out.append(f'<div class="cat"><div class="cat-h">{esc(label)} <span class="count">{len(items)}</span></div>')
        if not items:
            out.append('<div class="muted small">—</div></div>')
            continue
        out.append('<div class="chips">')
        for t in items:
            out.append(_chip(t, css, by_id))
        out.append("</div></div>")
    return "".join(out)


def _chip(t: Dict, css: str, by_id: Dict[str, Dict]) -> str:
    src = str(t.get("src") or "inferred")
    badge, bcss = _src_badge(src)
    verdict = t.get("verdict")
    conf = t.get("conf")
    conf_s = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "—"
    if src == "rule":
        vicon, vcss, vtip = "⚙", "ok", "由規則判定（附觸發依據）"
    elif css == "metric":
        vicon, vcss, vtip = "✔", "ok", "由檢驗值直接產生（非 LLM）"
    elif src == "inferred" and not verdict:
        vicon, vcss, vtip = "~", "muted", "模型推論，未引用資料"
    elif verdict in VERDICT:
        vicon, vcss, vtip = VERDICT[verdict]
    else:
        vicon, vcss, vtip = "·", "muted", "無查核紀錄"
    detail = []
    if src == "rule" and t.get("evidence"):
        detail.append(f'<div class="ev-h">觸發依據</div><div class="ev">{esc(t["evidence"])}</div>')
    elif src.startswith("kb:"):
        cid = src[3:]
        c = by_id.get(cid)
        if c:
            link = ""
            if c.get("url"):
                link = f'<div class="ev-url">來源：{esc(c["url"])}</div>'
            detail.append(f'<div class="ev-h">引用段落 [{esc(cid)}]</div>'
                          f'<div class="ev passage">{esc(c["text"][:700])}</div>{link}')
        else:
            detail.append(f'<div class="ev muted">引用 {esc(cid)} 不在本次檢索結果中</div>')
    elif css == "metric":
        detail.append('<div class="ev">由程式依檢驗值與參考值直接產生，無 LLM 參與。</div>')
    elif src == "report":
        detail.append('<div class="ev">標籤內容可在報告或檢驗值中找到。</div>')
    else:
        detail.append('<div class="ev">由模型推論，未直接引用資料（信任度較低）。</div>')
    detail.append(f'<div class="ev-meta">來源：{esc(src)}　查核：{esc(vtip)}　信心：{conf_s}</div>')
    low = " low" if isinstance(conf, (int, float)) and conf < 0.55 else ""
    return (f'<details class="chip c-{css}{low}"><summary>'
            f'<span class="v {vcss}" title="{esc(vtip)}">{vicon}</span>'
            f'<span class="chip-t">{esc(t.get("text", ""))}</span>'
            f'<span class="badge {bcss}">{badge}</span></summary>'
            f'<div class="chip-body">{"".join(detail)}</div></details>')


def _status_cell(f: Dict) -> str:
    st = f.get("status") or f.get("direction") or "unknown"
    label, arrow = STATUS_ZH.get(st, (st, ""))
    return f'<span class="st st-{esc(st)}">{esc(label)} {esc(arrow)}</span>'


def render_findings(a: Analysis, only_abnormal: bool = False) -> str:
    r = a.result
    if not r.findings:
        return '<div class="muted">沒有擷取到檢驗值。</div>'
    out = []
    conflicts = [f for f in r.findings if f.get("range_conflict")]
    if conflicts:
        out.append('<div class="conflict-box"><div class="conflict-h">⚠ 參考值衝突 '
                   f'<span class="count">{len(conflicts)}</span></div>'
                   '<div class="conflict-lead">報告印製的參考值與一般參考範圍（公開資料彙整）對同一數值給出不同判定。'
                   '各檢驗單位的參考值可能因儀器、方法或族群而不同；本工具<b>以報告參考值為準</b>，'
                   '並列出一般參考範圍的判定供您與醫師討論。</div><ul>')
        for f in conflicts:
            rep = STATUS_ZH.get(f.get("report_status") or "", (f.get("report_status") or "—", ""))[0]
            kb = STATUS_ZH.get(f.get("kb_status") or "", (f.get("kb_status") or "—", ""))[0]
            out.append(f'<li><b>{esc(f.get("display_name") or f.get("name"))}</b> = '
                       f'{esc(_fmt_num(f.get("value")))} {esc(f.get("unit"))}：'
                       f'報告參考值 <code>{esc(f.get("report_range") or "—")}</code> → <b>{esc(rep)}</b>；'
                       f'一般參考範圍 <code>{esc(f.get("kb_range") or "—")}</code> → <b>{esc(kb)}</b></li>')
        out.append("</ul></div>")
    rows = [f for f in r.findings if not only_abnormal or f.get("direction") in ("high", "low")
            or f.get("status") == "positive"]
    out.append('<div class="tbl-wrap"><table class="ftable"><thead><tr><th>項目</th><th>數值</th>'
               '<th>判定</th><th>報告參考值</th><th>一般參考範圍</th><th>判定依據</th></tr></thead><tbody>')
    for f in rows:
        st = f.get("status") or f.get("direction") or "unknown"
        conflict = f.get("range_conflict")
        name = f.get("display_name") or f.get("name")
        raw = f.get("name") or ""
        sub = f'<div class="raw">{esc(raw)}</div>' if raw and raw != name else ""
        flag = (f'<span class="cflag" title="{esc(f.get("range_conflict_note"))}">⚠</span>'
                if conflict else "")
        sexn = ' <span class="sexn" title="參考值因性別而異，但性別未知">⚧?</span>' if f.get("sex_unknown") else ""
        val = esc(_fmt_num(f.get("value")))
        if f.get("censored") and isinstance(f.get("value"), float):
            val = esc(f["censored"]) + val
        out.append(
            f'<tr class="row-{esc(st)}{" row-conflict" if conflict else ""}">'
            f'<td><b>{esc(name)}</b>{sub}</td>'
            f'<td class="num">{val} <span class="unit">{esc(f.get("unit"))}</span></td>'
            f'<td>{_status_cell(f)}{flag}{sexn}</td>'
            f'<td>{esc(f.get("report_range") or "—")}</td>'
            f'<td>{esc(f.get("kb_range") or "—")}</td>'
            f'<td>{esc(RANGE_SOURCE_ZH.get(f.get("range_source"), "—"))}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def render_references(a: Analysis) -> str:
    chunks = a.result.chunks
    if not chunks:
        return '<div class="muted">本次沒有檢索到參考資料（例如所有檢驗值皆正常時）。</div>'
    groups: Dict[str, List[Dict]] = {}
    for c in chunks:
        key = c.get("for_finding") or a.fact_for.get(c["id"]) or "一般保健"
        groups.setdefault(key, []).append(c)
    src_zh = {"checkitem": ("項目說明卡", "b-rule"), "kb": ("知識庫文章", "b-kb"), "web": ("網路", "b-web")}
    cited = set(a.result.cited_ids or [])
    out = []
    for name, cs in groups.items():
        out.append(f'<div class="refgrp"><div class="cat-h">{esc(name)} <span class="count">{len(cs)}</span></div>')
        for c in cs:
            label, css = src_zh.get(c.get("source"), (c.get("source") or "?", "b-inferred"))
            url = (f'<div class="ev-url">🔗 {esc(c["url"])}</div>' if c.get("url") else "")
            used = '<span class="badge b-report">已送入 LLM</span>' if c["id"] in cited else ""
            score = c.get("score")
            score_s = f"score {score:.3f}" if isinstance(score, (int, float)) and score else ""
            out.append(f'<details class="ref"><summary><span class="badge {css}">{esc(label)}</span>'
                       f'<code>{esc(c["id"])}</code> {used}<span class="muted small">{esc(score_s)}</span>'
                       f'<div class="ref-prev">{esc(c["text"][:90])}…</div></summary>'
                       f'<div class="ev passage">{esc(c["text"])}</div>{url}</details>')
        out.append("</div>")
    return "".join(out)


def render_perf(a: Analysis, res: Resources) -> str:
    r = a.result
    tm = r.timing
    vals = [(zh, float(tm.get(f"{k}_s", 0.0) or 0.0), f"{k}_s" in tm) for k, zh in STAGES]
    total = max(sum(v for _, v, _ in vals), 1e-6)
    out = ['<div class="perf"><div class="cat-h">各階段耗時</div>']
    for (zh, v, ran), (k, _) in zip(vals, STAGES):
        pct = 100 * v / total
        note = "" if ran else "（未執行）"
        if a.stub and k in ("llm",):
            note = "（示範模式略過）"
        out.append(f'<div class="bar-row"><div class="bar-l">{esc(zh)}</div>'
                   f'<div class="bar"><div class="bar-f bar-{k}" style="width:{pct:.1f}%"></div></div>'
                   f'<div class="bar-v">{v:.2f} s <span class="muted small">{note}</span></div></div>')
    first = tm.get("first_result_s")
    extra = f"　首批結果（檢驗值＋規則標籤）{first:.2f} s" if isinstance(first, (int, float)) else ""
    if a.cached:
        extra = f"　（結果快取命中；原始分析 {tm.get('original_total_s', 0):.2f} s）"
    out.append(f'<div class="muted small">總計 {tm.get("total_s", total):.2f} s{extra}</div></div>')

    ver = (a.tags or {}).get("_verification") or {}
    rate = ver.get("citation_support_rate")
    rate_s = f"{rate * 100:.0f}%" if isinstance(rate, (int, float)) else "—"
    items = [("引用支持率", rate_s), ("知識庫引用數", ver.get("kb_checked", 0)),
             ("字詞吻合", ver.get("kb_lexical_ok", 0)), ("LLM 查核通過", ver.get("kb_llm_ok", 0)),
             ("未能查核", ver.get("kb_unverified", 0)), ("報告標籤降級", ver.get("report_downgraded", 0)),
             ("不支持而刪除", ver.get("kb_dropped", 0)), ("推論超額刪除", ver.get("inferred_capped", 0)),
             ("使用 OCR", "是" if r.used_ocr else "否")]
    out.append('<div class="cat-h">引用查核（grounding）</div><div class="kpis">' + "".join(
        f'<div class="kpi"><div class="kpi-v">{esc(v)}</div><div class="kpi-l">{esc(k)}</div></div>'
        for k, v in items) + "</div>")
    dbg = r.rag_debug or {}
    errs = [e for e in (dbg.get("errors") or []) if e]
    out.append('<div class="cat-h">檢索設定</div><table class="kv">' + "".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in res.status.items()) +
        f"<tr><td>說明卡命中</td><td>{esc(dbg.get('fact_hits', 0))}</td></tr>"
        f"<tr><td>知識庫命中</td><td>{esc(dbg.get('kb_hits', 0))}</td></tr>"
        f"<tr><td>網路命中</td><td>{esc(dbg.get('web_hits', 0))}</td></tr>"
        + (f"<tr><td>檢索錯誤</td><td>{esc('；'.join(map(str, errs))[:300])}</td></tr>" if errs else "")
        + "</table>")
    return "".join(out)


def render_audit(a: Analysis) -> str:
    w = a.web_audit
    if not w.get("enabled"):
        return '<div class="muted">網路補充搜尋未開啟：本次分析沒有任何資料離開本機。</div>'
    demo = ('<div class="note note-info">示範模式：以下查詢<b>沒有</b>真正送出，僅展示實際模式會送出的內容。</div>'
            if w.get("demo") else "")
    sent = "".join(f"<li><code>{esc(q)}</code></li>" for q in w["sent"]) or "<li class='muted'>（無）</li>"
    cached = "".join(f"<li><code>{esc(q)}</code></li>" for q in w["cached"])
    return (demo + '<div>只會送出「標準化項目名稱＋偏高/偏低＋衛教」，不含數值、姓名或報告文字。</div>'
            f'<div class="cat-h">本次送出的查詢</div><ul>{sent}</ul>'
            + (f'<div class="cat-h">使用本機快取（未送出）</div><ul>{cached}</ul>' if cached else "")
            + (f'<div class="muted small">另有 {w["blocked"]} 個項目因不在標準名稱表中而未送出。</div>'
               if w.get("blocked") else ""))


# ─── Export ──────────────────────────────────────────────────────────────────

_FORMULA = ("=", "+", "-", "@", "\t", "\r")


def _csv_cell(v) -> object:
    """Neutralise spreadsheet formula injection in text cells."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    s = "" if v is None else str(v)
    return "'" + s if s.startswith(_FORMULA) else s


def export_dict(a: Analysis) -> Dict:
    r = a.result
    tags = a.tags or {}
    keep = ("name", "display_name", "canonical_key", "value", "unit", "status", "direction",
            "report_range", "kb_range", "range_source", "range_conflict", "range_conflict_note",
            "report_status", "kb_status", "sex_unknown")
    return {
        "tool": TITLE, "mode": "demo-stub" if a.stub else "full",
        "report_date": a.report_date, "sex": r.sex,
        "summary": tags.get("summary", ""),
        "tags": {k: tags.get(k, []) for k, _, _ in CATEGORIES},
        "verification": tags.get("_verification", {}),
        "findings": [{k: f.get(k) for k in keep} for f in r.findings],
        "references": [{"id": c["id"], "source": c.get("source"), "for_finding":
                        c.get("for_finding") or a.fact_for.get(c["id"]), "url": c.get("url")}
                       for c in r.chunks],
        "timing_s": r.timing, "used_ocr": r.used_ocr, "error": r.error,
        "disclaimer": DISCLAIMER,
    }


def export_csv(a: Analysis) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow(["type", "category", "item", "value", "unit", "status", "source", "verdict",
                "confidence", "detail"])
    for key, label, _ in CATEGORIES:
        for t in (a.tags or {}).get(key, []) or []:
            w.writerow([_csv_cell(x) for x in (
                "tag", label, t.get("text", ""), "", "", "", t.get("src", ""), t.get("verdict", ""),
                t.get("conf", ""), t.get("evidence", ""))])
    for f in a.result.findings:
        detail = f.get("range_conflict_note") if f.get("range_conflict") else (f.get("report_range") or "")
        w.writerow([_csv_cell(x) for x in (
            "finding", "檢驗值", f.get("display_name") or f.get("name"), f.get("value"), f.get("unit", ""),
            f.get("status") or f.get("direction"), f.get("range_source") or "", "", "", detail)])
    return buf.getvalue()


def write_exports(a: Analysis) -> Tuple[str, str]:
    d = Path(tempfile.mkdtemp(prefix="hrt_export_"))
    jp, cp = d / "health_report_tags.json", d / "health_report_tags.csv"
    jp.write_text(json.dumps(export_dict(a), ensure_ascii=False, indent=2), encoding="utf-8")
    cp.write_text(export_csv(a), encoding="utf-8-sig")  # BOM so Excel reads UTF-8
    return str(jp), str(cp)


# ─── Trends ──────────────────────────────────────────────────────────────────

def trend_choices(patient_id: str, runs_db: str = RUNS_DB) -> List[Tuple[str, str]]:
    if not patient_id:
        return []
    inds = storage.list_indicators_for_patient(patient_id, db_path=runs_db)
    return [(f'{i["display"]}（{i["occurrences"]} 次）', i["key"]) for i in inds]


def trend_frame(patient_id: str, key: str, runs_db: str = RUNS_DB):
    import pandas as pd
    series = storage.time_series_for_indicator(patient_id, key, db_path=runs_db) if patient_id and key else []
    rows = [{"日期": s["date"], "數值": float(s["value"]), "項目": s["raw_name"], "判定":
             STATUS_ZH.get(s["direction"], (s["direction"], ""))[0], "單位": s["unit"]} for s in series]
    df = pd.DataFrame(rows, columns=["日期", "數值", "項目", "判定", "單位"])
    if not df.empty:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.dropna(subset=["日期"])
    return df


def trend_outputs(patient_id: str, key: Optional[str], runs_db: str = RUNS_DB):
    """(plot frame with datetimes, table frame with ISO date strings)."""
    df = trend_frame(patient_id, key or "", runs_db)
    table = df.copy()
    if not table.empty:
        table["日期"] = table["日期"].dt.strftime("%Y-%m-%d")
    return df, table


# ─── Gradio app ──────────────────────────────────────────────────────────────

CSS = """
:root { --ok:#1f8a5b; --hi:#c0392b; --lo:#2463b0; --warn:#b7791f; }
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
#hdr h1 { font-size: 1.6rem; margin: 0; }
#hdr .sub { opacity: .75; font-size: .92rem; margin-top: 2px; }
.muted { opacity: .65; } .small { font-size: .8rem; }
.note { padding: 8px 12px; border-radius: 8px; margin: 6px 0; font-size: .9rem; border: 1px solid; }
.note-info { background: rgba(36,99,176,.08); border-color: rgba(36,99,176,.35); }
.note-warn { background: rgba(183,121,31,.12); border-color: rgba(183,121,31,.5); }
.note-err  { background: rgba(192,57,43,.10); border-color: rgba(192,57,43,.5); }
.note-ok   { background: rgba(31,138,91,.10); border-color: rgba(31,138,91,.45); }
.kpis { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; margin: 6px 0 12px; }
.kpi { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 8px 10px;
       background: var(--background-fill-secondary); }
.kpi-v { font-size: 1.25rem; font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi-v.kpi-sm { font-size: 1rem; line-height: 1.9rem; } .kpi-v small { font-size: .7rem; font-weight: 400; opacity: .7; }
.kpi-l { font-size: .78rem; opacity: .7; }
.summary { border-left: 4px solid var(--color-accent); background: var(--background-fill-secondary);
           padding: 12px 16px; border-radius: 6px; line-height: 1.7; margin-bottom: 10px; }
.summary-h { font-weight: 650; font-size: .8rem; opacity: .7; margin-bottom: 2px; }
.legend { font-size: .8rem; margin: 4px 0 10px; opacity: .9; }
.cat { margin: 10px 0 12px; }
.cat-h { font-weight: 650; margin: 12px 0 6px; font-size: .95rem; }
.count { font-size: .72rem; padding: 1px 7px; border-radius: 999px; background: var(--background-fill-secondary);
         border: 1px solid var(--border-color-primary); font-weight: 500; margin-left: 4px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: flex-start; }
details.chip { border-radius: 14px; border: 1px solid; font-size: .88rem; max-width: 100%; }
details.chip > summary { list-style: none; cursor: pointer; padding: 4px 10px; display: flex; gap: 6px; align-items: center; }
details.chip > summary::-webkit-details-marker { display: none; }
details.chip[open] { flex-basis: 100%; }
details.chip.low { border-style: dashed; opacity: .8; }
.chip-body { padding: 4px 12px 10px; font-size: .82rem; border-top: 1px dashed currentColor; }
.c-cond  { background: rgba(192,57,43,.09);  border-color: rgba(192,57,43,.45); }
.c-risk  { background: rgba(214,137,16,.10); border-color: rgba(214,137,16,.5); }
.c-metric{ background: rgba(155,89,182,.09); border-color: rgba(155,89,182,.45); }
.c-life  { background: rgba(36,99,176,.08);  border-color: rgba(36,99,176,.4); }
.c-food  { background: rgba(39,174,96,.09);  border-color: rgba(39,174,96,.45); }
.c-exer  { background: rgba(94,84,196,.09);  border-color: rgba(94,84,196,.45); }
.c-supp  { background: rgba(22,160,133,.09); border-color: rgba(22,160,133,.45); }
.c-avoid { background: rgba(211,84,0,.09);   border-color: rgba(211,84,0,.45); }
.badge { font-size: .68rem; padding: 1px 6px; border-radius: 6px; border: 1px solid; white-space: nowrap; margin-right: 4px; }
.b-report { border-color: rgba(31,138,91,.6); color: var(--ok); }
.b-rule   { border-color: rgba(94,84,196,.6); color: #6c63d9; }
.b-kb     { border-color: rgba(36,99,176,.6); color: #3a7bd5; }
.b-web    { border-color: rgba(211,84,0,.6);  color: #d35400; }
.b-inferred { border-color: rgba(127,127,127,.6); opacity: .8; }
.v { font-weight: 700; width: 1em; text-align: center; }
.v.ok { color: var(--ok); } .v.warn { color: var(--warn); } .v.muted { opacity: .55; }
.ev-h { font-weight: 650; margin-top: 6px; font-size: .78rem; opacity: .75; }
.ev { margin-top: 2px; line-height: 1.6; }
.passage { white-space: pre-wrap; background: var(--background-fill-secondary); padding: 8px 10px;
           border-radius: 6px; border: 1px solid var(--border-color-primary); max-height: 260px; overflow: auto; }
.ev-url { font-size: .75rem; word-break: break-all; margin-top: 4px; opacity: .8; }
.ev-meta { font-size: .72rem; opacity: .6; margin-top: 6px; }
.conflict-box { border: 1.5px solid rgba(183,121,31,.7); background: rgba(183,121,31,.08);
                border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; }
.conflict-h { font-weight: 700; color: var(--warn); font-size: 1rem; }
.conflict-lead { font-size: .85rem; margin: 4px 0; opacity: .9; }
.conflict-box ul { margin: 6px 0 0 18px; font-size: .88rem; } .conflict-box li { margin: 3px 0; }
.conflict-box code, .ftable code { font-size: .8rem; }
.tbl-wrap { overflow-x: auto; }
table.ftable { width: 100%; border-collapse: collapse; font-size: .88rem; }
.ftable th { text-align: left; font-weight: 650; padding: 6px 8px; border-bottom: 2px solid var(--border-color-primary);
             position: sticky; top: 0; background: var(--background-fill-primary); }
.ftable td { padding: 5px 8px; border-bottom: 1px solid var(--border-color-primary); vertical-align: top; }
.ftable .num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.ftable .unit, .ftable .raw { opacity: .6; font-size: .75rem; }
.row-high td:first-child, .row-positive td:first-child { box-shadow: inset 3px 0 0 var(--hi); }
.row-low td:first-child { box-shadow: inset 3px 0 0 var(--lo); }
.row-conflict { background: rgba(183,121,31,.08); }
.st { font-weight: 600; white-space: nowrap; }
.st-high, .st-positive { color: var(--hi); } .st-low { color: var(--lo); }
.st-normal, .st-negative { color: var(--ok); font-weight: 500; } .st-unknown { opacity: .6; font-weight: 500; }
.cflag { color: var(--warn); font-weight: 700; margin-left: 6px; cursor: help; }
.sexn { font-size: .75rem; color: var(--warn); cursor: help; }
.refgrp { margin-bottom: 8px; }
details.ref { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 6px 10px; margin: 4px 0; }
details.ref > summary { cursor: pointer; }
.ref-prev { font-size: .8rem; opacity: .7; margin-top: 2px; }
.bar-row { display: grid; grid-template-columns: 110px 1fr 150px; gap: 8px; align-items: center; margin: 5px 0; }
.bar { height: 14px; border-radius: 7px; background: var(--background-fill-secondary);
       border: 1px solid var(--border-color-primary); overflow: hidden; }
.bar-f { height: 100%; border-radius: 7px; min-width: 2px; }
.bar-extract { background: #3a7bd5; } .bar-abnormal { background: #16a085; } .bar-retrieve { background: #8e44ad; }
.bar-llm { background: #d35400; } .bar-verify { background: #27ae60; }
.bar-v { font-variant-numeric: tabular-nums; font-size: .85rem; }
table.kv { font-size: .85rem; border-collapse: collapse; border: none !important; }
.kv td { padding: 3px 12px 3px 0; border: none !important; background: transparent !important; }
.kv td:first-child { opacity: .7; }
#footer { text-align: center; font-size: .82rem; opacity: .75; padding: 12px 0 4px;
          border-top: 1px solid var(--border-color-primary); margin-top: 12px; }
"""


def _theme(gr):
    fonts = ["Noto Sans TC", "Microsoft JhengHei", "PingFang TC", "system-ui", "sans-serif"]
    return gr.themes.Soft(primary_hue="teal", secondary_hue="sky", neutral_hue="slate",
                          font=fonts, font_mono=["Consolas", "ui-monospace", "monospace"])


def build_app(res: Resources, runs_db: str = RUNS_DB):
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no telemetry calls
    import gradio as gr

    sample_map = {label: str(SAMPLES_DIR / fn) for label, fn in SAMPLES if (SAMPLES_DIR / fn).exists()}
    blocks_kw = {"title": "健檢報告標籤器", "delete_cache": (3600, 3600)}
    launch_kw = {}
    if "theme" in inspect.signature(gr.Blocks.__init__).parameters and int(gr.__version__.split(".")[0]) < 6:
        blocks_kw.update(theme=_theme(gr), css=CSS)
    else:  # Gradio 6 moved theme/css to launch()
        launch_kw.update(theme=_theme(gr), css=CSS)

    with gr.Blocks(**blocks_kw) as demo:
        state = gr.State(None)
        gr.HTML(f'<div id="hdr"><h1>🩺 {esc(TITLE)}</h1><div class="sub">上傳健檢報告 PDF，自動擷取檢驗值、'
                f'判定異常，並產生有引用依據的健康標籤。全部在本機執行。'
                f'{"　<b>（示範模式）</b>" if res.stub else ""}</div></div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                pdf = gr.File(label="上傳健檢報告（PDF）", file_types=[".pdf"], type="filepath")
                sample = gr.Dropdown(choices=list(sample_map), value=None, label="或試用合成範例報告",
                                     info="完全合成的測試資料，不含真實個案")
                sex = gr.Radio(SEX_CHOICES, value=SEX_CHOICES[0], label="性別",
                               info="預設從報告判斷；判斷不到時請手動選擇")
                alias = gr.Textbox(label="個人代號（選填，用於趨勢）", placeholder="例如：我、爸爸-2026",
                                   info="只儲存代號的加鹽雜湊，不會儲存代號本身、檔名或報告文字")
                rdate = gr.Textbox(label="報告日期（選填）", placeholder="YYYY-MM-DD，留空則自動從報告讀取")
                web = gr.Checkbox(value=False, label="允許網路補充搜尋（僅送出標準化項目名稱）")
                run = gr.Button("開始分析", variant="primary")
                status = gr.Markdown("")
                with gr.Accordion("網路查詢稽核紀錄", open=False):
                    audit = gr.HTML('<div class="muted">尚未分析。</div>')
                with gr.Accordion("系統狀態", open=False):
                    gr.HTML('<table class="kv">' + "".join(
                        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in res.status.items())
                        + "</table>")
            with gr.Column(scale=3):
                notices = gr.HTML("")
                overview = gr.HTML("")
                with gr.Tabs():
                    with gr.Tab("摘要 & 標籤"):
                        tags_html = gr.HTML('<div class="muted">請上傳報告或選擇範例後按「開始分析」。</div>')
                    with gr.Tab("檢驗值"):
                        only_abn = gr.Checkbox(value=False, label="只顯示異常")
                        findings_html = gr.HTML("")
                    with gr.Tab("參考資料"):
                        refs_html = gr.HTML("")
                    with gr.Tab("效能"):
                        perf_html = gr.HTML("")
                    with gr.Tab("趨勢"):
                        trend_note = gr.Markdown("輸入個人代號並分析報告後，這裡會顯示數值型檢驗項目的歷次變化"
                                                 "（定性結果如「陰性」不列入）。")
                        with gr.Row():
                            trend_item = gr.Dropdown(choices=[], label="檢驗項目", scale=3)
                            trend_load = gr.Button("載入此代號的趨勢", scale=1)
                        trend_plot = gr.LinePlot(x="日期", y="數值", title="歷次數值", height=320,
                                                 tooltip=["日期", "數值", "判定"])
                        trend_table = gr.Dataframe(interactive=False, label="資料點")
                        with gr.Row():
                            del_btn = gr.Button("刪除此代號的所有紀錄", variant="stop")
                            del_msg = gr.Markdown("")
                with gr.Row():
                    dl_json = gr.DownloadButton("下載 JSON", value=None, interactive=False)
                    dl_csv = gr.DownloadButton("下載 CSV", value=None, interactive=False)
        gr.HTML(f'<div id="footer">⚕ {esc(DISCLAIMER)}任何健康疑慮請諮詢醫師或專業醫療人員。</div>')

        sample.change(lambda s: sample_map.get(s), inputs=sample, outputs=pdf)

        outputs = [status, notices, overview, tags_html, findings_html, refs_html, perf_html, audit,
                   dl_json, dl_csv, state, trend_item, trend_plot, trend_table]

        def analyze(pdf_path, sex_choice, alias_v, rdate_v, web_v, only_abn_v, progress=gr.Progress()):
            skip = [gr.update()] * (len(outputs) - 1)
            if not pdf_path:
                yield ["⚠ 請先上傳 PDF 或選擇範例報告。", *skip]
                return
            if rdate_v and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", rdate_v.strip()):
                yield ["⚠ 報告日期格式應為 YYYY-MM-DD。", *skip]
                return
            data = Path(pdf_path).read_bytes()
            q: "queue.Queue" = queue.Queue()
            box: Dict = {}

            def worker():
                try:
                    box["a"] = run_analysis(data, res, sex_choice, alias_v, rdate_v or "", bool(web_v),
                                            on_stage=lambda s: q.put(("stage", s)), runs_db=runs_db,
                                            on_partial=lambda pa: q.put(("partial", pa)))
                except Exception as e:  # noqa: BLE001
                    log.exception("analysis failed")
                    box["err"] = e
                q.put(("done", None))

            threading.Thread(target=worker, daemon=True).start()
            names = dict(STAGES)
            order = [k for k, _ in STAGES]
            done_stages: List[str] = []
            shown_findings = False
            lines: List[str] = []
            while True:
                kind, val = q.get()
                if kind == "done":
                    break
                if kind == "partial":
                    # 1st partial: findings + rule tags (~1 s); later ones: streamed LLM advice
                    upd = [gr.update()] * len(outputs)
                    upd[outputs.index(tags_html)] = render_tags(val)
                    if not shown_findings:
                        shown_findings = True
                        upd[outputs.index(notices)] = render_notices(val)
                        upd[outputs.index(overview)] = render_overview(val)
                        upd[outputs.index(findings_html)] = render_findings(val, bool(only_abn_v))
                        first = val.result.timing.get("first_result_s")
                        lines.append(f"⚡ 檢驗值與規則判定已顯示（{first:.1f} 秒）；建議標籤生成中…")
                        upd[0] = "\n\n".join(lines)
                    yield upd
                    continue
                done_stages.append(val)
                i = order.index(val) if val in order else 0
                progress((i + 0.5) / len(order), desc=names.get(val, val))
                stage_lines = [f"{'⏳' if s == val else '✅'} {names.get(s, s)}" for s in done_stages]
                yield ["\n\n".join(stage_lines + lines), *skip]
            if "err" in box:
                yield [f"❌ 分析失敗：{esc(box['err'])}", *skip]
                return
            a: Analysis = box["a"]
            jp, cp = write_exports(a)
            choices = trend_choices(a.patient_id, runs_db) if a.patient_id else []
            first = choices[0][1] if choices else None
            plot_df, table_df = trend_outputs(a.patient_id, first, runs_db)
            tm = a.result.timing
            msg = (f"✅ 完成，用時 {tm.get('total_s', 0):.1f} 秒"
                   + (f"（首批結果 {tm['first_result_s']:.1f} 秒）" if tm.get("first_result_s") else "")
                   + ("（快取結果）" if a.cached else "")
                   if not a.result.error else f"⚠ {esc(a.result.error)}")
            yield [msg, render_notices(a), render_overview(a), render_tags(a),
                   render_findings(a, bool(only_abn_v)), render_references(a), render_perf(a, res),
                   render_audit(a), gr.update(value=jp, interactive=True),
                   gr.update(value=cp, interactive=True), a,
                   gr.update(choices=choices, value=first), plot_df, table_df]

        run.click(analyze, inputs=[pdf, sex, alias, rdate, web, only_abn], outputs=outputs,
                  concurrency_limit=1)
        only_abn.change(lambda a, o: render_findings(a, o) if a else "", inputs=[state, only_abn],
                        outputs=findings_html)

        def load_trend(alias_v):
            pid = storage.hash_patient_label(alias_v, db_path=runs_db) if (alias_v or "").strip() else ""
            if not pid:
                return (gr.update(choices=[], value=None), *trend_outputs("", None),
                        "⚠ 請先輸入個人代號。")
            choices = trend_choices(pid, runs_db)
            first = choices[0][1] if choices else None
            note = f"此代號共有 {len(choices)} 個數值型項目。" if choices else "此代號尚無紀錄。"
            return (gr.update(choices=choices, value=first), *trend_outputs(pid, first, runs_db), note)

        trend_load.click(load_trend, inputs=alias, outputs=[trend_item, trend_plot, trend_table, trend_note])

        def pick_item(alias_v, key):
            pid = storage.hash_patient_label(alias_v, db_path=runs_db) if (alias_v or "").strip() else ""
            return trend_outputs(pid, key, runs_db)

        trend_item.change(pick_item, inputs=[alias, trend_item], outputs=[trend_plot, trend_table])

        def delete_all(alias_v):
            if not (alias_v or "").strip():
                return ("⚠ 請先輸入要刪除的個人代號。", gr.update(choices=[], value=None),
                        *trend_outputs("", None))
            n = storage.delete_patient(storage.hash_patient_label(alias_v, db_path=runs_db), db_path=runs_db)
            return (f"已刪除 {n} 筆紀錄。", gr.update(choices=[], value=None), *trend_outputs("", None))

        del_btn.click(delete_all, inputs=alias, outputs=[del_msg, trend_item, trend_plot, trend_table])
    return demo, launch_kw


def main(argv=None):
    ap = argparse.ArgumentParser(description=TITLE)
    ap.add_argument("--demo-stub", action="store_true",
                    help="no LLM / embeddings / network: real extraction + canned tags")
    ap.add_argument("--host", default=os.environ.get("UI_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("UI_PORT", "7860")))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("loading resources (%s mode) ...", "demo-stub" if args.demo_stub else "full")
    res = build_resources(stub=args.demo_stub)
    demo, launch_kw = build_app(res)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, max_file_size="20mb", **launch_kw)


if __name__ == "__main__":
    main()
