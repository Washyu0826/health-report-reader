"""
storage.py — SQLite audit log of every analyze run, with per-patient grouping.

Identity model: the user picks a private label (e.g. "me", "dad-2026").
We hash it with sha256 and store only the hash as `patient_id`. The cleartext
never touches disk, so even if the DB leaks the patient identity stays opaque.
The trade-off is that a forgotten label = lost history.

`runs` is the single source of truth — analysis output, retrieval debug, the
structured findings list (per-indicator value+direction+ref) — so all
follow-up queries (trends, history, comparison) read from this one table.

Privacy rules (R7):
  * File names are never stored (they often contain the patient's real name);
    the `file_name` column is kept for schema compatibility but written empty,
    and legacy rows are scrubbed on connect.
  * Only findings resolved to a canonical lab item are stored, with a fixed
    field whitelist. Unmatched extracted names (which can carry phone numbers,
    dates of birth or names) and raw value text never reach disk.
  * Label hashing uses a per-install random salt (HMAC-SHA256), so short labels
    such as "me" or "dad" cannot be reversed with a dictionary.
  * Trend series contain numeric values only; qualitative results are skipped.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB = "./chroma_db/runs.sqlite3"


# ─── Connection / schema ─────────────────────────────────────────────────────

def _connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL,
            patient_id      TEXT,
            report_date     TEXT,
            file_name       TEXT,
            file_hash       TEXT,
            file_bytes      INTEGER,
            model           TEXT,
            prompt_version  TEXT,
            embed_model     TEXT,
            top_k           INTEGER,
            use_web         INTEGER,
            summary         TEXT,
            tags_json       TEXT,
            findings_json   TEXT,
            retrieval_json  TEXT,
            error           TEXT
        )
    """)
    _ensure_columns(conn, "runs", {
        "patient_id": "TEXT",
        "report_date": "TEXT",
    })
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_hash ON runs(file_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_patient ON runs(patient_id, report_date)")
    # Privacy scrub: older versions stored plain file names.
    conn.execute("UPDATE runs SET file_name = '' WHERE file_name IS NOT NULL AND file_name != ''")
    conn.commit()
    return conn


def _ensure_columns(conn: sqlite3.Connection, table: str, cols: Dict[str, str]) -> None:
    """Add columns missing from an older deployment of the schema."""
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, ddl in cols.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


# ─── Identity helpers ────────────────────────────────────────────────────────

SALT_FILE = ".patient_id_salt"


def _install_salt(db_path: str = DEFAULT_DB) -> bytes:
    """Per-install random salt (env PATIENT_ID_SALT overrides), created on first use."""
    env = os.environ.get("PATIENT_ID_SALT")
    if env:
        return env.encode("utf-8")
    p = Path(db_path).parent / SALT_FILE
    try:
        return p.read_bytes()
    except FileNotFoundError:
        p.parent.mkdir(parents=True, exist_ok=True)
        salt = secrets.token_hex(16).encode("ascii")
        p.write_bytes(salt)
        return salt


def hash_patient_label(label: str, db_path: str = DEFAULT_DB) -> str:
    """Stable, opaque ID derived from a user-chosen private label (salted HMAC)."""
    label = (label or "").strip()
    if not label:
        return ""
    return hmac.new(_install_salt(db_path), label.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def file_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


# ─── Sanitising ──────────────────────────────────────────────────────────────

_FINDING_FIELDS = ("canonical_key", "display_name", "matched_name", "value", "unit", "direction",
                   "status", "ref_low", "ref_high", "ref_unit", "range_source", "range_conflict")


def sanitize_findings(findings: List[Dict]) -> List[Dict]:
    """Keep only canonical (catalog-resolved) findings and a fixed field whitelist.

    `name` is replaced by the curated display name: the extracted name can
    carry adjacent text from the report (e.g. a patient name before 收縮壓).
    """
    out = []
    for f in findings or []:
        if not isinstance(f, dict) or not f.get("canonical_key"):
            continue
        row = {k: f.get(k) for k in _FINDING_FIELDS if k in f}
        row["name"] = f.get("display_name") or f.get("matched_name") or f["canonical_key"]
        out.append(row)
    return out


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ─── Writes ──────────────────────────────────────────────────────────────────

def record_run(
    *,
    file_name: str,
    file_bytes: bytes,
    model: str,
    prompt_version: str,
    embed_model: str,
    top_k: int,
    use_web: bool,
    tags_result: Dict[str, Any],
    findings: List[Dict],
    retrieval_debug: Dict,
    patient_id: str = "",
    report_date: str = "",
    error: str = "",
    db_path: str = DEFAULT_DB,
) -> int:
    """Insert a row and return the new run id.

    `file_name` is accepted for API compatibility but never stored; findings
    are sanitised with `sanitize_findings`.
    """
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO runs (
                ts, patient_id, report_date,
                file_name, file_hash, file_bytes,
                model, prompt_version, embed_model, top_k, use_web,
                summary, tags_json, findings_json, retrieval_json, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                patient_id or None,
                report_date or None,
                "",
                file_fingerprint(file_bytes),
                len(file_bytes),
                model,
                prompt_version,
                embed_model,
                top_k,
                1 if use_web else 0,
                tags_result.get("summary", "") if isinstance(tags_result, dict) else "",
                json.dumps(tags_result, ensure_ascii=False),
                json.dumps(sanitize_findings(findings), ensure_ascii=False),
                json.dumps(retrieval_debug, ensure_ascii=False),
                error,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def record_trend_point(patient_id: str, report_date: str, findings: List[Dict],
                       db_path: str = DEFAULT_DB) -> Optional[int]:
    """Minimal trend row: opaque patient id + report date + canonical findings.

    No file name, file hash, report text, summary or tags are stored.
    """
    if not patient_id:
        return None
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO runs (ts, patient_id, report_date, file_name, findings_json) "
            "VALUES (?, ?, ?, '', ?)",
            (_now(), patient_id, report_date or None,
             json.dumps(sanitize_findings(findings), ensure_ascii=False)),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def delete_patient(patient_id: str, db_path: str = DEFAULT_DB) -> int:
    """Delete every run stored under `patient_id`; returns the number of rows removed."""
    if not patient_id:
        return 0
    conn = _connect(db_path)
    try:
        cur = conn.execute("DELETE FROM runs WHERE patient_id = ?", (patient_id,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ─── Reads ───────────────────────────────────────────────────────────────────

def list_recent(limit: int = 20, db_path: str = DEFAULT_DB) -> List[Dict]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, ts, patient_id, file_name, model, error "
            "FROM runs ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_runs_for_patient(patient_id: str, db_path: str = DEFAULT_DB) -> List[Dict]:
    """Return all runs for one patient ordered by report_date (then ts)."""
    if not patient_id:
        return []
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, ts, report_date, file_name, summary, error,
                   findings_json, tags_json, retrieval_json
            FROM runs
            WHERE patient_id = ?
            ORDER BY COALESCE(report_date, '0000') ASC, ts ASC
            """,
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_run(run_id: int, db_path: str = DEFAULT_DB) -> Optional[Dict]:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ─── Trend extraction ────────────────────────────────────────────────────────

def list_indicators_for_patient(patient_id: str, db_path: str = DEFAULT_DB) -> List[Dict]:
    """Return indicators that appeared in at least 1 run for this patient.

    Each entry: {key, display, unit, ref_low, ref_high, occurrences}.
    The `key` is the canonical name we use to group across reports.
    """
    runs = list_runs_for_patient(patient_id, db_path)
    seen: Dict[str, Dict] = {}
    for run in runs:
        try:
            findings = json.loads(run["findings_json"] or "[]")
        except Exception:
            continue
        for f in findings:
            key = _indicator_key(f)
            if not key or not _is_number(f.get("value")):
                continue  # qualitative / string results are not plotted
            entry = seen.setdefault(key, {
                "key": key,
                "display": _display_name(f),
                "unit": f.get("unit") or f.get("ref_unit") or "",
                "ref_low": f.get("ref_low"),
                "ref_high": f.get("ref_high"),
                "occurrences": 0,
            })
            entry["occurrences"] += 1
            # Prefer non-empty unit / fill in ref if missing on later runs
            if not entry["unit"]:
                entry["unit"] = f.get("unit") or f.get("ref_unit") or ""
            if entry["ref_low"] is None:
                entry["ref_low"] = f.get("ref_low")
            if entry["ref_high"] is None:
                entry["ref_high"] = f.get("ref_high")

    out = list(seen.values())
    # Sort: more occurrences first, then alphabetical
    out.sort(key=lambda x: (-x["occurrences"], x["display"]))
    return out


def time_series_for_indicator(
    patient_id: str,
    indicator_key: str,
    db_path: str = DEFAULT_DB,
) -> List[Dict]:
    """Return chronological [{date, value, direction, run_id, ref_low, ref_high}, ...].

    Numeric values only: qualitative results ("陰性", "+") are skipped.
    """
    runs = list_runs_for_patient(patient_id, db_path)
    series: List[Dict] = []
    for run in runs:
        try:
            findings = json.loads(run["findings_json"] or "[]")
        except Exception:
            continue
        for f in findings:
            if _indicator_key(f) != indicator_key or not _is_number(f.get("value")):
                continue
            series.append({
                "date": run["report_date"] or run["ts"][:10],
                "value": f.get("value"),
                "direction": f.get("direction", "unknown"),
                "unit": f.get("unit") or f.get("ref_unit") or "",
                "ref_low": f.get("ref_low"),
                "ref_high": f.get("ref_high"),
                "run_id": run["id"],
                "raw_name": _display_name(f),
            })
            break  # at most one value per run for a given indicator
    series.sort(key=lambda x: x["date"])
    return series


def _indicator_key(f: Dict) -> str:
    """Group key: canonical_key, else the KB matched_name (legacy rows).

    Unmatched raw names are never used as keys (they may contain PII).
    """
    ck = (f.get("canonical_key") or "").strip()
    if ck:
        return ck
    matched = (f.get("matched_name") or "").strip()
    return matched.lower() if matched else ""


def _display_name(f: Dict) -> str:
    return (f.get("display_name") or f.get("matched_name") or f.get("canonical_key")
            or "").strip() or "(unknown)"
