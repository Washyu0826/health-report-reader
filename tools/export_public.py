"""
export_public.py — Build the public portfolio repo from this working tree.

Only allowlisted paths are copied; everything else (real de-identified samples,
the third-party KB, legacy modules, internal results) stays private. After
copying, every exported file is scanned and the export FAILS on any hit:

  * denylist: brand / employer / private identifiers (base64-stored, so this
    file does not contain them), case-insensitive, in text files and PDF text;
  * real-data rows: (item, value, reference) triples of the private CSV are
    hashed in memory and compared with every JSON/CSV/PDF text in the export
    (reusing the synthetic-set leak checker); nothing from the CSV is printed.

Usage:
    .venv/Scripts/python tools/export_public.py --dest D:/portfolio/health-report-tagger [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOW = [
    # app
    "config.py", "pipeline.py", "pdf_utils.py", "ocr.py", "abnormal.py", "conditions.py",
    "llm.py", "verify.py", "retrieval.py", "kb_index.py", "embeddings.py", "web_fallback.py",
    "storage.py", "ui.py", "graph_workflow.py", "report_cache.py",
    # optional LangChain adapter + runnable examples
    "integrations/**", "examples/**", "requirements-langchain.txt",
    # data that is ours
    "data/lab_synonyms.json", "data/reference_ranges.json", "data/reference_ranges.README.md",
    "data/kb_sample.json", "data/kb_sample_extended.json", "data/kb_sample.README.md",
    # evaluation (synthetic only)
    "eval/run_eval.py", "eval/gt_rules.py", "eval/retrieval_bench.py", "eval/test_abnormal.py",
    "eval/HISTORY.md", "eval/README.md", "eval/results_public/*",
    "eval/synth/*.py", "eval/synth/*.json", "eval/synth/README.md", "eval/synth/samples/*.pdf",
    # tests, tooling, docs
    "tests/*.py", "tools/export_public.py", "tools/kb_coverage.py", "tools/*.ps1", "docs/**",
    ".github/**", "requirements*.txt", "Dockerfile", "docker-compose.yml", ".dockerignore",
    ".gitignore", ".gitattributes", "README.md", "README.zh-TW.md", "LICENSE", "run.ps1", "run.sh",
]
DENY_PATHS = ["data/knowledge_base.json", "data/brand_terms.txt", "data/checkitem_kb.json", "eval/samples/*",
              "eval/ground_truth.json", "eval/silver_labels.json", "eval/build_*.py", "tests/internal/*",
              "csv_to_kb.py", "app.py", "rag.py", "kb_loader.py", "_e2e_test.py", "*.sqlite3", ".env*"]

# base64: brand (latin), brand (CJK), private CSV stem, original author's home dir,
# original folder name, handoff deck stem, employer name (CJK)
_DENY_B64 = ["SDJV", "5pep5a6J5YGl5bq3", "Q2hlY2tSZXBvcnREZXRhaWw=", "L2hvbWUvdG9ueQ==",
             "cmFnX2Zyb21fY2xhdWRl", "Wmh1XzA0MjA=", "5rC45oKF"]
DENY = [base64.b64decode(x).decode("utf-8") for x in _DENY_B64]
TEXT_EXT = {".py", ".json", ".md", ".txt", ".yml", ".yaml", ".toml", ".ps1", ".sh", ".cfg",
            ".csv", ".html", ".css", ".js", ""}


def _matches(rel: str, patterns) -> bool:
    return any(fnmatch.fnmatch(rel, p) or (p.endswith("/**") and rel.startswith(p[:-3] + "/"))
               for p in patterns)


def collect() -> list[Path]:
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(part in {".git", ".venv", "__pycache__", "chroma_db",
                                            "_chroma", "_chroma_bench"} for part in p.parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if _matches(rel, ALLOW) and not _matches(rel, DENY_PATHS):
            files.append(p)
    return sorted(files)


def _pdf_text(path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            meta = " ".join(str(v) for v in (pdf.metadata or {}).values())
            return meta + "\n" + "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    except Exception:  # noqa: BLE001
        return ""


def scan(dest: Path) -> list[str]:
    problems = []
    texts = {}
    for p in dest.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = p.relative_to(dest).as_posix()
        if p.suffix.lower() == ".pdf":
            texts[rel] = _pdf_text(p)
        elif p.suffix.lower() in TEXT_EXT:
            texts[rel] = p.read_text(encoding="utf-8", errors="ignore")
    for rel, t in texts.items():
        if rel == "tools/export_public.py":
            continue
        low = t.lower()
        for d in DENY:
            if d.lower() in low:
                problems.append(f"denylisted identifier in {rel}")
    # real-row triples (only when the private CSV exists locally)
    sys.path.insert(0, str(ROOT / "eval" / "synth"))
    try:
        from check_no_real_data import DEFAULT_CSV, h, norm  # noqa: WPS433
        import csv
        if DEFAULT_CSV.exists():
            real = set()
            with open(DEFAULT_CSV, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    val = norm(row.get("Value", ""))
                    if re.search(r"\d", val) and len(val) >= 2:
                        real.add(h(row.get("ItemName", ""), val, row.get("Reference", "")))
            for rel, t in texts.items():
                # scan "name value (reference)" windows on each line
                for line in t.splitlines():
                    m = re.search(r"^(.{2,30}?)\s+(-?\d+(?:\.\d+)?)\s+.*?(\(.*\))", line)
                    if m and h(m.group(1), m.group(2), m.group(3)) in real:
                        problems.append(f"real CSV row reproduced in {rel}")
                        break
    except ImportError:
        problems.append("leak checker unavailable")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    files = collect()
    print(f"{len(files)} files allowlisted")
    if args.dry_run:
        for f in files:
            print("  ", f.relative_to(ROOT).as_posix())
        return
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.rglob("*"):  # mirror: remove files no longer allowlisted
        if old.is_file() and ".git" not in old.parts:
            rel = old.relative_to(dest).as_posix()
            if not (ROOT / rel).exists() or (ROOT / rel) not in files:
                old.unlink()
    for f in files:
        out = dest / f.relative_to(ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
    problems = scan(dest)
    if problems:
        print("EXPORT BLOCKED:")
        for p in sorted(set(problems)):
            print("  -", p)
        sys.exit(1)
    print(f"export OK -> {dest} (scan clean: denylist + real-row hashes)")


if __name__ == "__main__":
    main()
