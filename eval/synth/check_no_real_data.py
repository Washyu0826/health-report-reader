"""
check_no_real_data.py — Assert that the synthetic dataset in eval/synth/ holds
no brand names and no rows copied from the private source CSV.

1. Denylist: brand / private identifiers (stored base64-encoded so this public
   file does not itself contain them) must not appear, case-insensitively, in
   any text file under eval/synth/ or in any PDF's text layer / metadata.
2. Row hashing (only if the private CSV is present locally; skipped otherwise):
   every CSV row is reduced IN MEMORY to salted SHA-256 hashes of
     (item name, value)  and  (item name, value, reference)
   after NFKC + whitespace normalisation. Nothing from the CSV is printed or
   written. Synthetic items are hashed the same way (printed name and canonical
   name). Assertions:
     a. no synthetic (name, value, reference) row with a non-generic value
        equals a real row (generic = qualitative results such as "陰性", "-",
        "Negative", "+", "2+" that every report shares);
     b. (name, value) pair collisions are no more frequent than chance: the
        same check is re-run on a null model where every numeric synthetic
        value is shifted by +-1..3 display steps (values that certainly were
        not copied); observed collisions must be <= 2 x null mean + 5.
        Low-cardinality values (integer blood pressure, "0.2") collide by chance;
     c. no synthetic report shares more than MAX_OVERLAP of its (name, value)
        pairs with any single real report (i.e. no report was copied).
   Only counts are printed.

Usage (from project root):
    .venv/Scripts/python eval/synth/check_no_real_data.py [--csv <path to the private source CSV>]
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SALT = b"synth-leak-check-v1"
MAX_OVERLAP = 0.20
TEXT_EXT = {".py", ".json", ".md", ".txt", ".csv"}

# base64 of: brand name (latin), brand name (CJK), private source file stem
_DENY_B64 = ["SDJV", "5pep5a6J5YGl5bq3", "Q2hlY2tSZXBvcnREZXRhaWw="]
DENY = [base64.b64decode(x).decode("utf-8") for x in _DENY_B64]
# the private CSV lives next to the repo folder (never inside it); name kept out of plain text
DEFAULT_CSV = HERE.parent.parent.parent / (DENY[2] + ".csv")
GENERIC_VALUES = {"陰性", "-", "negative", "neg", "+", "1+", "2+", "3+", "4+", "陽性", "positive", "(-)", "±", "+/-"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "")).strip().lower()


def h(*parts: str) -> bytes:
    return hashlib.sha256(SALT + "\x1f".join(norm(p) for p in parts).encode("utf-8")).digest()


def scan_denylist(root: Path):
    import pdfplumber
    hits = []
    deny = [d.lower() for d in DENY]
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == Path(__file__).name or "__pycache__" in p.parts:
            continue
        if p.suffix.lower() in TEXT_EXT:
            text = p.read_text(encoding="utf-8", errors="ignore")
        elif p.suffix.lower() == ".pdf":
            with pdfplumber.open(p) as doc:
                text = "\n".join((pg.extract_text() or "") for pg in doc.pages)
                text += "\n" + json.dumps(doc.metadata, ensure_ascii=False, default=str)
        else:
            continue
        low = unicodedata.normalize("NFKC", text).lower()
        for i, d in enumerate(deny):
            if d in low:
                hits.append((str(p.relative_to(root)), f"denylist#{i}"))
    return hits


def synthetic_rows(gt: dict):
    reports = {}
    for fname, e in gt["samples"].items():
        rows = []
        for it in e["items"]:
            for name in {it["printed_name"], it["canonical_name"]}:
                rows.append((name, it["value"], it.get("printed_range", ""), it.get("sex_applicable_range", "")))
        reports[fname] = rows
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    root = Path(args.dir)
    ok = True

    hits = scan_denylist(root)
    print(f"[denylist] {len(DENY)} terms, files scanned under {root.name}/ : {len(hits)} hit(s)")
    for f, t in hits:
        print(f"   HIT {t} in {f}")
    ok &= not hits

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("[rows] private CSV not present -> row-hash check skipped (expected on public clones)")
    else:
        pair_h, triple_h = set(), set()
        per_report = defaultdict(set)
        n_rows = 0
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                name, val, ref = r.get("ItemName", ""), r.get("Value", ""), r.get("Reference", "")
                if not norm(name) or not norm(val):
                    continue
                n_rows += 1
                ph = h(name, val)
                pair_h.add(ph)
                triple_h.add(h(name, val, ref))
                per_report[hashlib.sha256(SALT + (r.get("ReportId") or "").encode()).digest()].add(ph)
        gt = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
        syn = synthetic_rows(gt)
        n_syn = sum(len(v) for v in syn.values())
        triple_hits = pair_hits = generic_pair_hits = 0
        worst = 0.0
        null_hits = defaultdict(int)
        shifts = [-3, -2, -1, 1, 2, 3]
        for fname, rows in syn.items():
            s_pairs = set()
            for name, val, pref, sref in rows:
                ph = h(name, val)
                s_pairs.add(ph)
                generic = norm(val) in GENERIC_VALUES
                if not generic and (h(name, val, pref) in triple_h or h(name, val, sref) in triple_h):
                    triple_hits += 1
                if ph in pair_h:
                    if generic:
                        generic_pair_hits += 1
                    else:
                        pair_hits += 1
                m = re.fullmatch(r"(\d+)(?:\.(\d+))?", val)
                if m:
                    dec = len(m.group(2) or "")
                    for k in shifts:
                        sv = f"{float(val) + k * 10 ** -dec:.{dec}f}"
                        if h(name, sv) in pair_h:
                            null_hits[k] += 1
            for rp in per_report.values():
                if s_pairs:
                    worst = max(worst, len(s_pairs & rp) / len(s_pairs))
        null_mean = sum(null_hits.values()) / len(shifts)
        pair_limit = 2 * null_mean + 5
        print(f"[rows] real rows hashed: {n_rows} (in memory only); synthetic rows checked: {n_syn}")
        print(f"[rows] (a) exact (name,value,reference) matches, non-generic value: {triple_hits} (must be 0)")
        print(f"[rows] (b) exact (name,value) matches, non-generic value: {pair_hits}; "
              f"null model (values shifted +-1..3 steps) mean: {null_mean:.1f} -> limit {pair_limit:.1f}")
        print(f"[rows]     (name,value) matches with generic qualitative value (not counted): {generic_pair_hits}")
        print(f"[rows] (c) max share of one synthetic report's (name,value) pairs found in a single real report: "
              f"{worst:.3f} (limit {MAX_OVERLAP})")
        ok &= triple_hits == 0 and pair_hits <= pair_limit and worst <= MAX_OVERLAP

    print("PASS: no brand names, no real rows" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
