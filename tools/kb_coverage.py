"""
kb_coverage.py — Validate the tagged KB (default: data/kb_sample_extended.json) and print its
per-finding coverage. The default retrieval corpus, data/kb_sample.json, is a subset of it.

Every passage carries `applies_to`: a list of
  "<canonical_key>:<direction>"  (key from data/lab_synonyms.json; direction high|low|positive)
  "cond:<label>"                 (a rule-derived condition label emitted by conditions.py)
and `kind` (meaning | diet | lifestyle | followup). Coverage = for each key:direction,
which kinds have at least one passage. REQUIRED lists the findings that must have all
four kinds: every abnormal key:direction in the synthetic ground truth (eval/synth)
plus the common checkup findings.

Usage:
    .venv/Scripts/python tools/kb_coverage.py [--kb data/kb_sample_extended.json]
Exit code 1 on any schema error or a REQUIRED finding missing a kind.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("meaning", "diet", "lifestyle", "followup")
DIRECTIONS = ("high", "low", "positive")
MIN_CHARS, MAX_CHARS = 150, 400
REQUIRED_FIELDS = ("Title", "Content", "TagName", "kind", "applies_to", "Source")

# eval/synth/catalog.py item keys that differ from the canonical lab_synonyms keys
SYNTH_TO_CANONICAL = {
    "tc": "chol", "glu": "glucose_ac", "ua": "uric_acid", "cr": "creatinine", "tbil": "bil_total",
    "alb": "albumin", "tp": "total_protein", "ft4": "free_t4", "u_pro": "urine_protein",
    "u_glu": "urine_glucose", "u_ob": "urine_ob", "u_ket": "urine_ketone", "u_ph": "urine_ph",
    "u_sg": "urine_sg", "u_ubg": "urobilinogen",
}

# common checkup findings that must be fully covered (in addition to the synthetic set)
COMMON = [
    "ldl:high", "chol:high", "tg:high", "hdl:low", "glucose_ac:high", "hba1c:high", "sbp:high", "dbp:high",
    "uric_acid:high", "alt:high", "ast:high", "ggt:high", "alp:high", "bil_total:high", "bil_direct:high",
    "egfr:low", "creatinine:high", "bun:high", "urine_protein:positive", "urine_ob:positive",
    "urine_glucose:positive", "hb:low", "hct:low", "rbc:low", "mcv:low", "mch:low", "mchc:low",
    "hb:high", "rbc:high", "wbc:high", "wbc:low", "plt:high", "plt:low",
    "tsh:high", "tsh:low", "free_t4:low", "free_t4:high",
    "afp:high", "cea:high", "ca199:high", "ca125:high", "ca153:high", "psa:high",
    "bmi:high", "waist:high", "body_fat_pct:high", "hs_crp:high", "homocysteine:high",
    "na:high", "na:low", "k:high", "k:low", "ca:high", "ca:low", "bmd:low",
]


def canonical_keys(path: Path = ROOT / "data" / "lab_synonyms.json") -> Set[str]:
    return {it["key"] for it in json.loads(path.read_text(encoding="utf-8"))["items"]}


def condition_labels(path: Path = ROOT / "conditions.py") -> Set[str]:
    """Labels conditions.derive() can emit (the first argument of every cond(...) call)."""
    return set(re.findall(r'\bcond\(\s*"([^"]+)"', path.read_text(encoding="utf-8")))


def synth_abnormal(path: Path = ROOT / "eval" / "synth" / "ground_truth.json") -> Set[str]:
    """Canonical key:direction of every gold-abnormal item in the synthetic ground truth."""
    gt = json.loads(path.read_text(encoding="utf-8"))
    out = set()
    for s in gt["samples"].values():
        for it in s.get("items", []):
            if it.get("gold_status") in DIRECTIONS:
                out.add(f"{SYNTH_TO_CANONICAL.get(it['key'], it['key'])}:{it['gold_status']}")
    return out


def required_findings() -> List[str]:
    return sorted(set(COMMON) | synth_abnormal())


def load_kb(path: Path = ROOT / "data" / "kb_sample_extended.json") -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(passages: Iterable[Dict], keys: Set[str], conds: Set[str]) -> List[str]:
    errors = []
    titles = set()
    for i, p in enumerate(passages):
        tag = f"[{i}] {p.get('Title', '?')}"
        for f in REQUIRED_FIELDS:
            if not p.get(f):
                errors.append(f"{tag}: missing {f}")
        if p.get("Title") in titles:
            errors.append(f"{tag}: duplicate title")
        titles.add(p.get("Title"))
        n = len(p.get("Content") or "")
        if not MIN_CHARS <= n <= MAX_CHARS:
            errors.append(f"{tag}: content length {n} not in [{MIN_CHARS}, {MAX_CHARS}]")
        if p.get("kind") not in KINDS:
            errors.append(f"{tag}: kind {p.get('kind')!r} not in {KINDS}")
        if not str(p.get("Source", "")).startswith("https://"):
            errors.append(f"{tag}: Source is not an https URL")
        at = p.get("applies_to")
        if not isinstance(at, list) or not at:
            errors.append(f"{tag}: applies_to must be a non-empty list")
            continue
        for a in at:
            if a.startswith("cond:"):
                if a[5:] not in conds:
                    errors.append(f"{tag}: unknown condition label {a!r}")
                continue
            key, _, direction = a.partition(":")
            if key not in keys:
                errors.append(f"{tag}: unknown canonical key {key!r}")
            if direction not in DIRECTIONS:
                errors.append(f"{tag}: bad direction in {a!r}")
    return errors


def coverage(passages: Iterable[Dict]) -> Dict[str, Dict[str, int]]:
    """{applies_to token: {kind: n_passages}}"""
    cov: Dict[str, Dict[str, int]] = defaultdict(lambda: {k: 0 for k in KINDS})
    for p in passages:
        for a in p.get("applies_to") or []:
            cov[a][p.get("kind")] = cov[a].get(p.get("kind"), 0) + 1
    return cov


def missing_kinds(cov: Dict[str, Dict[str, int]], required: Iterable[str]) -> Dict[str, List[str]]:
    out = {}
    for r in required:
        miss = [k for k in KINDS if not cov.get(r, {}).get(k)]
        if miss:
            out[r] = miss
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=str(ROOT / "data" / "kb_sample_extended.json"))
    args = ap.parse_args()
    kb = load_kb(Path(args.kb))
    errors = validate(kb, canonical_keys(), condition_labels())
    cov = coverage(kb)
    req = required_findings()
    kinds = {k: sum(p["kind"] == k for p in kb) for k in KINDS}
    print(f"{len(kb)} passages; kinds {kinds}; {len(cov)} distinct applies_to tokens")
    print(f"\n{'finding':28s} " + " ".join(f"{k:>9s}" for k in KINDS) + "  req")
    for tok in sorted(cov, key=lambda t: (t.startswith("cond:"), t)):
        c = cov[tok]
        print(f"{tok:28s} " + " ".join(f"{c.get(k, 0):9d}" for k in KINDS) + ("    *" if tok in req else ""))
    miss = missing_kinds(cov, req)
    print(f"\nrequired findings: {len(req)}; fully covered: {len(req) - len(miss)}")
    for r, m in miss.items():
        print(f"  MISSING {r}: {', '.join(m)}")
    for e in errors:
        print(f"  ERROR {e}")
    return 1 if (errors or miss) else 0


if __name__ == "__main__":
    sys.exit(main())
