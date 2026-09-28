"""
verify.py — Sanity checks for the synthetic dataset in eval/synth/.

Checks
  1. every sample in ground_truth.json has a PDF;
  2. text PDFs: the pdfplumber text layer contains every item value
     (NFKC-normalised, so full-width digits count) and the fake PII header;
     also reports how often item name + value land on the same extracted line;
  3. image-only PDFs: zero text characters on every page, >= 1 image per page;
  4. gold statuses agree with the independent eval/gt_rules.classify() applied
     to (value, sex-applicable printed range);
  5. scenario -> expected-condition consistency, healthy reports all normal,
     one metric tag per abnormal item;
  6. quick.json covers every layout variant;
  7. determinism: regenerating with the same seed into a temp dir yields
     byte-identical PDFs, ground_truth.json and quick.json (skip: --no-regen).

Usage (from project root):  .venv/Scripts/python eval/synth/verify.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from catalog import SCENARIO_EXPECTS  # noqa: E402
from gt_rules import classify  # noqa: E402


def nf(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--no-regen", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import pdfplumber

    d = Path(args.dir)
    gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
    quick = json.loads((d / "quick.json").read_text(encoding="utf-8"))
    fails = []
    n_text = n_img = 0
    val_total = val_found = line_total = line_hit = 0
    rule_total = rule_agree = 0
    rule_disagree = []

    for fname, e in gt["samples"].items():
        pdf = d / "samples" / fname
        if not pdf.exists():
            fails.append(f"{fname}: missing PDF")
            continue
        with pdfplumber.open(pdf) as doc:
            pages = doc.pages
            nchars = [len(p.chars) for p in pages]
            nimgs = [len(p.images) for p in pages]
            text = "\n".join((p.extract_text() or "") for p in pages)
        if e["layout_features"]["image_only"]:
            n_img += 1
            if sum(nchars) or text.strip():
                fails.append(f"{fname}: image-only PDF has a text layer ({sum(nchars)} chars)")
            if min(nimgs) < 1:
                fails.append(f"{fname}: image-only page without image")
        else:
            n_text += 1
            flat = nf(text)
            lines = [nf(x) for x in text.splitlines()]
            for tok in (e["fake_pii"]["name"], e["fake_pii"]["id_no"], e["fake_pii"]["exam_date"]):
                if nf(tok) not in flat:
                    fails.append(f"{fname}: header token {tok!r} missing from text layer")
            for it in e["items"]:
                val_total += 1
                v = nf(it["value"])
                if v in flat:
                    val_found += 1
                else:
                    fails.append(f"{fname}: value {it['value']!r} of {it['printed_name']} not in text layer")
                # same-line check: value + a 2-char chunk of the printed name
                nm = nf(it["printed_name"])
                chunks = {nm[i:i + 2] for i in range(max(1, len(nm) - 1))}
                line_total += 1
                if any(v in ln and any(c in ln for c in chunks) for ln in lines):
                    line_hit += 1
        # independent rule cross-check
        for it in e["items"]:
            if it["gold_status"] == "no_range":
                continue
            rule_total += 1
            ref = it["sex_applicable_range"]
            got = classify(it["value"], ref, it["printed_name"])
            exp = {"positive": "high", "negative": "normal"}.get(it["gold_status"], it["gold_status"])
            if got == exp:
                rule_agree += 1
            else:
                rule_disagree.append(f"{fname} {it['key']}={it['value']} ref={ref} gold={it['gold_status']} gt_rules={got}")
        # scenario consistency
        cond = e["expected"]["conditions"]
        for s in e["scenarios"]:
            exp = SCENARIO_EXPECTS.get(s)
            if exp and not any(c in cond for c in exp):
                fails.append(f"{fname}: scenario {s} but conditions {cond}")
        if "healthy" in e["scenarios"]:
            bad = [i["key"] for i in e["items"] if i["gold_status"] in ("high", "low", "positive")]
            if bad or cond or e["expected"]["risks"]:
                fails.append(f"{fname}: healthy report has abnormal items/conditions {bad} {cond}")
        n_abn = sum(1 for i in e["items"] if i["gold_status"] in ("high", "low", "positive"))
        if n_abn != len(e["expected"]["metrics"]) or n_abn != len(e["metric_keys"]):
            fails.append(f"{fname}: metrics/abnormal count mismatch")

    fails += [f"gt_rules disagreement: {x}" for x in rule_disagree]

    layouts_q = {gt["samples"][f"{i}.pdf"]["layout"] for i in quick["ids"]}
    all_layouts = {e["layout"] for e in gt["samples"].values()}
    if layouts_q != all_layouts or len(quick["ids"]) != 10:
        fails.append(f"quick.json layouts {sorted(layouts_q)} != {sorted(all_layouts)}")

    regen = "skipped"
    if not args.no_regen:
        with tempfile.TemporaryDirectory() as td:
            g = gt["generator"]
            subprocess.run([sys.executable, str(HERE / "generate.py"), "--n", str(g["n"]), "--seed", str(g["seed"]),
                            "--out", td, "-q"], check=True, capture_output=True)
            diffs = []
            for f in ["ground_truth.json", "quick.json"] + [f"samples/{x}" for x in gt["samples"]]:
                if sha(d / f) != sha(Path(td) / f):
                    diffs.append(f)
            regen = "identical" if not diffs else f"DIFFERENT: {diffs[:5]}"
            if diffs:
                fails.append(f"non-deterministic output: {diffs[:5]}")

    print(f"samples: {len(gt['samples'])}  text PDFs: {n_text}  image-only PDFs: {n_img}")
    print(f"text layer: {val_found}/{val_total} item values found; "
          f"name+value on same line: {line_hit}/{line_total} ({line_hit / max(1, line_total):.1%})")
    print(f"gold status vs eval/gt_rules.classify: {rule_agree}/{rule_total} agree")
    print(f"quick.json layouts: {sorted(layouts_q)}")
    print(f"determinism (regenerate & compare sha256): {regen}")
    if fails:
        print(f"\nFAIL ({len(fails)}):")
        for f in fails[:50]:
            print("  -", f)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
