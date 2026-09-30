"""
eval/ocr_eval.py — What does scanning cost? A paired text-vs-scan evaluation of the OCR path.

For each synthetic text-layer report, every page is rendered to an image, degraded with a seeded
scan effect (level: clean / medium / heavy), and rebuilt as an image-only PDF. The pipeline then
runs without the LLM (run_llm=False), so only extraction, OCR and the findings are measured. The
same report's text-layer run is the paired control; gold labels come from
eval/synth/ground_truth.json (synthetic only, no real data).

Levels:  clean  = 200 dpi, straight, no noise (OCR itself, nothing else)
         medium = the synthetic set's own scan effect: 150 dpi, 0.6-1.6 deg skew, noise, blur, JPEG 72
         heavy  = 150 dpi, 1.8-2.6 deg skew, stronger noise and blur, JPEG 55, more specks

Per level: abnormal-item P/R/F1 against gold (the same v2 scorer as run_eval), value accuracy
(share of the text run's (item, value) pairs the scan run reproduces exactly), misread values,
missing items, OCR seconds per page.

    .venv/Scripts/python eval/ocr_eval.py                          # 20 reports x 3 levels
    .venv/Scripts/python eval/ocr_eval.py --n 60 --levels heavy    # all text-layer reports, one level
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from abnormal import load_checkitem_lookup  # noqa: E402
from pipeline import analyze_pdf  # noqa: E402
from run_eval import score_abnormal  # noqa: E402

SAMPLES = ROOT / "eval" / "synth" / "samples"
GT = ROOT / "eval" / "synth" / "ground_truth.json"
SEED = 20260930

LEVELS = {
    "clean": dict(dpi=200, angle=(0.0, 0.0), noise=0, blur=0.0, jpeg=92, speck=0.0),
    "medium": dict(dpi=150, angle=(0.6, 1.6), noise=9, blur=0.6, jpeg=72, speck=0.0015),
    "heavy": dict(dpi=150, angle=(1.8, 2.6), noise=14, blur=1.0, jpeg=55, speck=0.003),
}


def degrade(pdf_bytes: bytes, level: str, sample_id: str) -> tuple[bytes, int]:
    """Image-only PDF of the report under a seeded scan effect -> (pdf bytes, pages)."""
    p = LEVELS[level]
    prng = random.Random(f"{SEED}:{sample_id}:{level}")
    nrng = np.random.default_rng(int(hashlib.sha256(f"{SEED}:{sample_id}:{level}".encode()).hexdigest()[:8], 16))
    angle = prng.choice([-1, 1]) * prng.uniform(*p["angle"])
    doc = pdfium.PdfDocument(pdf_bytes)
    pages = []
    for i in range(len(doc)):
        img = doc[i].render(scale=p["dpi"] / 72).to_pil().convert("L")
        if angle:
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=245)
        if p["noise"] or p["speck"]:
            arr = np.asarray(img).astype(np.float32)
            arr = arr * 0.93 + 12 + nrng.normal(0, p["noise"], arr.shape)
            speck = nrng.random(arr.shape) < p["speck"]
            arr[speck] = nrng.integers(0, 90, int(speck.sum()))
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if p["blur"]:
            img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
        jpg = io.BytesIO()
        img.save(jpg, format="JPEG", quality=p["jpeg"])
        pages.append(Image.open(io.BytesIO(jpg.getvalue())).convert("L"))  # keep the JPEG artefacts
    n = len(doc)
    doc.close()
    out = io.BytesIO()
    pages[0].save(out, format="PDF", save_all=True, append_images=pages[1:], resolution=p["dpi"])
    return out.getvalue(), n


def value_sig(f: dict) -> str:
    v = f.get("value")
    if isinstance(v, (int, float)):
        return f"{(f.get('censored') or '')}{float(v):g}"
    return str(f.get("value_text") or v).strip().lower()


def pairs(findings: list) -> dict:
    return {f["canonical_key"]: value_sig(f) for f in findings if f.get("canonical_key")}


def pick(n: int, gt: dict) -> list:
    text_layer = sorted(k for k, s in gt["samples"].items() if s.get("layout") != "scanned")
    if n >= len(text_layer):
        return text_layer
    step = len(text_layer) / n
    return [text_layer[int(i * step)] for i in range(n)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="text-layer reports to scan (evenly spread over the 57)")
    ap.add_argument("--levels", default="clean,medium,heavy")
    ap.add_argument("--out", default="eval/results_ocr.json")
    ap.add_argument("--keep-dir", default="", help="also write the degraded PDFs here")
    args = ap.parse_args(argv)

    gt = json.loads(GT.read_text(encoding="utf-8"))
    lookup = load_checkitem_lookup()
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    files = pick(args.n, gt)
    keep = Path(args.keep_dir) if args.keep_dir else None
    if keep:
        keep.mkdir(parents=True, exist_ok=True)

    rows = []
    for fi, name in enumerate(files, 1):
        spec = gt["samples"][name]
        items = spec.get("items", [])
        src = (SAMPLES / name).read_bytes()
        ctl = analyze_pdf(src, lookup, run_llm=False, ocr=False)
        ctl_pairs = pairs(ctl.findings)
        ctl_score = score_abnormal(ctl.findings, items, v2=True)
        for level in levels:
            pdf, n_pages = degrade(src, level, spec["sample_id"])
            if keep:
                (keep / f"{Path(name).stem}_{level}.pdf").write_bytes(pdf)
            t = time.time()
            r = analyze_pdf(pdf, lookup, run_llm=False, ocr=True)
            secs = time.time() - t
            scan_pairs = pairs(r.findings)
            common = [k for k in ctl_pairs if k in scan_pairs]
            misread = [(k, ctl_pairs[k], scan_pairs[k]) for k in common if ctl_pairs[k] != scan_pairs[k]]
            missing = [k for k in ctl_pairs if k not in scan_pairs]
            sc = score_abnormal(r.findings, items, v2=True) if not r.error else None
            row = {"file": name, "layout": spec.get("layout"), "level": level, "pages": n_pages,
                   "used_ocr": r.used_ocr, "error": r.error, "seconds": round(secs, 2),
                   "sec_per_page": round(secs / max(n_pages, 1), 2),
                   "items_text": len(ctl_pairs), "items_scan": len(scan_pairs),
                   "values_exact": len(common) - len(misread), "misread": misread, "missing": missing,
                   "control_abnormal": {k: ctl_score[k] for k in ("tp", "fp", "fn")},
                   "scan_abnormal": {k: sc[k] for k in ("tp", "fp", "fn")} if sc else None}
            rows.append(row)
            acc = row["values_exact"] / max(row["items_text"], 1)
            print(f"[{fi}/{len(files)}] {name} {level:6s} {n_pages}p {secs:5.1f}s  values {acc:5.1%}  "
                  f"misread {len(misread)} missing {len(missing)}  "
                  + (f"abn tp/fp/fn {sc['tp']}/{sc['fp']}/{sc['fn']}" if sc else f"ERROR {r.error[:60]}"),
                  flush=True)

    summary = {}
    for level in levels:
        rs = [r for r in rows if r["level"] == level]
        ok = [r for r in rs if r["scan_abnormal"]]
        tp = sum(r["scan_abnormal"]["tp"] for r in ok)
        fp = sum(r["scan_abnormal"]["fp"] for r in ok)
        fn = sum(r["scan_abnormal"]["fn"] for r in ok) + sum(r["control_abnormal"]["tp"] for r in rs
                                                              if not r["scan_abnormal"])
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        errs = Counter(k for r in rs for k, _, _ in r["misread"])
        summary[level] = {
            "reports": len(rs), "failed": len(rs) - len(ok),
            "abnormal_precision": round(p, 3), "abnormal_recall": round(rec, 3),
            "abnormal_f1": round(2 * p * rec / (p + rec), 3) if p + rec else 0.0,
            "value_accuracy": round(sum(r["values_exact"] for r in rs) / max(sum(r["items_text"] for r in rs), 1), 3),
            "misread_values": sum(len(r["misread"]) for r in rs),
            "missing_items": sum(len(r["missing"]) for r in rs),
            "sec_per_page_p50": round(statistics.median(r["sec_per_page"] for r in rs), 1) if rs else None,
            "most_misread_items": errs.most_common(8),
            "misread_examples": [m for r in rs for m in r["misread"]][:12],
        }
    ctl_tp = sum(r["control_abnormal"]["tp"] for r in rows if r["level"] == levels[0])
    ctl_fp = sum(r["control_abnormal"]["fp"] for r in rows if r["level"] == levels[0])
    ctl_fn = sum(r["control_abnormal"]["fn"] for r in rows if r["level"] == levels[0])
    cp = ctl_tp / (ctl_tp + ctl_fp) if ctl_tp + ctl_fp else 0.0
    cr = ctl_tp / (ctl_tp + ctl_fn) if ctl_tp + ctl_fn else 0.0
    summary["text_layer_control"] = {"reports": len(files),
                                     "abnormal_f1": round(2 * cp * cr / (cp + cr), 3) if cp + cr else 0.0}
    out = {"seed": SEED, "levels": {k: LEVELS[k] for k in levels}, "files": files, "summary": summary,
           "rows": rows}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
