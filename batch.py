"""
batch.py — Analyse many health-check PDFs at once and write standardised tables.

    python batch.py reports/                     # every PDF in the folder
    python batch.py a.pdf b.pdf --out results/
    python batch.py reports/ --no-llm            # extraction + findings + rule tags only; no Ollama
    python batch.py reports/ --fhir              # also <name>.fhir.json (FHIR R4 Bundle, LOINC-coded)

Writes to --out (default ./batch_out):
  * <name>.json / <name>.csv   one per report, same schema as the UI's downloads
  * summary.csv                one row per report (abnormal count, rule conditions, advice tags,
                               citation support, seconds, error)
  * findings_all.csv           every lab value of every report in ONE long table with canonical
                               item keys, so reports from different clinics and layouts line up
  * <name>.fhir.json           with --fhir: the findings as a FHIR R4 Bundle (see fhir_export.py)

Runs locally like the UI (same models and settings from config.py). Report text is not written
anywhere; output file names follow the input file names, so keep --out as private as the PDFs.
Exit code 1 if any report failed (e.g. a scanned PDF with --no-llm, which skips OCR).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import config
import ui
from fhir_export import to_fhir_bundle, write_fhir
from pipeline import NO_TEXT_ERROR, analyze_pdf

SUMMARY_COLS = ["file", "status", "sex", "findings", "abnormal", "range_conflicts", "conditions", "risks",
                "advice_tags", "citation_support", "seconds", "error"]
FINDING_COLS = ["file", "canonical_key", "item", "reported_name", "value", "unit", "status", "range_source",
                "report_range", "reference_range", "range_conflict"]
ADVICE_KEYS = ["lifestyle", "food", "exercise", "supplements", "avoid"]


def collect_pdfs(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() == ".pdf")
        elif p.suffix.lower() == ".pdf" and p.exists():
            out.append(p)
        else:
            raise SystemExit(f"not a PDF or folder: {s}")
    return out


class Analyzer:
    """Full mode: the UI's resources (retrieval index, LLM warm-up). --no-llm: rules only."""

    def __init__(self, no_llm: bool):
        self.no_llm = no_llm
        if no_llm:
            self.lookup = ui.load_checkitem_lookup(config.REFERENCE_RANGES_PATH)
            self.facts = ui.CheckitemFacts(config.REFERENCE_RANGES_PATH)
        else:
            self.res = ui.build_resources(stub=False)

    def __call__(self, data: bytes, sex_choice: str) -> "ui.Analysis":
        if not self.no_llm:
            return ui.run_analysis(data, self.res, sex_choice, use_cache=False)
        t = time.time()
        r = analyze_pdf(data, self.lookup, retriever=ui.Retriever(None, facts=self.facts), run_llm=False,
                        ocr=False, sex=ui._parse_sex(sex_choice))
        r.timing["total_s"] = round(time.time() - t, 3)
        if r.error == NO_TEXT_ERROR:
            r.error += " — OCR needs the local model: run without --no-llm"
        tags = {} if r.error else {k: v for k, v in ui.preliminary_tags(r).items() if k != "summary"}
        return ui.Analysis(r, tags, {}, dict(ui._NO_WEB), ui.report_date_from_text(r.text), False)


def summary_row(name: str, a: "ui.Analysis") -> Dict:
    r, tags = a.result, a.tags or {}
    ver = tags.get("_verification") or {}
    return {
        "file": name, "status": "error" if r.error else "ok", "sex": r.sex or "",
        "findings": len(r.findings), "abnormal": len(r.abnormals),
        "range_conflicts": sum(bool(f.get("range_conflict")) for f in r.findings),
        "conditions": "、".join(t["text"] for t in tags.get("conditions", [])),
        "risks": "、".join(t["text"] for t in tags.get("risks", [])),
        "advice_tags": sum(len(tags.get(k, [])) for k in ADVICE_KEYS),
        "citation_support": ver.get("citation_support_rate") if ver.get("citation_support_rate") is not None
        else "",
        "seconds": r.timing.get("total_s", ""), "error": r.error,
    }


def finding_rows(name: str, a: "ui.Analysis") -> List[Dict]:
    return [{"file": name, "canonical_key": f.get("canonical_key"), "item": f.get("display_name"),
             "reported_name": f.get("name"), "value": f.get("value"), "unit": f.get("unit", ""),
             "status": f.get("status") or f.get("direction"), "range_source": f.get("range_source") or "",
             "report_range": f.get("report_range") or "", "reference_range": f.get("kb_range") or "",
             "range_conflict": bool(f.get("range_conflict"))} for f in a.result.findings]


def write_csv(path: Path, cols: List[str], rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:  # BOM: Excel reads UTF-8
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: ui._csv_cell(row.get(k)) for k in cols})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyse health-check PDFs in bulk.")
    ap.add_argument("inputs", nargs="+", help="PDF files and/or folders of PDFs")
    ap.add_argument("--out", default="batch_out", help="output folder (created if missing)")
    ap.add_argument("--no-llm", action="store_true",
                    help="no Ollama: extraction, abnormal findings and rule conditions/risks only "
                         "(no advice, no OCR for scanned PDFs)")
    ap.add_argument("--fhir", action="store_true",
                    help="also write <name>.fhir.json: a FHIR R4 Bundle with LOINC-coded Observations")
    ap.add_argument("--sex", choices=["auto", "M", "F"], default="auto",
                    help="patient sex for sex-specific ranges (default: read it from each report)")
    args = ap.parse_args(argv)

    pdfs = collect_pdfs(args.inputs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sex_choice = {"M": "男", "F": "女"}.get(args.sex, "")
    analyze = Analyzer(args.no_llm)
    summaries, findings, stems = [], [], set()
    for i, pdf in enumerate(pdfs, 1):
        stem = pdf.stem
        while stem in stems:  # same file name in two folders
            stem += "_"
        stems.add(stem)
        try:
            a = analyze(pdf.read_bytes(), sex_choice)
        except Exception as e:  # noqa: BLE001 — one bad file must not stop the batch
            summaries.append({"file": pdf.name, "status": "error", "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(pdfs)}] {pdf.name}: ERROR {e}")
            continue
        d = ui.export_dict(a)
        if args.no_llm:
            d["mode"] = "no-llm"
        (out / f"{stem}.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / f"{stem}.csv").write_text(ui.export_csv(a), encoding="utf-8-sig")
        if args.fhir and a.result.findings:
            write_fhir(out / f"{stem}.fhir.json",
                       to_fhir_bundle(a.result.findings, sex=a.result.sex, report_date=a.report_date))
        row = summary_row(pdf.name, a)
        summaries.append(row)
        findings += finding_rows(pdf.name, a)
        print(f"[{i}/{len(pdfs)}] {pdf.name}: {row['status']} — {row['abnormal']} abnormal of "
              f"{row['findings']}, {row['seconds']} s" + (f" ({row['error']})" if row["error"] else ""))
    write_csv(out / "summary.csv", SUMMARY_COLS, summaries)
    write_csv(out / "findings_all.csv", FINDING_COLS, findings)
    failed = sum(r["status"] == "error" for r in summaries)
    print(f"done: {len(pdfs) - failed}/{len(pdfs)} ok -> {out.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
