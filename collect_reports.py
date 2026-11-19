#!/usr/bin/env python3
"""Put the step 3-10 reports (and step7_timing.json, diag_escapes.txt, sealed protocol) of every configuration
into one file: all_reports.txt"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ["step3_report.txt", "step4_report.txt", "step5_report.txt", "step6_report.txt",
         "step7_report.txt", "step7_timing.json", "diag_escapes.txt", "step8_report.txt",
         "step9_report.txt", "sealed/protocol.json", "step10_report.txt", "step10b_report.txt"]
dirs = [("mot17_yolov8s_bytetrack", ROOT / "results")]
runs = ROOT / "runs"
if runs.exists():
    dirs += [(d.name, d) for d in sorted(runs.iterdir()) if d.is_dir() and d.name != "cache"]
out = []
for name, d in dirs:
    out.append(f"\n{'=' * 100}\n### CONFIG {name}\n{'=' * 100}")
    for f in FILES:
        p = d / f
        out.append(f"\n----- {f} -----")
        out.append(p.read_text(encoding="utf-8") if p.exists() else "(missing)")
st = ROOT / "runs" / "grid_status.txt"
if st.exists():
    out.append(f"\n----- grid_status.txt -----\n{st.read_text(encoding='utf-8')}")
(ROOT / "all_reports.txt").write_text("\n".join(out), encoding="utf-8")
print(f"wrote {ROOT / 'all_reports.txt'} ({len(dirs)} configurations)")
