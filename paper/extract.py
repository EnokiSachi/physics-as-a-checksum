#!/usr/bin/env python3
"""Parse all_reports_final.txt (collect_reports.py output, 8 configurations) into summary.json."""
import json
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "reported_results" / "all_reports.txt")
OUT = Path(__file__).with_name("summary.json")
NAMES = {"mot17_yolov8s_bytetrack": "YOLOv8s", "mot17_yolov8n_bytetrack": "YOLOv8n",
         "mot17_yolov8m_bytetrack": "YOLOv8m", "mot17_yolo11s_bytetrack": "YOLO11s",
         "mot17_yolov8s_sort": "YOLOv8s+SORT", "mot20_yolov8s_bytetrack": "MOT20",
         "kitti_yolov8s_bytetrack": "KITTI", "mot17_rtdetr-l_bytetrack": "RT-DETR"}
NUM = r"(-?[\d.]+)"


def section(blk, name):
    m = re.search(r"----- " + re.escape(name) + r" -----\n(.*?)(?=\n----- |\Z)", blk, re.S)
    return m.group(1) if m else ""


def policy_tables(text, recovered_col):
    """rate -> policy -> dict(HOTA, sd, dHOTA, compute, alarms) from the 'rate X:' tables of steps 9/10.
    Step 9 rows end in 'compute% alarms% false%'; step 10 rows in '[recovered%] compute% alarms%'."""
    out = {}
    for m in re.finditer(r"rate ([\d.]+):\s+policy[^\n]*\n((?:[ \t]{6,}\S[^\n]*\n?)+)", text):
        rate = m.group(1)
        for line in m.group(2).splitlines():
            p = re.match(r"\s+(\S+)\s+" + NUM + r" ±" + NUM + r"\s+" + NUM + r"(.*)$", line)
            if not p:
                continue
            pct = [float(x) for x in re.findall(r"(-?[\d.]+)%", p.group(5))]
            if recovered_col:
                comp, alarms = pct[-2], pct[-1]
            else:
                comp, alarms = pct[0], pct[1]
            out.setdefault(rate, {})[p.group(1)] = {
                "HOTA": float(p.group(2)), "sd": float(p.group(3)), "dHOTA": float(p.group(4)),
                "compute": comp, "alarms": alarms}
    return out


def dev_step4(text):
    out = {}
    for m in re.finditer(r"## fault rate ([\d.]+) per inference\n[^\n]*\n((?:[ \t]+\S[^\n]*\n?)+)", text):
        for line in m.group(2).splitlines():
            p = re.match(r"\s+(\S+)\s+" + NUM + r" ±" + NUM + r"\s+" + NUM + r"\s+" + NUM + r"\s+" + NUM
                         + r"\s+" + NUM + r"%\s+" + NUM + r"%", line)
            if p:
                out.setdefault(m.group(1), {})[p.group(1)] = {
                    "HOTA": float(p.group(2)), "dHOTA": float(p.group(4)), "compute": float(p.group(7))}
    return out


def parse(blk):
    r = {}
    s9 = section(blk, "step9_report.txt")
    m = re.search(r"fixed \(split-q12\)\s+1%\s+" + NUM + r"%\s+\[\s*" + NUM + r",\s*" + NUM + r"\]%\s+" + NUM, s9)
    r["far_fixed"] = [float(m.group(i)) for i in (1, 2, 3, 4)]
    m = re.search(r"adaptive gamma=0.005\s+1%\s+" + NUM + r"%\s+\[\s*" + NUM + r",\s*" + NUM + r"\]%", s9)
    r["far_adaptive"] = [float(m.group(i)) for i in (1, 2, 3)]
    m = re.search(r"fixed \(split-q12\)\s+0.1%\s+" + NUM + r"%", s9)
    r["far_strict"] = float(m.group(1))
    m = re.search(r"(\d+) activation faults: detector output changed (\d+) \(" + NUM + r"%\), tracking changed (\d+), "
                  r"propagated (\d+); tracker fully masked " + NUM + "%", s9)
    r["campaign"] = {"n": int(m.group(1)), "changed": int(m.group(2)), "tracking": int(m.group(4)),
                     "propagated": int(m.group(5)), "masked": float(m.group(6))}
    m = re.search(r"propagated \(sdc_prop\)\s+(\d+)\s+" + NUM + r"\s+fixed alpha 1%\s+" + NUM + r"% \[\s*" + NUM
                  + r",\s*" + NUM + r"\]\s+" + NUM + "%", s9)
    r["tpr"] = {"n": int(m.group(1)), "auc": float(m.group(2)), "tpr": float(m.group(3)),
                "lo": float(m.group(4)), "hi": float(m.group(5)), "damage": float(m.group(6))}
    m = re.search(r"\n\s+adaptive alpha 1%\s+" + NUM + r"% \[\s*" + NUM + r",\s*" + NUM + r"\]\s+" + NUM + "%", s9)
    r["tpr_adaptive"] = {"tpr": float(m.group(1)), "damage": float(m.group(4))}
    m = re.search(r"fault-free HOTA " + NUM, s9)
    r["clean_hota"] = float(m.group(1))
    m = re.search(r"with sanitizer " + NUM, s9)
    r["clean_hota_sanitized"] = float(m.group(1)) if m else None
    r["e2e"] = policy_tables(s9.split("## 4.")[1], False)
    san = s9.split("## 3.")[1].split("## 4.")[0]
    r["sanitizer_sealed"] = san.strip()
    s10 = section(blk, "step10_report.txt")
    m = re.search(r"changed by the clamp alone: (\d+) of (\d+)", s10)
    r["clamp_changed"] = [int(m.group(1)), int(m.group(2))]
    r["far_range_q1"] = float(re.search(r"range \(white-box, q1 threshold\)\s+" + NUM, s10).group(1))
    r["e2e_exp"] = policy_tables(s10.split("## 3.")[1].split("## 4.")[0], True)
    r["e2e_all"] = policy_tables(s10.split("## 4.")[1], True)
    s10b = section(blk, "step10b_report.txt")
    r["far_range_q12"] = float(re.search(r"range-q12 \(white-box, split-q12, \d+ q2 frames\)\s+" + NUM, s10b).group(1))
    for rate, pols in policy_tables(s10b, True).items():
        r["e2e_exp"].setdefault(rate, {})["range-q12"] = pols["range-q12"]
    t7 = section(blk, "step7_timing.json")
    if t7.strip():
        j = json.loads(t7[t7.index("{"):t7.rindex("}") + 1])
        r["checksum"] = {"ms": j["checksum_ms"], "frame_ms": j["frame_ms"], "layerwise_ms": j["checksum_layerwise_ms"],
                         "caught": j.get("flips_caught_all", j["flips_caught"]), "tested": j["flips_tested"],
                         "n_params": j.get("n_params_all", j["n_weights"])}
    s7 = section(blk, "step7_report.txt")
    r["checksum_period"] = [(int(a), float(b), float(c)) for a, b, c in
                            re.findall(r"\n\s+(\d+) frames\s+" + NUM + r"%\s+" + NUM + "%", s7)]
    s5 = section(blk, "step5_report.txt")
    m = re.search(r"combined\s+split-q12\s+" + NUM + r"%\s+\[[^\]]*\]%\s+" + NUM + r"%\s+" + NUM + "%", s5)
    r["dev_far"], r["dev_tpr"] = float(m.group(1)), float(m.group(3))
    r["dev_e2e"] = dev_step4(section(blk, "step4_report.txt"))
    return r


def main():
    t = SRC.read_text(encoding="utf-8")
    res = {}
    for blk in t.split("### CONFIG ")[1:]:
        name = blk.split("\n")[0].strip()
        res[NAMES[name]] = parse(blk)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {OUT}: {', '.join(res)}")


if __name__ == "__main__":
    main()
