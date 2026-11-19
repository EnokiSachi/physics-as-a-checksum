#!/usr/bin/env python3
"""
Step 10b: the white-box range monitor with the same split calibration as the black-box monitor.

In step 10 the range monitor kept its step 3 threshold: activation ranges profiled on the first half of
q1 and the threshold taken on the second half of q1. On the test split that gave 5-7% false alarms at a
nominal 1%, so its extra compute was overstated and the comparison favoured the black-box monitor.
Here the range monitor is calibrated the way the combined monitor is (split-q12):

  range-q12   activation ranges profiled on all of q1; threshold = the (1 - alpha) empirical quantile
              ("higher") of the range violation on clean q2 frames, alpha 1%; one re-execution on alarm

Only development data (q1, q2) set it; the protocol is written to sealed/baselines/range_q12_protocol.json
before the test split is touched. Same fault process as steps 9 and 10 (common random numbers), so the
none / combined rows (step 9) and range / ranger+combined rows (step 10) are shown next to it.

Usage:  python step10b_range.py            run + report
        python step10b_range.py --analyze  report only
Output: results/sealed/baselines/range_q12/..., results/step10b_report.txt
"""
import argparse
import json
import time

import numpy as np

import common as C
from evaltools import evaluate, prepare_gt, write_mot
from monitors import range_violation
from step2_campaign import load_golden
from step3_analyze import clean_rows
from step4_e2e import Executor
from step9_sealed import OUT as SEALED, PROTOCOL as P9, Monitor, calibrate, make_det
from step10_baselines import OUT as B10, run_baseline, table, wide_bounds

OUT = B10 / "range_q12"
PROTOCOL = {"range_q12": "ranges profiled on all of q1; threshold = (1-alpha) 'higher' quantile of the "
                         "range violation on clean q2; alpha 0.01; one re-execution",
            "exp_rates": [0.1, 0.3, 1.0], "seeds": 3}
ALPHA = 0.01


class RangeQ12:
    """Interface of step3_analyze.Scorer as far as run_baseline('range', ...) uses it."""

    def __init__(self, root):
        rows1, _ = clean_rows(load_golden(root, "q1"))
        rows2, _ = clean_rows(load_golden(root, "q2"))
        self.lo = np.min([r["stats"]["amin"] for r in rows1], 0).astype(float)
        self.hi = np.max([r["stats"]["amax"] for r in rows1], 0).astype(float)
        v2 = np.sort([self.violation(r.get("stats")) for r in rows2])
        self.thr = {"range": {ALPHA: float(np.quantile(v2, 1 - ALPHA, method="higher"))}}
        self.n_q2 = len(v2)

    def violation(self, st):
        if st is None or not np.isfinite(st["amax"]).any():
            return float("inf")
        return range_violation(st, self.lo, self.hi)

    def scores(self, feat):
        return {"range": self.violation(feat.get("stats"))}


def run(a, rq):
    det, loader, g = make_det(a), C.FrameLoader(a.mot17, capacity=50), load_golden(a.mot17, "test")
    det.set_bounds(*wide_bounds())
    nb = det.nbits
    bits = list(range(nb - 6, nb))
    results, t0 = {}, time.monotonic()
    for rate in PROTOCOL["exp_rates"]:
        for seed in range(PROTOCOL["seeds"]):
            E = Executor(det, loader, g, rate, 100 + seed, bits, a.backend)
            tag = f"r{rate:g}_s{seed}"
            per_seq, st = run_baseline("range", E, g, None, rq)
            for s, rows in per_seq.items():
                write_mot(OUT / tag / "range-q12" / "data" / f"{s}.txt", rows)
            ev = evaluate("test", OUT / tag, ["range-q12"])
            results.setdefault(str(rate), {}).setdefault("range-q12", []).append(dict(ev["range-q12"], **st))
            print(f"  rate {rate:g} seed {seed}: range-q12 {ev['range-q12']['HOTA']:.2f}, extra compute "
                  f"{100 * st['extra_compute']:.1f}%  ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    (OUT / "summary.json").write_text(json.dumps({"results": results}, indent=1), encoding="utf-8")


def report(a, rq):
    gt = load_golden(a.mot17, "test")
    rows_t, _ = clean_rows(gt)
    cal = calibrate(a.mot17)
    mon = Monitor(cal, P9["alpha"], 0.0)
    far_q12 = np.mean([rq.violation(r.get("stats")) > rq.thr["range"][ALPHA] for r in rows_t])
    far_c = np.mean([cal.score(r, "combined") > mon.threshold(r["seq"]) for r in rows_t])
    L = [f"# Step 10b report: white-box range monitor with split-q12 calibration ({C.RESULTS.name})", "",
         "frozen before the test split was touched: " + json.dumps(PROTOCOL), "",
         "## 1. False alarms on clean test frames at nominal alpha 1%",
         f"  range-q12 (white-box, split-q12, {rq.n_q2} q2 frames)  {100 * far_q12:.2f}%",
         f"  combined  (black-box, split-q12)                     {100 * far_c:.2f}%", ""]
    sp = OUT / "summary.json"
    if sp.exists():
        s9 = json.loads((SEALED / "e2e" / "summary.json").read_text())
        clean_hota = s9["clean"]["clean"]["HOTA"]
        ref = {r: {p: v[p] for p in ("none", "combined") if p in v} for r, v in s9["results"].items()}
        s10 = B10 / "exp" / "summary.json"
        if s10.exists():
            for r, v in json.loads(s10.read_text())["results"].items():
                ref.setdefault(r, {}).update({p: v[p] for p in ("range", "ranger+combined") if p in v})
        L.append("## 2. End-to-end, bit model 'exp' (same faults as steps 9 and 10)")
        L.append(f"  fault-free HOTA {clean_hota:.2f}; range = step 10 (q1 threshold), range-q12 = this step")
        table(L, json.loads(sp.read_text())["results"], clean_hota, ref)
        L.append("")
    L.append("  sealed result: nothing in this report was used to change any setting.")
    (C.RESULTS / "step10b_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--backend", default="yolo", choices=["yolo", "fake"])
    ap.add_argument("--weights", default=C.WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=C.IMGSZ)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not (SEALED / "e2e" / "summary.json").exists():
        raise SystemExit("run step 9 first (sealed/e2e/summary.json is missing)")
    OUT.mkdir(parents=True, exist_ok=True)
    pf = B10 / "range_q12_protocol.json"
    if pf.exists() and json.loads(pf.read_text()) != PROTOCOL:
        raise SystemExit(f"protocol changed since the run started ({pf}); refusing to continue")
    pf.write_text(json.dumps(PROTOCOL, indent=1), encoding="utf-8")
    t0 = time.monotonic()
    prepare_gt(a.mot17)
    rq = RangeQ12(a.mot17)
    if not a.analyze and (a.force or not (OUT / "summary.json").exists()):
        print("running range-q12, bit model exp", flush=True)
        run(a, rq)
    report(a, rq)
    print(f"step 10b done in {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
