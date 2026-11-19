#!/usr/bin/env python3
"""
Step 5: false alarm control with split-conformal calibration. No GPU needed.

Problem found in steps 3-4: the black-box monitors were calibrated by building the p-value tables
and the alarm threshold on the SAME clean q1 frames. The threshold then sits on scores that the
tables have already seen, so it is optimistic, and the realized false alarm rate on q2 came out
above the target (physics: 1.71% at alpha = 1%).

Fix (split-conformal calibration):
  reference part  builds the per-statistic p-value tables (and the NIS reference)
  calibration part  only used to score frames with those tables; the alarm threshold is the
                    ceil((n+1)(1-alpha))-th smallest calibration score
If calibration frames and later frames are exchangeable, the false alarm rate is at most alpha in
expectation (finite-sample guarantee, no distributional assumption). Video frames are neither
independent nor identically distributed over time, so this report also measures how far the
guarantee holds on frames that come later in the same sequences.

Calibrations compared (all on clean frames, faults never used):
  same-data   current method: tables and threshold both from all of q1
  split-q1    tables from the first half of q1, threshold from the second half of q1
  split-q12   tables from q1, threshold from q2 (the configuration proposed for the sealed test)
Checked on:
  q2 clean frames (out of sample for same-data and split-q1; in-sample for split-q12)
  and, for detection power, the propagated transient faults of the step 2 campaign.
FAR confidence intervals use a moving-block bootstrap (blocks of BLOCK frames inside each
sequence), because neighbouring frames are strongly correlated.

Usage:  python step5_calibration.py
Output: results/step5_report.txt, results/calibration_split_q12.json
"""
import json
import math

import numpy as np

import common as C
from monitors import FAMILIES, FEATURES, Calibrator, extract, finite
from step2_campaign import golden_tracking, load_golden
from step3_analyze import clean_rows, read_csv, unpack, wilson

ALPHAS = (0.01, 0.001)
FAMS = ["frame", "temporal", "physics", "combined"]
BLOCK = 60
N_BOOT = 2000


def fit_split(ref_rows, cal_rows):
    """Calibrator whose p-value tables come from ref_rows and whose score reference comes from cal_rows."""
    ref = {k: np.sort(np.array([r[k] for r in ref_rows], float)) for k in FEATURES}
    nis = np.sort(np.asarray([v for r in ref_rows for v in r["nis"]], float))
    c = Calibrator(ref, nis)
    for fam in FAMILIES:
        c.xref[fam] = np.sort(np.array([c.score(r, fam) for r in cal_rows]))
    return c


def conformal_threshold(sorted_scores, alpha):
    n = len(sorted_scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(sorted_scores[k - 1])


def halves(rows):
    """Split every sequence's clean rows in time: first half -> tables, second half -> threshold."""
    by = {}
    for r in rows:
        by.setdefault(r["seq"], []).append(r)
    first, second = [], []
    for rs in by.values():
        rs = sorted(rs, key=lambda r: r["frame"])
        first += rs[:len(rs) // 2]; second += rs[len(rs) // 2:]
    return first, second


def block_bootstrap(flags_by_seq, rng):
    blocks = []
    for f in flags_by_seq.values():
        f = np.asarray(f, float)
        blocks += [f[i:i + BLOCK] for i in range(0, len(f), BLOCK)]
    sums = np.array([b.sum() for b in blocks]); lens = np.array([len(b) for b in blocks])
    idx = rng.integers(0, len(blocks), size=(N_BOOT, len(blocks)))
    est = sums[idx].sum(1) / lens[idx].sum(1)
    return np.quantile(est, [0.025, 0.975])


def main():
    cmeta = json.loads((C.RESULTS / "campaign" / "meta.json").read_text())
    mot17 = cmeta["args"]["mot17"]
    g1, g2 = load_golden(mot17, "q1"), load_golden(mot17, "q2")
    rows1, nis1 = clean_rows(g1)
    rows2, _ = clean_rows(g2)
    A, B = halves(rows1)

    cals = {}
    same = Calibrator.fit(rows1, nis1)
    cals["same-data"] = (same, lambda fam, a: same.threshold(fam, a))
    sq1 = fit_split(A, B)
    cals["split-q1"] = (sq1, lambda fam, a, c=sq1: conformal_threshold(c.xref[fam], a))
    sq12 = fit_split(rows1, rows2)
    cals["split-q12"] = (sq12, lambda fam, a, c=sq12: conformal_threshold(c.xref[fam], a))

    # propagated transient faults (positives), features recomputed from the stored detections
    act = read_csv(C.RESULTS / "campaign" / "act.csv")
    crit = [r for r in act if r["outcome"] == "sdc_prop"]
    store = unpack(C.RESULTS / "campaign" / "act_store.npz")
    by_seq = {}
    for r in crit:
        by_seq.setdefault(r["seq"], []).append(r)
    pos_feats = []
    for s, rs in by_seq.items():
        gs = g2[s]
        _, snaps = golden_tracking(gs, {int(r["frame"]) for r in rs})
        for r in rs:
            t = int(r["frame"])
            d, _ = store[(int(r["id"]), C.SEQS.index(s), t)]
            pos_feats.append(extract(snaps[t], d, finite(gs["dets"][t - 1]), t, gs["info"]["width"], gs["info"]["height"]))

    rng = np.random.default_rng(0)
    L = ["# Step 5 report: split-conformal false alarm control", "",
         f"clean frames: q1 {len(rows1)} (first half {len(A)}, second half {len(B)}), q2 {len(rows2)}; "
         f"propagated faults {len(pos_feats)}", "",
         "  FAR = false alarm rate on clean q2 frames, 95% moving-block bootstrap interval (blocks of "
         f"{BLOCK} frames)", "  TPR = detected fraction of propagated transient faults (Wilson interval)",
         "  split-q12 uses q2 to set the threshold, so its q2 FAR is in-sample by construction",
         "  a conformal threshold needs at least 1/alpha - 1 calibration frames; with fewer it is infinite (no alarms)", ""]
    summary = {}
    for a in ALPHAS:
        L.append(f"## alpha = {100 * a:g}%")
        L.append(f"  {'monitor':<10}{'calibration':<13}{'FAR q2':>9}{'95% CI':>18}{'worst seq':>11}"
                 f"{'TPR':>8}{'95% CI':>17}")
        for fam in FAMS:
            for name, (cal, thr_fn) in cals.items():
                t = thr_fn(fam, a)
                flags = {}
                for r in rows2:
                    flags.setdefault(r["seq"], []).append(cal.score(r, fam) > t)
                allf = np.concatenate([np.asarray(v) for v in flags.values()])
                lo, hi = block_bootstrap(flags, rng)
                worst = max(np.mean(v) for v in flags.values())
                k = sum(cal.score(f, fam) > t for f in pos_feats)
                tlo, thi = wilson(k, len(pos_feats))
                L.append(f"  {fam:<10}{name:<13}{100 * allf.mean():8.2f}%  [{100 * lo:5.2f},{100 * hi:6.2f}]%"
                         f"{100 * worst:10.2f}%{100 * k / max(len(pos_feats), 1):7.1f}%  [{100 * tlo:4.1f},{100 * thi:5.1f}]")
                summary.setdefault(str(a), {}).setdefault(fam, {})[name] = {
                    "threshold": t, "far_q2": float(allf.mean()), "far_ci": [float(lo), float(hi)],
                    "worst_seq_far": float(worst), "tpr": k / max(len(pos_feats), 1)}
            L.append("")

    # exchangeability check: same split-q1 calibrator, FAR on held-out q1 frames vs on q2
    L.append("## Does the guarantee survive the move from q1 to q2?  (split-q1, alpha = 1%)")
    L.append("  the split-q1 threshold is only guaranteed for frames exchangeable with the second half of q1.")
    L.append(f"  {'monitor':<10}{'FAR q1 2nd half':>17}{'FAR q2':>9}")
    for fam in FAMS:
        t = conformal_threshold(sq1.xref[fam], 0.01)
        f_in = np.mean([sq1.score(r, fam) > t for r in B])
        f_out = np.mean([sq1.score(r, fam) > t for r in rows2])
        L.append(f"  {fam:<10}{100 * f_in:16.2f}%{100 * f_out:8.2f}%")
    L.append("")

    out = {"note": "tables from q1, thresholds from q2 (split-conformal); proposed for the sealed test",
           "thresholds": {str(a): {fam: conformal_threshold(sq12.xref[fam], a) for fam in FAMILIES} for a in ALPHAS},
           "summary": summary}
    (C.RESULTS / "calibration_split_q12.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (C.RESULTS / "step5_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
