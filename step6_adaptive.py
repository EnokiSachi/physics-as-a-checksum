#!/usr/bin/env python3
"""
Step 6: adaptive false alarm control with feedback from re-execution. No GPU needed.

Problem left by step 5: a threshold fixed on q1 holds the false alarm rate only for frames that look
like q1. Later frames drift (scene, crowd density), and the physics monitor's realized rate on q2 went
from 0.82% (held-out q1) to 1.47%.

Idea: the protection loop already tells us which alarms were false. An alarm triggers a re-execution
of the frame; if the re-run gives the same output, the first run was (with overwhelming probability)
correct, so the alarm was a false alarm. That is exactly the feedback adaptive conformal inference
(Gibbs and Candes, NeurIPS 2021) needs. Per camera stream:

    threshold_t = conformal quantile of the calibration scores at level 1 - alpha_t
    err_t       = 1 if the alarm at t was false (re-run agreed), else 0
    alpha_t+1   = alpha_t + gamma * (alpha - err_t)

For any gamma > 0 the long-run false alarm rate converges to alpha on any sequence of frames,
without assuming the frames are exchangeable with the calibration set (the guarantee is
deterministic: |mean(err) - alpha| <= (max(alpha_1, 1 - alpha) + gamma) / (gamma * T)).
gamma = 0 is the fixed threshold of step 5.

Evaluated here on clean q2 streams (every alarm is false) with the split-q1 calibration of step 5,
and on the propagated transient faults of the step 2 campaign, using the threshold in force at the
fault's frame. Faults are rare, so they barely move the controller; they are not fed back.

Usage:  python step6_adaptive.py
Output: results/step6_report.txt
"""
import json
import math

import numpy as np

import common as C
from monitors import FAMILIES, extract, finite
from step2_campaign import golden_tracking, load_golden
from step3_analyze import clean_rows, read_csv, unpack, wilson
from step5_calibration import BLOCK, block_bootstrap, fit_split, halves

ALPHAS = (0.01, 0.001)
GAMMAS = (0.0, 0.0005, 0.002, 0.005, 0.02)
FAMS = ["physics", "temporal", "combined"]
WIN = 500  # frames per window for the local false alarm rate


def quantile_at(sorted_scores, a):
    """Conformal threshold at level 1 - a; a <= 0 means never alarm, a >= 1 means always alarm."""
    n = len(sorted_scores)
    if a >= 1:
        return -math.inf
    k = math.ceil((n + 1) * (1 - a))
    return math.inf if k > n else float(sorted_scores[k - 1])


def run_stream(scores, xref, alpha, gamma):
    """Adaptive threshold over one stream of clean scores. Returns alarms and thresholds used."""
    a_t, alarms, thr = alpha, np.zeros(len(scores), bool), np.zeros(len(scores))
    for i, x in enumerate(scores):
        t = quantile_at(xref, a_t)
        thr[i] = t
        alarms[i] = x > t
        a_t += gamma * (alpha - float(alarms[i]))
    return alarms, thr


def main():
    cmeta = json.loads((C.RESULTS / "campaign" / "meta.json").read_text())
    root = cmeta["args"]["mot17"]
    g1, g2 = load_golden(root, "q1"), load_golden(root, "q2")
    rows1, _ = clean_rows(g1)
    rows2, _ = clean_rows(g2)
    A, B = halves(rows1)
    cal = fit_split(A, B)

    # propagated transient faults, features recomputed from the stored detections (as in step 5)
    act = read_csv(C.RESULTS / "campaign" / "act.csv")
    crit = [r for r in act if r["outcome"] == "sdc_prop"]
    store = unpack(C.RESULTS / "campaign" / "act_store.npz")
    by_seq = {}
    for r in crit:
        by_seq.setdefault(r["seq"], []).append(r)
    pos = []
    for s, rs in by_seq.items():
        gs = g2[s]
        _, snaps = golden_tracking(gs, {int(r["frame"]) for r in rs})
        for r in rs:
            t = int(r["frame"])
            d, _ = store[(int(r["id"]), C.SEQS.index(s), t)]
            pos.append((s, t, extract(snaps[t], d, finite(gs["dets"][t - 1]), t, gs["info"]["width"], gs["info"]["height"])))

    streams = {}
    for r in rows2:
        streams.setdefault(r["seq"], []).append(r)
    for s in streams:
        streams[s].sort(key=lambda r: r["frame"])

    rng = np.random.default_rng(0)
    L = ["# Step 6 report: adaptive false alarm control with re-execution feedback", "",
         f"calibration: split-q1 (tables from the first half of q1, scores from the second half, "
         f"{len(B)} frames); test: clean q2 streams ({len(rows2)} frames in {len(streams)} sequences), "
         f"{len(pos)} propagated transient faults", "",
         "  gamma 0 = fixed threshold (step 5). FAR = false alarm rate on clean q2 (95% moving-block bootstrap,",
         f"  blocks of {BLOCK}); worst seq = highest per-sequence rate; window p95 = 95th percentile of the rate over",
         f"  windows of {WIN} frames; TPR = propagated faults detected with the threshold in force at their frame", ""]
    for a in ALPHAS:
        L.append(f"## alpha = {100 * a:g}%")
        L.append(f"  {'monitor':<10}{'gamma':>8}{'FAR':>9}{'95% CI':>18}{'worst seq':>11}{'window p95':>12}"
                 f"{'TPR':>8}{'95% CI':>16}")
        for fam in FAMS:
            xref = cal.xref[fam]
            for gm in GAMMAS:
                flags, thr_at = {}, {}
                for s, rs in streams.items():
                    sc = np.array([cal.score(r, fam) for r in rs])
                    al, th = run_stream(sc, xref, a, gm)
                    flags[s] = al
                    for r, t in zip(rs, th):
                        thr_at[(s, r["frame"])] = t
                allf = np.concatenate(list(flags.values()))
                lo, hi = block_bootstrap(flags, rng)
                worst = max(np.mean(v) for v in flags.values())
                win = [np.mean(v[i:i + WIN]) for v in flags.values() for i in range(0, max(len(v) - WIN + 1, 1), WIN // 2)]
                k = 0
                for s, t, feat in pos:
                    th = thr_at.get((s, t), quantile_at(xref, a))
                    k += cal.score(feat, fam) > th
                n = max(len(pos), 1)
                tlo, thi = wilson(k, n)
                L.append(f"  {fam:<10}{gm:8g}{100 * allf.mean():8.2f}%  [{100 * lo:5.2f},{100 * hi:6.2f}]%"
                         f"{100 * worst:10.2f}%{100 * np.quantile(win, 0.95):11.2f}%{100 * k / n:7.1f}%  [{100 * tlo:4.1f},{100 * thi:5.1f}]")
            L.append("")
    L += ["  what to compare: FAR and worst seq should move toward alpha as gamma grows (gamma 0 is the step 5",
          "  baseline), and TPR shows what the adaptation costs in detection power. Guarantee per stream of T frames:",
          "  |realized rate - alpha| <= (max(alpha, 1 - alpha) + gamma) / (gamma * T).", ""]
    (C.RESULTS / "step6_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
