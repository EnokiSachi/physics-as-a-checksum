#!/usr/bin/env python3
"""
Step 8: an output sanitizer between detector and tracker. CPU only; reuses the step 2 campaign.

Why: on RT-DETR (no NMS) about a third of the propagated faults the black-box monitors miss only
ADD a box that duplicates a real object (IoU >= 0.5) with a score just above the tracker's 0.5
threshold. Such a fault stays inside the clean output's variability, so a monitor cannot flag it
without false alarms (diag_escapes.py), yet it costs about as much tracking damage as a caught fault.
Instead of detecting these faults, make them harmless: class-agnostic greedy NMS on the detections
before they reach the tracker (keep the higher-scoring box of any pair with IoU > tau). For
detectors that already run NMS at IoU 0.7 (YOLO), tau >= 0.7 changes nothing on clean frames.

Measured for each tau in {none, 0.8, 0.7, 0.6, 0.5}:
  clean cost   HOTA / IDF1 / MOTA of the fault-free tracker on q2 with the sanitizer
  fault damage every activation fault that changed the detector output (step 2) is replayed with the
               sanitizer applied to the faulty frame and to the golden frames around it; damage =
               FN + FP + ID switches over the window against the sanitized fault-free run (the
               reference is the protected system itself). Reported: faults that still change
               tracking, total damage, damage by detector-level effect, and the residual damage
               of the faults the combined monitor misses at alpha 1% (monitor sees the raw output)

tau is chosen on development data here; the chosen value must be confirmed on the sealed test split.

Usage:  python step8_sanitize.py           current configuration
        python step8_sanitize.py --all     every configuration of run_grid.py -> runs/step8_all.txt
Output: <results>/step8_report.txt
"""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import common as C
from evaltools import GT_ROOT, evaluate, prepare_gt, write_mot
from faults import compare_tracks, same_dets
from monitors import extract, finite
from step2_campaign import load_golden
from step3_analyze import Scorer, read_csv, unpack
from trackers import make_tracker
from trackers.bytetrack import iou_matrix

TAUS = (None, 0.8, 0.7, 0.6, 0.5)


def sanitize(d, tau):
    """Class-agnostic greedy NMS: drop any box that overlaps a higher-scoring kept box with IoU > tau."""
    if tau is None or d is None or len(d) < 2:
        return d
    d = d[np.isfinite(d).all(1)]
    order = np.argsort(-d[:, 4], kind="stable")
    d = d[order]
    io = iou_matrix(d[:, :4], d[:, :4])
    keep = np.ones(len(d), bool)
    for i in range(len(d)):
        if keep[i]:
            sup = io[i] > tau
            sup[: i + 1] = False
            keep &= ~sup
    return d[keep]


def track_split(gs, tau, need=()):
    trk, outs, snaps = make_tracker(), {}, {}
    for f in range(gs["a"], gs["b"] + 1):
        if f in need:
            snaps[f] = copy.deepcopy(trk)
        outs[f] = trk.step(f, finite(sanitize(gs["dets"][f], tau)))
    return outs, snaps


def kind(r):
    if r["n_missing"] > 0 and r["n_spurious"] == 0:
        return "missing only"
    if r["n_spurious"] > 0 and r["n_missing"] == 0:
        return "added only"
    if r["n_spurious"] > 0:
        return "both"
    return "no confident change"


def one():
    cmeta = json.loads((C.RESULTS / "campaign" / "meta.json").read_text())
    root, win = cmeta["args"]["mot17"], int(cmeta["args"]["win"])
    g1, g2 = load_golden(root, "q1"), load_golden(root, "q2")
    if not (GT_ROOT / "q2").exists():
        prepare_gt(root)
    sc = Scorer(g1)
    thr = sc.thr["combined"][0.01]
    act = read_csv(C.RESULTS / "campaign" / "act.csv")
    store = unpack(C.RESULTS / "campaign" / "act_store.npz")
    changed = [r for r in act if r["outcome"] not in ("masked_det", "due")]
    by_seq = {}
    for r in changed:
        by_seq.setdefault(r["seq"], []).append(r)

    # monitor verdict on the raw faulty output (unchanged by the sanitizer)
    for s, rs in by_seq.items():
        gs = g2[s]
        _, snaps = track_split(gs, None, {int(r["frame"]) for r in rs})
        for r in rs:
            t = int(r["frame"])
            d, st = store[(int(r["id"]), C.SEQS.index(s), t)]
            f = extract(snaps[t], d, finite(gs["dets"][t - 1]), t, gs["info"]["width"], gs["info"]["height"])
            f["stats"] = st
            r["flag"] = sc.cal.score(f, "combined") > thr

    ev_root = C.RESULTS / "sanitize"
    res = {}
    for tau in TAUS:
        tag = "none" if tau is None else f"nms{tau:g}"
        dmg, harmful = [], 0
        for s, rs in by_seq.items():
            gs = g2[s]
            outs, snaps = track_split(gs, tau, {int(r["frame"]) for r in rs})
            for r in rs:
                t = int(r["frame"])
                d, _ = store[(int(r["id"]), C.SEQS.index(s), t)]
                frames = list(range(t, min(t + win, gs["b"] + 1)))
                ds = sanitize(d, tau)
                if same_dets(sanitize(gs["dets"][t], tau), ds):
                    r[tag] = 0.0
                    continue
                trk = copy.deepcopy(snaps[t])
                fo = [trk.step(f, finite(ds if f == t else sanitize(gs["dets"][f], tau))) for f in frames]
                cmp = compare_tracks([outs[f] for f in frames], fo)
                r[tag] = float(cmp["fn"] + cmp["fp"] + cmp["idsw"])
        # clean cost
        for s in C.SEQS:
            gs = g2[s]
            outs, _ = track_split(gs, tau)
            write_mot(ev_root / tag / tag / "data" / f"{s}.txt",
                      [(f - gs["a"] + 1, *o) for f in range(gs["a"], gs["b"] + 1) for o in outs[f]])
        res[tag] = evaluate("q2", ev_root / tag, [tag])[tag]
        print(f"  {tag}: HOTA {res[tag]['HOTA']:.2f}", flush=True)

    n_raw = sum(r["outcome"] in ("sdc_local", "sdc_prop") for r in changed)
    base = res["none"]
    L = [f"# Step 8 report: output sanitizer (class-agnostic NMS before the tracker), {C.RESULTS.name}", "",
         f"activation faults that changed the detector output: {len(changed)} (of {len(act)}); "
         f"replay window {win} frames", "",
         "  clean = fault-free tracker on q2 with the sanitizer; damage = FN + FP + ID switches over the window",
         "  against the sanitized fault-free run; residual = damage of faults the combined monitor misses at 1%", "",
         f"  {'sanitizer':<10}{'HOTA':>7}{'dHOTA':>7}{'IDF1':>7}{'MOTA':>7}{'harmful faults':>16}{'damage':>8}"
         f"{'residual':>10}   damage by effect: missing / added / both / none"]
    for tau in TAUS:
        tag = "none" if tau is None else f"nms{tau:g}"
        h = [r for r in changed if r[tag] > 0]
        tot = sum(r[tag] for r in changed)
        resid = sum(r[tag] for r in changed if not r["flag"])
        byk = " / ".join(f"{sum(r[tag] for r in changed if kind(r) == k):.0f}"
                         for k in ("missing only", "added only", "both", "no confident change"))
        e = res[tag]
        L.append(f"  {tag:<10}{e['HOTA']:7.2f}{e['HOTA'] - base['HOTA']:7.2f}{e['IDF1']:7.2f}{e['MOTA']:7.2f}"
                 f"{len(h):16d}{tot:8.0f}{resid:10.0f}   {byk}")
    L += ["", f"  check: harmful faults without sanitizer in the step 2 campaign = {n_raw} "
              f"(should equal the 'none' row up to float tolerance in the replay)",
          "  development numbers; tau must be confirmed on the sealed test split.", ""]
    (C.RESULTS / "step8_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def run_all():
    from run_grid import GRID, results_dir
    root = Path(__file__).resolve().parent
    out = []
    for cfg in GRID:
        env = dict(os.environ, PC_DATASET=cfg[0], PC_DETECTOR=cfg[1], PC_TRACKER=cfg[2])
        print(f"=== {'/'.join(cfg)}", flush=True)
        p = subprocess.run([sys.executable, str(root / "step8_sanitize.py")], cwd=root, env=env)
        d = results_dir(cfg)
        f = d / "step8_report.txt" if d else None
        out.append(f"{'=' * 90}\n### {'/'.join(cfg)}\n" + (f.read_text(encoding="utf-8") if f and f.exists() and p.returncode == 0
                                                          else f"(failed, exit code {p.returncode})"))
    (root / "runs").mkdir(exist_ok=True)
    (root / "runs" / "step8_all.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {root / 'runs' / 'step8_all.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    run_all() if a.all else one()
