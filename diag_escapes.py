#!/usr/bin/env python3
"""
diag_escapes.py: why do the black-box monitors miss some propagated faults, and which additional
statistics would catch them without raising false alarms? CPU only; reads the step 2 campaign.

Motivation: on RT-DETR the combined monitor caught 60% of propagated transient faults at 1% false
alarms (YOLO models: 83-96%). The misses were faults that only ADD confident boxes (14/14 missed) or
change nothing confident (12/12 missed). monitors.py now computes six candidate statistics aimed at
added boxes (n_dup, n_crowd, n_birth_mid, lo_birth, t_dn_lo, score_drop); none is used by any monitor
yet. This script measures, for the current configuration:

  1. what the missed faults look like at detector level (score and overlap of the added boxes)
  2. each statistic alone: clean false alarm rate on q2 at its q1 99th percentile, and the share of
     propagated faults / of escapes it flags
  3. an extended monitor combined+ = combined + the six statistics (Fisher's method, q1-calibrated),
     against combined, at alpha 1% and 0.1%
  4. damage-weighted coverage: share of the total tracking damage (FN + FP + ID switches over the
     window) that comes from faults the combined monitor flags, for propagated faults and for all
     faults that changed tracking (sdc_local + sdc_prop); counting faults treats a fault that costs
     one ID switch the same as one that loses ten tracks

Development data only (q1 calibration, q2 evaluation, as in step 3). A statistic chosen here must be
confirmed on the sealed test split later, so this is not a final result.

Usage:  python diag_escapes.py           current configuration (PC_DATASET / PC_DETECTOR / PC_TRACKER)
        python diag_escapes.py --all     every configuration of run_grid.py -> runs/diag_escapes_all.txt
Output: <results>/diag_escapes.txt
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import common as C
import monitors as M
from faults import HIGH as F_HIGH
from trackers.bytetrack import iou_matrix, match

NEW_STATS = ["n_dup", "n_crowd", "n_birth_mid", "lo_birth", "t_dn_lo", "score_drop"]
M.FAMILIES["combined+"] = M.FAMILIES["combined"] + [(k, "up") for k in NEW_STATS]

from step2_campaign import golden_tracking, load_golden  # noqa: E402
from step3_analyze import ALPHAS, Scorer, auc, clean_rows, read_csv, unpack, wilson  # noqa: E402


def added_boxes(gold, faulty):
    """Confident faulty boxes not matched to a confident golden box (IoU 0.5): score, best IoU with any golden box."""
    g = gold[gold[:, 4] >= F_HIGH]
    f = faulty[np.isfinite(faulty).all(1)] if faulty is not None else np.zeros((0, 5))
    f = f[f[:, 4] >= F_HIGH]
    _, _, uf = match(1 - iou_matrix(g[:, :4], f[:, :4]), 0.5)
    ga = gold[gold[:, 4] >= 0.1]
    io = iou_matrix(f[list(uf), :4], ga[:, :4]) if len(uf) else np.zeros((0, len(ga)))
    best = io.max(1) if io.shape[1] else np.zeros(len(uf))
    return [(float(f[j, 4]), float(b)) for j, b in zip(uf, best)]


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
    root = cmeta["args"]["mot17"]
    g1, g2 = load_golden(root, "q1"), load_golden(root, "q2")
    sc = Scorer(g1)
    rows1, _ = clean_rows(g1)
    clean2, _ = clean_rows(g2)

    act = read_csv(C.RESULTS / "campaign" / "act.csv")
    harm = [r for r in act if r["outcome"] in ("sdc_local", "sdc_prop")]
    crit = [r for r in harm if r["outcome"] == "sdc_prop"]
    store = unpack(C.RESULTS / "campaign" / "act_store.npz")
    by_seq = {}
    for r in harm:
        by_seq.setdefault(r["seq"], []).append(r)
    for s, rs in by_seq.items():
        gs = g2[s]
        _, snaps = golden_tracking(gs, {int(r["frame"]) for r in rs})
        for r in rs:
            t = int(r["frame"])
            d, st = store[(int(r["id"]), C.SEQS.index(s), t)]
            f = M.extract(snaps[t], d, M.finite(gs["dets"][t - 1]), t, gs["info"]["width"], gs["info"]["height"])
            f["stats"] = st
            r["feat"] = f
            r["added"] = added_boxes(gs["dets"][t], d)
    thr = {fam: {a: sc.cal.threshold(fam, a) for a in ALPHAS} for fam in ("combined", "combined+")}
    for r in harm:
        r["S"] = {fam: sc.cal.score(r["feat"], fam) for fam in thr}
    esc = [r for r in crit if r["S"]["combined"] <= thr["combined"][0.01]]
    n = max(len(crit), 1)

    L = [f"# Escape diagnosis: {C.RUN if hasattr(C, 'RUN') else C.RESULTS.name}", "",
         f"propagated transient faults {len(crit)}; missed by combined at 1%: {len(esc)}", ""]

    L.append("## 1. Missed faults by detector-level effect (added = confident box with no golden match)")
    for k in ("missing only", "added only", "both", "no confident change"):
        a = [r for r in crit if kind(r) == k]
        e = [r for r in esc if kind(r) == k]
        L.append(f"  {k:<22} missed {len(e):>3} / {len(a):>3}")
    add = [b for r in esc for b in r["added"]]
    if add:
        sc_ = np.array([b[0] for b in add]); io = np.array([b[1] for b in add])
        L += ["", f"  added boxes in missed faults: {len(add)}; score median {np.median(sc_):.2f}, "
                  f"share with score < 0.6: {100 * np.mean(sc_ < 0.6):.0f}%",
              f"  overlap with a real (golden) box: IoU >= 0.2 in {100 * np.mean(io >= 0.2):.0f}%, "
              f"IoU >= 0.5 (duplicate) in {100 * np.mean(io >= 0.5):.0f}%"]
    L.append("")

    L.append("## 2. Each statistic alone (threshold = q1 clean 99th percentile, strict exceedance)")
    L.append(f"  {'statistic':<13}{'q1 p99':>9}{'clean q2 FAR':>14}{'all propagated':>16}{'escapes':>10}")
    for k in ["n_birth", "n_death_hi", "t_birth", "t_dn"] + NEW_STATS:
        ref = np.array([r[k] for r in rows1], float)
        t = float(np.quantile(ref, 0.99, method="higher"))
        far = np.mean([r[k] > t for r in clean2])
        tp = np.mean([r["feat"][k] > t for r in crit]) if crit else 0.0
        te = np.mean([r["feat"][k] > t for r in esc]) if esc else 0.0
        L.append(f"  {k:<13}{t:9.2f}{100 * far:13.2f}%{100 * tp:15.1f}%{100 * te:9.1f}%")
    L.append("")

    L.append("## 3. Extended monitor (q1-calibrated, Fisher), clean q2 FAR and propagated-fault TPR")
    L.append(f"  {'monitor':<11}{'AUC':>7}" + "".join(f"{'TPR@' + str(100 * a).rstrip('0').rstrip('.') + '%':>22}{'FAR':>8}" for a in ALPHAS))
    for fam in ("combined", "combined+"):
        neg = np.array([sc.cal.score(r, fam) for r in clean2])
        pos = np.array([r["S"][fam] for r in crit])
        cells = ""
        for a in ALPHAS:
            k = int(np.sum(pos > thr[fam][a])); lo, hi = wilson(k, n)
            cells += f"{100 * k / n:9.1f} [{100 * lo:4.1f},{100 * hi:5.1f}]{100 * np.mean(neg > thr[fam][a]):7.2f}%"
        L.append(f"  {fam:<11}{auc(neg, pos) if len(pos) else float('nan'):7.3f}{cells}")
    L += ["", "  development numbers (q1 calibration, q2 evaluation); a monitor adopted from this table must be",
          "  confirmed on the sealed test split before it is reported as a result.", ""]

    L.append("## 4. Damage-weighted coverage of the combined monitor (damage = FN + FP + ID switches over the window)")
    L.append(f"  {'faults':<26}{'n':>5}{'damage':>9}  " + "".join(f"{'@' + str(100 * a).rstrip('0').rstrip('.') + '%: faults / damage caught':>34}" for a in ALPHAS)
             + f"{'median damage caught / missed (1%)':>38}")
    dmg = lambda r: float(r["fn"]) + float(r["fp"]) + float(r["idsw"])
    for lab, rs in (("propagated (sdc_prop)", crit), ("all tracking changes", harm)):
        if not rs:
            continue
        tot = sum(dmg(r) for r in rs)
        cells = ""
        for a in ALPHAS:
            hit = [r for r in rs if r["S"]["combined"] > thr["combined"][a]]
            cells += f"{100 * len(hit) / len(rs):23.1f}% / {100 * sum(dmg(r) for r in hit) / max(tot, 1e-9):5.1f}%"
        hit = [dmg(r) for r in rs if r["S"]["combined"] > thr["combined"][0.01]]
        mis = [dmg(r) for r in rs if r["S"]["combined"] <= thr["combined"][0.01]]
        med = f"{(np.median(hit) if hit else float('nan')):.1f} / {(np.median(mis) if mis else float('nan')):.1f}"
        L.append(f"  {lab:<26}{len(rs):5d}{tot:9.0f}  {cells}{med:>38}")
    L.append("  by detector-level effect (propagated, alpha 1%): " + ", ".join(
        f"{k} {sum(dmg(r) for r in esc if kind(r) == k):.0f}/{sum(dmg(r) for r in crit if kind(r) == k):.0f} damage missed"
        for k in ("missing only", "added only", "both", "no confident change")))
    L.append("")
    (C.RESULTS / "diag_escapes.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def run_all():
    from run_grid import GRID, results_dir
    root = Path(__file__).resolve().parent
    out = []
    for cfg in GRID:
        env = dict(os.environ, PC_DATASET=cfg[0], PC_DETECTOR=cfg[1], PC_TRACKER=cfg[2])
        print(f"=== {'/'.join(cfg)}", flush=True)
        p = subprocess.run([sys.executable, str(root / "diag_escapes.py")], cwd=root, env=env)
        d = results_dir(cfg)
        f = d / "diag_escapes.txt" if d else None
        out.append(f"{'=' * 90}\n### {'/'.join(cfg)}\n" + (f.read_text(encoding="utf-8") if f and f.exists() and p.returncode == 0
                                                          else f"(failed, exit code {p.returncode})"))
    (root / "runs" / "diag_escapes_all.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {root / 'runs' / 'diag_escapes_all.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    run_all() if a.all else one()
