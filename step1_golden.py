#!/usr/bin/env python3
"""
Step 1: golden (fault-free) runs and monitor calibration.

For the q1 and q2 splits of every MOT17 training sequence:
  1. run the detector with recording hooks: detections + per-layer activation min/max
  2. run the tracker on the golden detections and compute every monitor statistic per frame
Then, using q1 only:
  - fit the Calibrator (empirical p-values, Fisher thresholds) -> results/calibration.json
  - profile per-layer activation ranges (range detector, range-restriction baseline)
    -> results/range_bounds.npz
and on q2 measure how well the q1 thresholds hold: realized false alarm rate on clean frames.

Usage:  python step1_golden.py --mot17 E:\\Datasets\\MOT17
Output: cache/golden/<split>/<seq>.npz, results/clean_<split>.csv, results/step1_report.txt
"""
import argparse
import csv
import json
import time

import numpy as np

import common as C
from detector import make_detector
from faults import same_dets
from monitors import FAMILIES, FEATURES, Calibrator, extract, finite, range_violation
from trackers import make_tracker


def golden_split(det, loader, mot17, split, backend):
    out = {}
    for s in C.SEQS:
        info = C.seqinfo(mot17, s)
        a, b = C.split_range(info["length"], split)
        dets, amin, amax, ms = {}, [], [], []
        for f in range(a, b + 1):
            img = loader(s, f) if backend != "fake" else None
            t = time.perf_counter()
            d, st = det.infer(img, key=(s, f), record=True)
            ms.append(1000 * (time.perf_counter() - t))
            if d is None:
                raise RuntimeError(f"fault-free inference failed on {s} frame {f}")
            dets[f] = d; amin.append(st["amin"]); amax.append(st["amax"])
        z = C.pack_dets(dets)
        z.update(amin=np.array(amin, np.float32), amax=np.array(amax, np.float32), infer_ms=np.array(ms, np.float32))
        p = C.golden_path(split, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **z)
        out[s] = (dets, z["amin"], z["amax"], info, a, b, float(np.median(ms)))
        print(f"  {split} {s}: frames {a}-{b}, median {np.median(ms):.1f} ms", flush=True)
    return out


def clean_features(gold):
    rows, nis = [], []
    for s, (dets, _, _, info, a, b, _) in gold.items():
        trk, prev = make_tracker(), np.zeros((0, 5), np.float32)
        for f in range(a, b + 1):
            if f >= a + C.WARMUP:
                r = extract(trk, dets[f], prev, f, info["width"], info["height"])
                nis += r["nis"]
                r.update(seq=s, frame=f)
                rows.append(r)
            trk.step(f, finite(dets[f]))
            prev = finite(dets[f])
    return rows, nis


def write_rows(path, rows, extra=()):
    cols = ["seq", "frame"] + FEATURES + list(extra)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--backend", default="yolo", choices=["yolo", "fake"])
    ap.add_argument("--weights", default=C.WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=C.IMGSZ)
    ap.add_argument("--fp32", action="store_true", help="run in FP32 instead of FP16")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    det = make_detector(args.backend, weights=args.weights, imgsz=args.imgsz, half=not args.fp32, device=args.device)
    loader = C.FrameLoader(args.mot17)

    # determinism: the same frame three times must give identical output
    s0 = C.SEQS[0]
    img = loader(s0, 1) if args.backend != "fake" else None
    runs = [det.infer(img, key=(s0, 1))[0] for _ in range(3)]
    deterministic = all(same_dets(runs[0], r, tol=0.0) for r in runs[1:])
    near = all(same_dets(runs[0], r) for r in runs[1:])

    gold = {sp: golden_split(det, loader, args.mot17, sp, args.backend) for sp in ("q1", "q2")}
    rows1, nis1 = clean_features(gold["q1"])
    rows2, _ = clean_features(gold["q2"])
    cal = Calibrator.fit(rows1, nis1)
    cal.save(C.RESULTS / "calibration.json")
    lo = np.min(np.concatenate([g[1] for g in gold["q1"].values()]), 0)
    hi = np.max(np.concatenate([g[2] for g in gold["q1"].values()]), 0)
    np.savez(C.RESULTS / "range_bounds.npz", lo=lo, hi=hi)
    for rows, sp in ((rows1, "q1"), (rows2, "q2")):
        for r in rows:
            s, f = r["seq"], r["frame"]
            dets, amin, amax, info, a, b, _ = gold[sp][s]
            r["range_v"] = range_violation({"amin": amin[f - a], "amax": amax[f - a]}, lo, hi)
            for fam in FAMILIES:
                r["X_" + fam] = cal.score(r, fam)
        write_rows(C.RESULTS / f"clean_{sp}.csv", rows, ["range_v"] + ["X_" + f for f in FAMILIES])
    meta = dict(det.meta, layers=det.names, act_numel=det.act_numel.tolist(), w_numel=det.w_numel.tolist(),
                nbits=det.nbits, backend=args.backend)
    (C.RESULTS / "detector_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    L = ["# Step 1 report: golden runs and calibration", "",
         f"detector: {det.meta.get('weights')} imgsz={args.imgsz} dtype={det.dtype} conv layers={det.L}",
         f"activation elements per 1080p frame: {int(det.act_numel.sum()):,}   conv weights: {int(det.w_numel.sum()):,}",
         f"deterministic (bit-exact over 3 runs): {deterministic}   within 0.01 px: {near}",
         "median inference ms per split: " + ", ".join(
             f"{sp} " + "/".join(f"{g[6]:.1f}" for g in gold[sp].values()) for sp in gold),
         f"clean frames after warm-up: q1 {len(rows1)}, q2 {len(rows2)}   matched track innovations in q1: {len(nis1)}",
         "", "Clean statistics (mean per frame, q1 / q2):"]
    for k in ["n_det", "n_tracks", "n_match", "n_birth", "n_death", "t_birth", "t_death", "n_invalid", "nis_max"]:
        L.append(f"  {k:<10} {np.mean([r[k] for r in rows1]):8.3f} / {np.mean([r[k] for r in rows2]):8.3f}")
    L += ["", "False alarm rate on clean q2 frames with thresholds fixed on q1 (target -> realized):",
          "  family      alpha=1%    alpha=0.1%"]
    for fam in FAMILIES:
        vals = [float(np.mean([r["X_" + fam] > cal.threshold(fam, a) for r in rows2])) for a in (0.01, 0.001)]
        L.append(f"  {fam:<10} {100 * vals[0]:8.2f}%   {100 * vals[1]:8.3f}%")
    rv = np.array([r["range_v"] for r in rows2])
    L.append(f"  range det. (any activation outside q1 range): {100 * np.mean(rv > 0):.2f}% of q2 frames")
    L += ["", f"time: {time.monotonic() - t0:.0f}s"]
    (C.RESULTS / "step1_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
