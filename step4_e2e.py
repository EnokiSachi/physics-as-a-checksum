#!/usr/bin/env python3
"""
Step 4: end-to-end protection under a continuous stream of transient faults.

Fault process: every inference call is hit, independently with probability --rates, by one bit
flip at a random activation element (chosen as in step 2). Two bit models:
  exp  the flipped bit is uniform over the FP16 exponent and sign bits (10-15). Timing errors from
       undervolting hit the long carry chains, i.e. high-order bits, so this is the stress model.
  all  uniform over all 16 bits (the random-upset model of step 2).

Protection policies, all running the same tracker:
  none        use every detector output as is
  dmr         run every frame twice; if the two outputs differ, run a third time and take the
              majority (dual modular redundancy with a tie-breaker)
  oracle      re-run exactly the frames whose output is corrupted, until it is correct
              (ideal detector; lower bound on the overhead needed for full correction)
  <monitor>   run once; if the monitor alarms (threshold at --alpha, fixed on q1 clean frames),
              re-run once and keep whichever of the two outputs the monitor scores as more normal.
              Monitors: frame, temporal, physics, combined (black-box) and range (white-box).
Monitors are not applied during the first WARMUP frames of a split, when no tracks exist yet.

Common random numbers: whether an attempt (sequence, frame, attempt number) is faulty, and where,
depends only on the seed and rate, so every policy sees exactly the same faults. Fault-free attempts
reuse the golden detections; only faulty attempts run on the GPU.

Usage:  python step4_e2e.py                       (development split q2)
Output: results/e2e/q2/... tracker files, results/step4_report.txt, results/e2e/q2/summary.json
"""
import argparse
import json
import time

import numpy as np

import common as C
from detector import make_detector
from evaltools import GT_ROOT, evaluate, prepare_gt, write_mot
from faults import same_dets
from monitors import extract, finite
from step2_campaign import load_golden
from step3_analyze import Scorer
from trackers import make_tracker

MONITOR_POLICIES = ["frame", "temporal", "physics", "combined", "range"]
POLICIES = ["none", "dmr", "oracle"] + MONITOR_POLICIES


class Executor:
    """Memoized faulty / fault-free inference with common random numbers across policies."""

    def __init__(self, det, loader, g, rate, seed, bits, backend):
        self.det, self.loader, self.g, self.rate, self.seed, self.bits, self.backend = det, loader, g, rate, seed, bits, backend
        self.p_layer = det.act_numel / det.act_numel.sum()
        self.memo, self.gpu_calls = {}, 0

    def __call__(self, s, f, attempt):
        key = (s, f, attempt)
        if key not in self.memo:
            rng = np.random.default_rng([self.seed, int(round(self.rate * 1e4)), C.SEQS.index(s), f, attempt])
            gs = self.g[s]
            if rng.random() >= self.rate:
                i = f - gs["a"]
                self.memo[key] = (gs["dets"][f], {"amin": gs["amin"][i], "amax": gs["amax"][i]}, False)
            else:
                layer = int(rng.choice(len(self.p_layer), p=self.p_layer))
                site = (layer, float(rng.random()), int(rng.choice(self.bits)))
                img = self.loader(s, f) if self.backend != "fake" else None
                d, st = self.det.infer(img, key=(s, f), act_sites=[site], record=True)
                self.gpu_calls += 1
                self.memo[key] = (d, st, True)
        return self.memo[key]


def run_policy(policy, E, g, sc, alpha):
    per_seq, cost, alarms, bad, nframes = {}, 0, 0, 0, 0
    for s in C.SEQS:
        gs = g[s]
        W, H = gs["info"]["width"], gs["info"]["height"]
        trk, prev, rows = make_tracker(), np.zeros((0, 5), np.float32), []

        def score(d, st, f):
            if d is None:
                return float("inf")
            feat = extract(trk, d, prev, f, W, H)
            feat["stats"] = st
            return sc.scores(feat)[policy]

        for f in range(gs["a"], gs["b"] + 1):
            gold = gs["dets"][f]
            d, st, _ = E(s, f, 0)
            cost += 1
            if policy == "dmr":
                d1, _, _ = E(s, f, 1)
                cost += 1
                if not same_dets(d, d1):
                    d2, _, _ = E(s, f, 2)
                    cost += 1
                    d = d if same_dets(d, d2) else (d1 if same_dets(d1, d2) else d2)
            elif policy == "oracle":
                k = 0
                while not same_dets(d, gold) and k < 8:
                    k += 1
                    d, st, _ = E(s, f, k)
                    cost += 1
            elif policy in MONITOR_POLICIES and (policy == "range" or f >= gs["a"] + C.WARMUP):
                s0 = score(d, st, f)
                if s0 > sc.thr[policy][alpha]:
                    alarms += 1
                    d1, st1, _ = E(s, f, 1)
                    cost += 1
                    if score(d1, st1, f) <= s0:
                        d = d1
            bad += not same_dets(d, gold)
            nframes += 1
            out = trk.step(f, finite(d))
            prev = finite(d)
            rows += [(f - gs["a"] + 1, *o) for o in out]
        per_seq[s] = rows
    return per_seq, {"extra_compute": cost / nframes - 1, "alarms": alarms / nframes, "bad_used": bad / nframes,
                     "frames": nframes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--backend", default="yolo", choices=["yolo", "fake"])
    ap.add_argument("--weights", default=C.WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=C.IMGSZ)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default="q2")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bits", default="exp", choices=["exp", "all"])
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES)
    ap.add_argument("--out", default=None, help="report path (default results/step4_report.txt)")
    args = ap.parse_args()
    t0 = time.monotonic()
    if not (GT_ROOT / args.split).exists():
        prepare_gt(args.mot17)
    det = make_detector(args.backend, weights=args.weights, imgsz=args.imgsz, half=not args.fp32, device=args.device)
    loader = C.FrameLoader(args.mot17, capacity=50)
    g1, g = load_golden(args.mot17, "q1"), load_golden(args.mot17, args.split)
    sc = Scorer(g1)
    nb = det.nbits
    bits = list(range(nb - 6, nb)) if args.bits == "exp" else list(range(nb))
    root = C.RESULTS / "e2e" / args.split
    results = {}
    # fault-free reference
    E0 = Executor(det, loader, g, 0.0, 0, bits, args.backend)
    ref, _ = run_policy("none", E0, g, sc, args.alpha)
    for s, rows in ref.items():
        write_mot(root / "clean" / "clean" / "data" / f"{s}.txt", rows)
    clean = evaluate(args.split, root / "clean", ["clean"])["clean"]
    for rate in args.rates:
        for seed in range(args.seeds):
            E = Executor(det, loader, g, rate, seed, bits, args.backend)
            tag = f"r{rate:g}_s{seed}"
            stats = {}
            for pol in args.policies:
                per_seq, st = run_policy(pol, E, g, sc, args.alpha)
                for s, rows in per_seq.items():
                    write_mot(root / tag / pol / "data" / f"{s}.txt", rows)
                stats[pol] = st
            ev = evaluate(args.split, root / tag, args.policies)
            for pol in args.policies:
                results.setdefault(str(rate), {}).setdefault(pol, []).append(dict(ev[pol], **stats[pol]))
            print(f"  rate {rate:g} seed {seed}: {E.gpu_calls} faulty inferences, "
                  + ", ".join(f"{p} {ev[p]['HOTA']:.2f}" for p in args.policies), flush=True)
    (root / "summary.json").write_text(json.dumps({"clean": clean, "results": results, "args": vars(args)}, indent=1),
                                       encoding="utf-8")
    L = [f"# Step 4 report: end-to-end protection ({args.split}, bit model '{args.bits}', alpha {args.alpha:g}, "
         f"{args.seeds} seeds)", "",
         f"fault-free: HOTA {clean['HOTA']:.2f}  MOTA {clean['MOTA']:.2f}  IDF1 {clean['IDF1']:.2f}", "",
         "  rate = probability that an inference call is hit by one bit flip",
         "  extra compute = additional inference calls per frame; wrong frames = frames whose used output differs from fault-free",
         ""]
    for rate, pols in results.items():
        L.append(f"## fault rate {float(rate):g} per inference")
        L.append(f"  {'policy':<10}{'HOTA':>14}{'dHOTA':>8}{'IDF1':>8}{'MOTA':>8}{'extra compute':>15}{'alarms':>9}{'wrong frames':>14}")
        for pol, runs in pols.items():
            h = np.array([r["HOTA"] for r in runs])
            m = lambda k: np.mean([r[k] for r in runs])
            L.append(f"  {pol:<10}{h.mean():8.2f} ±{h.std():4.2f}{h.mean() - clean['HOTA']:8.2f}{m('IDF1'):8.2f}{m('MOTA'):8.2f}"
                     f"{100 * m('extra_compute'):14.1f}%{100 * m('alarms'):8.2f}%{100 * m('bad_used'):13.2f}%")
        L.append("")
    L.append(f"time: {(time.monotonic() - t0) / 60:.1f} min")
    out = args.out or (C.RESULTS / "step4_report.txt")
    from pathlib import Path
    Path(out).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
