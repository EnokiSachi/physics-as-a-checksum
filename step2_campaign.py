#!/usr/bin/env python3
"""
Step 2: fault injection campaigns on the development split (q2).

Activation faults (transient, one inference):
  pick a frame t and a site (conv layer, element, bit), run the faulty inference on frame t only.
  The tracker restarts from the golden state just before t, gets the faulty detections at t and
  golden detections afterwards, for --win frames, and its output is compared with the golden run.
  The same fault is also run with two range-restriction baselines (every conv output clamped):
    r_  tight: clamp to the per-layer range seen on q1 clean frames (Ranger-style)
    w_  wide : the q1 range widened by its own width on each side (fixed in advance, not tuned)
  and the clamped fault-free output is recorded, so changes caused by clamping alone are visible.

Weight faults (persistent):
  flip one weight bit at frame t0 and keep it for --wwin frames. The tracker runs on the faulty
  detections. The bit is flipped back afterwards and the weight checksum is verified.

Everything the monitors need is stored (faulty detections and per-layer activation ranges), so the
monitors are evaluated afterwards in step3 without the GPU and can be changed without rerunning this.

Usage:  python step2_campaign.py
        (defaults: 6000 activation faults, 1500 weight faults, about 35 minutes on an RTX 3070 Ti)
Output: results/campaign/act.csv, act_store.npz, weight.csv, weight_frames.csv, weight_store.npz, meta.json
"""
import argparse
import copy
import csv
import json
import time

import numpy as np

import common as C
from detector import block_index, make_detector
from faults import classify, compare_tracks, det_diff, same_dets, sample_sites
from monitors import finite
from trackers import make_tracker

SPLIT = "q2"
EMPTY = np.zeros((0, 5), np.float32)


def load_golden(mot17, split):
    g = {}
    for s in C.SEQS:
        z = np.load(C.golden_path(split, s))
        info = C.seqinfo(mot17, s)
        a, b = C.split_range(info["length"], split)
        g[s] = {"dets": C.dets_by_frame(z), "info": info, "a": a, "b": b,
                "amin": z["amin"], "amax": z["amax"]}
    return g


def golden_tracking(g, need):
    """Tracker over the split on golden detections: per-frame outputs and the state before each frame in `need`."""
    trk, outs, snaps = make_tracker(), {}, {}
    for f in range(g["a"], g["b"] + 1):
        if f in need:
            snaps[f] = copy.deepcopy(trk)
        outs[f] = trk.step(f, finite(g["dets"][f]))
    return outs, snaps


def run_window(trk, frames, dets_seq):
    return [trk.step(f, finite(d)) for f, d in zip(frames, dets_seq)]


def eligible(g):
    return [(s, f) for s in C.SEQS for f in range(g[s]["a"] + C.WARMUP, g[s]["b"])]


class Store:
    """Variable-length detections + per-layer ranges, saved as flat arrays."""

    def __init__(self, L):
        self.keys, self.n, self.dets, self.amin, self.amax, self.L = [], [], [], [], [], L

    def add(self, key, dets, stats):
        self.keys.append(key)
        if dets is None:
            self.n.append(-1)
        else:
            self.n.append(len(dets)); self.dets.append(dets)
        nan = np.full(self.L, np.nan, np.float32)
        self.amin.append(stats["amin"] if stats is not None else nan)
        self.amax.append(stats["amax"] if stats is not None else nan)

    def save(self, path):
        np.savez_compressed(path, keys=np.array(self.keys, np.int64).reshape(len(self.keys), -1),
                            n=np.array(self.n, np.int64),
                            dets=np.concatenate(self.dets).astype(np.float32) if self.dets else EMPTY,
                            amin=np.array(self.amin, np.float32), amax=np.array(self.amax, np.float32))


def act_campaign(args, det, loader, g, bounds, rng, names):
    pool = eligible(g)
    pick = rng.choice(len(pool), size=args.n_act, replace=True)
    sites = sample_sites(rng, det.act_numel, det.nbits, args.n_act)
    faults = sorted([(pool[i][0], pool[i][1], k, sites[k]) for k, i in enumerate(pick)])
    cols = ["id", "seq", "frame", "layer", "block", "layer_name", "u", "bit", "outcome", "fn", "fp", "idsw",
            "n_missing", "n_spurious", "worst_iou", "n_gold", "n_faulty"]
    for p in ("r_", "w_"):
        cols += [p + "outcome", p + "fn", p + "fp", p + "idsw", p + "clean_same"]
    fh = open(args.out_dir / "act.csv", "w", newline="")
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    store = Store(det.L)
    t0, done = time.monotonic(), 0
    for s in C.SEQS:
        fs = [x for x in faults if x[0] == s]
        if not fs:
            continue
        gs = g[s]
        outs, snaps = golden_tracking(gs, {x[1] for x in fs})
        for _, t, k, (layer, u, bit) in fs:
            img = loader(s, t) if args.backend != "fake" else None
            gold_t = gs["dets"][t]
            frames = list(range(t, min(t + args.win, gs["b"] + 1)))
            gold_outs = [outs[f] for f in frames]
            row = {"id": k, "seq": s, "frame": t, "layer": layer, "block": block_index(names[layer]),
                   "layer_name": names[layer], "u": round(u, 6), "bit": bit}

            def outcome(d, gref, prefix=""):
                if d is None:
                    return {prefix + "outcome": "due"}
                if same_dets(gref, d):
                    return {prefix + "outcome": "masked_det", prefix + "fn": 0, prefix + "fp": 0, prefix + "idsw": 0}
                seq_d = [d] + [gs["dets"][f] for f in frames[1:]]
                cmp = compare_tracks(gold_outs, run_window(copy.deepcopy(snaps[t]), frames, seq_d))
                return {prefix + "outcome": classify(False, cmp, 1), prefix + "fn": cmp["fn"],
                        prefix + "fp": cmp["fp"], prefix + "idsw": cmp["idsw"]}

            d, st = det.infer(img, key=(s, t), act_sites=[(layer, u, bit)], record=True)
            row.update(outcome(d, gold_t))
            row.update(det_diff(gold_t, d if d is not None else EMPTY))
            store.add((k, C.SEQS.index(s), t), d, st)
            for p, b in ((() if args.no_clamp else (("r_", "tight"), ("w_", "wide")))):
                det.set_bounds(*bounds[b])
                dr, _ = det.infer(img, key=(s, t), act_sites=[(layer, u, bit)], clip=True)
                dc, _ = det.infer(img, key=(s, t), clip=True)
                row.update(outcome(dr, gold_t, p))
                row[p + "clean_same"] = int(same_dets(gold_t, dc))
            w.writerow(row)
            done += 1
            if done % 500 == 0:
                fh.flush()
                el = time.monotonic() - t0
                print(f"  act {done}/{len(faults)}  {el / 60:.1f} min, eta {el / done * (len(faults) - done) / 60:.1f} min", flush=True)
    fh.close()
    store.save(args.out_dir / "act_store.npz")
    return done


def weight_campaign(args, det, loader, g, rng, names):
    pool = eligible(g)
    pick = rng.choice(len(pool), size=args.n_weight, replace=True)
    sites = sample_sites(rng, det.w_numel, det.nbits, args.n_weight)
    faults = sorted([(pool[i][0], pool[i][1], k, sites[k]) for k, i in enumerate(pick)])
    base_ck = det.weight_checksum()
    fsum = open(args.out_dir / "weight.csv", "w", newline="")
    ffr = open(args.out_dir / "weight_frames.csv", "w", newline="")
    ws = csv.DictWriter(fsum, fieldnames=["id", "seq", "t0", "layer", "block", "layer_name", "u", "idx", "bit",
                                          "n_frames", "outcome", "fn", "fp", "idsw", "first_changed", "n_changed", "due"])
    wf = csv.DictWriter(ffr, fieldnames=["id", "k", "frame", "changed", "n_missing", "n_spurious"])
    ws.writeheader(); wf.writeheader()
    store = Store(det.L)
    t0, done = time.monotonic(), 0
    for s in C.SEQS:
        fs = [x for x in faults if x[0] == s]
        if not fs:
            continue
        gs = g[s]
        outs, snaps = golden_tracking(gs, {x[1] for x in fs})
        for _, t, k, (layer, u, bit) in fs:
            frames = list(range(t, min(t + args.wwin, gs["b"] + 1)))
            idx = det.flip_weight(layer, u, bit)
            trk = copy.deepcopy(snaps[t])
            fouts, changed, due = [], [], 0
            try:
                for kk, f in enumerate(frames):
                    img = loader(s, f) if args.backend != "fake" else None
                    d, st = det.infer(img, key=(s, f), record=True)
                    if d is None:
                        due += 1
                    ch = not same_dets(gs["dets"][f], d)
                    changed.append(ch)
                    dd = det_diff(gs["dets"][f], d if d is not None else EMPTY)
                    wf.writerow({"id": k, "k": kk, "frame": f, "changed": int(ch),
                                 "n_missing": dd["n_missing"], "n_spurious": dd["n_spurious"]})
                    store.add((k, kk, f), d, st)
                    fouts.append(trk.step(f, finite(d)))
            finally:
                det.flip_weight(layer, u, bit)
            cmp = compare_tracks([outs[f] for f in frames], fouts)
            oc = classify(not any(changed), cmp, len(frames))
            ws.writerow({"id": k, "seq": s, "t0": t, "layer": layer, "block": block_index(names[layer]),
                         "layer_name": names[layer], "u": round(u, 6), "idx": idx, "bit": bit,
                         "n_frames": len(frames), "outcome": "sdc" if oc.startswith("sdc") else oc,
                         "fn": cmp["fn"], "fp": cmp["fp"], "idsw": cmp["idsw"],
                         "first_changed": changed.index(True) if any(changed) else -1,
                         "n_changed": int(sum(changed)), "due": due})
            done += 1
            if done % 100 == 0:
                fsum.flush(); ffr.flush()
                if det.weight_checksum() != base_ck:
                    raise RuntimeError("weights were not restored exactly; stopping")
                el = time.monotonic() - t0
                print(f"  weight {done}/{len(faults)}  {el / 60:.1f} min, eta {el / done * (len(faults) - done) / 60:.1f} min", flush=True)
    fsum.close(); ffr.close()
    if det.weight_checksum() != base_ck:
        raise RuntimeError("weights were not restored exactly")
    store.save(args.out_dir / "weight_store.npz")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--backend", default="yolo", choices=["yolo", "fake"])
    ap.add_argument("--weights", default=C.WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=C.IMGSZ)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--kind", nargs="+", default=["act", "weight"], choices=["act", "weight"])
    ap.add_argument("--n-act", type=int, default=6000)
    ap.add_argument("--n-weight", type=int, default=1500)
    ap.add_argument("--win", type=int, default=30, help="tracking window after a transient fault (frames)")
    ap.add_argument("--wwin", type=int, default=25, help="frames a persistent weight fault stays active")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--split", default=SPLIT, choices=["q2", "test"])
    ap.add_argument("--out", default=None, help="output folder (default results/campaign)")
    ap.add_argument("--no-clamp", action="store_true", help="skip the two range-restriction baselines")
    args = ap.parse_args()
    from pathlib import Path
    args.out_dir = Path(args.out) if args.out else C.RESULTS / "campaign"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_det = json.loads((C.RESULTS / "detector_meta.json").read_text())
    det = make_detector(args.backend, weights=args.weights, imgsz=args.imgsz, half=not args.fp32, device=args.device)
    if det.names != meta_det["layers"]:
        raise SystemExit("detector differs from the one used in step 1 (layer list mismatch)")
    g = load_golden(args.mot17, args.split)
    q1 = [np.load(C.golden_path("q1", s)) for s in C.SEQS]
    lo = np.min(np.concatenate([z["amin"] for z in q1]), 0).astype(float)
    hi = np.max(np.concatenate([z["amax"] for z in q1]), 0).astype(float)
    bounds = {"tight": (lo, hi), "wide": (lo - (hi - lo), hi + (hi - lo))}
    loader = C.FrameLoader(args.mot17)
    t0 = time.monotonic()
    counts = {}
    if "act" in args.kind:
        counts["act"] = act_campaign(args, det, loader, g, bounds, np.random.default_rng(args.seed), det.names)
    if "weight" in args.kind:
        counts["weight"] = weight_campaign(args, det, loader, g, np.random.default_rng(args.seed + 1), det.names)
    meta = {"args": {k: (str(v) if k == "out_dir" else v) for k, v in vars(args).items()}, "counts": counts, "minutes": round((time.monotonic() - t0) / 60, 1),
            "detector": det.meta, "seqs": C.SEQS}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"done: {counts}, {meta['minutes']} min")


if __name__ == "__main__":
    main()
