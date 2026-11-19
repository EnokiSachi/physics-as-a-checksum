#!/usr/bin/env python3
"""
Step 7: protecting the weights with a periodic checksum.

Step 3 showed that persistent weight faults are the part the black-box monitors cannot handle well:
a flipped weight bit corrupts every later frame, but the output only looks implausible at the onset
(single-frame tests catch about half, CUSUM less). Weights do not change at inference time, so they
can be checked exactly instead of statistically.

Check: the weights are stored as 16-bit (or 32-bit) words. Summing the raw words as integers in
int64 gives a value that changes by exactly +-2^b when bit b of any single word flips, and int64
cannot overflow on 11M words, so the check detects every single bit flip with no false alarms.
Repair: reload the weights from a clean copy (host memory) and re-run the current frame.

Part 1 (--measure, needs the GPU): time of one full checksum and of one inference, on the real model.
  Two versions: per layer (one reduction per weight tensor; first run: 3.8 ms, launch-bound), and with
  all weights moved into one contiguous buffer so the check is a single reduction kernel.
Part 2 (always, no GPU): replay the weight-fault campaign of step 2. For a check every P frames, a
fault that appears at a random point is caught after d = 0..P-1 faulty inferences (equally likely);
the tracker runs on the faulty detections for those d frames and on golden detections after the
repair. Reported: share of corrupting faults that still change tracking, corrupted frames, and the
compute overhead checksum_time / (P * frame_time).

Usage:  python step7_weights.py --measure      (GPU timing, then the replay)
        python step7_weights.py                (replay only; uses saved timing if present)
Output: results/step7_report.txt, results/step7_timing.json
"""
import argparse
import copy
import json
import time

import numpy as np

import common as C
from faults import compare_tracks
from monitors import finite
from step2_campaign import golden_tracking, load_golden
from step3_analyze import read_csv, unpack

PERIODS = (1, 2, 5, 10, 25)


def int_checksum_torch(convs):
    import torch
    s = None
    for _, m in convs:
        w = m.weight.detach()
        v = w.view(torch.int16) if w.element_size() == 2 else w.view(torch.int32)
        t = v.to(torch.int64).sum()
        s = t if s is None else s + t
    return s


def flatten_weights(convs):
    """Move all weights into one contiguous buffer (each module's weight becomes a view into it), so
    the checksum is a single reduction kernel instead of one small kernel chain per layer."""
    import torch
    ws = [m.weight for _, m in convs]
    flat = torch.empty(sum(w.numel() for w in ws), dtype=ws[0].dtype, device=ws[0].device)
    off = 0
    with torch.no_grad():
        for w in ws:
            n = w.numel()
            flat[off:off + n].copy_(w.reshape(-1))
            w.data = flat[off:off + n].view_as(w)
            off += n
    return flat


def flatten_all(net):
    """Every floating-point parameter and buffer of the network (conv/linear weights, biases, norm
    layers, attention in_proj weights, ...) moved into one contiguous buffer per dtype."""
    import torch
    seen, groups = set(), {}
    tensors = list(net.parameters()) + [b for b in net.buffers() if b.is_floating_point()]
    for t in tensors:
        if id(t) in seen or not t.is_floating_point():
            continue
        seen.add(id(t))
        groups.setdefault(t.dtype, []).append(t)
    flats = []
    with torch.no_grad():
        for dt, ts in groups.items():
            flat = torch.empty(sum(t.numel() for t in ts), dtype=dt, device=ts[0].device)
            off = 0
            for t in ts:
                n = t.numel()
                flat[off:off + n].copy_(t.reshape(-1))
                t.data = flat[off:off + n].view_as(t)
                off += n
            flats.append(flat)
    return flats


def multi_checksum(flats):
    s = None
    for f in flats:
        t = flat_checksum(f)
        s = t if s is None else s + t
    return s


def _flips_caught_flat(flats, ck, ref, n=200):
    from detector import _xor_bits
    import torch
    rng = np.random.default_rng(1)
    sizes = np.array([f.numel() for f in flats], float)
    caught = 0
    for _ in range(n):
        k = int(rng.choice(len(flats), p=sizes / sizes.sum()))
        f = flats[k]
        nb = 8 * f.element_size()
        idx = torch.tensor([int(rng.integers(f.numel()))], device=f.device)
        bit = [int(rng.integers(nb))]
        _xor_bits(f, idx, bit)
        caught += int(ck().item()) != ref
        _xor_bits(f, idx, bit)
    if int(ck().item()) != ref:
        raise RuntimeError("parameters not restored")
    return caught


def flat_checksum(flat):
    import torch
    v = flat.view(torch.int16) if flat.element_size() == 2 else flat.view(torch.int32)
    return v.sum(dtype=torch.int64)


def _time(fn, n):
    import torch
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / n


def _flips_caught(det, ck, ref, n=200):
    rng = np.random.default_rng(0)
    caught = 0
    for _ in range(n):
        layer = int(rng.integers(det.L)); u = float(rng.random()); bit = int(rng.integers(det.nbits))
        det.flip_weight(layer, u, bit)
        caught += int(ck().item()) != ref
        det.flip_weight(layer, u, bit)
    if int(ck().item()) != ref:
        raise RuntimeError("weights not restored")
    return caught


def measure(args):
    from detector import make_detector
    det = make_detector(args.backend, weights=args.weights, imgsz=args.imgsz, half=not args.fp32, device=args.device)
    out = {"backend": args.backend}
    if args.backend == "fake":
        out.update(checksum_ms=0.05, frame_ms=11.5, note="synthetic numbers (fake backend)")
    else:
        import torch
        loader = C.FrameLoader(args.mot17, capacity=60)
        s = C.SEQS[0]
        imgs = [loader(s, f) for f in range(1, 41)]
        for img in imgs[:10]:
            det.infer(img)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for img in imgs:
            det.infer(img)
        torch.cuda.synchronize()
        out["frame_ms"] = 1000 * (time.perf_counter() - t0) / len(imgs)
        # 1) per-layer checksum (one reduction chain per weight tensor)
        ck1 = lambda: int_checksum_torch(det.convs)
        ref1 = int(ck1().item())
        out["checksum_layerwise_ms"] = _time(ck1, 200)
        out["flips_caught_layerwise"] = _flips_caught(det, ck1, ref1)
        out["n_layers"] = det.L
        # 2) all weights in one contiguous buffer: a single reduction kernel
        dets_before = [det.infer(img)[0] for img in imgs[:5]]
        flat = flatten_weights(det.convs)
        ck2 = lambda: flat_checksum(flat)
        ref2 = int(ck2().item())
        out["checksum_ms"] = _time(ck2, 500)
        out["flips_caught"] = _flips_caught(det, ck2, ref2)
        out["flips_tested"] = 200
        same = all(np.array_equal(a, b) for a, b in zip(dets_before, [det.infer(img)[0] for img in imgs[:5]]))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for img in imgs:
            det.infer(img)
        torch.cuda.synchronize()
        out["frame_ms_after_flatten"] = 1000 * (time.perf_counter() - t0) / len(imgs)
        out.update(outputs_unchanged_after_flatten=bool(same), n_weights=int(det.w_numel.sum()), dtype=det.dtype)
        # 3) every parameter and floating buffer (biases, norms, attention projections), not only conv/linear weights
        flats = flatten_all(det.net)
        ck3 = lambda: multi_checksum(flats)
        ref3 = int(ck3().item())
        out["n_params_all"] = int(sum(f.numel() for f in flats))
        out["checksum_convlin_ms"] = out["checksum_ms"]
        out["checksum_ms"] = _time(ck3, 500)
        out["flips_caught_all"] = _flips_caught_flat(flats, ck3, ref3)
        out["outputs_unchanged_after_flatten_all"] = bool(all(
            np.array_equal(a, b) for a, b in zip(dets_before, [det.infer(img)[0] for img in imgs[:5]])))
    (C.RESULTS / "step7_timing.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def replay(timing):
    cmeta = json.loads((C.RESULTS / "campaign" / "meta.json").read_text())
    g2 = load_golden(cmeta["args"]["mot17"], "q2")
    wsum = read_csv(C.RESULTS / "campaign" / "weight.csv")
    wstore = unpack(C.RESULTS / "campaign" / "weight_store.npz")
    frames_of = {}
    for (fid, k, f), v in wstore.items():
        frames_of.setdefault(fid, []).append((k, f, v[0]))
    sdc = [r for r in wsum if r["outcome"] == "sdc" and r["due"] == 0]
    need = {}
    for r in sdc:
        need.setdefault(r["seq"], set()).add(int(r["t0"]))
    gold_out, snaps = {}, {}
    for s, fs in need.items():
        gold_out[s], snaps[s] = golden_tracking(g2[s], fs)

    res = {P: {"still": 0.0, "corrupt_frames": 0.0} for P in PERIODS}
    base_corrupt = 0.0
    for r in sdc:
        s, t0 = r["seq"], int(r["t0"])
        gs = g2[s]
        seq = sorted(frames_of[int(r["id"])])
        frames = [f for _, f, _ in seq]
        faulty = [d for _, _, d in seq]
        gold = [gs["dets"][f] for f in frames]
        base_corrupt += int(r["n_changed"])
        cache = {}
        for P in PERIODS:
            for dly in range(P):
                if dly not in cache:
                    dl = min(dly, len(frames))
                    trk = copy.deepcopy(snaps[s][t0])
                    outs = [trk.step(f, finite(d)) for f, d in zip(frames, faulty[:dl] + gold[dl:])]
                    cmp = compare_tracks([gold_out[s][f] for f in frames], outs)
                    changed = sum(1 for a, b in zip(faulty[:dl], gold[:dl])
                                  if a is None or a.shape != b.shape or (len(a) and np.max(np.abs(a - b)) > 1e-2))
                    cache[dly] = (cmp["fn"] + cmp["fp"] + cmp["idsw"] > 0, changed)
                bad, ch = cache[dly]
                res[P]["still"] += bad / P
                res[P]["corrupt_frames"] += ch / P

    n = max(len(sdc), 1)
    ck, fr = timing.get("checksum_ms"), timing.get("frame_ms")
    L = ["# Step 7 report: weight checksum", ""]
    if ck is not None:
        if "checksum_layerwise_ms" in timing:
            L.append(f"per-layer checksum ({timing.get('n_layers', '?')} small reductions): {timing['checksum_layerwise_ms']:.3f} ms, "
                     f"{timing['flips_caught_layerwise']}/200 flips caught")
            L.append(f"weights moved into one buffer: outputs unchanged {timing['outputs_unchanged_after_flatten']}, "
                     f"inference {timing['frame_ms_after_flatten']:.2f} ms per frame afterwards")
        if "n_params_all" in timing:
            L.append(f"conv/linear weights only: {timing['n_weights']:,} values, {timing['checksum_convlin_ms']:.3f} ms; "
                     f"all parameters and buffers: {timing['n_params_all']:,} values "
                     f"({100 * timing['n_weights'] / timing['n_params_all']:.1f}% were covered before), "
                     f"{timing['flips_caught_all']}/200 random flips anywhere caught, outputs unchanged "
                     f"{timing['outputs_unchanged_after_flatten_all']}")
        L.append(f"timing ({timing.get('backend')}): integer checksum of all parameters {ck:.3f} ms, inference {fr:.2f} ms per frame"
                 + (f"; {timing['flips_caught']}/{timing['flips_tested']} random single bit flips changed the checksum"
                    if "flips_tested" in timing else "") + (f"  [{timing['note']}]" if "note" in timing else ""))
    else:
        L.append("timing: not measured yet (run with --measure on the GPU); overhead column left empty")
    L += ["", f"corrupting weight faults replayed: {len(sdc)} (of {len(wsum)}), window {int(np.median([r['n_frames'] for r in wsum]))} frames;",
          f"without any check they corrupt detector output on {base_corrupt / n:.1f} frames on average and all of them change tracking", "",
          f"  {'check every':<14}{'overhead':>10}{'still change tracking':>24}{'corrupted frames':>19}"]
    for P in PERIODS:
        ov = f"{100 * ck / (P * fr):8.2f}%" if ck is not None else f"{'-':>9}"
        L.append(f"  {str(P) + ' frames':<14}{ov:>10}{100 * res[P]['still'] / n:23.1f}%{res[P]['corrupt_frames'] / n:19.2f}")
    L += ["", "  reading: the checksum has no false alarms and catches every weight fault, harmful or not; the only",
          "  question is how often to run it. Black-box monitors stay responsible for transient (activation) faults,",
          "  which a weight checksum cannot see. Together they cover both fault types at a known cost.", ""]
    (C.RESULTS / "step7_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--backend", default="yolo", choices=["yolo", "fake"])
    ap.add_argument("--weights", default=C.WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=C.IMGSZ)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    tpath = C.RESULTS / "step7_timing.json"
    if args.measure:
        timing = measure(args)
    else:
        timing = json.loads(tpath.read_text()) if tpath.exists() else {}
    replay(timing)


if __name__ == "__main__":
    main()
