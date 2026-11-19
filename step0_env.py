#!/usr/bin/env python3
"""
Step 0: environment check and unit tests of the fault injector on the real GPU.

Checks, in order:
  1. torch / CUDA / ultralytics versions, MOT17 folder
  2. bit flips on known values (FP16 and FP32, CPU and GPU): 1.0 -> -1.0 (sign), -> inf (exponent MSB),
     -> 1.0009765625 (FP16 mantissa LSB); flipping twice restores the exact bits
  3. the detector runs in the requested precision, all conv layers are hooked
  4. fault-free inference is deterministic, injected activation faults change the output, and the
     next fault-free call is clean again (no leftover fault)
  5. a weight bit flip changes the checksum and is undone exactly by the second flip
  6. range restriction with bounds taken from the same frame leaves the output unchanged
  7. timing of plain / recording / clamped inference

Usage:  python step0_env.py --mot17 E:\\Datasets\\MOT17
Output: results/step0_report.txt  (every line must say OK before running step 1)
"""
import argparse
import time

import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--weights", default=C.WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=C.IMGSZ)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    L, ok_all = [], True

    def check(name, ok, info=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        L.append(f"[{'OK' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))
        print(L[-1], flush=True)

    import torch
    import ultralytics
    cuda = torch.cuda.is_available()
    check("torch + CUDA", cuda, f"torch {torch.__version__}, ultralytics {ultralytics.__version__}, "
          f"{torch.cuda.get_device_name(0) if cuda else 'no GPU'}")
    missing = [s for s in C.SEQS if not C.seq_dir(args.mot17, s).exists()]
    check(f"{C.DATASET} sequences found", not missing, f"missing: {missing}" if missing else args.mot17)
    if missing:
        C.RESULTS.mkdir(parents=True, exist_ok=True)
        (C.RESULTS / "step0_report.txt").write_text("\n".join(L), encoding="utf-8")
        raise SystemExit(f"dataset not found under {args.mot17}")

    from detector import FaultyYOLO, _xor_bits
    for dev in ["cpu"] + (["cuda"] if cuda else []):
        x = torch.tensor([1.0, 1.0, 1.0, 2.0], dtype=torch.float16, device=dev)
        ref = x.clone()
        _xor_bits(x, torch.tensor([0, 1, 2], device=dev), [15, 14, 0])
        v = x.float().cpu().tolist()
        good = v[0] == -1.0 and np.isinf(v[1]) and v[2] == 1.0009765625 and v[3] == 2.0
        _xor_bits(x, torch.tensor([0, 1, 2], device=dev), [15, 14, 0])
        check(f"FP16 bit flips on {dev}", good and torch.equal(x.view(torch.int16), ref.view(torch.int16)), str(v))
        y = torch.tensor([1.0, 1.0], dtype=torch.float32, device=dev)
        _xor_bits(y, torch.tensor([0, 1], device=dev), [31, 30])
        v = y.cpu().tolist()
        check(f"FP32 bit flips on {dev}", v[0] == -1.0 and np.isinf(v[1]), str(v))

    det = FaultyYOLO(args.weights, args.imgsz, half=not args.fp32, device=args.device, classes=tuple(C.CLASSES))
    want = "float32" if args.fp32 else "float16"
    check("detector precision", det.dtype == want, f"{det.dtype}, network at {det.net_path}, {det.L} conv layers, "
          f"{int(det.act_numel.sum()):,} activations at 1080p, {int(det.w_numel.sum()):,} weights")
    check("every conv layer saw activations", bool((det.act_numel > 0).all()))

    img = C.FrameLoader(args.mot17)(C.SEQS[min(1, len(C.SEQS) - 1)], 1)
    clean = [det.infer(img)[0] for _ in range(3)]
    exact = all(np.array_equal(clean[0], c) for c in clean[1:])
    check("fault-free inference deterministic", exact, f"{len(clean[0])} detections")

    head = det.L - 1
    rng = np.random.default_rng(0)
    for layer in sorted({0, det.L // 2, head}):
        sites = [(layer, float(u), 14) for u in rng.random(200)]
        d, _ = det.infer(img, act_sites=sites)
        changed = d is None or not np.array_equal(d, clean[0])
        check(f"200 exponent faults in layer {layer} change the output", changed, det.names[layer])
    after = det.infer(img)[0]
    check("no leftover fault after an injected call", np.array_equal(after, clean[0]))

    ck = det.weight_checksum()
    det.flip_weight(0, 0.37, 14)
    ck_f = det.weight_checksum()
    det.flip_weight(0, 0.37, 14)
    check("weight flip changes checksum and is undone exactly", ck_f != ck and det.weight_checksum() == ck)
    check("output after weight restore equals clean", np.array_equal(det.infer(img)[0], clean[0]))

    d, st = det.infer(img, record=True)
    check("recording hooks: per-layer ranges finite", st["amax"].shape == (det.L,) and np.isfinite(st["amax"]).all())
    det.set_bounds(st["amin"].astype(float), st["amax"].astype(float))
    dc, _ = det.infer(img, clip=True)
    check("range restriction with own bounds leaves output unchanged", np.allclose(dc, clean[0], atol=1e-2))

    T = {}
    for name, kw in (("plain", {}), ("record", {"record": True}), ("clamp", {"clip": True})):
        ts = []
        for _ in range(20):
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter(); det.infer(img, **kw)
            if cuda:
                torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t))
        T[name] = np.median(ts)
    L.append("timing (median ms per 1080p frame): " + ", ".join(f"{k} {v:.1f}" for k, v in T.items()))
    L.append("ALL OK" if ok_all else "SOME CHECKS FAILED: send this report before continuing")
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "step0_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-2:]))
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
