#!/usr/bin/env python3
"""
Step 10: baselines and bit-model robustness on the sealed test split.

Step 9 compared the frozen system with no protection, DMR and an oracle. A reviewer will also ask
(1) how it compares with range restriction, the standard cheap defence against activation faults,
(2) whether it adds anything on top of range restriction, (3) how it compares with a white-box
activation-range monitor, and (4) whether the results depend on the exponent-bit fault model.
Everything below is fixed before this script touches the test split (written to
sealed/baselines_protocol.json) and uses only settings chosen on q1 / q2 in steps 2-6:

  ranger           every conv output clamped to the q1 activation range widened by its own width on
                   each side ("wide" clamp of step 2; the tight q1 range changed 19% of clean outputs
                   on q2, so it is not a usable deployment setting). Always on, no re-execution.
  ranger+combined  ranger, plus the step 9 combined monitor (fixed threshold, alpha 1%) with one
                   re-execution on alarm. The wide clamp leaves clean outputs unchanged (checked in
                   phase A), so the step 9 calibration still applies.
  range            white-box monitor of step 3: activation range violation, profiled on the first half
                   of q1 and thresholded on the second half at alpha 1%; one re-execution on alarm.
  bit model 'all'  step 9 policies (none / dmr / oracle / combined / protected) with the flipped bit
                   uniform over all 16 bits instead of the exponent and sign bits, at rate 1.

Common random numbers: the fault process depends only on (seed, rate, sequence, frame, attempt), as in
step 9, so the 'exp' runs here see exactly the faults of the step 9 runs, and the step 9 numbers for
none / combined are shown next to the baselines.

Phases (skipped if their output exists, unless --force):
  A clamp check   wide clamp on every 5th clean test frame: share of outputs it changes (GPU)
  B baselines     exp bit model, rates 0.1 / 0.3 / 1, 3 seeds: range, ranger, ranger+combined (GPU)
  C all-bit       bit model 'all', rate 1, 3 seeds: none, dmr, oracle, combined, protected (GPU)
  D analysis      clean false alarm rate of the range monitor on test, tables (CPU)

Usage:  python step10_baselines.py            all phases
        python step10_baselines.py --analyze  phase D only
Output: results/sealed/baselines/..., results/step10_report.txt
"""
import argparse
import json
import time

import numpy as np

import common as C
from evaltools import evaluate, prepare_gt, write_mot
from faults import same_dets
from monitors import extract, finite
from step2_campaign import load_golden
from step3_analyze import Scorer, clean_rows
from step4_e2e import Executor
from step9_sealed import OUT as SEALED, PROTOCOL as P9, Monitor, calibrate, make_det, run_policy as run_policy9
from trackers import make_tracker

OUT = SEALED / "baselines"
PROTOCOL = {
    "ranger_bounds": "q1 activation range widened by its own width on each side (step 2 'wide')",
    "ranger_combined": "combined monitor, fixed split-q12 threshold, alpha 0.01, one re-execution",
    "range_monitor": "step 3 range violation, profile q1 first half, threshold q1 second half, alpha 0.01",
    "exp_rates": [0.1, 0.3, 1.0], "allbit_rates": [1.0], "seeds": 3, "clamp_check_every": 5,
}
EXP_POLS = ["range", "ranger", "ranger+combined"]
ALL_POLS = ["none", "dmr", "oracle", "combined", "protected"]


def wide_bounds():
    q1 = [np.load(C.golden_path("q1", s)) for s in C.SEQS]
    lo = np.min(np.concatenate([z["amin"] for z in q1]), 0).astype(float)
    hi = np.max(np.concatenate([z["amax"] for z in q1]), 0).astype(float)
    return lo - (hi - lo), hi + (hi - lo)


class ClipExecutor(Executor):
    """Same fault process as Executor, but every faulty inference runs with the wide clamp.
    Fault-free attempts reuse the golden output (the clamp leaves it unchanged; phase A checks this)."""

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
                d, st = self.det.infer(img, key=(s, f), act_sites=[site], record=True, clip=True)
                self.gpu_calls += 1
                self.memo[key] = (d, st, True)
        return self.memo[key]


def run_baseline(policy, E, g, mon, sc):
    """range: white-box monitor on the plain detector. ranger: clamped detector, no monitor.
    ranger+combined: clamped detector + combined monitor with one re-execution."""
    per_seq, cost, alarms, bad, nframes = {}, 0, 0, 0, 0
    thr_range = sc.thr["range"][0.01]
    for s in C.SEQS:
        gs = g[s]
        W, H = gs["info"]["width"], gs["info"]["height"]
        trk, prev, rows = make_tracker(), np.zeros((0, 5), np.float32), []

        def score(d, st, f):
            if d is None:
                return float("inf")
            feat = extract(trk, d, prev, f, W, H)
            if policy == "range":
                feat["stats"] = st
                return sc.scores(feat)["range"]
            return mon.cal.score(feat, "combined")

        for f in range(gs["a"], gs["b"] + 1):
            gold = gs["dets"][f]
            d, st, _ = E(s, f, 0)
            cost += 1
            check = policy == "range" or (policy == "ranger+combined" and f >= gs["a"] + C.WARMUP)
            if check:
                s0 = score(d, st, f)
                if s0 > (thr_range if policy == "range" else mon.threshold(s)):
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
    return per_seq, {"extra_compute": cost / nframes - 1, "alarms": alarms / nframes, "bad_used": bad / nframes}


def phase_clamp_check(a, det, loader, g):
    det.set_bounds(*wide_bounds())
    n = changed = 0
    for s in C.SEQS:
        gs = g[s]
        for f in range(gs["a"], gs["b"] + 1, PROTOCOL["clamp_check_every"]):
            img = loader(s, f) if a.backend != "fake" else None
            d, _ = det.infer(img, key=(s, f), clip=True)
            n += 1
            changed += not same_dets(gs["dets"][f], d)
    res = {"frames": n, "changed": changed}
    (OUT / "clamp_check.json").write_text(json.dumps(res), encoding="utf-8")
    print(f"  wide clamp changed {changed} of {n} clean test outputs", flush=True)


def phase_runs(a, det, loader, g, cal, sc, kind):
    nb = det.nbits
    if kind == "exp":
        bits, rates, pols, root = list(range(nb - 6, nb)), PROTOCOL["exp_rates"], EXP_POLS, OUT / "exp"
    else:
        bits, rates, pols, root = list(range(nb)), PROTOCOL["allbit_rates"], ALL_POLS, OUT / "allbit"
    tau = P9["sanitizer_tau"]
    det.set_bounds(*wide_bounds())
    results = {}
    t0 = time.monotonic()
    for rate in rates:
        for seed in range(PROTOCOL["seeds"]):
            E = Executor(det, loader, g, rate, 100 + seed, bits, a.backend)
            EC = ClipExecutor(det, loader, g, rate, 100 + seed, bits, a.backend)
            tag = f"r{rate:g}_s{seed}"
            stats = {}
            for pol in pols:
                if kind == "exp":
                    mon = Monitor(cal, P9["alpha"], 0.0)
                    per_seq, st = run_baseline(pol, EC if pol.startswith("ranger") else E, g, mon, sc)
                else:
                    mon = Monitor(cal, P9["alpha"], P9["gamma"] if pol == "protected" else 0.0)
                    per_seq, st = run_policy9(pol, E, g, mon, tau)
                for s, rows in per_seq.items():
                    write_mot(root / tag / pol / "data" / f"{s}.txt", rows)
                stats[pol] = st
            ev = evaluate("test", root / tag, pols)
            for pol in pols:
                results.setdefault(str(rate), {}).setdefault(pol, []).append(dict(ev[pol], **stats[pol]))
            print(f"  {kind} rate {rate:g} seed {seed}: " + ", ".join(f"{p} {ev[p]['HOTA']:.2f}" for p in pols)
                  + f"  ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    (root / "summary.json").write_text(json.dumps({"results": results}, indent=1), encoding="utf-8")


def table(L, results, clean_hota, extra=None):
    for rate, pols in results.items():
        L.append(f"  rate {float(rate):g}:  {'policy':<16}{'HOTA':>14}{'dHOTA':>8}{'recovered':>11}{'extra compute':>15}{'alarms':>9}")
        rows = dict((extra or {}).get(rate, {}))
        rows.update(pols)
        none = np.mean([r["HOTA"] for r in rows["none"]]) if "none" in rows else None
        for pol, runs in rows.items():
            h = np.array([r["HOTA"] for r in runs])
            m = lambda k: np.mean([r.get(k, 0.0) for r in runs])
            rec = ("" if none is None or pol == "none" or none >= clean_hota
                   else f"{100 * (h.mean() - none) / (clean_hota - none):9.1f}%")
            L.append(f"              {pol:<16}{h.mean():8.2f} ±{h.std():4.2f}{h.mean() - clean_hota:8.2f}{rec:>11}"
                     f"{100 * m('extra_compute'):14.1f}%{100 * m('alarms'):8.2f}%")


def phase_analysis(a, cal, sc):
    gt = load_golden(a.mot17, "test")
    L = [f"# Step 10 report: baselines and bit model on the sealed test split ({C.RESULTS.name})", "",
         "frozen before the test split was touched: " + json.dumps(PROTOCOL), ""]
    cc = OUT / "clamp_check.json"
    if cc.exists():
        c = json.loads(cc.read_text())
        L += ["## 1. Wide clamp on clean test frames",
              f"  outputs changed by the clamp alone: {c['changed']} of {c['frames']} sampled frames "
              f"({100 * c['changed'] / max(c['frames'], 1):.2f}%)", ""]
    rows_t, _ = clean_rows(gt)
    thr = sc.thr["range"][0.01]
    far = np.mean([sc.scores(r)["range"] > thr for r in rows_t])
    far_c = np.mean([cal.score(r, "combined") > Monitor(cal, P9["alpha"], 0.0).threshold(r["seq"]) for r in rows_t])
    L += ["## 2. False alarms on clean test frames at nominal alpha 1%",
          f"  range (white-box, q1 threshold)  {100 * far:.2f}%",
          f"  combined (black-box, split-q12)  {100 * far_c:.2f}%", ""]
    s9 = json.loads((SEALED / "e2e" / "summary.json").read_text())
    clean_hota = s9["clean"]["clean"]["HOTA"]
    ref = {rate: {p: pols[p] for p in ("none", "combined") if p in pols} for rate, pols in s9["results"].items()}
    sp = OUT / "exp" / "summary.json"
    if sp.exists():
        L.append("## 3. End-to-end, bit model 'exp' (same faults as step 9; none / combined copied from step 9)")
        L.append(f"  fault-free HOTA {clean_hota:.2f}; recovered = share of the unprotected HOTA loss that is recovered")
        L.append("  extra compute counts extra inference calls; the clamp itself (one elementwise op per conv output) is not counted")
        table(L, json.loads(sp.read_text())["results"], clean_hota, ref)
        L.append("")
    sp = OUT / "allbit" / "summary.json"
    if sp.exists():
        L.append("## 4. End-to-end, bit model 'all' (flipped bit uniform over all 16 bits)")
        table(L, json.loads(sp.read_text())["results"], clean_hota)
        L.append("")
    L.append("  sealed result: nothing in this report was used to change any setting.")
    (C.RESULTS / "step10_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--backend", default="yolo", choices=["yolo", "fake"])
    ap.add_argument("--weights", default=C.WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=C.IMGSZ)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--analyze", action="store_true", help="phase D only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not (SEALED / "e2e" / "summary.json").exists():
        raise SystemExit("run step 9 first (sealed/e2e/summary.json is missing)")
    OUT.mkdir(parents=True, exist_ok=True)
    pf = OUT / "baselines_protocol.json"
    if pf.exists() and json.loads(pf.read_text()) != PROTOCOL:
        raise SystemExit(f"protocol changed since the baseline run started ({pf}); refusing to continue")
    pf.write_text(json.dumps(PROTOCOL, indent=1), encoding="utf-8")
    t0 = time.monotonic()
    prepare_gt(a.mot17)
    cal = calibrate(a.mot17)
    sc = Scorer(load_golden(a.mot17, "q1"))
    if not a.analyze:
        det, loader, g = make_det(a), C.FrameLoader(a.mot17, capacity=50), load_golden(a.mot17, "test")
        if a.force or not (OUT / "clamp_check.json").exists():
            print("phase A: wide clamp on clean test frames", flush=True)
            phase_clamp_check(a, det, loader, g)
        if a.force or not (OUT / "exp" / "summary.json").exists():
            print("phase B: baselines, bit model exp", flush=True)
            phase_runs(a, det, loader, g, cal, sc, "exp")
        if a.force or not (OUT / "allbit" / "summary.json").exists():
            print("phase C: bit model all", flush=True)
            phase_runs(a, det, loader, g, cal, sc, "all")
    print("phase D: analysis", flush=True)
    phase_analysis(a, cal, sc)
    print(f"step 10 done in {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
