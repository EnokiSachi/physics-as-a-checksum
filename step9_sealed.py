#!/usr/bin/env python3
"""
Step 9: sealed test. Every choice is frozen before this script touches the test split, and the test
split is used once.

Test split: the second half of every sequence ("test" in common.split_range). Steps 1-8 used only
the first half (q1 for calibration, q2 for development), so nothing was tuned on these frames.

Frozen protocol (written to sealed/protocol.json before any test frame is processed):
  monitor        combined (physics + temporal statistics, Fisher), black-box
  calibration    split conformal: p-value tables from clean q1, conformal scores from clean q2
                 (step 5 "split-q12"); target false alarm rate alpha = 1% (0.1% also reported)
  adaptation     adaptive conformal inference, gamma = 0.005 (step 6); feedback = re-execution:
                 an alarm whose re-run gives the same output counts as a false alarm
  sanitizer      class-agnostic NMS at IoU 0.6 before the tracker, only for detectors without NMS
                 (RT-DETR); YOLO already runs NMS at 0.45 (step 8)
  weights        full-parameter integer checksum every frame (step 7; measured there, not here)

Phases (each is skipped if its output exists, unless --force):
  A golden     fault-free detections on the test split (GPU)
  B campaign   6000 activation faults on the test split, no clamping baselines (GPU, via step 2)
  C e2e        end-to-end under continuous faults, rates 0.1 / 0.3 / 1 per inference, 3 seeds,
               policies none / dmr / oracle / combined (fixed threshold) / protected (the frozen
               system: adaptive threshold + sanitizer where it applies) (GPU)
  D analysis   false alarms on clean test frames, detection and damage-weighted coverage of the
               test campaign, sanitizer effect, e2e table (CPU)

Usage:  python step9_sealed.py            all phases
        python step9_sealed.py --analyze  phase D only (after A-C exist)
Output: results/sealed/..., results/step9_report.txt
"""
import argparse
import copy
import json
import subprocess
import sys
import time

import numpy as np

import common as C
from evaltools import evaluate, prepare_gt, write_mot
from faults import compare_tracks, same_dets
from monitors import extract, finite
from step2_campaign import load_golden
from step3_analyze import auc, clean_rows, read_csv, unpack, wilson
from step5_calibration import block_bootstrap, conformal_threshold, fit_split
from step6_adaptive import quantile_at, run_stream
from step8_sanitize import sanitize, track_split
from trackers import make_tracker

OUT = C.RESULTS / "sealed"
PROTOCOL = {
    "monitor": "combined",
    "calibration": "split conformal: tables from clean q1, conformal scores from clean q2",
    "alpha": 0.01, "alpha_strict": 0.001, "gamma": 0.005,
    "sanitizer_tau": 0.6 if "rtdetr" in C.DETECTOR else None,
    "e2e_rates": [0.1, 0.3, 1.0], "e2e_seeds": 3, "e2e_bits": "exp",
    "n_act": 6000, "win": 30, "campaign_seed": 2027,
}
FAM = "combined"


def det_args(a):
    return ["--mot17", a.mot17, "--backend", a.backend, "--weights", a.weights, "--imgsz", str(a.imgsz),
            "--device", a.device] + (["--fp32"] if a.fp32 else [])


def make_det(a):
    from detector import make_detector
    return make_detector(a.backend, weights=a.weights, imgsz=a.imgsz, half=not a.fp32, device=a.device)


# ------------------------------------------------------------------ calibration (frozen)
def calibrate(root):
    rows1, _ = clean_rows(load_golden(root, "q1"))
    rows2, _ = clean_rows(load_golden(root, "q2"))
    return fit_split(rows1, rows2)


class Monitor:
    """Combined monitor with a fixed or an adaptive (ACI) threshold, one state per sequence."""

    def __init__(self, cal, alpha, gamma):
        self.cal, self.alpha, self.gamma = cal, alpha, gamma
        self.xref = cal.xref[FAM]
        self.a = {}

    def threshold(self, s):
        return quantile_at(self.xref, self.a.setdefault(s, self.alpha))

    def feedback(self, s, false_alarm):
        if self.gamma > 0:
            self.a[s] = self.a.setdefault(s, self.alpha) + self.gamma * (self.alpha - float(false_alarm))


# ------------------------------------------------------------------ phase C
def run_policy(policy, E, g, mon, tau):
    per_seq, cost, alarms, false_alarms, bad, nframes = {}, 0, 0, 0, 0, 0
    for s in C.SEQS:
        gs = g[s]
        W, H = gs["info"]["width"], gs["info"]["height"]
        trk, prev, rows = make_tracker(), np.zeros((0, 5), np.float32), []

        def score(d, st, f):
            if d is None:
                return float("inf")
            return mon.cal.score(extract(trk, d, prev, f, W, H), FAM)

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
            elif policy in ("combined", "protected") and f >= gs["a"] + C.WARMUP:
                s0 = score(d, st, f)
                if s0 > mon.threshold(s):
                    alarms += 1
                    d1, st1, _ = E(s, f, 1)
                    cost += 1
                    same = same_dets(d, d1)
                    false_alarms += same
                    if policy == "protected":
                        mon.feedback(s, same)
                    if score(d1, st1, f) <= s0:
                        d = d1
                elif policy == "protected":
                    mon.feedback(s, False)
            bad += not same_dets(d, gold)
            nframes += 1
            use = sanitize(d, tau) if policy == "protected" else d
            out = trk.step(f, finite(use))
            prev = finite(d)
            rows += [(f - gs["a"] + 1, *o) for o in out]
        per_seq[s] = rows
    return per_seq, {"extra_compute": cost / nframes - 1, "alarms": alarms / nframes,
                     "false_alarms": false_alarms / nframes, "bad_used": bad / nframes}


def phase_e2e(a, cal):
    from step4_e2e import Executor
    det = make_det(a)
    loader = C.FrameLoader(a.mot17, capacity=50)
    g = load_golden(a.mot17, "test")
    tau = PROTOCOL["sanitizer_tau"]
    nb = det.nbits
    bits = list(range(nb - 6, nb))
    root = OUT / "e2e"
    # fault-free references: plain, and with the sanitizer the protected system uses
    clean = {}
    for tag, t in (("clean", None), ("clean_sanitized", tau)):
        for s in C.SEQS:
            gs = g[s]
            outs, _ = track_split(gs, t)
            write_mot(root / tag / tag / "data" / f"{s}.txt",
                      [(f - gs["a"] + 1, *o) for f in range(gs["a"], gs["b"] + 1) for o in outs[f]])
        clean[tag] = evaluate("test", root / tag, [tag])[tag]
    pols = ["none", "dmr", "oracle", "combined", "protected"]
    results = {}
    for rate in PROTOCOL["e2e_rates"]:
        for seed in range(PROTOCOL["e2e_seeds"]):
            E = Executor(det, loader, g, rate, 100 + seed, bits, a.backend)
            tag = f"r{rate:g}_s{seed}"
            stats = {}
            for pol in pols:
                mon = Monitor(cal, PROTOCOL["alpha"], PROTOCOL["gamma"] if pol == "protected" else 0.0)
                per_seq, st = run_policy(pol, E, g, mon, tau)
                for s, rows in per_seq.items():
                    write_mot(root / tag / pol / "data" / f"{s}.txt", rows)
                stats[pol] = st
            ev = evaluate("test", root / tag, pols)
            for pol in pols:
                results.setdefault(str(rate), {}).setdefault(pol, []).append(dict(ev[pol], **stats[pol]))
            print(f"  rate {rate:g} seed {seed}: " + ", ".join(f"{p} {ev[p]['HOTA']:.2f}" for p in pols), flush=True)
    (root / "summary.json").write_text(json.dumps({"clean": clean, "results": results}, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ phase D
def phase_analysis(a, cal):
    rng = np.random.default_rng(0)
    alpha, gamma = PROTOCOL["alpha"], PROTOCOL["gamma"]
    xref = cal.xref[FAM]
    gt = load_golden(a.mot17, "test")
    rows_t, _ = clean_rows(gt)
    streams = {}
    for r in rows_t:
        streams.setdefault(r["seq"], []).append(r)
    L = [f"# Step 9 report: sealed test ({C.RESULTS.name})", "",
         "frozen protocol: " + json.dumps(PROTOCOL), "",
         f"clean test frames: {len(rows_t)} in {len(streams)} sequences; calibration: {len(xref)} conformal scores (q2)", ""]

    # 1. false alarms on clean test frames
    L.append("## 1. False alarms on clean test frames (every alarm is false; re-execution overhead = alarm rate)")
    L.append(f"  {'threshold':<22}{'alpha':>7}{'FAR':>9}{'95% CI':>18}{'worst seq':>11}")
    thr_at = {}
    for lab, al, gm in (("fixed (split-q12)", alpha, 0.0), ("adaptive gamma=0.005", alpha, gamma),
                        ("fixed (split-q12)", PROTOCOL["alpha_strict"], 0.0)):
        flags = {}
        for s, rs in streams.items():
            rs = sorted(rs, key=lambda r: r["frame"])
            sc = np.array([cal.score(r, FAM) for r in rs])
            al_, th = run_stream(sc, xref, al, gm)
            flags[s] = al_
            if al == alpha:
                for r, t in zip(rs, th):
                    thr_at[(lab, s, r["frame"])] = t
        allf = np.concatenate(list(flags.values()))
        lo, hi = block_bootstrap(flags, rng)
        L.append(f"  {lab:<22}{100 * al:6g}%{100 * allf.mean():8.2f}%  [{100 * lo:5.2f},{100 * hi:6.2f}]%"
                 f"{100 * max(np.mean(v) for v in flags.values()):10.2f}%")
    L.append("")

    # 2. test campaign
    cdir = OUT / "campaign"
    act = read_csv(cdir / "act.csv")
    store = unpack(cdir / "act_store.npz")
    win = int(json.loads((cdir / "meta.json").read_text())["args"]["win"])
    changed = [r for r in act if r["outcome"] not in ("masked_det", "due")]
    harm = [r for r in changed if r["outcome"] in ("sdc_local", "sdc_prop")]
    crit = [r for r in harm if r["outcome"] == "sdc_prop"]
    by_seq = {}
    for r in changed:
        by_seq.setdefault(r["seq"], []).append(r)
    for s, rs in by_seq.items():
        gs = gt[s]
        _, snaps = track_split(gs, None, {int(r["frame"]) for r in rs})
        for r in rs:
            t = int(r["frame"])
            d, _ = store[(int(r["id"]), C.SEQS.index(s), t)]
            r["S"] = cal.score(extract(snaps[t], d, finite(gs["dets"][t - 1]), t, gs["info"]["width"], gs["info"]["height"]), FAM)
    n_ch = len(changed)
    L.append("## 2. Test fault campaign")
    L.append(f"  {len(act)} activation faults: detector output changed {n_ch} ({100 * n_ch / len(act):.1f}%), "
             f"tracking changed {len(harm)}, propagated {len(crit)}; tracker fully masked "
             f"{100 * (1 - len(harm) / max(n_ch, 1)):.1f}% of the output changes")
    neg = np.array([cal.score(r, FAM) for r in rows_t])
    dmg = lambda r: float(r["fn"]) + float(r["fp"]) + float(r["idsw"])
    L.append(f"  {'positives':<24}{'n':>5}{'AUC':>7}   {'threshold':<22}{'TPR':>7}{'95% CI':>15}{'damage caught':>15}")
    for lab, rs in (("propagated (sdc_prop)", crit), ("all tracking changes", harm)):
        if not rs:
            continue
        pos = np.array([r["S"] for r in rs])
        tot = sum(dmg(r) for r in rs)
        first = True
        for tl, getthr in (("fixed alpha 1%", lambda r: conformal_threshold(xref, alpha)),
                           ("adaptive alpha 1%", lambda r: thr_at.get(("adaptive gamma=0.005", r["seq"], int(r["frame"])),
                                                                      conformal_threshold(xref, alpha))),
                           ("fixed alpha 0.1%", lambda r: conformal_threshold(xref, PROTOCOL["alpha_strict"]))):
            hit = [r for r in rs if r["S"] > getthr(r)]
            k = len(hit); lo, hi = wilson(k, len(rs))
            head = f"  {lab:<24}{len(rs):5d}{auc(neg, pos):7.3f}   " if first else " " * 39
            L.append(f"{head}{tl:<22}{100 * k / len(rs):6.1f}% [{100 * lo:4.1f},{100 * hi:5.1f}]"
                     f"{100 * sum(dmg(r) for r in hit) / max(tot, 1e-9):14.1f}%")
            first = False
    L.append("")

    # 3. sanitizer on the test campaign (system damage replay, as step 8)
    tau = PROTOCOL["sanitizer_tau"]
    L.append(f"## 3. Sanitizer (frozen tau = {tau}) on the test campaign")
    if tau is None:
        L.append("  not used for this detector (it runs NMS at IoU 0.45 itself)")
    else:
        fixed = conformal_threshold(xref, alpha)
        for t_ in (None, tau):
            tot, resid, nh = 0.0, 0.0, 0
            for s, rs in by_seq.items():
                gs = gt[s]
                outs, snaps = track_split(gs, t_, {int(r["frame"]) for r in rs})
                for r in rs:
                    t = int(r["frame"])
                    d, _ = store[(int(r["id"]), C.SEQS.index(s), t)]
                    ds = sanitize(d, t_)
                    if same_dets(sanitize(gs["dets"][t], t_), ds):
                        continue
                    frames = list(range(t, min(t + win, gs["b"] + 1)))
                    trk = copy.deepcopy(snaps[t])
                    fo = [trk.step(f, finite(ds if f == t else sanitize(gs["dets"][f], t_))) for f in frames]
                    c = compare_tracks([outs[f] for f in frames], fo)
                    dm = float(c["fn"] + c["fp"] + c["idsw"])
                    tot += dm; nh += dm > 0
                    resid += dm if r["S"] <= fixed else 0.0
            L.append(f"  {'none' if t_ is None else f'nms{t_:g}':<10} harmful faults {nh:5d}   damage {tot:8.0f}   "
                     f"residual after monitor (fixed 1%) {resid:8.0f}")
    L.append("")

    # 4. end-to-end
    sp = OUT / "e2e" / "summary.json"
    if sp.exists():
        S = json.loads(sp.read_text())
        cl, cls_ = S["clean"]["clean"], S["clean"]["clean_sanitized"]
        L.append("## 4. End-to-end on the test split (bit model 'exp', 3 seeds)")
        L.append(f"  fault-free HOTA {cl['HOTA']:.2f} (with sanitizer {cls_['HOTA']:.2f}); dHOTA is against the plain fault-free run")
        for rate, pols in S["results"].items():
            L.append(f"  rate {float(rate):g}:  {'policy':<10}{'HOTA':>14}{'dHOTA':>8}{'extra compute':>15}{'alarms':>9}{'false':>8}")
            for pol, runs in pols.items():
                h = np.array([r["HOTA"] for r in runs])
                m = lambda k: np.mean([r.get(k, 0.0) for r in runs])
                L.append(f"              {pol:<10}{h.mean():8.2f} ±{h.std():4.2f}{h.mean() - cl['HOTA']:8.2f}"
                         f"{100 * m('extra_compute'):14.1f}%{100 * m('alarms'):8.2f}%{100 * m('false_alarms'):7.2f}%")
        L.append("")
    L.append("  sealed result: nothing in this report was used to change any setting.")
    (C.RESULTS / "step9_report.txt").write_text("\n".join(L), encoding="utf-8")
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
    OUT.mkdir(parents=True, exist_ok=True)
    pf = OUT / "protocol.json"
    if pf.exists() and json.loads(pf.read_text()) != PROTOCOL:
        raise SystemExit(f"protocol changed since the sealed run started ({pf}); refusing to continue")
    pf.write_text(json.dumps(PROTOCOL, indent=1), encoding="utf-8")
    t0 = time.monotonic()
    prepare_gt(a.mot17)
    cal = calibrate(a.mot17)
    if not a.analyze:
        if a.force or not all(C.golden_path("test", s).exists() for s in C.SEQS):
            print("phase A: golden run on the test split", flush=True)
            from step1_golden import golden_split
            golden_split(make_det(a), C.FrameLoader(a.mot17), a.mot17, "test", a.backend)
        if a.force or not (OUT / "campaign" / "meta.json").exists():
            print("phase B: activation fault campaign on the test split", flush=True)
            r = subprocess.run([sys.executable, "step2_campaign.py", *det_args(a), "--split", "test", "--kind", "act",
                                "--n-act", str(PROTOCOL["n_act"]), "--win", str(PROTOCOL["win"]),
                                "--seed", str(PROTOCOL["campaign_seed"]), "--out", str(OUT / "campaign"), "--no-clamp"],
                               cwd=C.ROOT)
            if r.returncode != 0:
                raise SystemExit("phase B failed")
        if a.force or not (OUT / "e2e" / "summary.json").exists():
            print("phase C: end-to-end on the test split", flush=True)
            phase_e2e(a, cal)
    print("phase D: analysis", flush=True)
    phase_analysis(a, cal)
    print(f"step 9 done in {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
