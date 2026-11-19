#!/usr/bin/env python3
"""
Step 3: analysis of the development campaigns (q2). No GPU needed.

All monitor statistics are recomputed here from the stored detections, so monitors can change
without rerunning step 2. Calibration always uses q1 clean frames only; q2 is the development set.

  A. Where do transient faults die?  Outcome funnel detector -> tracker, by bit and network part,
     and the two range-restriction baselines on the same faults.
  B. Persistent weight faults: outcomes by bit and network part.
  C. Detecting transient faults: ROC per monitor. Positives = faults that corrupted the tracking
     output; negatives = clean q2 frames; thresholds fixed on q1 (out-of-sample false alarm rates).
     Range detector, two versions:
       range-naive  any activation outside the full q1 range (no false alarm control)
       range        ranges profiled on the first half of q1, alarm threshold on the second half,
                    so it has the same false alarm control as the black-box monitors
  D. Detecting persistent weight faults: single-frame tests and a CUSUM on the Fisher score.
  E. Escapes: propagated faults the physics and combined monitors miss.

Usage:  python step3_analyze.py
Output: results/step3_report.txt, results/figures/roc_act.png, results/figures/outcome_by_bit.png
"""
import copy
import csv
import json
import math

import numpy as np

import common as C
from detector import block_index
from monitors import FAMILIES, Calibrator, extract, finite, range_violation
from step2_campaign import golden_tracking, load_golden
from trackers import make_tracker

ALPHAS = (0.01, 0.001)
MONITORS = list(FAMILIES) + ["range", "range-naive"]
CUSUM_RATE = 1e-3  # target clean alarm rate per frame for the sequential test


def read_csv(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if k not in ("seq", "layer_name") and not k.endswith("outcome") else v
            except (TypeError, ValueError):
                pass
    return rows


def unpack(path):
    with np.load(path) as z:  # read every array once; indexing z[...] inside the loop re-reads the file
        keys, n, dets, amin, amax = z["keys"], z["n"], z["dets"], z["amin"], z["amax"]
    off = np.r_[0, np.cumsum(np.maximum(n, 0))]
    out = {}
    for i, key in enumerate(keys):
        d = None if n[i] < 0 else dets[off[i]:off[i + 1]]
        out[tuple(int(x) for x in key)] = (d, {"amin": amin[i], "amax": amax[i]})
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def auc(neg, pos):
    from scipy.stats import rankdata
    x = np.nan_to_num(np.r_[neg, pos], nan=np.inf)
    rp = rankdata(x)[len(neg):]
    return float((rp.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "    -"


def group_of(block, head):
    return "other" if block < 0 else ("backbone" if block <= 9 else ("head" if block == head else "neck"))


def outcome_table(rows, key, keys, outs, field="outcome"):
    L = [f"  {key:<9}{'n':>6}" + "".join(f"{o:>12}" for o in outs)]
    for kv in list(keys) + ["all"]:
        sub = rows if kv == "all" else [r for r in rows if r[key] == kv]
        if sub:
            L.append(f"  {str(kv):<9}{len(sub):6d}" + "".join(f"{pct(sum(r[field] == o for r in sub), len(sub)):>12}" for o in outs))
    return L


# ------------------------------------------------------------------ statistics
def clean_rows(g):
    rows, nis = [], []
    for s, gs in g.items():
        trk, prev = make_tracker(), np.zeros((0, 5), np.float32)
        W, H = gs["info"]["width"], gs["info"]["height"]
        for f in range(gs["a"], gs["b"] + 1):
            d = gs["dets"][f]
            if f >= gs["a"] + C.WARMUP:
                r = extract(trk, d, prev, f, W, H)
                nis += r["nis"]
                r.update(seq=s, frame=f, stats={"amin": gs["amin"][f - gs["a"]], "amax": gs["amax"][f - gs["a"]]})
                rows.append(r)
            trk.step(f, finite(d))
            prev = finite(d)
    return rows, nis


class Scorer:
    def __init__(self, g1):
        rows, nis = clean_rows(g1)
        self.cal = Calibrator.fit(rows, nis)
        # range detector with false alarm control: profile on first half of q1, threshold on second half
        A = [r for r in rows if r["frame"] < (g1[r["seq"]]["a"] + g1[r["seq"]]["b"]) / 2]
        B = [r for r in rows if r["frame"] >= (g1[r["seq"]]["a"] + g1[r["seq"]]["b"]) / 2]
        self.lo_a = np.min([r["stats"]["amin"] for r in A], 0).astype(float)
        self.hi_a = np.max([r["stats"]["amax"] for r in A], 0).astype(float)
        self.lo = np.min([r["stats"]["amin"] for r in rows], 0).astype(float)
        self.hi = np.max([r["stats"]["amax"] for r in rows], 0).astype(float)
        vb = np.sort([range_violation(r["stats"], self.lo_a, self.hi_a) for r in B])
        self.thr = {fam: {a: self.cal.threshold(fam, a) for a in ALPHAS} for fam in FAMILIES}
        self.thr["range"] = {a: float(np.quantile(vb, 1 - a, method="higher")) for a in ALPHAS}
        self.thr["range-naive"] = {a: 0.0 for a in ALPHAS}
        self.n_q1 = len(rows)
        # CUSUM parameters on q1 clean streams
        self.cus = {}
        for fam in FAMILIES:
            x = {}
            for r in rows:
                x.setdefault(r["seq"], []).append(self.cal.score(r, fam))
            k = float(np.quantile(np.concatenate(list(x.values())), 0.95))
            gs = np.concatenate([self._cusum(np.array(v), k, np.inf)[1] for v in x.values()])
            self.cus[fam] = (k, float(np.quantile(gs, 1 - CUSUM_RATE, method="higher")))

    @staticmethod
    def _cusum(x, k, h):
        g, gs, alarms = 0.0, [], []
        for v in x:
            g = max(0.0, g + min(v, 1e6) - k)
            gs.append(g)
            if g > h:
                alarms.append(True); g = 0.0
            else:
                alarms.append(False)
        return np.array(alarms), np.array(gs)

    def scores(self, r):
        out = {fam: self.cal.score(r, fam) for fam in FAMILIES}
        st = r.get("stats")
        if st is None or not np.isfinite(st["amax"]).any():
            out["range"] = out["range-naive"] = float("inf")
        else:
            out["range"] = range_violation(st, self.lo_a, self.hi_a)
            out["range-naive"] = range_violation(st, self.lo, self.hi)
        return out


# ------------------------------------------------------------------ main
def main():
    meta = json.loads((C.RESULTS / "detector_meta.json").read_text())
    names, nb = meta["layers"], meta["nbits"]
    head = max(block_index(n) for n in names)
    cmeta = json.loads((C.RESULTS / "campaign" / "meta.json").read_text())
    mot17 = cmeta["args"]["mot17"]
    g1, g2 = load_golden(mot17, "q1"), load_golden(mot17, "q2")
    sc = Scorer(g1)
    clean2, _ = clean_rows(g2)
    neg = {m: np.array([sc.scores(r)[m] for r in clean2]) for m in MONITORS}

    act = read_csv(C.RESULTS / "campaign" / "act.csv")
    astore = unpack(C.RESULTS / "campaign" / "act_store.npz")
    by_seq = {}
    for r in act:
        r["group"] = group_of(int(r["block"]), head)
        by_seq.setdefault(r["seq"], []).append(r)
    for s, rs in by_seq.items():
        gs = g2[s]
        _, snaps = golden_tracking(gs, {int(r["frame"]) for r in rs})
        for r in rs:
            t = int(r["frame"])
            d, st = astore[(int(r["id"]), C.SEQS.index(s), t)]
            f = extract(snaps[t], d, finite(gs["dets"][t - 1]), t, gs["info"]["width"], gs["info"]["height"])
            f["stats"] = st
            r["S"] = sc.scores(f)
            r["feat"] = f

    L = ["# Step 3 report: development campaigns (q2)", "",
         f"detector {meta['weights']} ({meta['dtype']}), {meta['n_conv']} conv layers; {len(act)} activation faults; "
         f"clean frames q1 {sc.n_q1} (calibration), q2 {len(clean2)} (false alarm check)", ""]

    # ---------------------------------------------------------------- A
    outs = ["masked_det", "masked_trk", "sdc_local", "sdc_prop", "due"]
    L += ["## A. Transient activation faults: where they are masked", ""]
    L += outcome_table(act, "group", ["backbone", "neck", "head", "other"], outs)
    L += [""] + outcome_table(act, "bit", range(nb - 1, -1, -1), outs)
    ch = [r for r in act if r["outcome"] not in ("masked_det", "due")]
    crit = [r for r in act if r["outcome"] == "sdc_prop"]
    if ch:
        L += ["", f"  of {len(ch)} faults that changed the detector output: tracker fully masked "
                  f"{pct(sum(r['outcome'] == 'masked_trk' for r in ch), len(ch))}, propagated {pct(len(crit), len(ch))}"]
    if crit:
        L.append(f"  propagated faults: mean FN {np.mean([r['fn'] for r in crit]):.1f}, FP {np.mean([r['fp'] for r in crit]):.1f}, "
                 f"ID switches {np.mean([r['idsw'] for r in crit]):.2f} over the window")
    L += ["", "  Range restriction on the same faults (tight = q1 range, wide = q1 range +/- its width):",
          f"  {'':<15}" + "".join(f"{o:>12}" for o in outs) + "   clean output changed by clamping"]
    for lab, p in (("no protection", ""), ("tight clamp", "r_"), ("wide clamp", "w_")):
        extra = pct(sum(r[p + "clean_same"] == 0 for r in act), len(act)) if p else ""
        L.append(f"  {lab:<15}" + "".join(f"{pct(sum(r[p + 'outcome'] == o for r in act), len(act)):>12}" for o in outs) + f"   {extra}")

    # ---------------------------------------------------------------- B
    wsum = read_csv(C.RESULTS / "campaign" / "weight.csv")
    wstore = unpack(C.RESULTS / "campaign" / "weight_store.npz")
    for r in wsum:
        r["group"] = group_of(int(r["block"]), head)
        if r["due"] > 0:
            r["outcome"] = "due"
    L += ["", "## B. Persistent weight faults (active for the whole window)", ""]
    wouts = ["masked_det", "masked_trk", "sdc", "due"]
    L += outcome_table(wsum, "group", ["backbone", "neck", "head", "other"], wouts)
    L += [""] + outcome_table(wsum, "bit", range(nb - 1, -1, -1), wouts)

    # ---------------------------------------------------------------- C
    L += ["", "## C. Detecting transient faults (alarm at the faulty frame)", "",
          "  thresholds fixed on q1; FAR = realized false alarm rate on clean q2 frames.",
          "  the alarm rate on clean frames is also the re-execution overhead if every alarm triggers a re-run", ""]
    roc_data = {}
    for label, posset in (("propagated (sdc_prop)", {"sdc_prop"}), ("any tracking error (sdc_*)", {"sdc_local", "sdc_prop"})):
        pos_rows = [r for r in act if r["outcome"] in posset]
        benign = [r for r in act if r["outcome"] == "masked_trk"]
        L.append(f"  positives: {label}, n={len(pos_rows)}")
        L.append(f"  {'monitor':<12}{'AUC':>7}   {'TPR@1%':>17} {'FAR':>7}   {'TPR@0.1%':>17} {'FAR':>7}   alarms on benign (1%)")
        for m in MONITORS:
            pos = np.array([r["S"][m] for r in pos_rows])
            if not len(pos):
                continue
            cells = []
            for a in ALPHAS:
                t = sc.thr[m][a]
                k = int(np.sum(pos > t)); lo, hi = wilson(k, len(pos))
                cells.append(f"{100 * k / len(pos):5.1f} [{100 * lo:4.1f},{100 * hi:5.1f}] {100 * np.mean(neg[m] > t):6.2f}%")
            ben = pct(sum(r["S"][m] > sc.thr[m][0.01] for r in benign), len(benign))
            L.append(f"  {m:<12}{auc(neg[m], pos):7.3f}   {cells[0]}   {cells[1]}   {ben}")
            if label.startswith("propagated"):
                roc_data[m] = (neg[m], pos)
        L.append("")

    # ---------------------------------------------------------------- D
    L += ["## D. Detecting persistent weight faults", ""]
    sdc = [r for r in wsum if r["outcome"] == "sdc"]
    frames_of = {}
    for (fid, k, f), v in wstore.items():
        frames_of.setdefault(fid, []).append((k, f, v))
    need = {}
    for r in sdc:
        need.setdefault(r["seq"], set()).add(int(r["t0"]))
    snaps_all = {s: golden_tracking(g2[s], fs)[1] for s, fs in need.items()}
    streams = {}
    for r in sdc:
        s, t0 = r["seq"], int(r["t0"])
        gs = g2[s]
        trk, prev = copy.deepcopy(snaps_all[s][t0]), finite(gs["dets"][t0 - 1])
        seq_scores = []
        for k, f, (d, st) in sorted(frames_of[int(r["id"])]):
            feat = extract(trk, d, prev, f, gs["info"]["width"], gs["info"]["height"])
            feat["stats"] = st
            seq_scores.append(sc.scores(feat))
            trk.step(f, finite(d)); prev = finite(d)
        streams[int(r["id"])] = seq_scores
    n_win = int(np.median([r["n_frames"] for r in wsum]))
    clean_streams = {}
    for r in clean2:
        clean_streams.setdefault(r["seq"], []).append(r)
    L.append(f"  corrupting weight faults: {len(sdc)} (of {len(wsum)}); window {n_win} frames")
    L.append(f"  {'test':<26}{'clean alarms / 1000 frames':>27}{'detected':>10}{'at onset':>10}{'median delay':>14}")
    tests = [(f"{m} single-frame 1%", m, "shew", 0.01) for m in ("physics", "combined", "temporal", "range")]
    tests += [(f"{m} single-frame 0.1%", m, "shew", 0.001) for m in ("physics", "combined", "temporal", "range")]
    tests += [(f"{m} CUSUM", m, "cusum", None) for m in ("physics", "combined", "temporal")]
    for name, m, kind, a in tests:
        if kind == "shew":
            far = 1000 * np.mean(neg[m] > sc.thr[m][a])
            alarm_fn = lambda xs, m=m, a=a: np.array([x[m] > sc.thr[m][a] for x in xs])
        else:
            kk, hh = sc.cus[m]
            far = 1000 * np.mean(np.concatenate([Scorer._cusum(np.array([sc.cal.score(r, m) for r in v]), kk, hh)[0]
                                                 for v in clean_streams.values()]))
            alarm_fn = lambda xs, m=m, kk=kk, hh=hh: Scorer._cusum(np.array([x[m] for x in xs]), kk, hh)[0]
        det_n, onset, lat = 0, 0, []
        for r in sdc:
            al = np.flatnonzero(alarm_fn(streams[int(r["id"])]))
            fc = int(r["first_changed"])
            al = al[al >= fc]
            if len(al):
                det_n += 1; onset += al[0] == fc; lat.append(al[0] - fc)
        L.append(f"  {name:<26}{far:27.2f}{pct(det_n, len(sdc)):>10}{pct(onset, len(sdc)):>10}"
                 f"{(np.median(lat) if lat else float('nan')):14.1f}")
    L += ["  reference: a periodic checksum of the weights detects every weight fault (including the "
          f"{pct(len(wsum) - len(sdc), len(wsum)).strip()} that are harmless) with delay up to the checking period.", ""]

    # ---------------------------------------------------------------- E
    if crit:
        L += ["## E. Escapes: propagated transient faults missed at alpha=1%", ""]
        kinds = {"missing only": lambda r: r["n_missing"] > 0 and r["n_spurious"] == 0,
                 "spurious only": lambda r: r["n_spurious"] > 0 and r["n_missing"] == 0,
                 "both": lambda r: r["n_spurious"] > 0 and r["n_missing"] > 0,
                 "no confident change": lambda r: r["n_spurious"] == 0 and r["n_missing"] == 0}
        for m in ("physics", "combined", "temporal"):
            miss = [r for r in crit if r["S"][m] <= sc.thr[m][0.01]]
            L.append(f"  {m}: missed {len(miss)} of {len(crit)}; by detector-level effect: " + ", ".join(
                f"{k} {sum(f(r) for r in miss)}/{sum(f(r) for r in crit)}" for k, f in kinds.items()))
        L.append("")

    (C.RESULTS / "step3_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    try:
        figures(act, roc_data, nb)
    except Exception as e:
        print(f"(figures skipped: {e})")


def figures(act, roc_data, nb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = C.RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for m, (neg, pos) in roc_data.items():
        ts = np.unique(np.r_[neg, pos]); ts = ts[np.isfinite(ts)]
        far = np.array([np.mean(neg > t) for t in ts]); tpr = np.array([np.mean(pos > t) for t in ts])
        ax.plot(np.maximum(far, 1e-4), tpr, label=m, ls="--" if m == "range-naive" else "-")
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("false alarm rate on clean frames"); ax.set_ylabel("detected fraction of propagated faults")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "roc_act.png", dpi=160); plt.close(fig)

    outs = ["masked_det", "masked_trk", "sdc_local", "sdc_prop", "due"]
    bits = list(range(nb))
    frac = np.array([[np.mean([r["outcome"] == o for r in act if r["bit"] == b]) if any(r["bit"] == b for r in act) else 0
                      for b in bits] for o in outs])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bottom = np.zeros(len(bits))
    for o, f in zip(outs, frac):
        ax.bar(bits, f, bottom=bottom, label=o); bottom += f
    ax.set_xlabel("flipped bit (FP16: 15 sign, 10-14 exponent, 0-9 mantissa)"); ax.set_ylabel("fraction of faults")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout(); fig.savefig(out / "outcome_by_bit.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
