"""
faults.py: fault sampling and outcome classification (golden-run comparison).

Every faulty run is compared with the fault-free ("golden") run of the same frames, as is standard
in fault injection studies. Outcomes are classified at two levels:

  detector level   masked_det  detector output identical to golden (within float tolerance)
                   det_changed output differs
  tracking level   (only for det_changed) the tracker is restarted from the golden state just
                   before the fault and run over a window of frames:
                   masked_trk   tracker output identical to golden over the whole window
                   sdc_local    differs only on frames that received faulty detections
                   sdc_prop     the error outlives the faulty frames (missing / extra / switched
                                tracks afterwards), i.e. it propagated through the tracker state
  due              inference raised an exception (a detected, unrecoverable error)
"""
import numpy as np

from trackers.bytetrack import iou_matrix, match

HIGH = 0.5


def sample_sites(rng, numel, nbits, n):
    """n fault sites (layer, u, bit): layer chosen proportional to element count, element uniform
    within the layer (u in [0,1)), bit uniform over the storage format."""
    p = numel / numel.sum()
    layers = rng.choice(len(numel), size=n, p=p)
    return [(int(l), float(u), int(b)) for l, u, b in zip(layers, rng.random(n), rng.integers(0, nbits, n))]


def same_dets(a, b, tol=1e-2):
    if a is None or b is None:
        return a is b
    if a.shape != b.shape:
        return False
    if len(a) == 0:
        return True
    with np.errstate(invalid="ignore"):
        return bool(np.all(np.isfinite(a)) and np.max(np.abs(a - b)) <= tol)


def det_diff(gold, faulty):
    """Detector-level change on confident detections: missing, spurious, worst IoU of matched pairs."""
    g = gold[gold[:, 4] >= HIGH]
    f = faulty[faulty[:, 4] >= HIGH] if faulty is not None else np.zeros((0, 5))
    f = f[np.isfinite(f).all(1)]
    pairs, ug, uf = match(1 - iou_matrix(g[:, :4], f[:, :4]), 0.5)
    iou = iou_matrix(g[:, :4], f[:, :4])
    worst = min([iou[i, j] for i, j in pairs], default=1.0)
    return {"n_missing": len(ug), "n_spurious": len(uf), "worst_iou": float(worst),
            "n_gold": int(len(g)), "n_faulty": int(len(f))}


def compare_tracks(gold, faulty, iou_thr=0.5):
    """gold / faulty: per-frame lists of (id, x1, y1, x2, y2, score). Returns FN, FP, ID switches
    and the frame offsets (0 = fault frame) where the outputs differ."""
    fn = fp = 0
    diff_frames = []
    g2f, f2g = {}, {}
    for k, (go, fo) in enumerate(zip(gold, faulty)):
        gb = np.array([o[1:5] for o in go], float).reshape(-1, 4)
        fb = np.array([o[1:5] for o in fo], float).reshape(-1, 4)
        pairs, ug, uf = match(1 - iou_matrix(gb, fb), 1 - iou_thr)
        bad = len(ug) + len(uf)
        fn += len(ug); fp += len(uf)
        for i, j in pairs:
            gi, fj = go[i][0], fo[j][0]
            g2f.setdefault(gi, set()).add(fj)
            f2g.setdefault(fj, set()).add(gi)
        if bad:
            diff_frames.append(k)
    idsw = sum(len(v) - 1 for v in g2f.values()) + sum(len(v) - 1 for v in f2g.values())
    return {"fn": fn, "fp": fp, "idsw": idsw, "diff_frames": diff_frames}


def classify(det_same_all, cmp, n_faulty_frames):
    if not det_same_all:
        if cmp["fn"] + cmp["fp"] + cmp["idsw"] == 0:
            return "masked_trk"
        late = [k for k in cmp["diff_frames"] if k >= n_faulty_frames]
        return "sdc_prop" if (late or cmp["idsw"]) else "sdc_local"
    return "masked_det"
