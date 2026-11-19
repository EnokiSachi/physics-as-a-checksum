"""
monitors.py: black-box consistency monitors for a detection + tracking pipeline.

Every monitor only sees what leaves the detector (boxes and scores) plus, for the temporal ones,
earlier outputs. None of them looks inside the network. Each produces a few frame-level
statistics; a Calibrator turns them into empirical p-values using clean calibration frames and
combines them with Fisher's method, X = -2 * sum(log p). An alarm is raised when X exceeds the
clean (1 - alpha) quantile, so the false alarm rate is set in advance rather than tuned.

  physics   Kalman innovation tests. The tracker already predicts where every established track
            should be (mean and covariance). Statistics:
              nis_max   largest normalized innovation squared y' S^-1 y over matched tracks
                        (p-value Bonferroni-corrected for the number of matches)
              n_birth   confident detections that no existing track predicted
              n_death   established, fully visible tracks that no detection (score >= 0.1) explains
              n_death_hi  established, fully visible tracks without a confident detection (>= 0.5)
                        (added after the first dev campaign: faults that collapse many scores
                        below 0.5 left low-score boxes behind, which n_death counted as explained)
  temporal  the same idea without a motion model: compare with the previous frame's detections
              t_birth, t_death, t_dn (change in detection count)
  frame     single-frame plausibility checks (no history)
              n_det, mean_score (two-sided), max_area, n_invalid
  combined  physics + temporal (everything black-box that uses history)

The white-box range detector (activation outside its profiled range) is computed from detector
hooks, not here; see step2_campaign.py.
"""
import json

import numpy as np

from trackers.bytetrack import TRACKED, iou_matrix, match, xyxy_to_xyah

HIGH, NEW, LOW, GATE_IOU, MIN_AGE = 0.5, 0.6, 0.1, 0.2, 5

FAMILIES = {
    "frame": [("n_det", "two"), ("mean_score", "two"), ("max_area", "up"), ("n_invalid", "up")],
    "temporal": [("t_birth", "up"), ("t_death", "up"), ("t_dn", "up")],
    "physics": [("nis", "nis"), ("n_birth", "up"), ("n_death", "up"), ("n_death_hi", "up")],
}
FAMILIES["combined"] = FAMILIES["physics"] + FAMILIES["temporal"]
FEATURES = ["n_det", "mean_score", "max_area", "n_invalid", "n_nonfinite", "t_birth", "t_death", "t_dn",
            "nis_max", "n_match", "n_birth", "n_death", "n_death_hi", "n_tracks",
            "n_dup", "n_crowd", "n_birth_mid", "lo_birth", "t_dn_lo", "score_drop"]
# Candidate statistics added after the RT-DETR development campaign (not yet in any family; evaluated by
# diag_escapes.py). RT-DETR faults often ADD boxes (duplicates of real objects, boxes just above the
# confidence threshold, or low-score boxes that ByteTrack's second stage uses) instead of removing them:
#   n_dup        confident boxes overlapping another confident box (IoU > 0.5); NMS / set prediction
#                makes this rare on clean frames
#   n_crowd      confident boxes left unmatched by the established tracks although they overlap a
#                predicted track (IoU >= 0.2): a second box claiming an object that is already explained
#   n_birth_mid  unpredicted births with score in [0.5, 0.6), below n_birth's threshold
#   lo_birth     low-score boxes (0.1-0.5) with no predicted track and no box in the previous frame nearby
#   t_dn_lo      change in the number of boxes with score >= 0.1 from the previous frame
#   score_drop   largest drop of a matched established track's score (track score - detection score)


def finite(dets):
    """What any pipeline would do anyway: drop rows with NaN/inf before they reach the tracker."""
    if dets is None or len(dets) == 0:
        return np.zeros((0, 5), np.float32)
    return dets[np.isfinite(dets).all(1)]


def _inside_frac(b, W, H):
    w, h = b[2] - b[0], b[3] - b[1]
    iw = max(0.0, min(b[2], W) - max(b[0], 0)); ih = max(0.0, min(b[3], H) - max(b[1], 0))
    return iw * ih / max(w * h, 1e-9)


def extract(trk, dets_raw, prev_dets, frame, W, H):
    """Frame statistics. trk = tracker state BEFORE stepping this frame (not modified).
    dets_raw = detector output for this frame; prev_dets = previous frame's output (finite)."""
    raw = np.zeros((0, 5), np.float32) if dets_raw is None else dets_raw
    d = finite(raw)
    out = {"n_nonfinite": int(len(raw) - len(d))}
    # ---- frame: single-frame plausibility
    lo = d[d[:, 4] >= LOW]
    wh = (lo[:, 2:4] - lo[:, 0:2]) if len(lo) else np.zeros((0, 2))
    area = wh.prod(1) / (W * H) if len(lo) else np.zeros(0)
    invalid = (wh.min(1) < 2) | (area > 0.5) | (wh[:, 0] > 3 * np.maximum(wh[:, 1], 1e-6)) if len(lo) else np.zeros(0, bool)
    out.update(n_det=int(len(lo)), mean_score=float(lo[:, 4].mean()) if len(lo) else 0.0,
               max_area=float(area[lo[:, 4] >= HIGH].max()) if (lo[:, 4] >= HIGH).any() else 0.0,
               n_invalid=int(invalid.sum() + out["n_nonfinite"]))
    # ---- temporal without a motion model
    hi = d[d[:, 4] >= HIGH]
    ph = prev_dets[prev_dets[:, 4] >= HIGH] if len(prev_dets) else prev_dets
    pl = prev_dets[prev_dets[:, 4] >= LOW] if len(prev_dets) else prev_dets
    iou_new = iou_matrix(hi[:, :4], pl[:, :4])
    iou_old = iou_matrix(ph[:, :4], lo[:, :4])
    out["t_birth"] = int(((iou_new.max(1) if iou_new.shape[1] else np.zeros(len(hi))) < GATE_IOU)[hi[:, 4] >= NEW].sum())
    out["t_death"] = int(((iou_old.max(1) if iou_old.shape[1] else np.zeros(len(ph))) < GATE_IOU).sum())
    out["t_dn"] = abs(len(hi) - len(ph))
    # ---- physics: Kalman innovation
    kf = trk.kf
    preds = []
    for t in trk.tracks:
        m = t.mean.copy()
        if t.state != TRACKED:
            m[7] = 0.0
        m, P = kf.predict(m, t.cov)
        cx, cy, a, h = m[:4]
        w = a * h
        preds.append((t, m, P, np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])))
    est = [p for p in preds if p[0].state == TRACKED and p[0].last_frame == frame - 1
           and frame - p[0].start_frame >= MIN_AGE]
    eb = np.array([p[3] for p in est]).reshape(-1, 4)
    pairs, _, _ = match(1 - iou_matrix(eb, hi[:, :4]), 1 - GATE_IOU)
    nis = []
    for i, j in pairs:
        _, m, P, _ = est[i]
        z = xyxy_to_xyah(hi[j, :4])
        hgt = m[3]
        R = np.diag(np.square([kf.wp * hgt, kf.wp * hgt, 1e-1, kf.wp * hgt]))
        S = kf.H @ P @ kf.H.T + R
        y = z - kf.H @ m
        nis.append(float(y @ np.linalg.solve(S, y)))
    out["nis"] = nis
    out["nis_max"] = max(nis) if nis else 0.0
    out["n_match"] = len(nis)
    ab = np.array([p[3] for p in preds]).reshape(-1, 4)
    hn = hi[hi[:, 4] >= NEW]
    ib = iou_matrix(hn[:, :4], ab)
    out["n_birth"] = int(((ib.max(1) if ib.shape[1] else np.zeros(len(hn))) < GATE_IOU).sum())
    vis = np.array([_inside_frac(b, W, H) >= 0.9 for b in eb], bool)
    idd = iou_matrix(eb, lo[:, :4])
    unexplained = (idd.max(1) if idd.shape[1] else np.zeros(len(eb))) < GATE_IOU
    out["n_death"] = int((unexplained & vis).sum())
    idh = iou_matrix(eb, hi[:, :4])
    no_conf = (idh.max(1) if idh.shape[1] else np.zeros(len(eb))) < GATE_IOU
    out["n_death_hi"] = int((no_conf & vis).sum())
    out["n_tracks"] = len(est)
    # ---- candidate statistics for detectors whose faults add boxes (see FEATURES)
    ihh = iou_matrix(hi[:, :4], hi[:, :4])
    if len(hi):
        np.fill_diagonal(ihh, 0.0)
    out["n_dup"] = int((ihh.max(1) > 0.5).sum()) if len(hi) > 1 else 0
    matched_hi = {j for _, j in pairs}
    iha = iou_matrix(hi[:, :4], ab)
    near = (iha.max(1) if iha.shape[1] else np.zeros(len(hi))) >= GATE_IOU
    out["n_crowd"] = int(sum(1 for j in range(len(hi)) if j not in matched_hi and near[j]))
    mid = (hi[:, 4] < NEW) & ~near
    out["n_birth_mid"] = int(mid.sum())
    lb = d[(d[:, 4] >= LOW) & (d[:, 4] < HIGH)]
    ila = iou_matrix(lb[:, :4], ab)
    ilp = iou_matrix(lb[:, :4], pl[:, :4])
    far_a = (ila.max(1) if ila.shape[1] else np.zeros(len(lb))) < GATE_IOU
    far_p = (ilp.max(1) if ilp.shape[1] else np.zeros(len(lb))) < GATE_IOU
    out["lo_birth"] = int((far_a & far_p).sum())
    out["t_dn_lo"] = abs(len(lo) - len(pl))
    drops = [est[i][0].score - hi[j, 4] for i, j in pairs]
    out["score_drop"] = float(max(drops)) if drops else 0.0
    return out


def _tail_up(sorted_ref, x):
    n = len(sorted_ref)
    return (1 + n - np.searchsorted(sorted_ref, x, side="left")) / (n + 1)


def _tail_down(sorted_ref, x):
    n = len(sorted_ref)
    return (1 + np.searchsorted(sorted_ref, x, side="right")) / (n + 1)


class Calibrator:
    """Empirical p-values from clean calibration frames, Fisher combination, alarm thresholds."""

    def __init__(self, ref=None, nis_ref=None, xref=None):
        self.ref, self.nis_ref, self.xref = ref or {}, nis_ref, xref or {}

    @classmethod
    def fit(cls, rows, nis_values):
        ref = {k: np.sort(np.array([r[k] for r in rows], float)) for k in FEATURES}
        c = cls(ref, np.sort(np.asarray(nis_values, float)))
        for fam in FAMILIES:
            c.xref[fam] = np.sort(np.array([c.score(r, fam) for r in rows]))
        return c

    def pvalues(self, r, fam):
        ps = []
        for name, kind in FAMILIES[fam]:
            if kind == "nis":
                m = r["n_match"]
                p = 1.0 if m == 0 else min(1.0, m * _tail_up(self.nis_ref, r["nis_max"]))
            elif kind == "up":
                p = _tail_up(self.ref[name], r[name])
            else:
                p = min(1.0, 2 * min(_tail_up(self.ref[name], r[name]), _tail_down(self.ref[name], r[name])))
            ps.append(max(float(p), 1e-300))
        return ps

    def score(self, r, fam):
        if r.get("n_nonfinite", 0) > 0 and fam == "frame":
            return float("inf")
        return float(-2 * np.sum(np.log(self.pvalues(r, fam))))

    def threshold(self, fam, alpha):
        x = self.xref[fam]
        return float(np.quantile(x, 1 - alpha, method="higher"))

    def save(self, path):
        d = {"ref": {k: v.tolist() for k, v in self.ref.items()}, "nis_ref": self.nis_ref.tolist(),
             "xref": {k: v.tolist() for k, v in self.xref.items()}}
        path.write_text(json.dumps(d), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls({k: np.array(v) for k, v in d["ref"].items()}, np.array(d["nis_ref"]),
                   {k: np.array(v) for k, v in d["xref"].items()})


def range_violation(stats, lo, hi):
    """White-box baseline: how far any conv output left its profiled [lo, hi] range, relative to the
    range width. > 0 means out of range; NaN/inf activations give +inf."""
    amin, amax = stats["amin"].astype(float), stats["amax"].astype(float)
    if not (np.isfinite(amin).all() and np.isfinite(amax).all()):
        return float("inf")
    width = np.maximum(hi - lo, 1e-6)
    return float(np.max(np.maximum((amax - hi) / width, (lo - amin) / width)))
