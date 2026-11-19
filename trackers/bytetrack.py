"""
bytetrack.py: a self-contained Kalman filter + Hungarian matching multi-object tracker (SORT and ByteTrack modes).

Why a re-implementation: experiments need budget controls such as "detect every k frames and only predict in
between", which existing libraries do not support, and every design choice here can be stated and ablated.

State x = [cx, cy, a, h, vx, vy, va, vh] (centre, aspect ratio, height and their velocities), constant-velocity
model; process and measurement noise scale with the target height (as in the DeepSORT / ByteTrack code).

ByteTrack association (Zhang et al., ECCV 2022):
  1. confirmed tracks (including lost ones) <-> high-score detections, IoU distance + Hungarian algorithm;
  2. still unmatched tracked tracks <-> low-score detections (recovers occluded objects);
  3. tentative tracks <-> remaining high-score detections;
  4. remaining detections with a high enough score start new tracks; tracks lost for more than `buffer`
     frames are removed.
SORT mode = step 2 removed (low_thresh = high_thresh).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

TENTATIVE, TRACKED, LOST, REMOVED = 0, 1, 2, 3


class KalmanXYAH:
    def __init__(self):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.wp, self.wv = 1.0 / 20, 1.0 / 160

    def initiate(self, z):
        h = z[3]
        mean = np.r_[z, np.zeros(4)]
        std = [2 * self.wp * h, 2 * self.wp * h, 1e-2, 2 * self.wp * h,
               10 * self.wv * h, 10 * self.wv * h, 1e-5, 10 * self.wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.wp * h, self.wp * h, 1e-2, self.wp * h, self.wv * h, self.wv * h, 1e-5, self.wv * h]
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(np.square(std))
        return mean, cov

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self.wp * h, self.wp * h, 1e-1, self.wp * h]))
        S = self.H @ cov @ self.H.T + R
        K = np.linalg.solve(S, (cov @ self.H.T).T).T
        mean = mean + K @ (z - self.H @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=float)


def iou_matrix(a, b):
    """a: [N,4], b: [M,4] xyxy -> [N,M] IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match(cost, thresh):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    r, c = linear_sum_assignment(cost)
    pairs = [(i, j) for i, j in zip(r, c) if cost[i, j] <= thresh]
    mr = {i for i, _ in pairs}
    mc = {j for _, j in pairs}
    return pairs, [i for i in range(cost.shape[0]) if i not in mr], [j for j in range(cost.shape[1]) if j not in mc]


class Track:
    __slots__ = ("id", "mean", "cov", "score", "state", "hits", "last_frame", "start_frame")

    def __init__(self, tid, mean, cov, score, frame, state):
        self.id, self.mean, self.cov, self.score = tid, mean, cov, score
        self.state, self.hits, self.last_frame, self.start_frame = state, 1, frame, frame

    def xyxy(self):
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class ByteTracker:
    def __init__(self, high_thresh=0.5, low_thresh=0.1, new_thresh=0.6, match_thresh=0.8,
                 second_thresh=0.5, unconfirmed_thresh=0.7, buffer=30, n_init=2, use_low=True):
        self.high, self.low = high_thresh, (low_thresh if use_low else high_thresh)
        self.new, self.m1, self.m2, self.m3 = new_thresh, match_thresh, second_thresh, unconfirmed_thresh
        self.buffer, self.n_init, self.use_low = buffer, n_init, use_low
        self.kf = KalmanXYAH()
        self.tracks, self.next_id, self.first_frame = [], 1, None

    def _boxes(self, tracks):
        return np.array([t.xyxy() for t in tracks]).reshape(-1, 4)

    def step(self, frame, dets=None):
        """dets: ndarray[N,5] (x1, y1, x2, y2, score), or None if this frame is not detected (predict only).
        Returns this frame's output [(track_id, x1, y1, x2, y2, score), ...]."""
        if self.first_frame is None:
            self.first_frame = frame
        for t in self.tracks:
            if t.state != TRACKED:
                t.mean[7] = 0.0
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        if dets is None:  # budget control: a frame without detection outputs only the predictions of confirmed tracked tracks
            self._prune(frame)
            return [(t.id, *t.xyxy(), t.score) for t in self.tracks if t.state == TRACKED]

        dets = np.asarray(dets, dtype=float).reshape(-1, 5)
        hi = dets[dets[:, 4] >= self.high]
        lo = dets[(dets[:, 4] >= self.low) & (dets[:, 4] < self.high)] if self.use_low else dets[:0]

        confirmed = [t for t in self.tracks if t.state in (TRACKED, LOST)]
        tentative = [t for t in self.tracks if t.state == TENTATIVE]

        # 1) confirmed (including lost) <-> high score
        pairs, um_t, um_d = match(1 - iou_matrix(self._boxes(confirmed), hi[:, :4]), self.m1)
        for i, j in pairs:
            self._update(confirmed[i], hi[j], frame)
        # 2) still tracked <-> low score
        rest = [confirmed[i] for i in um_t if confirmed[i].state == TRACKED]
        pairs2, um_t2, _ = match(1 - iou_matrix(self._boxes(rest), lo[:, :4]), self.m2)
        for i, j in pairs2:
            self._update(rest[i], lo[j], frame)
        for i in um_t2:
            rest[i].state = LOST
        # 3) tentative <-> remaining high score
        hi_rest = hi[um_d]
        pairs3, um_t3, um_d3 = match(1 - iou_matrix(self._boxes(tentative), hi_rest[:, :4]), self.m3)
        for i, j in pairs3:
            t = tentative[i]
            self._update(t, hi_rest[j], frame)
            t.hits += 1
            if t.hits >= self.n_init:
                t.state = TRACKED
        for i in um_t3:
            tentative[i].state = REMOVED
        # 4) new tracks
        for j in um_d3:
            d = hi_rest[j]
            if d[4] < self.new:
                continue
            mean, cov = self.kf.initiate(xyxy_to_xyah(d[:4]))
            state = TRACKED if frame == self.first_frame else TENTATIVE
            self.tracks.append(Track(self.next_id, mean, cov, float(d[4]), frame, state))
            self.next_id += 1
        self._prune(frame)
        return [(t.id, *t.xyxy(), t.score) for t in self.tracks if t.state == TRACKED and t.last_frame == frame]

    # ---- read-only queries for detection scheduling (do not change the tracker state) ----
    def tracked(self):
        return [t for t in self.tracks if t.state == TRACKED]

    def predicted_boxes(self):
        """Predicted boxes of confirmed tracked tracks for the next frame (state unchanged)."""
        out = []
        for t in self.tracked():
            cx, cy, a, h = (self.kf.F @ t.mean)[:4]
            w = a * h
            out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        return np.array(out, dtype=float).reshape(-1, 4)

    def norm_speeds(self):
        """Speed of each tracked track, in its own heights moved per frame."""
        return np.array([np.hypot(t.mean[4], t.mean[5]) / max(t.mean[3], 1.0) for t in self.tracked()])

    def _update(self, t, d, frame):
        t.mean, t.cov = self.kf.update(t.mean, t.cov, xyxy_to_xyah(d[:4]))
        t.score, t.last_frame = float(d[4]), frame
        if t.state == LOST:
            t.state = TRACKED

    def _prune(self, frame):
        for t in self.tracks:
            if t.state == LOST and frame - t.last_frame > self.buffer:
                t.state = REMOVED
        self.tracks = [t for t in self.tracks if t.state != REMOVED]
