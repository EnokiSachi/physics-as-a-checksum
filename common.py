"""
common.py: run configuration, datasets, time splits, frame loading.

A run is one (dataset, detector, tracker) configuration, chosen with environment variables
(run.py sets them for you):
  PC_DATASET   mot17 (default) | mot20 | kitti
  PC_DETECTOR  yolov8s (default) | yolov8n | yolov8m | yolo11s | rtdetr-l
  PC_TRACKER   bytetrack (default) | sort
The default configuration keeps the original folders results/ and cache/, so earlier runs stay
valid. Every other configuration writes to runs/<dataset>_<detector>_<tracker>/ and keeps its
detector caches in runs/cache/<dataset>_<detector>/ (the tracker does not change detections).

Datasets (training sets only, ground truth needed):
  mot17  MOT17 train, FRCNN folder of each sequence, pedestrians     root E:\\Datasets\\MOT17
  mot20  MOT20 train, crowded pedestrians                            root E:\\Datasets\\MOT20
  kitti  KITTI tracking training set, cars (class Car)               root E:\\Datasets\\KITTI_tracking
         layout: training/image_02/0000/000000.png, training/label_02/0000.txt
Override a root with MOT17_ROOT / MOT20_ROOT / KITTI_ROOT or --mot17 <path> on any step.

Time splits of every sequence:
  q1   = first 25%   calibration only (clean statistics, activation ranges)
  q2   = 25%-50%     development: fault campaigns, design decisions, ablations
  test = last 50%    sealed, evaluated once under the frozen protocol in <results>/sealed/protocol.json
"""
import configparser
import json
import os
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WARMUP = 15  # frames at the start of each split before tracks are considered established

DATASETS = {
    "mot17": {"root": os.environ.get("MOT17_ROOT", r"E:\Datasets\MOT17"), "benchmark": "MOT17", "classes": [0],
              "seqs": ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]},
    "mot20": {"root": os.environ.get("MOT20_ROOT", r"E:\Datasets\MOT20"), "benchmark": "MOT20", "classes": [0],
              "seqs": ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]},
    "kitti": {"root": os.environ.get("KITTI_ROOT", r"E:\Datasets\KITTI_tracking"), "benchmark": "MOT17", "classes": [2],
              "seqs": [f"{i:04d}" for i in range(21)]},
}
DETECTORS = {"yolov8n": ("yolov8n.pt", 960), "yolov8s": ("yolov8s.pt", 960), "yolov8m": ("yolov8m.pt", 960),
             "yolo11s": ("yolo11s.pt", 960), "rtdetr-l": ("rtdetr-l.pt", 640)}
TRACKERS = ("bytetrack", "sort")

DATASET = os.environ.get("PC_DATASET", "mot17").lower()
DETECTOR = os.environ.get("PC_DETECTOR", "yolov8s").lower()
TRACKER = os.environ.get("PC_TRACKER", "bytetrack").lower()
if DATASET not in DATASETS or DETECTOR not in DETECTORS or TRACKER not in TRACKERS:
    raise SystemExit(f"unknown configuration {DATASET}/{DETECTOR}/{TRACKER}; "
                     f"datasets {list(DATASETS)}, detectors {list(DETECTORS)}, trackers {list(TRACKERS)}")
DS = DATASETS[DATASET]
SEQS = DS["seqs"]
CLASSES = DS["classes"]
WEIGHTS, IMGSZ = DETECTORS[DETECTOR]
RUN = f"{DATASET}_{DETECTOR}_{TRACKER}"
if RUN == "mot17_yolov8s_bytetrack":
    RESULTS, CACHE = ROOT / "results", ROOT / "cache"
else:
    RESULTS, CACHE = ROOT / "runs" / RUN, ROOT / "runs" / "cache" / f"{DATASET}_{DETECTOR}"
DEFAULT_MOT17 = DS["root"]  # name kept for compatibility: every step's --mot17 is the dataset root
VARIANT = "FRCNN"


def seq_dir(root, seq):
    if DATASET == "mot17":
        return Path(root) / "train" / f"{seq}-{VARIANT}"
    if DATASET == "mot20":
        return Path(root) / "train" / seq
    return Path(root) / "training" / "image_02" / seq


def seqinfo(root, seq):
    if DATASET in ("mot17", "mot20"):
        cp = configparser.ConfigParser()
        cp.read(seq_dir(root, seq) / "seqinfo.ini")
        s = cp["Sequence"]
        return {"name": seq, "fps": float(s["frameRate"]), "length": int(s["seqLength"]),
                "width": int(s["imWidth"]), "height": int(s["imHeight"]),
                "imdir": s.get("imDir", "img1"), "ext": s.get("imExt", ".jpg")}
    # KITTI has no seqinfo.ini: count the images once and read the size of the first one
    cache = CACHE / "kitti_seqinfo.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    if seq not in known:
        files = sorted(seq_dir(root, seq).glob("*.png"))
        if not files:
            raise FileNotFoundError(seq_dir(root, seq))
        import cv2
        h, w = cv2.imread(str(files[0])).shape[:2]
        known[seq] = {"name": seq, "fps": 10.0, "length": len(files), "width": w, "height": h,
                      "imdir": "", "ext": ".png"}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=1))
    return known[seq]


def frame_path(root, info, f):
    """f is 1-based everywhere in this project; KITTI files are numbered from 0."""
    if DATASET == "kitti":
        return seq_dir(root, info["name"]) / f"{f - 1:06d}.png"
    return seq_dir(root, info["name"]) / info["imdir"] / f"{f:06d}{info['ext']}"


def load_gt(root, seq):
    """Ground truth in MOTChallenge format: frame, id, x, y, w, h, conf, class, visibility (1-based frames).
    KITTI: class Car only, converted from label_02 (frame is 0-based there, boxes are x1 y1 x2 y2)."""
    import numpy as np
    if DATASET in ("mot17", "mot20"):
        return np.loadtxt(seq_dir(root, seq) / "gt" / "gt.txt", delimiter=",", ndmin=2)
    rows = []
    with open(Path(root) / "training" / "label_02" / f"{seq}.txt") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 10 or p[2] != "Car":
                continue
            fr, tid = int(p[0]) + 1, int(p[1])
            x1, y1, x2, y2 = map(float, p[6:10])
            rows.append([fr, tid + 1, x1, y1, x2 - x1, y2 - y1, 1, 1, 1])
    return np.array(rows, float).reshape(-1, 9)


def split_range(length, split):
    """(first, last) frame of a split, 1-based, inclusive."""
    half, quarter = length // 2, length // 4
    return {"tune": (1, half), "test": (half + 1, length),
            "q1": (1, quarter), "q2": (quarter + 1, half)}[split]


class FrameLoader:
    """Decoded-frame cache. Campaigns visit frames grouped by sequence, so a small LRU is enough."""

    def __init__(self, mot17, capacity=400):
        self.mot17, self.capacity, self.cache, self.infos = mot17, capacity, OrderedDict(), {}

    def info(self, seq):
        if seq not in self.infos:
            self.infos[seq] = seqinfo(self.mot17, seq)
        return self.infos[seq]

    def __call__(self, seq, f):
        import cv2
        key = (seq, f)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        img = cv2.imread(str(frame_path(self.mot17, self.info(seq), f)))
        if img is None:
            raise FileNotFoundError(frame_path(self.mot17, self.info(seq), f))
        self.cache[key] = img
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return img


def golden_path(split, seq):
    return CACHE / "golden" / split / f"{seq}.npz"


def dets_by_frame(z):
    """npz with det_frame / boxes / scores -> {frame: ndarray[N,5]} (every frame of the split present)."""
    import numpy as np
    frames, boxes, scores = z["det_frame"], z["boxes"], z["scores"]
    out = {int(f): np.zeros((0, 5), np.float32) for f in z["frames"]}
    order = np.argsort(frames, kind="stable")
    frames, boxes, scores = frames[order], boxes[order], scores[order]
    if len(frames):
        uniq, starts = np.unique(frames, return_index=True)
        ends = list(starts[1:]) + [len(frames)]
        for f, a, b in zip(uniq, starts, ends):
            out[int(f)] = np.hstack([boxes[a:b], scores[a:b, None]]).astype(np.float32)
    return out


def pack_dets(dets):
    """{frame: ndarray[N,5]} -> arrays for np.savez."""
    import numpy as np
    fr, bx, sc = [], [], []
    for f in sorted(dets):
        d = dets[f]
        if len(d):
            fr.append(np.full(len(d), f, np.int32)); bx.append(d[:, :4]); sc.append(d[:, 4])
    return {"frames": np.array(sorted(dets), np.int32),
            "det_frame": np.concatenate(fr) if fr else np.zeros(0, np.int32),
            "boxes": (np.concatenate(bx) if bx else np.zeros((0, 4))).astype(np.float32),
            "scores": (np.concatenate(sc) if sc else np.zeros(0)).astype(np.float32)}
