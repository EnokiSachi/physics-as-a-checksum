"""
evaltools.py: ground truth for the time splits and HOTA / MOTA / IDF1 with the official TrackEval
code (vendored in third_party/trackeval, see COMMIT.txt there).

Frames are renumbered from 1 inside each split, both in the ground truth and in tracker output.
"""
import contextlib
import io
import json
import sys

import numpy as np

import common as C

GT_ROOT = C.RESULTS / "gt"


def prepare_gt(mot17, splits=("q2", "test")):
    info_all = json.loads((GT_ROOT / "splits.json").read_text()) if (GT_ROOT / "splits.json").exists() else {}
    for s in C.SEQS:
        info = C.seqinfo(mot17, s)
        gt = C.load_gt(mot17, s)
        rec = info_all.setdefault(s, {})
        for split in splits:
            a, b = C.split_range(info["length"], split)
            sub = gt[(gt[:, 0] >= a) & (gt[:, 0] <= b)].copy()
            sub[:, 0] -= a - 1
            out = GT_ROOT / split / s / "gt" / "gt.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as fh:
                for r in sub:
                    fh.write(f"{int(r[0])},{int(r[1])},{r[2]:g},{r[3]:g},{r[4]:g},{r[5]:g},{int(r[6])},{int(r[7])},{r[8]:g}\n")
            rec[split] = {"first": a, "last": b, "n": b - a + 1}
    (GT_ROOT / "splits.json").write_text(json.dumps(info_all, indent=1), encoding="utf-8")


def write_mot(path, rows):
    """rows: (frame_in_split, id, x1, y1, x2, y2, score)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for fr, tid, x1, y1, x2, y2, s in rows:
            fh.write(f"{int(fr)},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.4f},-1,-1,-1\n")


def _trackeval():
    p = str(C.ROOT / "third_party")
    if p not in sys.path:
        sys.path.insert(0, p)
    for a, b in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, a):
            setattr(np, a, b)
    with contextlib.redirect_stdout(io.StringIO()):
        import trackeval  # noqa: E402
    return trackeval


def evaluate(split, trackers_root, names):
    """Returns {tracker: {metric: value}} over all sequences combined."""
    te = _trackeval()
    info = json.loads((GT_ROOT / "splits.json").read_text())
    ecfg = te.Evaluator.get_default_eval_config()
    ecfg.update(PRINT_RESULTS=False, PRINT_CONFIG=False, TIME_PROGRESS=False, OUTPUT_SUMMARY=False,
                OUTPUT_DETAILED=False, PLOT_CURVES=False, USE_PARALLEL=False, OUTPUT_EMPTY_CLASSES=False,
                DISPLAY_LESS_PROGRESS=True)
    dcfg = te.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update(GT_FOLDER=str(GT_ROOT / split), TRACKERS_FOLDER=str(trackers_root), SKIP_SPLIT_FOL=True,
                SEQ_INFO={s: info[s][split]["n"] for s in C.SEQS}, TRACKERS_TO_EVAL=list(names),
                PRINT_CONFIG=False, BENCHMARK=C.DS["benchmark"])
    metrics = [te.metrics.HOTA({"PRINT_CONFIG": False}), te.metrics.CLEAR({"PRINT_CONFIG": False}),
               te.metrics.Identity({"PRINT_CONFIG": False})]
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = te.Evaluator(ecfg).evaluate([te.datasets.MotChallenge2DBox(dcfg)], metrics)
    out = {}
    for t in names:
        r = res["MotChallenge2DBox"][t]["COMBINED_SEQ"]["pedestrian"]
        out[t] = {"HOTA": 100 * float(np.mean(r["HOTA"]["HOTA"])), "MOTA": 100 * float(r["CLEAR"]["MOTA"]),
                  "IDF1": 100 * float(r["Identity"]["IDF1"]), "IDSW": int(r["CLEAR"]["IDSW"]),
                  "FN": int(r["CLEAR"]["CLR_FN"]), "FP": int(r["CLEAR"]["CLR_FP"])}
    return out
