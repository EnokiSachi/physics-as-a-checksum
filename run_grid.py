#!/usr/bin/env python3
"""
run_grid.py: the generalization study. Starting from the main configuration (MOT17, YOLOv8s,
ByteTrack), change one factor at a time:

  detector   yolov8n, yolov8m (model size), yolo11s (newer architecture), rtdetr-l (transformer)
  tracker    sort (no low-score association: does ByteTrack's second stage hide faults?)
  dataset    mot20 (very crowded), kitti (driving, cars)

Every configuration runs steps 0-7 on its own. Finished steps are skipped, so the script can be
stopped (Ctrl+C) and started again at any time; it continues where it stopped. A configuration
that fails is reported and the grid moves on to the next one.

  python run_grid.py                     all configurations
  python run_grid.py --only mot20 kitti  configurations whose name contains any of these words
  python run_grid.py --list              show the configurations and their status
  python run_grid.py --steps 7 --force   rerun step 7 in every configuration

Writes a status table to runs/grid_status.txt after every configuration.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from run import STEPS, run_config

ROOT = Path(__file__).resolve().parent
GRID = [
    ("mot17", "yolov8s", "bytetrack"),  # main configuration (steps 1-5 done; 6-7 run here)
    ("mot17", "yolov8n", "bytetrack"),
    ("mot17", "yolov8m", "bytetrack"),
    ("mot17", "yolo11s", "bytetrack"),
    ("mot17", "yolov8s", "sort"),
    ("mot20", "yolov8s", "bytetrack"),
    ("kitti", "yolov8s", "bytetrack"),
    ("mot17", "rtdetr-l", "bytetrack"),
]


def results_dir(cfg):
    env = dict(os.environ, PC_DATASET=cfg[0], PC_DETECTOR=cfg[1], PC_TRACKER=cfg[2])
    out = subprocess.run([sys.executable, "-c", "import common as C; print(C.RESULTS)"], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    return Path(out.stdout.strip()) if out.returncode == 0 else None


def status(cfg):
    d = results_dir(cfg)
    if d is None:
        return "config error"
    done = [k for k in range(1, 8) if (d / STEPS[k][1]).exists()]
    return "finished" if len(done) == 7 else (f"steps done: {done}" if done else "not started")


def write_status():
    lines = [f"{'/'.join(c):<32} {status(c)}" for c in GRID]
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "grid_status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--force", action="store_true", help="rerun the given steps even if their reports exist")
    a = ap.parse_args()
    if a.list:
        print("\n".join(write_status()))
        return
    todo = [c for c in GRID if not a.only or any(w in "_".join(c) for w in a.only)]
    failed = []
    for cfg in todo:
        if not run_config(*cfg, steps=a.steps, force=a.force):
            failed.append("/".join(cfg))
        write_status()
    print("\n".join(["", "grid status:"] + write_status()))
    if failed:
        print("failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
