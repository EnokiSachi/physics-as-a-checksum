#!/usr/bin/env python3
"""
run.py: run the pipeline for one (dataset, detector, tracker) configuration.

  python run.py --dataset mot20 --detector yolov8s --tracker bytetrack
  python run.py --dataset kitti --steps 1 2            (only some steps)
  python run.py --dataset mot17 --detector yolov8n --force   (rerun steps that already finished)

Steps: 0 environment check, 1 golden runs, 2 fault campaigns, 3 analysis, 4 end-to-end protection,
5 split-conformal calibration,
6 adaptive false alarm control, 7 weight checksum, 9 sealed test, 10 baselines on the sealed test,
11 white-box range monitor with split-q12 calibration (step10b_range.py). A step is skipped when its report already exists (use --force to
rerun). Extra arguments after "--" go to every step, e.g.  -- --mot17 D:\\Data\\MOT17
Output folder: printed at the start (results/ for the default configuration, runs/<name>/ otherwise).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = {0: ("step0_env.py", "step0_report.txt"), 1: ("step1_golden.py", "step1_report.txt"),
         2: ("step2_campaign.py", "campaign/meta.json"), 3: ("step3_analyze.py", "step3_report.txt"),
         4: ("step4_e2e.py", "step4_report.txt"), 5: ("step5_calibration.py", "step5_report.txt"),
         6: ("step6_adaptive.py", "step6_report.txt"), 7: ("step7_weights.py", "step7_report.txt"),
         9: ("step9_sealed.py", "step9_report.txt"), 10: ("step10_baselines.py", "step10_report.txt"),
         11: ("step10b_range.py", "step10b_report.txt")}


def run_config(dataset, detector, tracker, steps, force=False, extra=(), backend=None):
    env = dict(os.environ, PC_DATASET=dataset, PC_DETECTOR=detector, PC_TRACKER=tracker)
    out = subprocess.run([sys.executable, "-c", "import common as C; print(C.RESULTS)"], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout, out.stderr)
        return False
    results = Path(out.stdout.strip())
    results.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {dataset} / {detector} / {tracker}  ->  {results}", flush=True)
    log = open(results / "run_log.txt", "a", encoding="utf-8")
    for k in steps:
        script, report = STEPS[k]
        if not force and k != 0 and (results / report).exists():
            print(f"  step {k}: already done, skipped ({report})", flush=True)
            continue
        args = [sys.executable, script, *(["--measure"] if k == 7 else []),
                *(extra if k in (0, 1, 2, 4, 7, 9, 10, 11) else [])]
        if backend and k in (1, 2, 4, 7, 9, 10, 11):
            args += ["--backend", backend]
        print(f"  step {k}: {script} ...", flush=True)
        t0 = time.monotonic()
        r = subprocess.run(args, cwd=ROOT, env=env)
        mins = (time.monotonic() - t0) / 60
        log.write(f"{time.strftime('%Y-%m-%d %H:%M')} step {k} exit {r.returncode} {mins:.1f} min\n")
        log.flush()
        if r.returncode != 0:
            print(f"  step {k} FAILED after {mins:.1f} min; stopping this configuration", flush=True)
            return False
        print(f"  step {k}: done in {mins:.1f} min", flush=True)
    return True


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mot17")
    ap.add_argument("--detector", default="yolov8s")
    ap.add_argument("--tracker", default="bytetrack")
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    ok = run_config(a.dataset, a.detector, a.tracker, a.steps, a.force, extra)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
