#!/usr/bin/env python3
"""
Pipeline self-test without GPU or MOT17: copies the code into a temporary folder, writes synthetic
MOT17-style sequence info, and runs steps 1-3 with the numpy FakeDetector. Checks that every stage
runs and that the outputs are internally consistent. Does not touch the real results/ folder.

Usage:  python tests/run_selftest.py
"""
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
SEQ_LEN = {"MOT17-02": 240, "MOT17-04": 260, "MOT17-05": 220, "MOT17-09": 200,
           "MOT17-10": 240, "MOT17-11": 260, "MOT17-13": 240}


def run(cwd, *args):
    r = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(args)}")
    return r.stdout


def run_env(cwd, env, *args):
    import os
    r = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, env=dict(os.environ, **env))
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED ({env}): {' '.join(args)}")
    return r.stdout


def check_other_configs(tmp, fake):
    """MOT20 layout with the SORT tracker, and KITTI layout (PNG frames numbered from 0, label_02 with
    cars) with ByteTrack, both through steps 1-5 with the fake detector. Checks that every non-default
    configuration writes into its own runs/ folder and leaves results/ alone."""
    import numpy as np
    m20 = tmp / "MOT20"
    for s, n in {"MOT20-01": 200, "MOT20-02": 220, "MOT20-03": 240, "MOT20-05": 200}.items():
        d = m20 / "train" / s
        (d / "gt").mkdir(parents=True)
        (d / "seqinfo.ini").write_text(f"[Sequence]\nname={s}\nimDir=img1\nframeRate=25\nseqLength={n}\n"
                                       f"imWidth=1920\nimHeight=1080\nimExt=.jpg\n")
        with open(d / "gt" / "gt.txt", "w") as fh:
            for f in range(1, n + 1):
                for tid, x, y, w, h in fake.ground_truth(s, f):
                    fh.write(f"{f},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")
    kit = tmp / "KITTI"
    import cv2
    tiny = np.zeros((375, 1242, 3), np.uint8)
    for i in range(21):
        s, n = f"{i:04d}", 80 + 4 * i
        d = kit / "training" / "image_02" / s
        d.mkdir(parents=True)
        for f in range(n):
            cv2.imwrite(str(d / f"{f:06d}.png"), tiny)
        lab = kit / "training" / "label_02"
        lab.mkdir(parents=True, exist_ok=True)
        with open(lab / f"{s}.txt", "w") as fh:
            for f in range(1, n + 1):
                for tid, x, y, w, h in fake.ground_truth(s, f):
                    fh.write(f"{f - 1} {tid - 1} Car 0 0 0 {x:.2f} {y:.2f} {x + w:.2f} {y + h:.2f} 1 1 1 0 0 0 0\n")
                fh.write(f"{f - 1} -1 DontCare -1 -1 -10 0 0 10 10 -1 -1 -1 -1000 -1000 -1000 -10\n")
    before = sorted(p.name for p in (tmp / "results").iterdir())
    for env, root in (({"PC_DATASET": "mot20", "PC_TRACKER": "sort"}, m20), ({"PC_DATASET": "kitti"}, kit)):
        run_env(tmp, env, "step1_golden.py", "--backend", "fake", "--mot17", str(root))
        run_env(tmp, env, "step2_campaign.py", "--backend", "fake", "--mot17", str(root), "--n-act", "300", "--n-weight", "60")
        run_env(tmp, env, "step3_analyze.py")
        run_env(tmp, env, "step5_calibration.py")
        run_env(tmp, env, "step6_adaptive.py")
        run_env(tmp, env, "step7_weights.py", "--measure", "--backend", "fake", "--mot17", str(root))
        o = run_env(tmp, env, "step4_e2e.py", "--backend", "fake", "--mot17", str(root), "--rates", "1.0", "--seeds", "1")
        name = f"{env['PC_DATASET']}_yolov8s_{env.get('PC_TRACKER', 'bytetrack')}"
        out = tmp / "runs" / name
        for f in ("step1_report.txt", "step3_report.txt", "step4_report.txt", "step5_report.txt", "step6_report.txt", "step7_report.txt"):
            assert (out / f).exists(), (name, f)
        assert "## fault rate 1 per inference" in o, o
        print(f"config {name}: OK")
    assert sorted(p.name for p in (tmp / "results").iterdir()) == before, "default results/ was modified"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pc_selftest_"))
    try:
        for p in SRC.glob("*.py"):
            shutil.copy2(p, tmp / p.name)
        shutil.copytree(SRC / "trackers", tmp / "trackers", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(SRC / "third_party", tmp / "third_party", ignore=shutil.ignore_patterns("__pycache__"))
        sys.path.insert(0, str(SRC))
        from detector import FakeDetector
        fake = FakeDetector()
        mot = tmp / "MOT17"
        for s, n in SEQ_LEN.items():
            d = mot / "train" / f"{s}-FRCNN"
            d.mkdir(parents=True)
            (d / "seqinfo.ini").write_text(
                f"[Sequence]\nname={s}-FRCNN\nimDir=img1\nframeRate=30\nseqLength={n}\n"
                f"imWidth=1920\nimHeight=1080\nimExt=.jpg\n")
            (d / "gt").mkdir()
            with open(d / "gt" / "gt.txt", "w") as fh:
                for f in range(1, n + 1):
                    for tid, x, y, w, h in fake.ground_truth(s, f):
                        fh.write(f"{f},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")
        o1 = run(tmp, "step1_golden.py", "--backend", "fake", "--mot17", str(mot))
        assert "deterministic (bit-exact over 3 runs): True" in o1, o1
        run(tmp, "step2_campaign.py", "--backend", "fake", "--mot17", str(mot), "--n-act", "600", "--n-weight", "120")
        o3 = run(tmp, "step3_analyze.py")
        res = tmp / "results"
        with open(res / "campaign" / "act.csv", newline="") as fa, open(res / "campaign" / "weight.csv", newline="") as fw:
            act, ws = list(csv.DictReader(fa)), list(csv.DictReader(fw))
        assert len(act) == 600 and len(ws) == 120
        oc = {r["outcome"] for r in act}
        assert {"masked_det", "sdc_prop"} <= oc, oc
        # mantissa bits must be fully masked in the fake detector
        assert all(r["outcome"] == "masked_det" for r in act if int(r["bit"]) <= 9)
        # every fault that changed nothing at the detector must not change tracking either
        assert all(r["fn"] == r["fp"] == r["idsw"] == "0" for r in act if r["outcome"] == "masked_det")
        for needle in ("## A.", "## B.", "## C.", "## D.", "## E.", "physics"):
            assert needle in o3, needle
        assert (res / "figures" / "roc_act.png").exists() or "figures skipped" in o3
        o5 = run(tmp, "step5_calibration.py")
        assert "split-q12" in o5 and (res / "calibration_split_q12.json").exists(), o5
        o6 = run(tmp, "step6_adaptive.py")
        assert "## alpha = 1%" in o6 and (res / "step6_report.txt").exists(), o6
        o8 = run(tmp, "step8_sanitize.py")
        assert "nms0.5" in o8 and (res / "step8_report.txt").exists(), o8
        print(o8)
        od = run(tmp, "diag_escapes.py")
        assert "combined+" in od and (res / "diag_escapes.txt").exists(), od
        print(od)
        o9 = run(tmp, "step9_sealed.py", "--backend", "fake", "--mot17", str(mot))
        assert "sealed result" in o9 and (res / "step9_report.txt").exists(), o9
        print(o9)
        o9b = run(tmp, "step9_sealed.py", "--backend", "fake", "--mot17", str(mot), "--analyze")
        assert "sealed result" in o9b, o9b
        o10 = run(tmp, "step10_baselines.py", "--backend", "fake", "--mot17", str(mot))
        assert "ranger+combined" in o10 and "bit model 'all'" in o10 and (res / "step10_report.txt").exists(), o10
        print(o10)
        o10b = run(tmp, "step10b_range.py", "--backend", "fake", "--mot17", str(mot))
        assert "range-q12" in o10b and (res / "step10b_report.txt").exists(), o10b
        print(o10b)
        o7 = run(tmp, "step7_weights.py", "--measure", "--backend", "fake", "--mot17", str(mot))
        assert "check every" in o7 and (res / "step7_timing.json").exists(), o7
        o4 = run(tmp, "step4_e2e.py", "--backend", "fake", "--mot17", str(mot), "--rates", "0.3", "1.0", "--seeds", "2")
        assert "## fault rate 1 per inference" in o4, o4
        check_other_configs(tmp, fake)
        print(o1.split("Clean statistics")[0])
        print(o3)
        print(o4)
        print(o5)
        print(o6)
        print(o7)
        print("SELFTEST OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
