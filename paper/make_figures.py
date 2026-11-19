#!/usr/bin/env python3
"""Paper figures and tables from summary.json (sealed test split, 8 configurations).

fig_pareto        recovered HOTA loss vs extra compute at fault rate 1, every policy (median over configs,
                  bars = min-max over configs)
fig_recovery      recovered HOTA loss per configuration at fault rate 1: white-box vs ours
fig_far           clean false alarm rate on the test split, fixed vs adaptive threshold, with 95% CI
tables.md         the paper's tables in Markdown (tables.tex: the same in LaTeX)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "summary.json").read_text())
ORDER = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLO11s", "RT-DETR", "YOLOv8s+SORT", "MOT20", "KITTI"]
CFG_LABEL = {"YOLOv8n": "MOT17 · YOLOv8n", "YOLOv8s": "MOT17 · YOLOv8s", "YOLOv8m": "MOT17 · YOLOv8m",
             "YOLO11s": "MOT17 · YOLO11s", "RT-DETR": "MOT17 · RT-DETR-L", "YOLOv8s+SORT": "MOT17 · YOLOv8s · SORT",
             "MOT20": "MOT20 · YOLOv8s", "KITTI": "KITTI · YOLOv8s"}

SHORT = {"YOLOv8n": "MOT17\nv8n", "YOLOv8s": "MOT17\nv8s", "YOLOv8m": "MOT17\nv8m", "YOLO11s": "MOT17\n11s",
         "RT-DETR": "MOT17\nRT-DETR", "YOLOv8s+SORT": "MOT17\nv8s SORT", "MOT20": "MOT20\nv8s", "KITTI": "KITTI\nv8s"}

# reference palette (light): text inks, surface, categorical slots 1-3 (validated all-pairs), neutral for baselines
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False})


def rec(h, none, clean):
    """Share of the unprotected HOTA loss that a policy recovers (100% = back to fault-free)."""
    return 100.0 * (h - none) / (clean - none)


def point(cfg, pol, rate="1"):
    """(recovered %, extra compute %) of one policy on one configuration, bit model 'exp'."""
    r = S[cfg]
    t9, t10 = r["e2e"][rate], r["e2e_exp"][rate]
    none, clean = t9["none"]["HOTA"], r["clean_hota"]
    row = t9.get(pol) or t10.get(pol)
    return rec(row["HOTA"], none, clean), row["compute"]


POLICIES = [  # key, label, color, marker, ours
    ("none", "No protection", MUTED, "o", False),
    ("ranger", "Range restriction (Ranger)", MUTED, "s", False),
    ("dmr", "DMR (run twice)", MUTED, "D", False),
    ("oracle", "Oracle re-execution", MUTED, "v", False),
    ("range-q12", "White-box monitor", AQUA, "^", False),
    ("combined", "Ours: black-box monitor", BLUE, "o", True),
    ("ranger+combined", "Ours + Ranger", ORANGE, "s", True),
]


def fig_pareto():
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.axhline(100, color=GRID, lw=1, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for key, label, col, mk, ours in POLICIES:
        pts = np.array([point(c, key) if key != "none" else (0.0, 0.0) for c in ORDER])
        y, x = pts[:, 0], pts[:, 1]
        ym, xm = np.median(y), np.median(x)
        ax.errorbar(xm, ym, yerr=[[ym - y.min()], [y.max() - ym]], xerr=[[xm - x.min()], [x.max() - xm]],
                    fmt=mk, ms=7 if (ours or key == "range-q12") else 6, color=col, mec=SURF, mew=1.5, elinewidth=1.2, capsize=0,
                    zorder=3 if ours else 2)
        dx, dy, ha = {"none": (0.3, 4, "left"), "ranger": (0.3, -9, "left"), "dmr": (0, -10, "right"),
                      "oracle": (0, -13, "center"), "range-q12": (0, -24, "center"),
                      "combined": (-8, -24, "right"), "ranger+combined": (-10, 10, "right")}[key]
        ax.annotate(label, (xm, ym), xytext=(dx * 10 if key in ("none", "ranger") else dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=7.5,
                    color=INK if ours else INK2, fontweight="bold" if ours else "normal")
    ax.set_xscale("symlog", linthresh=1, linscale=0.5)
    ax.set_xlim(-0.2, 300)
    ax.set_xticks([0, 1, 3, 10, 30, 100, 200])
    ax.set_xticklabels(["0", "1", "3", "10", "30", "100", "200"])
    ax.set_ylim(-5, 115)
    ax.set_xlabel("extra compute (% of inference calls)")
    ax.set_ylabel("HOTA loss recovered (%)")
    ax.set_title("Fault rate 1 per inference, sealed test, 8 configurations\n"
                 "marker = median, bars = range over configurations", fontsize=8, color=INK2, loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_pareto.{ext}", dpi=300)
    plt.close(fig)


def fig_recovery():
    series = [("range-q12", "White-box monitor", AQUA, "^"), ("combined", "Ours", BLUE, "o"),
              ("ranger+combined", "Ours + Ranger", ORANGE, "s")]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    yy = np.arange(len(ORDER))[::-1]
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for i, (key, label, col, mk) in enumerate(series):
        vals = [point(c, key) for c in ORDER]
        ax.scatter([v[0] for v in vals], yy + (1 - i) * 0.3, marker=mk, s=38, color=col, edgecolor=SURF,
                   linewidth=1.2, zorder=3, label=label)
    ax.axvline(100, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([CFG_LABEL[c] for c in ORDER], color=INK)
    ax.set_xlim(60, 115)
    ax.set_xlabel("HOTA loss recovered at fault rate 1 (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=3, fontsize=7.5, handletextpad=0.2,
              columnspacing=1.2, labelcolor=INK2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_recovery.{ext}", dpi=300)
    plt.close(fig)


def fig_far():
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    xx = np.arange(len(ORDER))
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.axhline(1.0, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.annotate("nominal α = 1%", (len(ORDER) - 0.5, 1.0), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=7, color=INK2)
    for off, key, label, col, mk in ((-0.15, "far_fixed", "Fixed split-conformal threshold", BLUE, "o"),
                                     (0.15, "far_adaptive", "Adaptive (ACI, γ = 0.005)", ORANGE, "s")):
        v = np.array([S[c][key][:3] for c in ORDER])
        ax.errorbar(xx + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, ms=5, color=col,
                    mec=SURF, mew=1.2, elinewidth=1.2, capsize=0, zorder=3, label=label)
    ax.set_xticks(xx)
    ax.set_xticklabels([SHORT[c] for c in ORDER], fontsize=7, color=INK)
    ax.set_ylabel("false alarms on clean frames (%)")
    ax.set_ylim(0, 6)
    ax.legend(loc="upper left", fontsize=7, labelcolor=INK2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_far.{ext}", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ tables
def tables():
    md, tex = [], []

    def add(title, head, rows, note=""):
        md.append(f"### {title}\n\n| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        md.extend("| " + " | ".join(r) + " |" for r in rows)
        md.append(f"\n{note}\n" if note else "")
        cols = "l" + "r" * (len(head) - 1)
        esc = lambda s: s.replace("%", r"\%").replace("&", r"\&").replace("±", r"$\pm$").replace("α", r"$\alpha$") \
            .replace("γ", r"$\gamma$").replace("·", r"$\cdot$").replace("–", "--").replace("≥", r"$\geq$")
        tex.append(f"% {title}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n" + " & ".join(esc(h) for h in head)
                   + r" \\" + "\n\\midrule\n" + "\n".join(" & ".join(esc(c) for c in r) + r" \\" for r in rows)
                   + "\n\\bottomrule\n\\end{tabular}\n")

    rows = []
    for c in ORDER:
        k = S[c]["campaign"]
        rows.append([CFG_LABEL[c], f"{k['changed'] / k['n'] * 100:.1f}%", f"{k['masked']:.1f}%",
                     f"{k['tracking']}", f"{k['propagated']}"])
    add("Table 1. Tracker masking of transient activation faults (6000 single bit flips per configuration, sealed test)",
        ["configuration", "detector output changed", "fully masked by tracker", "tracking changed", "propagated"], rows)

    rows = []
    for c in ORDER:
        r = S[c]
        f, t = r["far_fixed"], r["tpr"]
        rows.append([CFG_LABEL[c], f"{f[0]:.2f}% [{f[1]:.2f}, {f[2]:.2f}]", f"{r['far_adaptive'][0]:.2f}%",
                     f"{t['n']}", f"{t['tpr']:.1f}% [{t['lo']:.1f}, {t['hi']:.1f}]", f"{t['damage']:.1f}%",
                     f"{r['far_range_q12']:.2f}%"])
    add("Table 2. Monitor at nominal α = 1% on the sealed test split",
        ["configuration", "FAR fixed [95% CI]", "FAR adaptive", "propagated faults", "TPR [95% CI]",
         "damage caught", "FAR white-box"], rows,
        "FAR = false alarms on clean test frames. Damage = FN + FP + ID switches over 30 frames after the fault.")

    keys = [("none", "none"), ("ranger", "Ranger"), ("dmr", "DMR"), ("oracle", "oracle"),
            ("range-q12", "white-box"), ("combined", "ours"), ("ranger+combined", "ours + Ranger")]
    rows = []
    for c in ORDER:
        row = [CFG_LABEL[c], f"{S[c]['e2e']['1']['none']['dHOTA']:+.2f}"]
        for k, _ in keys[1:]:
            rv, cp = point(c, k)
            row.append(f"{rv:.0f}% / {cp:.1f}%")
        rows.append(row)
    add("Table 3. End to end at fault rate 1 (exponent and sign bits): HOTA loss recovered / extra compute",
        ["configuration", "dHOTA unprotected"] + [n for _, n in keys[1:]], rows,
        "3 seeds; identical fault sequences for every policy (common random numbers).")

    rows = []
    for c in ORDER:
        a = S[c]["e2e_all"]["1"]
        none, clean = a["none"]["HOTA"], S[c]["clean_hota"]
        cell = lambda p: f"{rec(a[p]['HOTA'], none, clean):.0f}% / {a[p]['compute']:.1f}%"
        rows.append([CFG_LABEL[c], f"{a['none']['dHOTA']:+.2f}", cell("dmr"), cell("oracle"), cell("combined")])
    add("Table 4. Bit model 'all' (flipped bit uniform over 16 bits), fault rate 1",
        ["configuration", "dHOTA unprotected", "DMR", "oracle", "ours"], rows)

    rows = []
    for det, cfg in (("YOLOv8n", "YOLOv8n"), ("YOLOv8s", "YOLOv8s"), ("YOLOv8m", "YOLOv8m"),
                     ("YOLO11s", "YOLO11s"), ("RT-DETR-L", "RT-DETR")):
        k = S[cfg]["checksum"]
        rows.append([det, f"{k['n_params'] / 1e6:.1f} M", f"{k['ms']:.2f} ms", f"{k['frame_ms']:.1f} ms",
                     f"{100 * k['ms'] / k['frame_ms']:.2f}%", f"{k['caught']}/{k['tested']}"])
    add("Table 5. Weight checksum (int64 sum of all FP16 words, one reduction per frame)",
        ["detector", "parameters", "checksum", "inference", "overhead", "random flips caught"], rows)

    rows = []
    for c in ORDER:
        r = S[c]
        d1, s1 = r["dev_e2e"]["1"], r["e2e"]["1"]
        dev_rec = rec(d1["combined"]["HOTA"], d1["none"]["HOTA"], d1["none"]["HOTA"] - d1["none"]["dHOTA"])
        rows.append([CFG_LABEL[c], f"{r['dev_far']:.2f}%", f"{r['far_fixed'][0]:.2f}%", f"{r['dev_tpr']:.1f}%",
                     f"{r['tpr']['tpr']:.1f}%", f"{dev_rec:.0f}%", f"{point(c, 'combined')[0]:.0f}%"])
    add("Table 6. Development (q2) vs sealed test, combined monitor, α = 1%",
        ["configuration", "FAR dev", "FAR test", "TPR dev", "TPR test", "recovered dev", "recovered test"], rows,
        "Dev FAR is in-sample for the threshold by construction; test is the frozen-protocol result.")

    (HERE / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (HERE / "tables.tex").write_text("% requires \\usepackage{booktabs}\n" + "\n".join(tex), encoding="utf-8")


if __name__ == "__main__":
    fig_pareto()
    fig_recovery()
    fig_far()
    tables()
    print("wrote", ", ".join(sorted(p.name for p in HERE.glob("fig_*.png"))), "tables.md tables.tex")
