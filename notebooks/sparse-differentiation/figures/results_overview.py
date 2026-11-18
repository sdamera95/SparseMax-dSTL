# The results overview: (a) the time of every control period of the closed loop after the hold-up, (b) one Hessian of the
# Lagrangian on the five-cell zone and (c) one constraint Jacobian on the 64-cell circle, from results/closed_loop.csv and
# results/counts.csv. Writes results-overview.svg and .pdf in ../out/ and prints every number drawn.
import csv
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

here = pathlib.Path(__file__).parent
out_dir = here.parent / "out"
out_dir.mkdir(exist_ok=True)
results = here.parent / "results"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2, "ytick.major.size": 2,
                     "ytick.minor.size": 0, "pdf.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
blue, grey, light = "#1f5fa8", "0.3", "0.62"
TS = (100, 200, 400)


def read(name):
    with open(results / name) as f:
        return list(csv.DictReader(f))


def loop(comparator, quantity):
    """(runs, 20): each run's time of the 20 control periods after the first."""
    r = [x for x in read("closed_loop.csv") if x["comparator"] == comparator and x["quantity"] == quantity]
    runs, periods = max(int(x["run"]) for x in r), max(int(x["period"]) for x in r) + 1
    t = np.full((runs, periods), np.nan)
    t[[int(x["run"]) - 1 for x in r], [int(x["period"]) for x in r]] = [float(x["value"]) for x in r]
    return t[:, 1:]


def times(scene, quantity, comparator):
    """Median, first and third quartile (s) at 100, 200 and 400 time steps; NaN where the cell was not measured."""
    out = np.full((3, 3), np.nan)
    for x in read("counts.csv"):
        if x["scene"] == scene and x["quantity"] == quantity and x["comparator"] == comparator:
            out[TS.index(int(x["horizon"]))] = [float(x["value"]), float(x["first_quartile"]), float(x["third_quartile"])]
    return out


ours = loop("SparseMax (ours)", "whole control period")
base = loop("Sound-LSE", "solve time")
names = ("dense AD", "Sound-LSE", "SparseMax (ours)")
HESS = {c: times("five-cell zone", "Hessian of the Lagrangian time", c) for c in names}
JAC = {c: times("64-cell circle", "constraint Jacobian time", c) for c in names}
k = np.arange(1, ours.shape[1] + 1)


def band(ax, t, color, marker, ms):
    lo, md, hi = t.min(0), np.median(t, 0), t.max(0)
    ax.fill_between(k, lo, hi, color=color, alpha=0.28, lw=0)
    ax.errorbar(k, md, yerr=[md - lo, hi - md], fmt="none", ecolor=color, elinewidth=0.6, capsize=1.3, capthick=0.5)
    ax.plot(k, md, "-" + marker, color=color, ms=ms, lw=0.9, mec="white", mew=0.4)


fig, axes = plt.subplots(1, 3, figsize=(5.76, 1.66), gridspec_kw={"width_ratios": [1.3, 1, 1]})
ax = axes[0]
band(ax, base, grey, "s", 3.2)
band(ax, ours, blue, "o", 3.4)
ax.axhline(0.5, color="black", lw=0.6, ls=(0, (4, 2)))
dense_hessian = HESS["dense AD"][0, 0]
ax.axhline(dense_hessian, color=light, lw=0.9, ls=(0, (1, 1.5)))
ax.text(20.3, dense_hessian * 0.86, "dense AD: one Hessian", ha="right", va="top", fontsize=6.5, color="0.4")
ax.set_yscale("log")
ax.set_ylim(0.07, 60)
ax.set_xlim(0.5, 20.5)
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlabel("control period after the hold-up", labelpad=1)
ax.set_ylabel("time per control period (s)")
ax.set_title("(a) closed loop, a new plan every 0.5 s", fontsize=7.5, pad=3)
for ax, d, title in ((axes[1], HESS, "(b) Hessian of the Lagrangian"), (axes[2], JAC, "(c) constraint Jacobian")):
    x = np.arange(3)
    for name, color, marker, ms in (("dense AD", light, "^", 3.4), ("Sound-LSE", grey, "s", 3.2), ("SparseMax (ours)", blue, "o", 3.4)):
        md, q1, q3 = d[name].T
        ax.errorbar(x, md, yerr=[md - q1, q3 - md], fmt="none", ecolor=color, elinewidth=0.6, capsize=1.3, capthick=0.5)
        ax.plot(x, md, "-" + marker, color=color, ms=ms, lw=0.9, mec="white", mew=0.4)
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 2e3)
    ax.set_xlim(-0.3, 2.3)
    ax.set_xticks(x)
    ax.set_xticklabels(["100", "200", "400"])
    ax.set_xlabel("time steps in the horizon", labelpad=1)
    ax.set_title(title, fontsize=7.5, pad=3)
axes[1].set_ylabel("time of one evaluation (s)")
axes[2].set_yticklabels([])
for ax in axes:
    ax.grid(axis="y", which="major", color="0.9", lw=0.4)
    ax.set_axisbelow(True)
    ax.minorticks_off()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
h = [plt.Line2D([], [], marker="o", color=blue, ms=3.4, lw=0.9, label="SparseMax, our patterns (JAX + asdex)"),
     plt.Line2D([], [], marker="s", color=grey, ms=3.2, lw=0.9, label="Sound-LSE (CasADi)"),
     plt.Line2D([], [], marker="^", color=light, ms=3.4, lw=0.9, label="dense AD (JAX)")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7, handlelength=1.8, columnspacing=1.6)
fig.subplots_adjust(left=0.08, right=0.995, bottom=0.205, top=0.78, wspace=0.34)
p1, p2 = axes[1].get_position(), axes[2].get_position()
axes[2].set_position([p1.x1 + 0.025, p2.y0, p2.x1 - p1.x1 - 0.025, p2.height])
for ext in ("svg", "pdf"):
    fig.savefig(out_dir / ("results-overview." + ext), metadata={"Creator": None, "Date": None} if ext == "svg" else {"Creator": None, "Producer": None, "CreationDate": None})
for name, t in (("SparseMax (ours), whole control period", ours), ("Sound-LSE, solve alone", base)):
    print(name, "| runs", t.shape[0], "| median per period", np.median(t, 0).round(3).tolist())
    print("  smallest", round(float(t.min()), 3), "largest", round(float(t.max()), 3))
for name, d in (("Hessian, five-cell zone", HESS), ("Jacobian, 64-cell circle", JAC)):
    for c in names:
        print(name, c, "| median, first and third quartile (s) at 100, 200, 400:", d[c].round(5).tolist())
