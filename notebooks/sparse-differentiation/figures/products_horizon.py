# Derivative products against the horizon on the two scenes, from results/products_horizon.csv: Hessian-vector products on
# the five-cell zone and Jacobian products on the 64-cell circle. Writes products-horizon.svg and .pdf in ../out/.
import csv
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

here = pathlib.Path(__file__).parent
out_dir = here.parent / "out"
out_dir.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "xtick.minor.size": 0, "ytick.minor.width": 0.4,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white", "savefig.facecolor": "white"})
T = [100, 200, 400]
with open(here.parent / "results" / "products_horizon.csv") as f:
    rows = list(csv.DictReader(f))


def series(scene, comparator):
    v = {int(r["horizon"]): float(r["value"]) for r in rows if r["scene"] == scene and r["comparator"] == comparator}
    return [v[t] for t in T]


def num(v):
    """The count with thousands commas; a median of two counts keeps its half."""
    whole = re.sub(r"(\d)(?=(\d{3})+$)", r"\1,", str(int(v)))
    return whole if v == int(v) else whole + ".5"


def panel(ax, lines, title, ylabel):
    for y, lab, ls, mk, col, off in lines:
        ax.plot(T, y, ls=ls, marker=mk, color=col, lw=1.0, ms=3.5, label=lab)
        for x, v in zip(T, y):
            ax.annotate(num(v), (x, v), xytext=(0, off), textcoords="offset points", ha="center", va="bottom" if off > 0 else "top", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(T, [str(t) + "\n" + str(t // 50) + " s" for t in T])
    ax.set_xticks([], minor=True)
    ax.set_xlim(80, 500)
    ax.set_ylim(4, 60000)
    ax.set_xlabel("time steps of the horizon (sampling period 0.02 s)", labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, fontsize=7.5, pad=31)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=6.5, frameon=False, handlelength=2.6, borderaxespad=0.1, labelspacing=0.25)


h = [series("five-cell zone", c) for c in ("dense AD", "prescribed pattern", "supports' pattern")]
j = [series("64-cell circle", c) for c in ("dense AD", "CasADi's detected pattern", "supports' pattern")]
fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.8))
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.175, top=0.78, wspace=0.28)
panel(axs[0], [(h[0], "dense AD", "-", "s", "0.15", 4), (h[1], "prescribed pattern (star coloring)", "--", "^", "#2b6cb0", 4),
               (h[2], "supports' pattern, sparsemax (star coloring)", ":", "o", "black", -4)],
      "Lagrangian Hessian, five-cell zone", "Hessian-vector products")
panel(axs[1], [(j[0], "dense AD (one product per constraint)", "-", "s", "0.15", 4),
               (j[1], "CasADi's coloring of its detected pattern", "--", "^", "#2b6cb0", 4),
               (j[2], "supports' pattern, sparsemax (row coloring)", ":", "o", "black", -4)],
      "constraint Jacobian, 64-cell circle", "Jacobian AD products")
for ext in ("svg", "pdf"):
    fig.savefig(out_dir / ("products-horizon." + ext), metadata={"Creator": None, "Date": None} if ext == "svg" else {"Creator": None, "Producer": None, "CreationDate": None})
print("Hessian", T, "dense", h[0], "prescribed", h[1], "supports", h[2])
print("Jacobian", T, "dense", j[0], "CasADi", j[1], "supports", j[2])
