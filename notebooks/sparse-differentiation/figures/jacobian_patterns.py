# The specification's rows of the constraint Jacobian at one control period of the 64-cell circle: the prescribed pattern above
# the supports' pattern. Reads ../out/patterns.npz (written by patterns.ipynb); writes jacobian-patterns.svg and .pdf there.
# A cell is one arm's 7 joint positions at one time step, shaded by the number of them a row reads; no row reads a velocity or
# an acceleration. The dynamics' rows are the same in both patterns and are not drawn; the products are the whole Jacobian's.
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

here = pathlib.Path(__file__).parent
out_dir = here.parent / "out"
out_dir.mkdir(exist_ok=True)
z = np.load(out_dir / "patterns.npz")
W, NJ = 302, 7  # figure width in points, joints per arm
Y0 = 28
H = Y0 + 2 * 40 + 11 + 12 + 10
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
                     "xtick.major.size": 1.5, "ytick.major.size": 1.5, "xtick.major.pad": 1, "ytick.major.pad": 1,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white", "savefig.facecolor": "white"})
NORM = BoundaryNorm(np.arange(0.5, NJ + 1), NJ)
CMAP = ListedColormap(plt.cm.Blues(np.linspace(0.3, 1.0, NJ)))
GROUPS = {"until": r"$\mathrm{out}_{o,z}\,\mathbf{U}_{[0,99]}\,\mathrm{out}_{v,z}$", "reach": r"$\mathbf{F}_{[0,d]}\,g$"}


def comma(n):
    return re.sub(r"(\d)(?=(\d{3})+$)", r"\1,", str(int(n)))


def axes_pt(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


T = int(z["jacobian_T"])
kind = z["jacobian_row_kind"]
R, D = kind.size, 28 * T
groups = [f for f in ("until", "reach") if (kind == f).any()]
label = {lev: ("prescribed pattern: " if lev == "prescribed" else "supports' pattern (SparseMax): ") + comma(z["jacobian_entries_" + lev]) + " entries, "
         + str(int(z["jacobian_products_" + lev])) + " AD products" for lev in ("prescribed", "supports")}
fig = plt.figure(figsize=(W / 72, H / 72))
L, aw, g, sw, ph = 66, 86, 4, 13, 40  # label room, arm panel width, gap, strip width, panel height
x_s = L + 2 * aw + g + 5
for i, (lev, y) in enumerate((("prescribed", Y0 + ph + 11), ("supports", Y0))):
    cells = z["jacobian_cells_" + lev]
    axs = [axes_pt(fig, L + a * (aw + g), y, aw, ph) for a in (0, 1)] + [axes_pt(fig, x_s, y, sw, ph)]
    fig.text(2 / W, (y + ph + 1.5) / H, label[lev], va="bottom", ha="left", fontsize=8)
    for a, ax in enumerate(axs):  # right arm, left arm, velocities and accelerations
        if a < 2:
            ax.pcolormesh(np.arange(T + 1), np.arange(R + 1), np.ma.masked_equal(cells[:, a], 0), cmap=CMAP, norm=NORM, edgecolors="face",
                          linewidth=0.1, rasterized=True)
            t_ = np.arange(0, T, T // 4)
            ax.set_xticks(t_ + 0.5, [str(v) for v in t_] if i == 1 else [])
        else:
            ax.text(0.5, R / 2, "no entries", rotation=90, ha="center", va="center", fontsize=7)
            ax.set_xticks([])
        ax.set_xlim(0, T if a < 2 else 1)
        ax.set_ylim(R, 0)
        ax.set_yticks([])
        for f in groups[1:]:
            ax.axhline(np.flatnonzero(kind == f)[0], color="0.45", lw=0.3)
        for s in ax.spines.values():
            s.set_linewidth(0.7)
    for f in groups:
        rows = np.flatnonzero(kind == f)
        txt = GROUPS[f] + "\n(" + str(rows.size) + " rows)" if f == "until" else GROUPS[f] + " (" + str(rows.size) + ")"
        mid = rows.mean() + 0.5 - 8 if f == "until" else R - 10
        axs[0].annotate(txt, xy=(0, rows.mean() + 0.5), xycoords=("axes fraction", "data"), xytext=(-3 / aw, mid), textcoords=("axes fraction", "data"),
                        ha="right", va="center", annotation_clip=False, fontsize=7, linespacing=1.0,
                        arrowprops={"arrowstyle": "-", "lw": 0.3, "color": "0.35", "shrinkA": 1, "shrinkB": 0})
    if i == 0:
        fig.text((L + aw / 2) / W, (y + ph + 12) / H, "right arm", ha="center", va="bottom", fontsize=8)
        fig.text((L + aw + g + aw / 2) / W, (y + ph + 12) / H, "left arm", ha="center", va="bottom", fontsize=8)
for a in (0, 1):
    fig.text((L + a * (aw + g) + aw / 2) / W, (Y0 - 10.5) / H, "joint positions by time step", ha="center", va="top", fontsize=7)
fig.text((x_s + sw / 2 + 4) / W, (Y0 - 1) / H, "velocities,\naccelerations\n(" + comma(D) + ")", ha="center", va="top", fontsize=7, linespacing=1.0)
cax = axes_pt(fig, x_s + sw + 6, Y0, 4, 2 * ph + 11)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, ticks=np.arange(1, NJ + 1))
cb.outline.set_linewidth(0.3)
cb.ax.tick_params(labelsize=7, pad=1, width=0.4, length=1.5)
cb.set_label("joints of the arm read at the time step", fontsize=7, labelpad=1)
for ext in ("svg", "pdf"):
    fig.savefig(out_dir / ("jacobian-patterns." + ext), dpi=600, metadata={"Creator": None, "Date": None} if ext == "svg" else {"Creator": None, "Producer": None, "CreationDate": None})
print("period", int(z["jacobian_period"]), "T", T, "rows", {f: int((kind == f).sum()) for f in groups})
print(label["prescribed"])
print(label["supports"])
