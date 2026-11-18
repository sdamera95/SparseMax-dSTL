# The specification's part of the Lagrangian Hessian at one control period of the five-cell zone, upper triangle: dense AD, the
# prescribed pattern and the supports' pattern side by side, and a stretch of 40 time steps enlarged. Reads ../out/patterns.npz
# (written by patterns.ipynb); writes hessian-patterns.svg and .pdf there. A cell is the 14 x 14 block of the joint positions
# at two time steps, shaded by the number of upper entries in it; every entry of both patterns lies in the joint positions. The
# entries are the drawn part's, the products those of the whole Lagrangian Hessian.
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

here = pathlib.Path(__file__).parent
out_dir = here.parent / "out"
out_dir.mkdir(exist_ok=True)
z = np.load(out_dir / "patterns.npz")
P, B, g, ZL = 53, 20, 16, 40  # panel side, panel bottom, gap (points); enlarged time steps
W = 320
H = B + P + 19 + 10
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
                     "xtick.major.size": 1.5, "ytick.major.size": 1.5, "xtick.major.pad": 1, "ytick.major.pad": 1,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white", "savefig.facecolor": "white"})
CMAP = ListedColormap(plt.cm.Blues(np.linspace(0.25, 1.0, 256)))


def comma(n):
    return re.sub(r"(\d)(?=(\d{3})+$)", r"\1,", str(int(n)))


def axes_pt(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def time_axis(ax, lo, hi, which, labels, n=4):
    t_ = np.linspace(lo, hi, n + 1).astype(int)[:-1]
    if which == "x":
        ax.set_xticks(t_ + 0.5, [str(v) for v in t_])
        ax.set_xlabel("time step", labelpad=0.5)
    else:
        ax.set_yticks(t_ + 0.5, [str(v) for v in t_] if labels else [])
        if labels:
            ax.set_ylabel("time step", labelpad=0.5)


T = int(z["hessian_T"])
nv = 42 * T
Hc = {lev: z["hessian_cells_" + lev] for lev in ("prescribed", "supports")}
head = {"dense": ("dense AD", comma(nv) + r" $\times$ " + comma(nv), comma(nv) + " products")}
for lev, title in (("prescribed", "prescribed pattern"), ("supports", "supports (SparseMax)")):
    head[lev] = (title, comma(z["hessian_upper_" + lev]) + " entries", comma(z["hessian_products_" + lev]) + " products")

# the enlarged stretch: the ZL time steps whose diagonal square holds the most cells of the supports' pattern
i2, j2 = np.nonzero(Hc["supports"])
s0 = np.arange(T - ZL + 1)
inside = (i2[None, :] >= s0[:, None]) & (j2[None, :] < s0[:, None] + ZL)
a = int(s0[np.argmax(inside.sum(1))])
b = a + ZL

fig = plt.figure(figsize=(W / 72, H / 72))
xs = [21 + i * (P + g) for i in range(3)] + [W - 16 - P]
for i, lev in enumerate(("dense", "prescribed", "supports")):
    ax = axes_pt(fig, xs[i], B, P, P)
    if lev == "dense":
        ax.add_patch(plt.Polygon([(0, 0), (T, 0), (T, T)], closed=True, color=CMAP(1.0), lw=0))
    else:
        ax.pcolormesh(np.arange(T + 1), np.arange(T + 1), np.ma.masked_equal(Hc[lev], 0), cmap=CMAP, vmin=1, vmax=196, edgecolors="face",
                      linewidth=0.1, rasterized=True)
        ax.add_patch(plt.Rectangle((a, a), ZL, ZL, fill=False, ec="black", lw=0.4, ls=(0, (2, 1))))
    ax.plot([0, T], [0, T], color="0.6", lw=0.25)
    ax.set_xlim(0, T)
    ax.set_ylim(T, 0)
    time_axis(ax, 0, T, "x", True)
    time_axis(ax, 0, T, "y", i == 0)
    x0 = (xs[i] + P / 2) / W
    fig.text(x0, (B + P + 2) / H, head[lev][1] + "\n" + head[lev][2], ha="center", va="bottom", fontsize=7, linespacing=1.0)
    fig.text(x0 + (0, -7, 7)[i] / W, (B + P + 19) / H, head[lev][0], ha="center", va="bottom", fontsize=8)
cax = axes_pt(fig, xs[2] + P + 3, B, 3.5, P)
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(1, 196)), cax=cax, ticks=[1, 98, 196])
cb.outline.set_linewidth(0.3)
cb.ax.tick_params(labelsize=7, pad=1, width=0.4, length=1.5)
ax = axes_pt(fig, xs[3], B, P, P)
Pz = (Hc["prescribed"][a:b, a:b] > 0).astype(float) + (Hc["supports"][a:b, a:b] > 0)
ax.pcolormesh(np.arange(a, b + 1), np.arange(a, b + 1), np.ma.masked_equal(Pz, 0), cmap=ListedColormap(["0.78", "black"]), vmin=1, vmax=2,
              edgecolors="white", linewidth=0.1)
ax.set_xlim(a, b)
ax.set_ylim(b, a)
time_axis(ax, a, b, "x", True)
time_axis(ax, a, b, "y", True)
ax.set_ylabel("")
ax.yaxis.tick_right()
fig.text((W - 1) / W, (B + P + 2) / H, "grey: prescribed only\nblack: supports' pattern", ha="right", va="bottom", fontsize=7, linespacing=1.0)
fig.text((W - 1) / W, (B + P + 19) / H, "time steps " + str(a) + " to " + str(b - 1), ha="right", va="bottom", fontsize=8)
for ext in ("svg", "pdf"):
    fig.savefig(out_dir / ("hessian-patterns." + ext), dpi=600, metadata={"Creator": None, "Date": None} if ext == "svg" else {"Creator": None, "Producer": None, "CreationDate": None})
print("period", int(z["hessian_period"]), "T", T, "enlarged time steps", a, "to", b - 1)
for lev in ("prescribed", "supports"):
    print(head[lev])
