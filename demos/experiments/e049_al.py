"""E049 round 2, Part 2: the synthesis with the single-shooting augmented-Lagrangian solver of the manipulator
study (planar_al: E029's merit and step, E040's one constraint per conjunct), the unicycle as the plant.

Usage:
    python experiments/e049_al.py OUT_DIR solve {jax|warp} [ROW ...]   the named rows (default every row), both margins, both guesses
    python experiments/e049_al.py OUT_DIR report             table.csv / table.md, agreement.csv, synthesis.svg/.pdf/.png

Problem: the STL specification of the 3 s stop of Part 1 (T = 447). Normalized commands V in [-1, 1]^2
(U = V * U_MAX), effort E = sum(V^2) / N, merit lam E (LAM = 0.01, as E029 and E033) for every row.
Constraint form (lse_plain, lse_sound, gm_pm01, gm_pm10, sparsemax): every conjunct's smoothed robustness at least c.
lse_sound (round 3) is the sound log-sum-exp, the Warp backend's 'lse' (planar_disk.matched('lse_sound') on the JAX chain).
Authors' form of the generalized-mean robustness (gm_pm01_authors, gm_pm10_authors; its eq. (11)): the
whole specification as one program, the cost lam E - REWARD eta and eta >= c, with REWARD = lam 200 / N, the
ratio of their Example 4 (J = (1/2) sum ||u||^2 against lambda = 100) for the time-averaged E. Margins
c = 0 and C_POS. Initial guesses: S1 and S2 of the 3 s stop. BUDGET updates for every row, no other stopping
test. eps = 0.1 m per node.
"""
import csv
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from sparsemax_diffstl.tasks import planar as P0  # noqa: E402
from sparsemax_diffstl.tasks import planar_al as A  # noqa: E402
from sparsemax_diffstl.tasks import planar_disk as D  # noqa: E402
from sparsemax_diffstl.tasks import planar_oracle as O  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
mode = sys.argv[2]
EPS, WAIT, LAM, ALPHA0, BUDGET = 0.1, 30, 0.01, 0.002, 400
C_POS = 0.2
MARGINS = (0.0, C_POS)
u1, u2, tm = D.trajectories(WAIT, EPS)
T = tm["T"]
N = T - 1
REWARD = LAM * 200.0 / N
spec, conj, until = D.specification(tm["a1"], tm["b1"], tm["a2"], tm["b2"], T)
ROWS = {"lse_plain": ("lse_plain", False), "lse_sound": ("lse", False), "gm_pm01": ("gm_pm01", False), "gm_pm10": ("gm_pm10", False), "sparsemax": ("sparsemax", False),
        "gm_pm01_authors": ("gm_pm01", True), "gm_pm10_authors": ("gm_pm10", True)}
V0 = np.stack([u1 / D.U_MAX, u2 / D.U_MAX])  # the two initial guesses


def chain_for(row, backend):
    method, authors = ROWS[row]
    progs = (spec,) if authors else conj
    if backend == "jax":
        return A.JaxChain(progs, T, method, EPS, D.Z0, D.REGIONS, 2)
    from sparsemax_diffstl.tasks.planar_warp import WarpChain
    return WarpChain(progs, T, method, EPS, D.Z0, D.REGIONS, 2)


# ------------------------------------------------------------------
# solve mode

if mode == "solve":
    backend = sys.argv[3]
    log = open(out / ("solve_" + backend + "_seconds.txt"), "a")
    for row in sys.argv[4:] or list(ROWS):  # over the rows (a handful)
        ch = chain_for(row, backend)
        r0 = ch.forward(V0)
        g0 = ch.pullback(np.ones_like(r0))
        np.savez(out / ("start_" + backend + "_" + row + ".npz"), r=r0, g=g0)
        for c in MARGINS:  # over the two margins
            t0 = time.time()
            V, rs, Ls = A.solve(ch, V0, BUDGET, LAM, c, ALPHA0, REWARD if ROWS[row][1] else 0.0)
            np.savez(out / ("solve_" + backend + "_" + row + "_c" + str(c) + ".npz"), V=V, r=rs, L=Ls)
            log.write(row + " c=" + str(c) + " " + str(round(time.time() - t0, 1)) + "\n")
            log.flush()
            print(backend, row, c, "final smoothed", np.round(rs[:, -1].min(-1), 4))
    sys.exit(0)


# ------------------------------------------------------------------
# report mode

def measure(V):
    """Exact robustness and its conjuncts (NumPy evaluation of the definitions), clearances and effort."""
    z = np.asarray(P0.rollout(D.Z0, V * D.U_MAX))
    S = np.asarray(D.scores(jnp.asarray(z[:, :2]), D.REGIONS))
    parts = [O.ev(c_, S, np.array([0]))[0] for c_ in conj]  # over the conjuncts
    red = -S[:, 0]
    ks = np.arange(tm["a1"], tm["b1"] + 1)
    inner = np.minimum(S[ks, 1], np.minimum.accumulate(red)[ks])
    t_star = ks[np.argmax(inner)]
    return z, {"exact": float(O.ev(spec, S, np.array([0]))[0]), "until": parts[0], "blue": parts[1], "obstacle_conj": parts[2], "boundary": parts[3],
               "deciding": ("until", "Blue", "Obstacle", "Boundary")[int(np.argmin(parts))],
               "red_before_green": float(red[: t_star + 1].min()), "red_depth_over_eps": float(max(0.0, -red[: t_star + 1].min()) / EPS),
               "obstacle_clearance": float(np.min(-S[:, 3])), "effort_E": float(np.sum(V * V) / N), "effort_J": float(0.5 * np.sum((V * D.U_MAX) ** 2)),
               "v_min": float(np.min(V[:, 0] * D.U_MAX[0]))}


label = {"lse_plain": "LSE", "lse_sound": "sound LSE", "gm_pm01": "GMR (0,1)", "gm_pm10": "GMR (-10,10)", "sparsemax": "sparsemax", "gm_pm01_authors": "GMR (0,1), authors' form",
         "gm_pm10_authors": "GMR (-10,10), authors' form"}
rows, agree, paths = [], [], {}
for row in ROWS:
    for c in MARGINS:
        res = {b: np.load(out / ("solve_" + b + "_" + row + "_c" + str(c) + ".npz")) for b in ("jax", "warp") if (out / ("solve_" + b + "_" + row + "_c" + str(c) + ".npz")).exists()}
        for g, gname in enumerate(("S1", "S2")):
            for b, d in res.items():
                V = d["V"][g]
                z, m = measure(V)
                smoothed = d["r"][g, -1]
                rows.append({"backend": b, "row": label[row], "c": c, "guess": gname, "smoothed_min_over_constraints": float(smoothed.min()),
                             "smoothed_values": " ".join(str(round(float(x), 4)) for x in smoothed), "inputs_in_box": bool(np.all(np.abs(V) <= 1 + 1e-12)), **m})
                paths[(b, row, c, gname)] = z
            if len(res) == 2:
                zj, zw = paths[("jax", row, c, gname)], paths[("warp", row, c, gname)]
                agree.append({"row": label[row], "c": c, "guess": gname, "max_abs_diff_inputs": float(np.max(np.abs(res["jax"]["V"][g] - res["warp"]["V"][g]))),
                              "max_abs_diff_positions_m": float(np.max(np.abs(zj[:, :2] - zw[:, :2]))),
                              "max_abs_diff_smoothed_last": float(np.max(np.abs(res["jax"]["r"][g, -1] - res["warp"]["r"][g, -1])))})
for row in ROWS:
    st = {b: np.load(out / ("start_" + b + "_" + row + ".npz")) for b in ("jax", "warp") if (out / ("start_" + b + "_" + row + ".npz")).exists()}
    if len(st) == 2:
        agree.append({"row": label[row], "c": "initial guess", "guess": "S1 and S2", "max_abs_diff_smoothed_start": float(np.max(np.abs(st["jax"]["r"] - st["warp"]["r"]))),
                      "max_abs_diff_gradient_start": float(np.max(np.abs(st["jax"]["g"] - st["warp"]["g"]))), "max_abs_gradient_start": float(np.max(np.abs(st["jax"]["g"])))})


def write_csv(path, rr):
    keys = list(dict.fromkeys(k for r in rr for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rr)


write_csv(out / "table.csv", rows)
write_csv(out / "agreement.csv", agree)


def r3(x):
    return round(float(x), 3)


md = ["eps = " + str(EPS) + " m per node; budget " + str(BUDGET) + " updates for every row, no other stopping test; merit lam E with lam = " + str(LAM)
      + ", E = sum(V^2)/N; authors' form reward " + str(round(REWARD, 6)) + ". Margins in metres.", ""]
for b in ("jax", "warp"):
    sub = [r for r in rows if r["backend"] == b]
    if not sub:
        continue
    md += [b.upper() + " chain.", "", "| row | c | guess | smoothed (min over constraints) | smoothed at the end (per constraint) | exact (margin obtained) | set by | Red before Green | Obstacle | effort E | J |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sub:
        md.append("| " + " | ".join([r["row"], str(r["c"]), r["guess"], str(r3(r["smoothed_min_over_constraints"])), r["smoothed_values"], str(r3(r["exact"])), r["deciding"],
                                      str(r3(r["red_before_green"])), str(r3(r["obstacle_clearance"])), str(round(r["effort_E"], 4)), str(r3(r["effort_J"]))]) + " |")
    md.append("")
if agree:
    md += ["JAX against Warp.", "", "| row | c | guess | inputs | positions (m) | smoothed at the end | smoothed at the start | gradient at the start (largest entry) |", "|---|---|---|---|---|---|---|---|"]
    for a in agree:
        md.append("| " + " | ".join(str(x) for x in (a["row"], a["c"], a["guess"], a.get("max_abs_diff_inputs", ""), a.get("max_abs_diff_positions_m", ""), a.get("max_abs_diff_smoothed_last", ""),
                                                      a.get("max_abs_diff_smoothed_start", ""), str(a.get("max_abs_diff_gradient_start", "")) + (" (" + str(r3(a["max_abs_gradient_start"])) + ")" if "max_abs_gradient_start" in a else ""))) + " |")
(out / "table.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

colors = {"Red": "#d9534f", "Green": "#5cb85c", "Blue": "#428bca", "Obstacle": "#777777"}
style = {"lse_plain": ("#e08214", "-"), "lse_sound": ("#8c510a", "--"), "gm_pm01": ("#2166ac", "-"), "gm_pm10": ("#4393c3", "--"), "sparsemax": ("k", "-"),
         "gm_pm01_authors": ("#1b7837", "-"), "gm_pm10_authors": ("#5aae61", "--")}
fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.4))
z0 = np.asarray(P0.rollout(D.Z0, u1))
for ax, c in zip(axs, MARGINS):
    for nm, (cx, cy, rad) in zip(D.NAMES, D.REGIONS):
        ax.add_patch(Circle((cx, cy), rad, color=colors[nm], alpha=0.3, lw=0))
        ax.text(cx, cy, nm, ha="center", va="center", fontsize=7, color="0.3")
    ax.plot(z0[:, 0], z0[:, 1], ":", color="0.6", lw=0.8, label="initial guess $S_1$")
    for row in ROWS:
        if ("jax", row, c, "S1") in paths:
            z = paths[("jax", row, c, "S1")]
            ax.plot(z[:, 0], z[:, 1], style[row][1], color=style[row][0], lw=1.0, label=label[row])
            ax.plot(z[:, 0], z[:, 1], "o", color=style[row][0], ms=0.8)
    ax.plot(*D.Z0[:2], "k^", ms=5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (m)")
    ax.set_title("every conjunct's smoothed robustness $\\geq " + str(c) + "$ m (authors' form: $\\eta \\geq " + str(c) + "$)", fontsize=7)
    ax.legend(loc="upper right", fontsize=5.5, frameon=False)
axs[0].set_ylabel("$y$ (m)")
fig.tight_layout()
fig.savefig(out / "synthesis.svg")
fig.savefig(out / "synthesis.pdf")
fig.savefig(out / "synthesis_preview.png", dpi=150)
