"""E049 round 2, Part 1: the disk-region example evaluated on two fixed trajectories (no solver).

Usage: python examples/e049_disk.py OUT_DIR

Writes to OUT_DIR: table.csv / table.md (values, node errors, weights with the closed forms), scan.csv
(the targets at eps 0.05, 0.1, 0.15, 0.2), range.csv (weights over eps with the trajectory fixed),
ranges.csv (outside the favourable range), checks.csv (every smoothed value against the NumPy evaluation,
every weight against finite differences of it), params.json, trajectories as CSV, workspace.svg/.pdf/.png.
"""
import csv
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from sparsemax_dstl.jax.operators import lower_max, lower_min  # noqa: E402
from sparsemax_dstl.stl import Atom, Until  # noqa: E402
from sparsemax_dstl.tasks import planar as P0  # noqa: E402
from sparsemax_dstl.tasks import planar_disk as D  # noqa: E402
from sparsemax_dstl.tasks import planar_disk_jax as Dj  # noqa: E402
from sparsemax_dstl.tasks import planar_jax as P0j  # noqa: E402
from sparsemax_dstl.tasks import planar_oracle as O  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
EPS = 0.1
WAITS = (30, 120, 480, 1300)  # samples standing at the wait spot: 3, 12, 48 and 130 s
DESIGN = {"depth": 0.15, "stand": 0.23, "clear": 3.0}  # in units of eps
METHODS = ("lse_plain", "gm01", "gm10", "sparsemax")
FD = 1e-6


def sim(u):
    z = np.asarray(P0j.rollout(D.Z0, u))
    return z, np.asarray(Dj.scores(jnp.asarray(z[:, :2]), D.REGIONS))


def until_f(tm):
    return Until((tm["a1"], tm["b1"]), Atom(0), Atom(1))


def weight(S, tm, m, eps, viol):
    """AD weight on the samples viol: sum of d(until at 0)/d(not Red at the sample), the operands being the
    predicates themselves (not Red = -S[:, 0], Green = S[:, 1])."""
    phi, psi = -S[:, 0], S[:, 1]
    sem = P0j.matched(m, eps)
    val, g = jax.value_and_grad(lambda ph: P0j.until_on_operands(ph, jnp.asarray(psi), tm["a1"], tm["b1"], sem))(jnp.asarray(phi))
    w = float(jnp.sum(jnp.where(viol, g, 0.0)))
    S2 = np.stack([phi, psi], 1)
    up = O.ev(until_f(tm), S2 + FD * np.stack([viol, 0 * viol], 1), np.array([0]), sem=m, eps=eps)[0]
    dn = O.ev(until_f(tm), S2 - FD * np.stack([viol, 0 * viol], 1), np.array([0]), sem=m, eps=eps)[0]
    return float(val), w, (up - dn) / (2 * FD)


def node_errors(S, tm, eps):
    """Sparsemax's realized error at each node on S1's deciding path, each node on its own (smoothed) inputs:
    the inner minimum at the deciding witness, the maximum over the witnesses, the top conjunction."""
    row, valid = O.until_rows(-S[:, 0], S[:, 1], tm["a1"], tm["b1"])
    gam = 2 * eps / (1 - 1 / valid.sum(1))
    inner_x = np.min(np.where(valid, row, np.inf), 1)
    inner_s = np.asarray(jax.vmap(lower_min)(jnp.asarray(np.where(valid, row, 0.0)), jnp.asarray(gam), jnp.asarray(valid)))
    M = len(inner_s)
    outer_s = float(lower_max(jnp.asarray(inner_s), 2 * eps / (1 - 1 / M)))
    j = int(np.argmax(inner_s))
    _, conj_s = Dj.values(jnp.asarray(S), tm, P0j.matched("sparsemax", eps))
    conj_s = np.asarray(conj_s)
    top_s = float(lower_min(jnp.asarray(conj_s), 2 * eps / (1 - 1 / 4)))
    return {"err_inner": float(inner_x[j] - inner_s[j]), "err_outer": float(inner_s.max() - outer_s), "err_top": float(conj_s.min() - top_s)}


def evaluate(wait, eps=EPS, design=DESIGN, smooth_eps=None):
    """Rows for S1 and S2 at one wait; the trajectories are built at eps, smoothed at smooth_eps (default eps)."""
    se = eps if smooth_eps is None else smooth_eps
    u1, u2, tm = D.trajectories(wait, eps, **design)
    T, a1, b1 = tm["T"], tm["a1"], tm["b1"]
    spec, conj, until = D.specification(a1, b1, tm["a2"], tm["b2"], T)
    rows, checks = [], []
    for name, u in (("S1", u1), ("S2", u2)):
        z, S = sim(u)
        row = {"trajectory": name, "wait": wait, "eps": se, "T": T, "a1": a1, "b1": b1}
        for m in ("exact",) + METHODS:
            sv, cv = Dj.values(jnp.asarray(S), tm, P0j.matched(m, se))
            row["spec_" + m] = float(sv)
            row["until_" + m] = float(np.asarray(cv)[0])
            osv = O.ev(spec, S, np.array([0]), sem=m, eps=se)[0]
            checks.append({"wait": wait, "eps": se, "trajectory": name, "method": m, "quantity": "spec value", "jax": float(sv), "reference": float(osv), "abs_diff": abs(float(sv) - osv)})
        row["obstacle_clearance"] = float(np.min(-S[:, 3]))
        row["red_clearance_before_window"] = float(np.min(-S[: b1 + 1, 0]))
        if name == "S2":
            row["gm01_over_exact"] = row["spec_gm01"] / row["spec_exact"]
            row["gm10_over_exact"] = row["spec_gm10"] / row["spec_exact"]
            row["lse_minus_exact_over_eps"] = (row["spec_lse_plain"] - row["spec_exact"]) / se
            row["sparsemax_over_exact"] = row["spec_sparsemax"] / row["spec_exact"]
        if name == "S1":
            phi = -S[:, 0]
            pre = np.arange(T) <= b1
            viol = pre & (phi < 0)
            kmid = (a1 + b1) // 2
            m_mid, d_mid = kmid + 2, kmid + 1
            mid = np.arange(T) <= kmid
            level = tm["stand"]
            standing = mid & (np.abs(phi - level) < 1e-9)
            lo = phi[viol].min()
            beta = np.log(m_mid) / se
            k_eff = np.sum(np.exp(-beta * (phi[viol] - lo)))
            tail = mid & ~viol
            gap = level - lo
            row.update({"k": int(viol.sum()), "k_full_depth": int(np.sum(viol & (np.abs(phi - lo) < 1e-6))), "N": int(standing.sum()), "m": m_mid, "d": d_mid,
                        "M": b1 - a1 + 1, "depth_over_eps": -lo / se, "gap_over_eps": gap / se, "nearest_outside_over_eps": float(phi[tail].min() / se), "k_eff": float(k_eff),
                        "pred_lse_plain": float(k_eff / (k_eff + standing.sum() * np.exp(-beta * gap))),
                        "pred_lse_plain_all": float(k_eff / (k_eff + np.sum(np.exp(-beta * (phi[tail] - lo)))))})
            v = -phi[viol]
            for m, q in (("gm01", 1.0), ("gm10", 10.0)):
                Mq = (np.sum(v ** q) / d_mid) ** (1 / q)
                row["pred_" + m] = float(2 ** (-1 / q) * np.sum(v ** (q - 1)) / d_mid / Mq ** (q - 1))
            gam = 2 * se / (1 - 1 / m_mid)
            row["sm_condition"] = float(np.sum(phi[tail].min() - phi[viol]) - gam)
            row["pred_sparsemax"] = 1.0 if row["sm_condition"] >= 0 else float("nan")
            for m in METHODS:
                _, w, fd = weight(S, tm, m, se, viol)
                row["weight_" + m] = w
                checks.append({"wait": wait, "eps": se, "trajectory": name, "method": m, "quantity": "weight vs finite difference of oracle", "jax": w, "reference": float(fd), "abs_diff": abs(w - fd)})
            row_e, valid_e = O.until_rows(phi, S[:, 1], a1, b1)
            inner_x = np.min(np.where(valid_e, row_e, np.inf), 1)
            inner_l = O.reduce(row_e, valid_e, "min", "lse_plain", se)
            e = float(np.max(inner_x - inner_l))
            n_tie = int(np.sum(np.abs(inner_x - inner_x.max()) < 1e-12))
            row.update({"inner_error_lse": e, "n_tied": n_tie, "flip_bound": float(inner_x.max() - e + se * np.log(n_tie) / np.log(b1 - a1 + 1))})
            if smooth_eps is None:
                row.update(node_errors(S, tm, se))
        rows.append(row)
    return rows, checks, None, (u1, u2, tm)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def r3(x):
    return round(float(x), 4)


# ------------------------------------------------------------------
# the table at the chosen eps

rows, checks, trajs = [], [], {}
for wait in WAITS:
    r, c, _, tr = evaluate(wait)
    rows += r
    checks += c
    trajs[wait] = tr
write_csv(out / "table.csv", rows)

# ------------------------------------------------------------------
# the scan of eps (trajectories built at each eps, the design in units of eps)

scan = []
for e in (0.05, 0.1, 0.15, 0.2):
    for wait in (WAITS[0], WAITS[-1]):
        r, c, _, _ = evaluate(wait, e)
        checks += c
        s1, s2 = r
        scan.append({"eps": e, "wait": wait, "S1_lse_over_eps": s1["spec_lse_plain"] / e, "S1_sparsemax_error_over_eps": (s1["spec_exact"] - s1["spec_sparsemax"]) / e,
                     "S1_sparsemax_negative": s1["spec_sparsemax"] < 0, "S2_lse_minus_exact_over_eps": s2["lse_minus_exact_over_eps"],
                     "S2_sparsemax_over_c0": s2["sparsemax_over_exact"], "S2_gm01_over_c0": s2["gm01_over_exact"], "S2_gm10_over_c0": s2["gm10_over_exact"],
                     "w_lse": s1["weight_lse_plain"], "w_sparsemax": s1["weight_sparsemax"], "w_gm01": s1["weight_gm01"], "w_gm10": s1["weight_gm10"]})
write_csv(out / "scan.csv", scan)

# ------------------------------------------------------------------
# the range of eps with the trajectories fixed (built at EPS)

rng_rows = []
for se in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2):
    for wait in WAITS:
        r, c, _, _ = evaluate(wait, EPS, smooth_eps=se)
        s1 = r[0]
        rng_rows.append({"smooth_eps": se, "wait": wait, "w_lse": s1["weight_lse_plain"], "pred_lse_all": s1["pred_lse_plain_all"], "w_sparsemax": s1["weight_sparsemax"],
                         "sm_condition": s1["sm_condition"], "S1_lse": s1["spec_lse_plain"], "gap_over_eps": s1["gap_over_eps"]})
write_csv(out / "range.csv", rng_rows)

# ------------------------------------------------------------------
# outside the favourable range (wait 120)

ranges = []
for label, kw in (("chosen design", {}), ("deeper violation, depth 1.5 eps", {"design": {**DESIGN, "depth": 1.5}}),
                  ("smaller eps at smoothing, 0.05", {"smooth_eps": 0.05}), ("larger eps at smoothing, 0.2", {"smooth_eps": 0.2})):
    r, c, _, _ = evaluate(WAITS[1], **kw)
    checks += c
    ranges.append({"case": label, **r[0]})
write_csv(out / "ranges.csv", ranges)
write_csv(out / "checks.csv", checks)
print("checks: max abs diff of values", max(c["abs_diff"] for c in checks if "value" in c["quantity"]))
print("checks: max abs diff of weights vs finite differences", max(c["abs_diff"] for c in checks if "finite" in c["quantity"]))

# ------------------------------------------------------------------
# markdown

s1rows = [r for r in rows if r["trajectory"] == "S1"]
s2rows = [r for r in rows if r["trajectory"] == "S2"]
md = ["eps = " + str(EPS) + " m per node. S1 cuts into Red by " + str(r3(s1rows[0]["depth_over_eps"] * EPS)) + " m; S2 passes Red at c0 = " + str(r3(s2rows[0]["spec_exact"])) + " m.", "",
      "| trajectory | wait (s) | exact | LSE | GMR (0,1) | GMR (-10,10) | sparsemax |", "|---|---|---|---|---|---|---|"]
for r in rows:
    md.append("| " + r["trajectory"] + " | " + str(round(r["wait"] * D.H, 1)) + " | " + " | ".join(str(r3(r["spec_" + m])) for m in ("exact",) + METHODS) + " |")
md += ["", "S1 against the targets: LSE / eps (target >= 0.2); sparsemax's realized error / eps (target <= 1.0) and its error at each node of the deciding path.", "",
       "| wait (s) | LSE / eps | sign-flip bound / eps | sparsemax error / eps | inner minimum | maximum over witnesses | top conjunction |", "|---|---|---|---|---|---|---|"]
for r in s1rows:
    md.append("| " + " | ".join(str(x) for x in (round(r["wait"] * D.H, 1), r3(r["spec_lse_plain"] / EPS), r3(r["flip_bound"] / EPS), r3((r["spec_exact"] - r["spec_sparsemax"]) / EPS),
                                                   r3(r["err_inner"] / EPS), r3(r["err_outer"] / EPS), r3(r["err_top"] / EPS))) + " |")
md += ["", "S2 against the targets: (LSE - c0) / eps (target >= 0.3); sparsemax / c0 (target >= 0.6); GMR / c0.", "",
       "| wait (s) | c0 | (LSE - c0) / eps | sparsemax / c0 | GMR (0,1) / c0 | GMR (-10,10) / c0 |", "|---|---|---|---|---|---|"]
for r in s2rows:
    md.append("| " + " | ".join(str(x) for x in (round(r["wait"] * D.H, 1), r3(r["spec_exact"]), r3(r["lse_minus_exact_over_eps"]), r3(r["sparsemax_over_exact"]),
                                                   r3(r["gm01_over_exact"]), r3(r["gm10_over_exact"]))) + " |")
md += ["", "S1: weight on the k samples inside Red, measured by AD / closed form. LSE: k_eff/(k_eff + N m^(-gap/eps)), N the standing samples (in brackets: every sample outside Red at its own gap); GMR: 2^(-1/q) (1/d) sum v_i^(q-1)/M_q^(q-1); sparsemax: 1 when sum(gap_min - delta_i) - gamma >= 0.", "",
       "| wait (s) | k (at full depth) | N | m | depth/eps | gap/eps | LSE | GMR (0,1) | GMR (-10,10) | sparsemax | condition (m) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in s1rows:
    md.append("| " + " | ".join(str(x) for x in (round(r["wait"] * D.H, 1), str(r["k"]) + " (" + str(r["k_full_depth"]) + ")", r["N"], r["m"], r3(r["depth_over_eps"]), r3(r["gap_over_eps"])))
              + " | " + str(r3(r["weight_lse_plain"])) + " / " + str(r3(r["pred_lse_plain"])) + " (" + str(r3(r["pred_lse_plain_all"])) + ")"
              + " | " + " | ".join(str(r3(r["weight_" + m])) + " / " + str(r3(r["pred_" + m])) for m in ("gm01", "gm10", "sparsemax")) + " | " + str(r3(r["sm_condition"])) + " |")
md += ["", "The range of eps (trajectories fixed, built at eps = " + str(EPS) + "): LSE weight at each wait, and sparsemax's weight (minimum over the waits).", "",
       "| smoothing eps | gap/eps | LSE weight at " + ", ".join(str(round(w * D.H)) + " s" for w in WAITS) + " | sparsemax weight (min) |", "|---|---|---|---|"]
for se in sorted(set(r["smooth_eps"] for r in rng_rows)):
    rr = [r for r in rng_rows if r["smooth_eps"] == se]
    md.append("| " + str(se) + " | " + str(r3(rr[0]["gap_over_eps"])) + " | " + ", ".join(str(r3(r["w_lse"])) for r in rr) + " | " + str(r3(min(r["w_sparsemax"] for r in rr))) + " |")
md += ["", "Outside the favourable range (S1, wait " + str(round(WAITS[1] * D.H)) + " s).", "",
       "| case | smoothing eps | depth/eps | gap/eps | exact | LSE | sparsemax | LSE weight (pred) | sparsemax weight | GMR (0,1) weight | GMR (-10,10) weight |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in ranges:
    md.append("| " + " | ".join(str(x) for x in (r["case"], r["eps"], r3(r["depth_over_eps"]), r3(r["gap_over_eps"]), r3(r["spec_exact"]), r3(r["spec_lse_plain"]), r3(r["spec_sparsemax"]),
                                                   str(r3(r["weight_lse_plain"])) + " (" + str(r3(r["pred_lse_plain_all"])) + ")", r3(r["weight_sparsemax"]), r3(r["weight_gm01"]), r3(r["weight_gm10"]))) + " |")
md += ["", "Scan of eps (trajectories built at each eps; design in units of eps): waits " + str(round(WAITS[0] * D.H)) + " s and " + str(round(WAITS[-1] * D.H)) + " s.", "",
       "| eps | wait (s) | S1 LSE/eps | S1 sparsemax error/eps | S2 (LSE-c0)/eps | S2 sparsemax/c0 | S2 GMR(0,1)/c0 | S2 GMR(-10,10)/c0 | w LSE | w sparsemax |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in scan:
    md.append("| " + " | ".join(str(x) for x in (r["eps"], round(r["wait"] * D.H), r3(r["S1_lse_over_eps"]), r3(r["S1_sparsemax_error_over_eps"]), r3(r["S2_lse_minus_exact_over_eps"]),
                                                   r3(r["S2_sparsemax_over_c0"]), r3(r["S2_gm01_over_c0"]), r3(r["S2_gm10_over_c0"]), r3(r["w_lse"]), r3(r["w_sparsemax"]))) + " |")
(out / "table.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

# ------------------------------------------------------------------
# trajectories, parameters, plot

for wait, (u1, u2, tm) in trajs.items():
    for name, u in (("S1", u1), ("S2", u2)):
        z, _ = sim(u)
        uu = np.concatenate([u, np.full((1, 2), np.nan)])
        np.savetxt(out / ("traj_" + name + "_wait" + str(wait) + ".csv"), np.column_stack([np.arange(len(z)) * D.H, z, uu]), delimiter=",",
                   header="t_s,x_m,y_m,theta_rad,v_mps,omega_radps", comments="")
params = {"eps_per_node_m": EPS, "h_s": D.H, "v_max_mps": P0.V_MAX, "omega_max_radps": P0.W_MAX, "z0": D.Z0, "regions_cx_cy_r": dict(zip(D.NAMES, D.REGIONS.tolist())),
          "boundary": [0, 10, 0, 10], "design_in_eps": DESIGN, "waits_samples": WAITS, "timing_samples": {w: t[2] for w, t in trajs.items()},
          "green_stop": D.GREEN_STOP, "blue_stop": D.BLUE_STOP, "green_heading": D.GREEN_HEADING, "blue_heading": D.BLUE_HEADING, "cruise_mps": D.V_CRUISE,
          "matching": "beta = log(m)/eps, gamma = 2 eps/(1 - 1/m), m the valid entries of each row; one entry exact"}
(out / "params.json").write_text(json.dumps(params, indent=1, default=float))

u1, u2, tm = trajs[WAITS[1]]
z1, _ = sim(u1)
z2, _ = sim(u2)
colors = {"Red": "#d9534f", "Green": "#5cb85c", "Blue": "#428bca", "Obstacle": "#777777"}
fig, ax = plt.subplots(figsize=(4.6, 4.6))
for nm, (cx, cy, r) in zip(D.NAMES, D.REGIONS):
    ax.add_patch(Circle((cx, cy), r, color=colors[nm], alpha=0.35, lw=0))
    ax.text(cx, cy, nm, ha="center", va="center", fontsize=8)
ax.plot(z1[:, 0], z1[:, 1], "-", color="k", lw=0.9, label="$S_1$")
ax.plot(z1[:, 0], z1[:, 1], "o", color="k", ms=1.0)
ax.plot(z2[:, 0], z2[:, 1], "--", color="#8a2be2", lw=0.9, label="$S_2$")
ax.plot(z2[:, 0], z2[:, 1], "s", color="#8a2be2", ms=0.9)
ax.plot(*D.Z0[:2], "k^", ms=6)
ax.text(D.Z0[0] + 0.15, D.Z0[1] - 0.35, "$t=0$", fontsize=8)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_xlabel("$x$ (m)")
ax.set_ylabel("$y$ (m)")
ax.legend(loc="upper right", fontsize=8, frameon=False)
ins = ax.inset_axes([0.38, 0.09, 0.3, 0.24])
ins.axhspan(-0.1, 0.0, color=colors["Red"], alpha=0.35, lw=0)
rr = np.hypot(z1[:, 0] - D.RED[0], z1[:, 1] - D.RED[1])
ang = np.degrees(np.arctan2(z1[:, 1] - D.RED[1], z1[:, 0] - D.RED[0])) % 360
sel = (ang > 140) & (ang < 230) & (np.arange(len(z1)) <= tm["b1"])
ins.plot(ang[sel], rr[sel] - D.RED[2], "o-", color="k", ms=1.6, lw=0.6)
ins.set_xlim(230, 140)
ins.set_ylabel("distance outside Red (m)", fontsize=5)
ins.set_ylim(-0.06, 0.12)
ins.set_xlabel("angle about Red's centre (deg)", fontsize=5)
ins.tick_params(labelsize=5)
ins.set_title("$S_1$ beside Red", fontsize=6)
fig.tight_layout()
fig.savefig(out / "workspace.svg")
fig.savefig(out / "workspace.pdf")
fig.savefig(out / "workspace_preview.png", dpi=150)
