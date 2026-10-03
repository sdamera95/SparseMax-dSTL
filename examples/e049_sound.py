"""E049 round 3, Part 1: the sound log-sum-exp as a fifth smoothing of the disk-region example, evaluated on
the fixed trajectories S1 and S2 of round 2 (no solver; trajectories, specification and eps unchanged).

Usage: python examples/e049_sound.py OUT_DIR [R2_TABLE_CSV]

Writes to OUT_DIR: table.csv (values of every smoothing on S1 and S2 at the four waits, the conjuncts of the
plain and the sound log-sum-exp, and on S1 the weights on the samples inside Red), node_errors.csv (the
realized error at each node of the path through the until: inner minimum, maximum over the witnesses, top
conjunction; sound and plain log-sum-exp and sparsemax), checks.csv (every JAX value against the NumPy
evaluation from the definitions, every weight against finite differences of it, and, with R2_TABLE_CSV,
the values and weights of the other smoothings against round 2's table), table.md.
"""
import csv
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from sparsemax_dstl.jax import budget  # noqa: E402
from sparsemax_dstl.stl import Atom, Until, compile_formula  # noqa: E402
from sparsemax_dstl.tasks import planar as P0  # noqa: E402
from sparsemax_dstl.tasks import planar_disk as D  # noqa: E402
from sparsemax_dstl.tasks import planar_oracle as O  # noqa: E402
from sparsemax_dstl.tasks import planar_oracle_sound as OS  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
EPS = 0.1
WAITS = (30, 120, 480, 1300)  # samples standing at the wait spot: 3, 12, 48 and 130 s
METHODS = ("exact", "lse_plain", "lse_sound", "gm01", "gm10", "sparsemax")
FD = 1e-6


def oracle(f, S, sem):
    if sem == "lse_sound":
        return OS.ev(f, S, np.array([0]), eps=EPS)[0]
    return O.ev(f, S, np.array([0]), sem=sem, eps=EPS)[0]


def reducer(sem):
    if sem == "lse_sound":
        return lambda z, v, kind: OS.reduce(z, v, kind, EPS)
    return lambda z, v, kind: O.reduce(z, v, kind, sem, EPS)


def node_errors(S, tm, spec, conj, sem):
    """Realized error at each node of the path through the until, each node on its own smoothed inputs
    (NumPy evaluation): the inner minimum at the witness with the largest smoothed inner value, the
    maximum over the witnesses, the top conjunction over the four conjuncts; and the whole error."""
    rd = reducer(sem)
    row, valid = O.until_rows(-S[:, 0], S[:, 1], tm["a1"], tm["b1"])
    inner_x = np.min(np.where(valid, row, np.inf), 1)
    inner_s = rd(row, valid, "min")
    outer_s = rd(inner_s[None], np.ones((1, len(inner_s)), bool), "max")[0]
    j = int(np.argmax(inner_s))
    conj_s = np.array([oracle(c, S, sem) for c in conj])  # over the four conjuncts
    top_s = rd(conj_s[None], np.ones((1, 4), bool), "min")[0]
    exact = oracle(spec, S, "exact")
    e = {"err_inner": inner_x[j] - inner_s[j], "err_outer": inner_s.max() - outer_s, "err_top": conj_s.min() - top_s}
    return {**{k: float(v) for k, v in e.items()}, "err_sum": float(sum(e.values())), "err_total": float(exact - top_s),
            "witness": int(tm["a1"] + j), "smoothed_until": float(outer_s), "smallest_conjunct": ("until", "Blue", "Obstacle", "Boundary")[int(np.argmin(conj_s))],
            "value_from_nodes": float(top_s), "value_oracle": float(oracle(spec, S, sem))}


def weight(S, tm, sem, viol):
    """AD weight on the samples viol: sum of d(until at 0)/d(not Red at the sample); and the central
    finite difference of the NumPy evaluation along viol (a test check)."""
    phi, psi = -S[:, 0], S[:, 1]
    g = jax.grad(lambda ph: P0.until_on_operands(ph, jnp.asarray(psi), tm["a1"], tm["b1"], D.matched(sem, EPS)))(jnp.asarray(phi))
    f = Until((tm["a1"], tm["b1"]), Atom(0), Atom(1))
    up = oracle(f, np.stack([phi + FD * viol, psi], 1), sem)
    dn = oracle(f, np.stack([phi - FD * viol, psi], 1), sem)
    return float(jnp.sum(jnp.where(viol, g, 0.0))), float((up - dn) / (2 * FD))


rows, nodes, checks = [], [], []
for wait in WAITS:  # over the four waits
    u1, u2, tm = D.trajectories(wait, EPS)
    T = tm["T"]
    spec, conj, _ = D.specification(tm["a1"], tm["b1"], tm["a2"], tm["b2"], T)
    B = float(budget(compile_formula(spec, T, reads=[(spec, [0])]), lambda m, param: np.where(np.asarray(m) > 1, EPS, 0.0))[0])
    for name, u in (("S1", u1), ("S2", u2)):
        z = np.asarray(P0.rollout(D.Z0, u))
        S = np.asarray(D.scores(jnp.asarray(z[:, :2]), D.REGIONS))
        row = {"trajectory": name, "wait": wait, "wait_s": round(wait * D.H, 1), "T": T, "a1": tm["a1"], "b1": tm["b1"], "budget_path": B}
        for m in METHODS:  # over the smoothings
            sv, cv = D.values(jnp.asarray(S), tm, D.matched(m, EPS))
            row["spec_" + m] = float(sv)
            ref = oracle(spec, S, m)
            checks.append({"wait": wait, "trajectory": name, "method": m, "quantity": "spec value", "jax": float(sv), "reference": float(ref), "abs_diff": abs(float(sv) - ref)})
            if m in ("lse_plain", "lse_sound"):
                for cn, c, v in zip(("until", "blue", "obstacle", "boundary"), conj, np.asarray(cv)):  # over the conjuncts
                    row[cn + "_" + m] = float(v)
                    ref = oracle(c, S, m)
                    checks.append({"wait": wait, "trajectory": name, "method": m, "quantity": cn + " value", "jax": float(v), "reference": float(ref), "abs_diff": abs(float(v) - ref)})
        for cn in ("until", "blue", "obstacle", "boundary"):
            row["shift_" + cn] = row[cn + "_lse_plain"] - row[cn + "_lse_sound"]
        row["sound_error"] = row["spec_exact"] - row["spec_lse_sound"]
        row["sparsemax_error"] = row["spec_exact"] - row["spec_sparsemax"]
        for m in ("lse_sound", "lse_plain", "sparsemax"):
            nodes.append({"trajectory": name, "wait": wait, "wait_s": row["wait_s"], "method": m, **node_errors(S, tm, spec, conj, m)})
        if name == "S1":
            viol = ((-S[:, 0]) < 0) & (np.arange(T) <= tm["b1"])
            row["k"] = int(viol.sum())
            for m in ("lse_plain", "lse_sound", "sparsemax"):
                w, fd = weight(S, tm, m, viol)
                row["weight_" + m] = w
                checks.append({"wait": wait, "trajectory": name, "method": m, "quantity": "weight vs finite difference of oracle", "jax": w, "reference": fd, "abs_diff": abs(w - fd)})
            row["weight_sound_minus_plain"] = row["weight_lse_sound"] - row["weight_lse_plain"]
        rows.append(row)

if len(sys.argv) > 2:
    with open(sys.argv[2]) as f:
        r2 = {(int(r["wait"]), r["trajectory"]): r for r in csv.DictReader(f)}
    for r in rows:  # reproduction of round 2's values and weights
        old = r2[(r["wait"], r["trajectory"])]
        for m in ("exact", "lse_plain", "gm01", "gm10", "sparsemax"):
            checks.append({"wait": r["wait"], "trajectory": r["trajectory"], "method": m, "quantity": "spec value vs round 2 table", "jax": r["spec_" + m],
                           "reference": float(old["spec_" + m]), "abs_diff": abs(r["spec_" + m] - float(old["spec_" + m]))})
        if r["trajectory"] == "S1":
            for m in ("lse_plain", "sparsemax"):
                checks.append({"wait": r["wait"], "trajectory": "S1", "method": m, "quantity": "weight vs round 2 table", "jax": r["weight_" + m],
                               "reference": float(old["weight_" + m]), "abs_diff": abs(r["weight_" + m] - float(old["weight_" + m]))})


def write_csv(path, rr):
    keys = list(dict.fromkeys(k for r in rr for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rr)


write_csv(out / "table.csv", rows)
write_csv(out / "node_errors.csv", nodes)
write_csv(out / "checks.csv", checks)
for label, sel in (("values against the NumPy evaluation", [c["abs_diff"] for c in checks if c["quantity"].endswith(" value")]),
                   ("weights against finite differences", [c["abs_diff"] for c in checks if "finite" in c["quantity"]]),
                   ("values and weights against round 2's table", [c["abs_diff"] for c in checks if "round 2" in c["quantity"]])):
    if sel:
        print("checks:", label, len(sel), "rows, max abs diff", max(sel))


# ------------------------------------------------------------------
# markdown

def r4(x):
    return round(float(x), 4)


md = ["eps = " + str(EPS) + " m per node; trajectories, specification and waits of round 2 (gate 2026-10-02T0854Z-r2-part1). Values in metres.", "",
      "| trajectory | wait (s) | exact | LSE | sound LSE | GMR (0,1) | GMR (-10,10) | sparsemax |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append("| " + r["trajectory"] + " | " + str(r["wait_s"]) + " | " + " | ".join(str(r4(r["spec_" + m])) for m in METHODS) + " |")
md += ["", "Sound LSE against the plain LSE, per conjunct: the plain value minus the sound value (the shift log(m)/beta = eps at the until's maximum over the witnesses and at the eventually of Blue; 0 for the conjuncts without a maximum node).", "",
       "| trajectory | wait (s) | until | Blue | Obstacle | Boundary | specification |", "|---|---|---|---|---|---|---|"]
for r in rows:
    md.append("| " + r["trajectory"] + " | " + str(r["wait_s"]) + " | " + " | ".join(str(r4(r["shift_" + c])) for c in ("until", "blue", "obstacle", "boundary"))
              + " | " + str(r4(r["spec_lse_plain"] - r["spec_lse_sound"])) + " |")
md += ["", "Realized error in units of eps at each node of the path through the until (each node on its own smoothed inputs; budget eps per node, " + str(r4(rows[0]["budget_path"] / EPS))
       + " eps along the deepest path). Negative: the node reports above the exact extremum.", "",
       "| trajectory | wait (s) | smoothing | whole error | inner minimum | maximum over witnesses | top conjunction | sum of the three | smallest smoothed conjunct |", "|---|---|---|---|---|---|---|---|---|"]
for n in nodes:
    md.append("| " + " | ".join(str(x) for x in (n["trajectory"], n["wait_s"], {"lse_sound": "sound LSE", "lse_plain": "LSE", "sparsemax": "sparsemax"}[n["method"]], r4(n["err_total"] / EPS),
                                                   r4(n["err_inner"] / EPS), r4(n["err_outer"] / EPS), r4(n["err_top"] / EPS), r4(n["err_sum"] / EPS), n["smallest_conjunct"])) + " |")
md += ["", "S1: weight on the k samples inside Red (derivative of the until at t = 0 with respect to not Red at those samples, by reverse-mode AD).", "",
       "| wait (s) | k | LSE | sound LSE | sound minus LSE | sound LSE, finite difference of the NumPy evaluation | sparsemax |", "|---|---|---|---|---|---|---|"]
for r in rows:
    if r["trajectory"] == "S1":
        fd = [c["reference"] for c in checks if c["wait"] == r["wait"] and c["method"] == "lse_sound" and "finite" in c["quantity"]][0]
        md.append("| " + " | ".join(str(x) for x in (r["wait_s"], r["k"], r4(r["weight_lse_plain"]), r4(r["weight_lse_sound"]), r["weight_sound_minus_plain"], r4(fd), r4(r["weight_sparsemax"]))) + " |")
(out / "table.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
