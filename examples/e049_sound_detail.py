"""E049 round 3, Part 2 detail: at each returned trajectory, which entry of the until decides the exact margin, and
the realized error of the until conjunct node by node (no solve; the saved inputs of round 2 and round 3 are read).

Usage: python examples/e049_sound_detail.py OUT_DIR R3_PART2_DIR R2_PART2_DIR

For the plain LSE, the sound LSE and sparsemax, both chains, levels 0 and 0.2 m and guesses S1 and S2: the exact
margin, the clearance from Red up to the deciding witness and Green's depth at that witness (exact, NumPy
evaluation), the smoothed until value (NumPy evaluation of the smoothing), its realized error (exact until minus
smoothed until) split into the inner minimum at the witness with the largest smoothed inner value and the maximum
over the witnesses. Writes detail.csv and detail.md to OUT_DIR.
"""
import csv
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from sparsemax_dstl.tasks import planar_disk as D  # noqa: E402
from sparsemax_dstl.tasks import planar_disk_jax as Dj  # noqa: E402
from sparsemax_dstl.tasks import planar_jax as P0j  # noqa: E402
from sparsemax_dstl.tasks import planar_oracle as O  # noqa: E402
from sparsemax_dstl.tasks import planar_oracle_sound as OS  # noqa: E402

out, r3, r2 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
EPS = 0.1
u1, u2, tm = D.trajectories(30, EPS)
a1, b1 = tm["a1"], tm["b1"]
spec, conj, _ = D.specification(a1, b1, tm["a2"], tm["b2"], tm["T"])


def reduce(z, valid, kind, sem):
    return OS.reduce(z, valid, kind, EPS) if sem == "lse_sound" else O.reduce(z, valid, kind, sem, EPS)


rows = []
for row, src in (("lse_plain", r2), ("lse_sound", r3), ("sparsemax", r2)):
    for c in (0.0, 0.2):
        for b in ("jax", "warp"):
            V = np.load(src / ("solve_" + b + "_" + row + "_c" + str(c) + ".npz"))["V"]
            for g, gname in enumerate(("S1", "S2")):
                z = np.asarray(P0j.rollout(D.Z0, V[g] * D.U_MAX))
                S = np.asarray(Dj.scores(jnp.asarray(z[:, :2]), D.REGIONS))
                rw, valid = O.until_rows(-S[:, 0], S[:, 1], a1, b1)
                inner_x = np.min(np.where(valid, rw, np.inf), 1)
                k = int(np.argmax(inner_x))
                inner_s = reduce(rw, valid, "min", row)
                outer_s = reduce(inner_s[None], np.ones((1, len(inner_s)), bool), "max", row)[0]
                j = int(np.argmax(inner_s))
                rows.append({"row": row, "c": c, "chain": b, "guess": gname, "exact": float(O.ev(spec, S, np.array([0]))[0]),
                             "exact_until": float(inner_x.max()), "red_clearance_to_witness": float(np.min(-S[: a1 + k + 1, 0])),
                             "green_depth_at_witness": float(S[a1 + k, 1]), "deciding_entry": "Red" if np.min(-S[: a1 + k + 1, 0]) <= S[a1 + k, 1] else "Green",
                             "smoothed_until": float(outer_s), "until_error": float(inner_x.max() - outer_s),
                             "err_inner": float(inner_x[j] - inner_s[j]), "err_outer": float(inner_s.max() - outer_s)})

keys = list(rows[0])
with open(out / "detail.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(rows)


def r4(x):
    return round(float(x), 4)


name = {"lse_plain": "LSE", "lse_sound": "sound LSE", "sparsemax": "sparsemax"}
md = ["Returned trajectories (saved inputs of " + r3.name + " and " + r2.name + "); metres, errors also in units of eps = " + str(EPS) + " m.", "",
      "| smoothing | c | chain | guess | exact margin | Red clearance to the witness | Green depth at the witness | decided by | smoothed until | until error / eps | inner minimum / eps | maximum over witnesses / eps |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append("| " + " | ".join(str(x) for x in (name[r["row"]], r["c"], r["chain"], r["guess"], r4(r["exact"]), r4(r["red_clearance_to_witness"]), r4(r["green_depth_at_witness"]),
                                                   r["deciding_entry"], r4(r["smoothed_until"]), r4(r["until_error"] / EPS), r4(r["err_inner"] / EPS), r4(r["err_outer"] / EPS))) + " |")
(out / "detail.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
