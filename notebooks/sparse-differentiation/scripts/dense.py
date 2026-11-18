# Dense AD of one control period's program (one reverse product per constraint, one forward-over-reverse product per
# variable) at the start point of the timed control period, beside the paper's (results/counts.csv). Memory and time grow
# with the horizon: at 400 time steps the Hessian takes about 30 minutes and 16 GB.
#   python scripts/dense.py <five-cell zone|64-cell circle> <T>
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")  # MUMPS and its BLAS on one thread, as the paper's solves ran
import pathlib
import sys
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from common import HES, JAC, SCENES, build, paper_rows, timed, timed_period  # noqa: E402
from sparsemax_dstl.mpc import baselines as B  # noqa: E402
from sparsemax_dstl.mpc import program as MP  # noqa: E402

name, T = sys.argv[1], int(sys.argv[2])
scene, conjs = build(SCENES[name](T))
k = timed_period(scene, conjs)
pb = MP.PeriodProgram(scene, conjs, k)
w, lam = jnp.asarray(pb.reference()), jnp.ones(pb.m)
jac, hess = B.dense_jacobian(pb), B.dense_hessian(pb)
print(name, "T", T, "control period", k, "| constraints", pb.m, "variables", pb.n)
t = time.perf_counter()
j = timed(lambda: jax.block_until_ready(jac(w)))
print("Jacobian", round(j, 3), "s | paper", [r["value"] for r in paper_rows("counts", scene=name, horizon=T, comparator="dense AD", quantity=JAC + " time")], "| with compilation", round(time.perf_counter() - t, 1), "s")
t = time.perf_counter()
h = timed(lambda: jax.block_until_ready(hess(w, jnp.asarray(1.0), lam)))
print("Hessian", round(h, 3), "s | paper", [r["value"] for r in paper_rows("counts", scene=name, horizon=T, comparator="dense AD", quantity=HES + " time")], "| with compilation", round(time.perf_counter() - t, 1), "s")
