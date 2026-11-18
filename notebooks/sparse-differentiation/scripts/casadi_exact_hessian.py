# Sound-LSE's Lagrangian Hessian in CasADi (its exact Hessian, CasADi's default evaluation) on the five-cell zone at the
# timed control period: "hessian" builds CasADi's Hessian of the Lagrangian (objective weight 1, every multiplier 1) and times
# one evaluation of its upper triangle at the start point (100 time steps: about
# 7 minutes and 14 GB; 200: 25 minutes and 46 GB); "solve" solves from the previous period's solution with the exact Hessian,
# twice, the second timed (100: 11 minutes and 14 GB; 200: 48 minutes and 47 GB; 400: 1 h 36 min and 89 GB). Beside the
# paper's (results/counts.csv, results/solves.csv).
#   python scripts/casadi_exact_hessian.py <T> <hessian|solve>
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")  # MUMPS and its BLAS on one thread, as the paper's solves ran
import pathlib
import sys
import time

import casadi as ca
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from common import HES, build, casadi_solve, paper_rows, timed, timed_period  # noqa: E402
from sparsemax_dstl.mpc import baselines as B  # noqa: E402
from sparsemax_dstl.tasks import dual_arm_scenes as S  # noqa: E402

T, what = int(sys.argv[1]), sys.argv[2]
scene, conjs = build(S.five_cell_zone(T))
k = timed_period(scene, conjs)
if what == "hessian":
    p = B.Period(scene, conjs, k, "lse")
    t = time.perf_counter()
    sigma, lam = ca.SX.sym("sigma"), ca.SX.sym("lam", p.m)
    H, _ = ca.hessian(sigma * p.f + ca.dot(lam, p.g), p.w)
    F = ca.Function("hess_l", [p.w, sigma, lam], [ca.triu(H)])
    print("T", T, "control period", k, "| build", round(time.perf_counter() - t, 1), "s, upper entries", F.sparsity_out(0).nnz())
    w, ones = p.reference(), np.ones(p.m)
    h = timed(lambda: F(w, 1.0, ones))
    rows = paper_rows("counts", scene="five-cell zone", horizon=T, comparator="Sound-LSE", quantity=HES + " time")
    print("Hessian", round(h, 3), "s | paper", " ".join(str(round(float(r["value"]), 3)) + " s" for r in rows) or "none at this horizon")
else:
    out = casadi_solve(scene, conjs, k, "lse", twice=True, hessian="exact")
    print("T", T, "control period", k, "|", out["status"], out["updates"], "updates, objective", out["objective"], ",", round(out["seconds"], 3), "s")
    m = dict(start="previous solution", horizon=T, comparator="Sound-LSE", variant="CasADi's exact Hessian")
    print("paper:", ", ".join(r["quantity"] + " " + r["value"] for r in paper_rows("solves", **m)))
