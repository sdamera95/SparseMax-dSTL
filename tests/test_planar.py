"""Tests of the unicycle's JAX rollout against a loop oracle and of the oracle's sparsemax reduction against a quadratic program."""
import jax
import numpy as np
from scipy.optimize import minimize

from sparsemax_dstl.tasks import planar as P
from sparsemax_dstl.tasks import planar_disk as D
from sparsemax_dstl.tasks import planar_jax as Pj
from sparsemax_dstl.tasks import planar_oracle as O

jax.config.update("jax_enable_x64", True)


def test_rollout_against_loop_oracle():
    u = D.trajectories(30)[0]
    z = np.asarray(Pj.rollout(D.Z0, u))
    ref = [np.array(D.Z0, float)]
    for v, w in u:  # brute-force oracle: one step at a time
        x, y, th = ref[-1]
        ref.append(np.array([x + P.H * v * np.cos(th), y + P.H * v * np.sin(th), th + P.H * w]))
    assert np.max(np.abs(z - np.array(ref))) < 1e-12


def test_sparsemax_against_qp():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(6, 9))
    gamma = rng.uniform(0.1, 2.0, size=6)
    valid = rng.uniform(size=(6, 9)) < 0.8
    valid[:, 0] = True
    val, _ = O.qmax(z, gamma, valid)
    for r in range(6):  # brute-force reference: a generic solver per row
        zr = z[r, valid[r]]
        obj = lambda p: -(p @ zr - gamma[r] / 2 * p @ p)  # noqa: E731
        res = minimize(obj, np.full(len(zr), 1 / len(zr)), method="SLSQP", bounds=[(0, 1)] * len(zr),
                       constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}], options={"ftol": 1e-14, "maxiter": 500})
        assert abs(-res.fun - val[r]) < 1e-7
