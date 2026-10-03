"""E049: the planar unicycle example against independent references.

- the JAX values of every smoothing on S1 and S2 against the NumPy oracle written from the
  definitions (sparsemax_dstl/tasks/planar_oracle.py);
- the AD derivative of the until on S1 against the closed forms of the reductions' derivatives
  (plain log-sum-exp, sparsemax) and against central finite differences of the oracle (a test
  check only);
- the oracle's sparsemax value against a generic solver of the quadratic program on the simplex;
- the rollout against a brute-force oracle (a Python loop over steps, labelled as such);
- the inputs inside the box, and the samples of S1 inside Red where the design puts them;
- the sparsemax value between the exact robustness minus its budget and the exact robustness.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from sparsemax_dstl.jax import budget
from sparsemax_dstl.stl import Atom, Not, Until, compile_formula
from sparsemax_dstl.tasks import planar as P
from sparsemax_dstl.tasks import planar_oracle as O

jax.config.update("jax_enable_x64", True)
EPS = 0.2


@pytest.fixture(scope="module")
def case():
    u1, u2, tm = P.trajectories(30)
    out = {"tm": tm, "u": {"S1": u1, "S2": u2}}
    for name, u in (("S1", u1), ("S2", u2)):
        z = np.asarray(P.rollout(P.Z0, u))
        out[name] = (z, P.scores(jnp.asarray(z[:, :2])))
    return out


def test_inputs_and_samples(case):
    """Inputs in the box, forward speed never negative, turn rate continuous (a change of at most
    0.2 rad/s per step); S1 dips at most depth into Red before the window and stands at the wait spot."""
    tm = case["tm"]
    for u in case["u"].values():
        assert np.all(u[:, 0] >= 0) and np.all(u[:, 0] <= P.V_MAX) and np.all(np.abs(u[:, 1]) <= P.W_MAX)
        assert np.max(np.abs(np.diff(u[:, 1]))) < 0.2
    z, S = case["S1"]
    pre = np.arange(tm["T"]) <= tm["b1"]
    red = O.ev(Not(P.box(0)), np.asarray(S), np.arange(tm["T"]))
    inside = pre & (red < 0)
    assert 1 <= inside.sum() <= 15
    assert abs(-red[inside].min() - tm["depth"]) < 2e-3
    standing = pre & (np.abs(red - tm["level"]) < 1e-9)
    assert standing.sum() >= tm["wait"]


def test_rollout_against_loop_oracle(case):
    u = case["u"]["S1"]
    z = np.asarray(P.rollout(P.Z0, u))
    ref = [np.array(P.Z0, float)]
    for v, w in u:  # brute-force oracle: one step at a time
        x, y, th = ref[-1]
        ref.append(np.array([x + P.H * v * np.cos(th), y + P.H * v * np.sin(th), th + P.H * w]))
    assert np.max(np.abs(z - np.array(ref))) < 1e-12


@pytest.mark.parametrize("method", P.METHODS)
def test_values_against_oracle(case, method):
    tm = case["tm"]
    spec, until = P.specification(tm["a1"], tm["b1"], tm["a2"], tm["b2"], tm["T"])
    for name in ("S1", "S2"):
        S = case[name][1]
        sv, uv = P.spec_values(S, tm, P.matched(method, EPS))
        assert abs(float(sv) - O.ev(spec, np.asarray(S), np.array([0]), sem=method, eps=EPS)[0]) < 1e-10
        assert abs(float(uv) - O.ev(until, np.asarray(S), np.array([0]), sem=method, eps=EPS)[0]) < 1e-10


@pytest.mark.parametrize("method", ["lse_plain", "gm01", "gm10", "sparsemax"])
def test_until_derivative(case, method):
    tm = case["tm"]
    S = case["S1"][1]
    T = tm["T"]
    phi_x = O.ev(Not(P.box(0)), np.asarray(S), np.arange(T))
    viol = (phi_x < 0) & (np.arange(T) <= tm["b1"])
    _, w, g = P.until_weight(S, tm, P.matched(method, EPS), jnp.asarray(viol))
    phi = O.ev(Not(P.box(0)), np.asarray(S), np.arange(T), sem=method, eps=EPS)
    psi = O.ev(P.box(1), np.asarray(S), np.arange(T), sem=method, eps=EPS)
    f = Until((tm["a1"], tm["b1"]), Atom(0), Atom(1))
    h = 1e-6
    up = O.ev(f, np.stack([phi + h * viol, psi], 1), np.array([0]), sem=method, eps=EPS)[0]
    dn = O.ev(f, np.stack([phi - h * viol, psi], 1), np.array([0]), sem=method, eps=EPS)[0]
    assert abs(float(w) - (up - dn) / (2 * h)) < 1e-6
    if method in ("lse_plain", "sparsemax"):
        gc = O.until_grad_closed(phi, psi, tm["a1"], tm["b1"], method, EPS)
        assert np.max(np.abs(np.asarray(g) - gc)) < 1e-10


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


def test_sparsemax_lower_bound(case):
    tm = case["tm"]
    spec, until = P.specification(tm["a1"], tm["b1"], tm["a2"], tm["b2"], tm["T"])
    prog = compile_formula(spec, tm["T"], reads=[(spec, [0]), (until, [0])])
    B = float(budget(prog, lambda m, param: np.where(np.asarray(m) > 1, EPS, 0.0))[0])
    for name in ("S1", "S2"):
        S = case[name][1]
        ex, _ = P.spec_values(S, tm, P.matched("exact", EPS))
        sm, _ = P.spec_values(S, tm, P.matched("sparsemax", EPS))
        assert float(ex) - B - 1e-12 <= float(sm) <= float(ex) + 1e-12
