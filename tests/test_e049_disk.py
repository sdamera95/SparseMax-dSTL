"""E049 round 2: the disk-region example and the unicycle chain against independent references.

- every smoothing's value of the STL specification on S1 and S2 against the NumPy evaluation from the
  definitions (planar_oracle);
- the AD weight on S1's samples inside Red against central finite differences of the NumPy evaluation
  (a test check only);
- S1 and S2: inputs in the box, forward speed never negative, the turn rate changing by at most
  0.2 rad/s per step; S1's samples at full depth tied, its standing samples at the stand-off;
- sparsemax between the exact robustness minus its budget and the exact robustness;
- the JAX chain's per-conjunct values equal planar_disk.values, its gradient equal to a finite
  difference along one direction (a test check only).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparsemax_dstl.stl import Atom, Until, budget, compile_formula
from sparsemax_dstl.tasks import planar as P0
from sparsemax_dstl.tasks import planar_al as A
from sparsemax_dstl.tasks import planar_disk as D
from sparsemax_dstl.tasks import planar_oracle as O

jax.config.update("jax_enable_x64", True)
EPS = 0.1


@pytest.fixture(scope="module")
def case():
    u1, u2, tm = D.trajectories(30, EPS)
    out = {"tm": tm, "u": {"S1": u1, "S2": u2}}
    for name, u in (("S1", u1), ("S2", u2)):
        z = np.asarray(P0.rollout(D.Z0, u))
        out[name] = (z, np.asarray(D.scores(jnp.asarray(z[:, :2]), D.REGIONS)))
    return out


def test_paths(case):
    tm = case["tm"]
    for u in case["u"].values():
        assert np.all(u[:, 0] >= 0) and np.all(u[:, 0] <= P0.V_MAX) and np.all(np.abs(u[:, 1]) <= P0.W_MAX)
        assert np.max(np.abs(np.diff(u[:, 1]))) < 0.2
    S = case["S1"][1]
    phi = -S[: tm["b1"] + 1, 0]
    full = np.abs(phi + tm["depth"]) < 1e-6
    assert full.sum() >= 10 and phi.min() >= -tm["depth"] - 1e-8
    assert np.sum(np.abs(phi - tm["stand"]) < 1e-9) >= tm["wait"]
    S2 = case["S2"][1]
    assert abs(np.min(-S2[: tm["b1"] + 1, 0]) - tm["clear"]) < 1e-6


@pytest.mark.parametrize("method", ["exact", "lse_plain", "gm01", "gm10", "sparsemax"])
def test_values_against_oracle(case, method):
    tm = case["tm"]
    spec, conj, _ = D.specification(tm["a1"], tm["b1"], tm["a2"], tm["b2"], tm["T"])
    for name in ("S1", "S2"):
        S = case[name][1]
        sv, cv = D.values(jnp.asarray(S), tm, P0.matched(method, EPS))
        assert abs(float(sv) - O.ev(spec, S, np.array([0]), sem=method, eps=EPS)[0]) < 1e-10
        for c, v in zip(conj, np.asarray(cv)):  # over the conjuncts
            assert abs(float(v) - O.ev(c, S, np.array([0]), sem=method, eps=EPS)[0]) < 1e-10


@pytest.mark.parametrize("method", ["lse_plain", "gm01", "gm10", "sparsemax"])
def test_weight_against_finite_difference(case, method):
    tm = case["tm"]
    S = case["S1"][1]
    phi, psi = -S[:, 0], S[:, 1]
    viol = (phi < 0) & (np.arange(tm["T"]) <= tm["b1"])
    sem = P0.matched(method, EPS)
    g = jax.grad(lambda ph: P0.until_on_operands(ph, jnp.asarray(psi), tm["a1"], tm["b1"], sem))(jnp.asarray(phi))
    w = float(jnp.sum(jnp.where(viol, g, 0.0)))
    f = Until((tm["a1"], tm["b1"]), Atom(0), Atom(1))
    h = 1e-6
    up = O.ev(f, np.stack([phi + h * viol, psi], 1), np.array([0]), sem=method, eps=EPS)[0]
    dn = O.ev(f, np.stack([phi - h * viol, psi], 1), np.array([0]), sem=method, eps=EPS)[0]
    assert abs(w - (up - dn) / (2 * h)) < 1e-6


def test_sparsemax_band(case):
    tm = case["tm"]
    spec, _, _ = D.specification(tm["a1"], tm["b1"], tm["a2"], tm["b2"], tm["T"])
    B = float(budget(compile_formula(spec, tm["T"], reads=[(spec, [0])]), lambda m, param: np.where(np.asarray(m) > 1, EPS, 0.0))[0])
    for name in ("S1", "S2"):
        S = case[name][1]
        ex, _ = D.values(jnp.asarray(S), tm, P0.matched("exact", EPS))
        sm, _ = D.values(jnp.asarray(S), tm, P0.matched("sparsemax", EPS))
        assert float(ex) - B - 1e-12 <= float(sm) <= float(ex) + 1e-12


@pytest.mark.parametrize("method", ["lse_plain", "gm_pm01", "gm_pm10", "sparsemax"])
def test_jax_chain(case, method):
    tm = case["tm"]
    _, conj, _ = D.specification(tm["a1"], tm["b1"], tm["a2"], tm["b2"], tm["T"])
    ch = A.JaxChain(conj, tm["T"], method, EPS, D.Z0, D.REGIONS, 1)
    V = (case["u"]["S1"] / D.U_MAX)[None]
    r = ch.forward(V)
    _, cv = D.values(jnp.asarray(case["S1"][1]), tm, P0.matched(A.JAX_NAMES[method], EPS))
    assert np.max(np.abs(r[0] - np.asarray(cv))) < 1e-10
    w = np.array([[1.0, 0.5, 0.25, 0.125]])
    g = ch.pullback(w)
    d = np.random.default_rng(1).normal(size=V.shape) * 1e-3
    h = 1e-4
    fd = (np.sum(w * ch.values(np.stack([V + h * d], 1))[:, 0]) - np.sum(w * ch.values(np.stack([V - h * d], 1))[:, 0])) / (2 * h)
    assert abs(np.sum(g * d) - fd) < 1e-6 * max(1.0, abs(fd))
