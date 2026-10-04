"""Tests of the sound log-sum-exp on the disk-region example against the NumPy oracle, the plain log-sum-exp and the Warp chain."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparsemax_dstl.jax import budget
from sparsemax_dstl.stl import Atom, Until, compile_formula
from sparsemax_dstl.tasks import planar_disk as D
from sparsemax_dstl.tasks import planar_disk_jax as Dj
from sparsemax_dstl.tasks import planar_jax as P0j
from sparsemax_dstl.tasks import planar_oracle_sound as OS
from sparsemax_dstl.tasks.planar_al_jax import JaxChain
from sparsemax_dstl.tasks.planar_warp import WarpChain

jax.config.update("jax_enable_x64", True)
EPS = 0.1


@pytest.fixture(scope="module")
def case():
    u1, u2, tm = D.trajectories(30, EPS)
    spec, conj, _ = D.specification(tm["a1"], tm["b1"], tm["a2"], tm["b2"], tm["T"])
    out = {"tm": tm, "spec": spec, "conj": conj, "u": {"S1": u1, "S2": u2}}
    for name, u in (("S1", u1), ("S2", u2)):
        z = np.asarray(P0j.rollout(D.Z0, u))
        out[name] = np.asarray(Dj.scores(jnp.asarray(z[:, :2]), D.REGIONS))
    return out


@pytest.mark.parametrize("name", ["S1", "S2"])
def test_values_against_oracle(case, name):
    S = case[name]
    sv, cv = Dj.values(jnp.asarray(S), case["tm"], Dj.matched("lse_sound", EPS))
    assert abs(float(sv) - OS.ev(case["spec"], S, np.array([0]), eps=EPS)[0]) < 1e-10
    for c, v in zip(case["conj"], np.asarray(cv)):
        assert abs(float(v) - OS.ev(c, S, np.array([0]), eps=EPS)[0]) < 1e-10


@pytest.mark.parametrize("name", ["S1", "S2"])
def test_below_exact_within_budget(case, name):
    tm = case["tm"]
    B = float(budget(compile_formula(case["spec"], tm["T"], reads=[(case["spec"], [0])]), lambda m, param: np.where(np.asarray(m) > 1, EPS, 0.0))[0])
    S = jnp.asarray(case[name])
    ex, _ = Dj.values(S, tm, Dj.matched("exact", EPS))
    so, _ = Dj.values(S, tm, Dj.matched("lse_sound", EPS))
    assert abs(B - 3 * EPS) < 1e-12
    assert float(ex) - B <= float(so) < float(ex)


@pytest.mark.parametrize("name", ["S1", "S2"])
def test_plain_minus_shifts(case, name):
    S = jnp.asarray(case[name])
    sp, cp = Dj.values(S, case["tm"], Dj.matched("lse_plain", EPS))
    ss, cs = Dj.values(S, case["tm"], Dj.matched("lse_sound", EPS))
    cp, cs = np.asarray(cp), np.asarray(cs)
    shift = np.array([EPS, EPS, 0.0, 0.0])  # until, eventually-always Blue, always not Obstacle, always Boundary
    assert np.max(np.abs(cs - (cp - shift))) < 1e-12
    beta = np.log(4) / EPS
    top = -np.log(np.sum(np.exp(-beta * (cp - shift)))) / beta  # log-sum-exp minimum over the four conjuncts
    assert abs(float(ss) - top) < 1e-12
    assert float(ss) < float(sp)


def test_weight_equals_plain(case):
    tm = case["tm"]
    S = case["S1"]
    phi, psi = -S[:, 0], S[:, 1]
    viol = (phi < 0) & (np.arange(tm["T"]) <= tm["b1"])
    assert viol.sum() == 21

    def grad(sem):
        return jax.grad(lambda ph: P0j.until_on_operands(ph, jnp.asarray(psi), tm["a1"], tm["b1"], Dj.matched(sem, EPS)))(jnp.asarray(phi))
    gs, gp = np.asarray(grad("lse_sound")), np.asarray(grad("lse_plain"))
    assert np.max(np.abs(gs - gp)) < 1e-12
    w = float(np.sum(np.where(viol, gs, 0.0)))
    f = Until((tm["a1"], tm["b1"]), Atom(0), Atom(1))
    h = 1e-6
    up = OS.ev(f, np.stack([phi + h * viol, psi], 1), np.array([0]), eps=EPS)[0]
    dn = OS.ev(f, np.stack([phi - h * viol, psi], 1), np.array([0]), eps=EPS)[0]
    assert abs(w - (up - dn) / (2 * h)) < 1e-6
    assert 0.6 < w < 0.75


def test_chains_agree_at_guesses(case):
    tm = case["tm"]
    V = np.stack([case["u"]["S1"], case["u"]["S2"]]) / D.U_MAX
    jc = JaxChain(case["conj"], tm["T"], "lse", EPS, D.Z0, D.REGIONS, 2)
    wc = WarpChain(case["conj"], tm["T"], "lse", EPS, D.Z0, D.REGIONS, 2)
    r_j, r_w = jc.forward(V), wc.forward(V)
    w = np.array([[1.0, 0.5, 0.25, 0.125], [0.3, 1.0, 0.7, 0.2]])
    assert np.max(np.abs(r_j - r_w)) < 1e-12
    assert np.max(np.abs(jc.pullback(w) - wc.pullback(w))) < 1e-12
    assert np.max(np.abs(jc.C)) > 1e-3  # the gradients compared are not all zero
    for g, name in enumerate(("S1", "S2")):
        _, cv = Dj.values(jnp.asarray(case[name]), tm, Dj.matched("lse_sound", EPS))
        assert np.max(np.abs(r_j[g] - np.asarray(cv))) < 1e-10
