"""The plain log-sum-exp ("lse_plain", E034) in the JAX evaluator and the Warp backend.

lse_plain is the sound lower log-sum-exp ("lse") without the shift -log(m)/beta at
maximum nodes: the two agree at minimum nodes, differ by exactly log(m)/beta at maximum
nodes, and have the same adjoint weights. Tolerances follow tests/test_stl_warp.py:
JAX against Warp allows 1024 eps relative to max(1, largest reference entry), eps the
unit roundoff of the dtype; identities that run the same code path in one backend are
compared for equality, and the shift identities in float64 to 1e-12.
"""
import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from sparsemax_dstl import stl
from sparsemax_dstl.core_study import methods
from sparsemax_dstl.stl.oracle import random_formula
from sparsemax_dstl.stl.warp_backend import Evaluator, evaluate_warp, matched_param, robustness_warp

wp.config.log_level = wp.LOG_WARNING
DTYPES = {np.float32: wp.float32, np.float64: wp.float64}


def x64(dtype):
    return jax.enable_x64(True) if dtype == np.float64 else contextlib.nullcontext()


def close(a, b, tol):
    return np.max(np.abs(a - b), initial=0.0) <= tol * max(1.0, np.max(np.abs(b), initial=0.0))


def warp_vjp(prog, z, sem, param, seed):
    dt = DTYPES[z.dtype.type]
    s = wp.array(z, dtype=dt, device="cpu", requires_grad=True)
    tape = wp.Tape()
    rho = robustness_warp(prog, s, sem, param, tape=tape)
    tape.backward(grads={rho: wp.array(seed, dtype=dt, device="cpu")})
    return rho.numpy(), s.grad.numpy()


def jax_vjp(prog, z, sem, param, seed):
    ref, vjp = jax.vjp(lambda x: stl.robustness(prog, x, sem, param), jnp.asarray(z))
    assert ref.dtype == z.dtype
    return np.asarray(ref), np.asarray(vjp(jnp.asarray(seed))[0])


def warp_steps(prog, z, sem, param):
    """Every step output of the Warp backend, as a list in program order."""
    vals, off = evaluate_warp(prog, wp.array(z, dtype=DTYPES[z.dtype.type], device="cpu"), sem, param)
    vals = vals.numpy()
    return [vals[:, o:o + st.length] for st, o in zip(prog.steps, off)]


def random_mask(rng, shape):
    """Boolean mask with a different number of valid entries (at least one) in each row."""
    n = rng.integers(1, shape[-1] + 1, shape[:-1])
    return np.arange(shape[-1]) < n[..., None]


def random_programs(seed, n):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f = random_formula(rng, 3, 1 + i % 3)
        out.append(stl.compile_formula(f, stl.horizon(stl.to_nnf(f)) + int(rng.integers(1, 5))))
    return out


# ------------------------------------------------------------------
# minimum nodes: lse_plain is lse

def test_min_equals_lse_jax():
    rng = np.random.default_rng(10)
    with jax.enable_x64(True):
        z = jnp.asarray(rng.standard_normal((6, 9)))
        mask = random_mask(rng, (6, 9))
        seed = jnp.asarray(rng.standard_normal(6))
        plain = stl.REDUCTIONS["lse_plain"][1]
        assert np.array_equal(plain(z, 3.0, mask), stl.lse_min(z, 3.0, mask))
        g1 = jax.grad(lambda x: jnp.sum(seed * plain(x, 3.0, mask)))(z)
        g2 = jax.grad(lambda x: jnp.sum(seed * stl.lse_min(x, 3.0, mask)))(z)
        assert np.array_equal(g1, g2)


def test_min_equals_lse_warp():
    # the Until's inner step is a minimum whose rows have 3 to 6 valid entries
    rng = np.random.default_rng(11)
    prog = stl.compile_formula(stl.Until((1, 4), stl.Atom(0), stl.Atom(1)), 12)
    inner = prog.inner[prog.formula]
    assert prog.steps[inner].kind == "min" and len(set(prog.steps[inner].count)) == 4
    z = rng.standard_normal((3, 12, 2))
    for param in (3.0, matched_param(prog, "lse", 0.3)):  # scalar and per-row parameters
        assert np.array_equal(warp_steps(prog, z, "lse_plain", param)[inner], warp_steps(prog, z, "lse", param)[inner])
    # a program of minimum nodes only: equal values and gradients
    prog = stl.compile_formula(stl.Always((0, 3), stl.And(stl.Atom(0), stl.Atom(1))), 12)
    seed = rng.standard_normal((3, prog.steps[-1].length))
    w1, g1 = warp_vjp(prog, z, "lse_plain", 3.0, seed)
    w2, g2 = warp_vjp(prog, z, "lse", 3.0, seed)
    assert np.array_equal(w1, w2) and np.array_equal(g1, g2)


# ------------------------------------------------------------------
# maximum nodes: plain minus sound is log(m)/beta, with the same gradients

def test_max_shift_jax():
    rng = np.random.default_rng(12)
    with jax.enable_x64(True):
        z = jnp.asarray(rng.standard_normal((7, 8)))
        mask = random_mask(rng, (7, 8))
        m = mask.sum(-1)
        d = stl.lse_plain_max(z, 2.5, mask) - stl.lse_max(z, 2.5, mask)
        assert np.max(np.abs(d - np.log(m) / 2.5)) <= 1e-12
        seed = jnp.asarray(rng.standard_normal(7))
        g1 = jax.grad(lambda x: jnp.sum(seed * stl.lse_plain_max(x, 2.5, mask)))(z)
        g2 = jax.grad(lambda x: jnp.sum(seed * stl.lse_max(x, 2.5, mask)))(z)
        assert np.max(np.abs(g1 - g2)) <= 1e-12
        # matched parameters: the shift is eps at every row with m > 1, and 0 at m = 1
        d = methods.lse_plain_max_matched(z, 0.3, mask) - methods.lse_max_matched(z, 0.3, mask)
        assert np.max(np.abs(d - np.where(m > 1, 0.3, 0.0))) <= 1e-12


def test_max_shift_warp():
    # the Release's inner step is a maximum whose rows have 3 to 6 valid entries
    rng = np.random.default_rng(13)
    prog = stl.compile_formula(stl.Release((1, 4), stl.Atom(0), stl.Atom(1)), 12)
    inner = prog.inner[prog.formula]
    step = prog.steps[inner]
    assert step.kind == "max" and len(set(step.count)) == 4
    z = rng.standard_normal((3, 12, 2))
    d = warp_steps(prog, z, "lse_plain", 2.5)[inner] - warp_steps(prog, z, "lse", 2.5)[inner]
    assert np.max(np.abs(d - np.log(step.count) / 2.5)) <= 1e-12
    # per-row parameters, with some rows at 0 (evaluated exactly, no shift)
    R = sum(st.length for st in prog.steps if st.kind != "atom")
    p = rng.uniform(1.0, 5.0, R)
    p[::5] = 0.0
    roff = sum(st.length for st in prog.steps[:inner] if st.kind != "atom")
    pr = p[roff:roff + step.length]
    d = warp_steps(prog, z, "lse_plain", p)[inner] - warp_steps(prog, z, "lse", p)[inner]
    expected = np.where(pr > 0, np.log(step.count) / np.where(pr > 0, pr, 1.0), 0.0)
    assert np.any(pr == 0) and np.max(np.abs(d - expected)) <= 1e-12
    # the adjoint weights are the same: when the maximum is the root (the Until's outer step,
    # over an inner minimum that both semantics evaluate alike) the shift only adds a
    # constant per row, so the tape gradients agree. Below a minimum they need not: there
    # the shift log(m)/beta differs between rows and changes the minimum's weights.
    prog = stl.compile_formula(stl.Until((1, 4), stl.Atom(0), stl.Atom(1)), 12)
    assert prog.steps[-1].kind == "max"
    seed = rng.standard_normal((3, prog.steps[-1].length))
    for param in (2.5, matched_param(prog, "lse", 0.3)):  # scalar and per-row parameters
        w1, g1 = warp_vjp(prog, z, "lse_plain", param, seed)
        w2, g2 = warp_vjp(prog, z, "lse", param, seed)
        assert np.any(w1 != w2) and np.array_equal(g1, g2)


# ------------------------------------------------------------------
# the synthetic Until: the plain value reports a violated specification as satisfied

T, EPS, DELTA = 100, 0.4, 0.1


def until_case():
    """Until([40, 90], zone, pick) over T = 100 samples, zone = atom 0, pick = atom 1.

    The zone score is +1 except at samples 5, 6 and 7, where it is -delta = -0.1; the
    pick score is 1 on samples 40 to 99 and -1 before. Every witness k in [40, 90] of
    the root at t = 0 reads the zone on [0, k], which contains the three dips, so every
    inner minimum is exactly -delta and the exact robustness is -delta. The outer
    maximum at t = 0 has 51 witnesses.

    Margins at eps = 0.4 with the matched parameters (beta = log(m)/eps per node): the
    inner minimum over m = k + 2 = 42 to 92 entries has three tied lowest entries, so
    the sound (and plain) inner value is about -delta - log(3)/beta_in, between -0.218
    (k = 40) and -0.197 (k = 90); the entries at +1 add under 2e-3. The sound outer
    maximum lies at or below the exact maximum of those values, so the sound value is
    about -0.205, negative. The plain outer maximum adds log(51)/beta_out = eps = 0.4,
    which exceeds delta plus the inner minimum's error (about 0.22), so the plain value
    is about +0.195. Sparsemax is a lower bound, so its value is at most -delta (about
    -0.371). Each sign holds with a margin of at least 0.1.
    """
    prog = stl.compile_formula(stl.Until((40, 90), stl.Atom(0), stl.Atom(1)), T)
    z = np.ones((1, T, 2))
    z[0, 5:8, 0] = -DELTA
    z[0, :40, 1] = -1.0
    return prog, z


def test_until_plain_reports_satisfied():
    prog, z = until_case()
    assert prog.steps[-1].kind == "max" and prog.steps[-1].count[0] == 51
    beta_out = np.log(51) / EPS
    with jax.enable_x64(True):
        exact = np.asarray(stl.robustness(prog, z))[0, 0]
        jx = {s: np.asarray(stl.robustness(prog, z, methods.SEMANTICS[s], EPS))[0, 0]
              for s in ("lse", "lse_plain", "sparsemax")}
    wx = {s: robustness_warp(prog, wp.array(z, dtype=wp.float64, device="cpu"), s, matched_param(prog, s, EPS)).numpy()[0, 0]
          for s in ("lse", "lse_plain", "sparsemax")}
    assert exact == -DELTA
    for r in (jx, wx):  # the two evaluators
        assert r["lse_plain"] > 0.1
        assert r["lse"] < -0.1 and r["sparsemax"] < -0.1
        # the outer node is the root: plain minus sound is log(51)/beta_out = eps, to 1e-12
        assert abs(r["lse_plain"] - r["lse"] - np.log(51) / beta_out) <= 1e-12
        assert abs(r["lse_plain"] - r["lse"] - EPS) <= 1e-12


# ------------------------------------------------------------------
# JAX against Warp

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("beta", [2.0, 10.0])
def test_jax_warp_scalar(dtype, beta):
    eps = np.finfo(dtype).eps
    rng = np.random.default_rng(14)
    prog, z0 = until_case()
    cases = [(prog, z0.astype(dtype))]
    for p in random_programs(0, 12):
        cases.append((p, rng.standard_normal((3, p.T, 3)).astype(dtype)))
    with x64(dtype):
        for prog, z in cases:  # programs
            seed = rng.standard_normal((z.shape[0], prog.steps[-1].length)).astype(dtype)
            w, gw = warp_vjp(prog, z, "lse_plain", beta, seed)
            j, gj = jax_vjp(prog, z, "lse_plain", beta, seed)
            assert close(w, j, 1024 * eps)
            assert close(gw, gj, 1024 * eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jax_warp_matched(dtype):
    # per-row parameters: Warp's Evaluator with matched_param against methods.SEMANTICS at eps
    eps = np.finfo(dtype).eps
    dt = DTYPES[dtype]
    rng = np.random.default_rng(15)
    prog, z0 = until_case()
    cases = [(prog, z0.astype(dtype), EPS)]
    for p in random_programs(1, 8):
        cases.append((p, rng.standard_normal((3, p.T, 3)).astype(dtype), 0.3))
    with x64(dtype):
        for prog, z, e in cases:  # programs
            B, P = z.shape[0], z.shape[2]
            seed = rng.standard_normal((B, prog.steps[-1].length)).astype(dtype)
            ev = Evaluator(prog, "lse_plain", matched_param(prog, "lse_plain", e), B, dt, "cpu", P=P)
            rho, grad = ev.gradient(wp.array(z, dtype=dt, device="cpu"), wp.array(seed, dtype=dt, device="cpu"))
            j, gj = jax_vjp(prog, z, methods.SEMANTICS["lse_plain"], e, seed)
            assert close(rho.numpy(), j, 1024 * eps)
            assert close(grad.numpy(), gj, 1024 * eps)
            assert close(ev.value().numpy(), j, 1024 * eps)


def test_registries():
    assert methods.METHODS == ("exact", "sparsemax", "lse", "gilpin", "dgmsr")
    assert methods.MATCHED == ("sparsemax", "lse")
    prog, _ = until_case()
    assert np.array_equal(matched_param(prog, "lse_plain", EPS), matched_param(prog, "lse", EPS))
