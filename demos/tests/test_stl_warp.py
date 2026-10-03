"""Warp STL backend against the JAX reference evaluator, values and tape gradients.

Tolerances, with eps the unit roundoff of the dtype under test:

- exact values are compared for equality: a maximum or minimum returns one of
  its inputs and a sign flip is exact, so both backends must return the same
  floating-point numbers, infinities included.
- exact gradients: each entry is a sum of seed * sign / n_ties products that
  the two backends accumulate in different orders; allow 64 eps relative to
  the largest reference gradient entry (or 1).
- lse values and gradients: exp and log come from different libraries (Warp's
  device math versus XLA), and the summation order differs; each reduction
  adds a few ulp and the weights exp(beta (y - c)) amplify an input error by
  up to beta |y|. Allow 1024 eps relative to max(1, largest reference entry).
- two-link chains add sin and cos from different libraries; same 1024 eps.

The measured maxima over a larger set are printed by experiments/e005_warp.py.
"""
import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from sparsemax_diffstl import stl
from sparsemax_diffstl.stl import twolink
from sparsemax_diffstl.stl.oracle import random_formula
from sparsemax_diffstl.stl.twolink_warp import twolink_scores
from sparsemax_diffstl.stl.warp_backend import evaluate_warp, robustness_warp

wp.config.log_level = wp.LOG_WARNING
DTYPES = {np.float32: wp.float32, np.float64: wp.float64}
CASES = [("exact", None), ("lse", 2.0), ("lse", 10.0)]


def warp_vjp(prog, z, sem, beta, seed):
    dt = DTYPES[z.dtype.type]
    s = wp.array(z, dtype=dt, device="cpu", requires_grad=True)
    tape = wp.Tape()
    rho = robustness_warp(prog, s, sem, beta, tape=tape)
    tape.backward(grads={rho: wp.array(seed, dtype=dt, device="cpu")})
    return rho.numpy(), s.grad.numpy()


def jax_vjp(prog, z, sem, beta, seed):
    ref, vjp = jax.vjp(lambda x: stl.robustness(prog, x, sem, beta), jnp.asarray(z))
    assert ref.dtype == z.dtype
    return np.asarray(ref), np.asarray(vjp(jnp.asarray(seed))[0])


def close(a, b, tol):
    return np.max(np.abs(a - b), initial=0.0) <= tol * max(1.0, np.max(np.abs(b), initial=0.0))


def x64(dtype):
    # float64 needs jax x64; float32 leaves the configuration as the run set it
    return jax.enable_x64(True) if dtype == np.float64 else contextlib.nullcontext()


def random_programs(seed, n, boundary="strict"):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f = random_formula(rng, 3, 1 + i % 3)
        h = stl.horizon(stl.to_nnf(f))
        T = h + int(rng.integers(1, 5)) if boundary == "strict" else int(rng.integers(1, h + 3))
        out.append(stl.compile_formula(f, T, boundary))
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sem,beta", CASES)
def test_random_strict(dtype, sem, beta):
    eps = np.finfo(dtype).eps
    rng = np.random.default_rng(1)
    with x64(dtype):
        for prog in random_programs(0, 12):
            z = rng.standard_normal((3, prog.T, 3)).astype(dtype)
            seed = rng.standard_normal((3, prog.steps[-1].length)).astype(dtype)
            w, gw = warp_vjp(prog, z, sem, beta, seed)
            j, gj = jax_vjp(prog, z, sem, beta, seed)
            if sem == "exact":
                assert np.array_equal(w, j)
                assert close(gw, gj, 64 * eps)
            else:
                assert close(w, j, 1024 * eps)
                assert close(gw, gj, 1024 * eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_clip_exact(dtype):
    eps = np.finfo(dtype).eps
    rng = np.random.default_rng(2)
    n_inf = 0
    with x64(dtype):
        for prog in random_programs(3, 16, "clip"):
            z = rng.standard_normal((3, prog.T, 3)).astype(dtype)
            j = np.asarray(stl.robustness(prog, jnp.asarray(z)))
            seed = np.where(np.isfinite(j), rng.standard_normal(j.shape), 0.0).astype(dtype)
            w, gw = warp_vjp(prog, z, "exact", None, seed)
            j, gj = jax_vjp(prog, z, "exact", None, seed)
            assert np.array_equal(w, j)
            assert np.all(np.isfinite(gw))
            assert close(gw, gj, 64 * eps)
            n_inf += int(np.sum(np.isinf(j)))
    assert n_inf > 0  # the set must exercise empty windows


def test_lse_rejects_empty_windows():
    prog = stl.compile_formula(stl.Eventually((1, 3), stl.Atom(0)), 2, "clip")
    s = wp.zeros((1, 2, 1), dtype=wp.float32, device="cpu")
    with pytest.raises(ValueError):
        evaluate_warp(prog, s, "lse", 2.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ties(dtype):
    eps = np.finfo(dtype).eps
    rng = np.random.default_rng(4)
    formulas = [stl.And(stl.Atom(0), stl.Atom(0)),
                stl.Until((0, 2), stl.Atom(0), stl.Atom(0)),
                stl.Release((1, 2), stl.Atom(1), stl.Atom(1))]
    with x64(dtype):
        for f in formulas:
            prog = stl.compile_formula(f, 6)
            # integer-valued scores also tie across time inside windows
            for z in (rng.standard_normal((3, 6, 2)), rng.integers(-1, 2, (3, 6, 2))):
                z = z.astype(dtype)
                seed = rng.standard_normal((3, prog.steps[-1].length)).astype(dtype)
                w, gw = warp_vjp(prog, z, "exact", None, seed)
                j, gj = jax_vjp(prog, z, "exact", None, seed)
                assert np.array_equal(w, j)
                assert close(gw, gj, 64 * eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sem,beta", CASES)
def test_twolink_chain(dtype, sem, beta):
    eps = np.finfo(dtype).eps
    dt = DTYPES[dtype]
    rng = np.random.default_rng(5)
    with x64(dtype):
        for prog in random_programs(6, 6):
            B, T = 3, prog.T
            X = rng.uniform(-np.pi, np.pi, (B, T, 2)).astype(dtype)
            seed = rng.standard_normal((B, prog.steps[-1].length)).astype(dtype)
            Xw = wp.array(X, dtype=dt, device="cpu", requires_grad=True)
            scores = wp.zeros((B, T, 3), dtype=dt, device="cpu", requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(twolink_scores, dim=(B, T), inputs=[Xw], outputs=[scores], device="cpu")
                rho = robustness_warp(prog, scores, sem, beta, tape=tape)
            tape.backward(grads={rho: wp.array(seed, dtype=dt, device="cpu")})

            def f(X):
                return stl.robustness(prog, stl.score_traces(twolink.PREDICATES, X), sem, beta)

            j, vjp = jax.vjp(f, jnp.asarray(X))
            gj = np.asarray(vjp(jnp.asarray(seed))[0])
            assert close(rho.numpy(), np.asarray(j), 1024 * eps)
            assert close(Xw.grad.numpy(), gj, 1024 * eps)
