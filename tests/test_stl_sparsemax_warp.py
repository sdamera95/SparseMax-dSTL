"""Warp sparsemax reductions and their weight adjoints against JAX.

Tolerance 1024 eps relative to max(1, largest reference entry), as in
test_stl_warp.py: the threshold search and the sums run in a different order
than JAX's sort and cumsum, and a weight is (y - theta)/gamma, so small gamma
scales rounding by 1/gamma. Near a support switch the two backends may place
an entry on different sides; weights are continuous there, so that stays
within rounding.
"""
import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

import twolink
from sparsemax_dstl import jax as stl_jax
from sparsemax_dstl import stl
from sparsemax_dstl.stl.oracle import random_formula
from sparsemax_dstl.warp.evaluator import evaluate_warp, robustness_warp
from twolink_warp import twolink_scores

wp.config.log_level = wp.LOG_WARNING
DTYPES = {np.float32: wp.float32, np.float64: wp.float64}


def x64(dtype):
    return jax.enable_x64(True) if dtype == np.float64 else contextlib.nullcontext()


def close(a, b, dtype):
    return np.max(np.abs(a - b), initial=0.0) <= 1024 * np.finfo(dtype).eps * max(1.0, np.max(np.abs(b), initial=0.0))


def compare(prog, z, gamma, seed):
    dt = DTYPES[z.dtype.type]
    s = wp.array(z, dtype=dt, device="cpu", requires_grad=True)
    tape = wp.Tape()
    rho = robustness_warp(prog, s, "sparsemax", gamma, tape=tape)
    tape.backward(grads={rho: wp.array(seed, dtype=dt, device="cpu")})
    ref, g = pullback(lambda x: stl_jax.robustness(prog, x, "sparsemax", gamma))(jnp.asarray(z), jnp.asarray(seed))
    assert ref.dtype == z.dtype
    return close(rho.numpy(), np.asarray(ref), z.dtype) and close(s.grad.numpy(), np.asarray(g), z.dtype)


def pullback(f):
    def pull(x, g):
        r, vjp = jax.vjp(f, x)
        return r, vjp(g)[0]
    return jax.jit(pull)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("gamma", [0.1, 1.0])
def test_random_programs(dtype, gamma):
    rng = np.random.default_rng(10)
    with x64(dtype):
        for i in range(12):  # formula count
            f = random_formula(rng, 3, 1 + i % 3)
            prog = stl.compile_formula(f, stl.horizon(f) + int(rng.integers(1, 5)))
            z = rng.standard_normal((3, prog.T, 3)).astype(dtype)
            assert compare(prog, z, gamma, rng.standard_normal((3, prog.steps[-1].length)).astype(dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ties(dtype):
    rng = np.random.default_rng(11)
    formulas = [stl.Always((0, 4), stl.Atom(0)), stl.Eventually((1, 3), stl.Atom(1)),
                stl.Until((0, 2), stl.Atom(0), stl.Atom(0)), stl.Release((1, 2), stl.Atom(1), stl.Atom(1))]
    with x64(dtype):
        for f in formulas:  # formula count
            prog = stl.compile_formula(f, 7)
            z = rng.integers(-1, 2, (3, 7, 2)).astype(dtype)
            assert compare(prog, z, 0.5, rng.standard_normal((3, prog.steps[-1].length)).astype(dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_twolink_chain(dtype):
    dt = DTYPES[dtype]
    rng = np.random.default_rng(12)
    with x64(dtype):
        for i in range(6):  # formula count
            f = random_formula(rng, 3, 1 + i % 3)
            prog = stl.compile_formula(f, stl.horizon(f) + 3)
            B, T = 3, prog.T
            X = rng.uniform(-np.pi, np.pi, (B, T, 2)).astype(dtype)
            seed = rng.standard_normal((B, prog.steps[-1].length)).astype(dtype)
            Xw = wp.array(X, dtype=dt, device="cpu", requires_grad=True)
            scores = wp.zeros((B, T, 3), dtype=dt, device="cpu", requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(twolink_scores, dim=(B, T), inputs=[Xw], outputs=[scores], device="cpu")
                rho = robustness_warp(prog, scores, "sparsemax", 0.5, tape=tape)
            tape.backward(grads={rho: wp.array(seed, dtype=dt, device="cpu")})
            f = lambda X: stl_jax.robustness(prog, stl_jax.score_traces(twolink.PREDICATES, X), "sparsemax", 0.5)
            ref, g = pullback(f)(jnp.asarray(X), jnp.asarray(seed))
            assert close(rho.numpy(), np.asarray(ref), dtype)
            assert close(Xw.grad.numpy(), np.asarray(g), dtype)


def test_worked_example_in_warp():
    # scores of the extension's example: mu = a + b and -nu = -(b + c) at samples 0..3
    z = np.array([[[0, 0], [-1, -1], [2, -3], [0.5, 0]]], dtype=np.float64)
    prog = stl.compile_formula(stl.And(stl.Eventually((1, 2), stl.Atom(0)), stl.Always((0, 1), stl.Atom(1, True))), 4)
    s = wp.array(z, dtype=wp.float64, device="cpu", requires_grad=True)
    tape = wp.Tape()
    vals, off = evaluate_warp(prog, s, "sparsemax", 1.0, tape=tape)
    v = vals.numpy()[0]
    rows = [i for i, st in enumerate(prog.steps) if st.label in ("F[1,2]", "G[0,1]")] + [prog.root]
    np.testing.assert_array_equal([v[off[r] + 1] for r in rows], [7 / 4, 1, 63 / 64])
    seed = np.zeros_like(vals.numpy())
    seed[0, off[prog.root] + 1] = 1.0
    tape.backward(grads={vals: wp.array(seed, dtype=wp.float64, device="cpu")})
    grad = np.zeros((4, 2))
    grad[2, 0], grad[1, 1] = 1 / 8, -7 / 8  # root weights times the edge weights (1, 0) below
    np.testing.assert_array_equal(s.grad.numpy()[0], grad)
