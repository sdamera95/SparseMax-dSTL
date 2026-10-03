import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparsemax_dstl import stl
from sparsemax_dstl.stl import (Always, And, Atom, Eventually, Or, Release, Until, compile_formula, oracle,
                                   robustness, score_traces, twolink)

X64 = jax.config.jax_enable_x64


def close(a, b, tol64=1e-9, tol32=3e-4):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    tol = (tol64 if X64 else tol32) * (1 + np.max(np.abs(b), initial=0))
    np.testing.assert_allclose(a, b, rtol=0, atol=tol)


def cases(seed, n, sems=("exact", "lse")):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):  # formula count
        f = oracle.random_formula(rng, 3, int(rng.integers(1, 4)), ops=("and", "or", "G", "F", "U", "R", "not"))
        T = stl.horizon(f) + int(rng.integers(1, 4))
        z = jnp.asarray(rng.normal(size=(T, 3)), float)
        out.append((compile_formula(f, T), z, sems[i % len(sems)], float(rng.choice([1.0, 3.0, 10.0]))))
    return out


@pytest.mark.parametrize("prog,z,sem,beta", cases(11, 16))
def test_reverse_matches_forward(prog, z, sem, beta):
    f = jax.jit(lambda z: robustness(prog, z, sem, beta))
    close(jax.jacrev(f)(z), jax.jacfwd(f)(z))
    v = jnp.asarray(np.random.default_rng(0).normal(size=z.shape), z.dtype)
    g = jax.grad(lambda z: f(z)[0])(z)
    close(jax.jvp(lambda z: f(z)[0], (z,), (v,))[1], jnp.sum(g * v))


@pytest.mark.parametrize("prog,z,sem,beta", cases(12, 10))
def test_second_order_modes_and_analytic_reference(prog, z, sem, beta):
    f = lambda z: robustness(prog, z, sem, beta)[0]
    n = z.size
    H = jax.jit(jax.hessian(f))(z).reshape(n, n)
    close(jax.jit(jax.jacrev(jax.jacfwd(f)))(z).reshape(n, n), H)
    close(jax.jit(jax.jacfwd(jax.jacrev(f)))(z).reshape(n, n), H)
    v = jnp.asarray(np.random.default_rng(1).normal(size=z.shape), z.dtype)
    hvp = jax.jit(lambda z, v: jax.jvp(jax.grad(f), (z,), (v,))[1])(z, v)
    close(hvp.reshape(n), H @ v.reshape(n))
    value, grad, hess = oracle.analytic_derivatives(prog, z, sem, beta)
    close(jax.jit(f)(z), value[0])
    close(jax.jit(jax.grad(f))(z).reshape(n), grad[0])
    close(H, hess[0])


@pytest.mark.parametrize("sem,beta", [("lse", 3.0), ("lse", 10.0), ("exact", None)])
def test_nonlinear_predicate_second_order(sem, beta):
    f = And(Eventually((1, 3), Atom(0)), Always((0, 4), Atom(1)), Until((0, 2), Atom(2), Atom(0)))
    T = 7
    X = np.random.default_rng(2).uniform(-2, 2, size=(T, 2))
    prog = compile_formula(f, T)
    rho = lambda X: robustness(prog, score_traces(twolink.PREDICATES, X), sem, beta)[0]
    Xj = jnp.asarray(X, float)
    grad_ref, hess_ref = twolink.analytic_chain(prog, X, sem, beta)
    H = jax.jit(jax.hessian(rho))(Xj).reshape(2 * T, 2 * T)
    close(jax.jit(jax.grad(rho))(Xj).reshape(-1), grad_ref, tol32=1e-3)
    close(H, hess_ref, tol32=1e-3)
    close(jax.jit(jax.jacrev(jax.jacfwd(rho)))(Xj).reshape(2 * T, 2 * T), H)
    v = jnp.asarray(np.random.default_rng(3).normal(size=X.shape), float)
    hvp = jax.jit(lambda X, v: jax.jvp(jax.grad(rho), (X,), (v,))[1])(Xj, v)
    close(hvp.reshape(-1), hess_ref @ np.asarray(v).reshape(-1), tol32=1e-3)
    if sem == "lse":
        assert np.max(np.abs(hess_ref)) > 1.0  # the check is not vacuous


@pytest.mark.parametrize("sem", ["exact", "lse"])
def test_jit_vmap_batches_equal_per_sample(sem):
    f = Or(Release((1, 2), Atom(0), Atom(1)), Eventually((0, 3), And(Atom(2), Atom(0, True))))
    T, N = 9, 32
    prog = compile_formula(f, T)
    Z = jnp.asarray(np.random.default_rng(4).normal(size=(N, T, 3)), float)
    V = jnp.asarray(np.random.default_rng(5).normal(size=(N, T, 3)), float)
    rho = lambda z: robustness(prog, z, sem, 4.0)
    first = lambda z: rho(z)[0]
    hvp = lambda z, v: jax.jvp(jax.grad(first), (z,), (v,))[1]
    per_sample = jax.jit(lambda Z, V: jax.lax.map(lambda a: (rho(a[0]), jax.grad(first)(a[0]),
                                                                 jax.hessian(first)(a[0]), hvp(a[0], a[1])), (Z, V)))
    batched = jax.jit(jax.vmap(lambda z, v: (rho(z), jax.grad(first)(z), jax.hessian(first)(z), hvp(z, v))))
    for a, b in zip(batched(Z, V), per_sample(Z, V)):
        close(a, b, tol64=1e-12, tol32=1e-6)
    close(jax.jit(rho)(Z), per_sample(Z, V)[0], tol64=1e-12, tol32=1e-6)  # leading batch axes natively
    close(jax.jit(jax.vmap(jax.jacfwd(rho)))(Z), jax.jit(jax.vmap(jax.jacrev(rho)))(Z))
