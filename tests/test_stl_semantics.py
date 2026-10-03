import jax
import jax.numpy as jnp
import numpy as np
import pytest

import twolink
from sparsemax_dstl import jax as stl_jax
from sparsemax_dstl import stl
from sparsemax_dstl.jax import robustness
from sparsemax_dstl.stl import Always, And, Atom, Eventually, Implies, Not, Or, Release, Until, compile_formula, oracle

# Scores are drawn in float32 so that exact semantics, which only select and
# negate inputs, must agree with the float64 oracle bit for bit in both precisions.


def scores(rng, T, P=3):
    return rng.normal(size=(T, P)).astype(np.float32)


def check_oracle(f, z, boundary):
    got = np.asarray(robustness(compile_formula(f, z.shape[0], boundary), z), np.float64)
    ref = oracle.trace(f, z.astype(np.float64), boundary)
    assert got.shape == ref.shape
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("boundary", ["strict", "clip"])
def test_exact_matches_brute_force_oracle(boundary):
    rng = np.random.default_rng(5 if boundary == "strict" else 6)
    for _ in range(80):  # formula count, not a time loop
        f = oracle.random_formula(rng, 3, int(rng.integers(1, 4)))
        T = stl.horizon(f) + int(rng.integers(1, 5))
        check_oracle(f, scores(rng, T), boundary)


def test_nnf_and_release_duality():
    rng = np.random.default_rng(7)
    f = Until((1, 3), Or(Atom(0), Not(Atom(1))), Implies(Atom(2), Always((0, 1), Atom(0))))
    nnf = stl.to_nnf(Not(f))
    assert nnf == Release((1, 3), And(Atom(0, True), Atom(1)),
                          And(Atom(2), Eventually((0, 1), Atom(0, True))))
    assert stl.to_nnf(Not(Not(f))) == stl.to_nnf(f)
    z = scores(rng, 9)
    for boundary in ("strict", "clip"):
        ref = oracle.trace(f, z.astype(np.float64), boundary)
        got = np.asarray(robustness(compile_formula(Not(f), 9, boundary), z), np.float64)
        np.testing.assert_array_equal(got, -ref)


def test_closed_prefix_until_probe():
    # M002 probe: left (1, 1, -1), right (-5, -5, 2), window [0, 2]. The closed
    # prefix needs the left operand at the witness too, so t = 0 gives -1; an
    # open prefix would give 1.
    z = jnp.array([[1.0, -5.0], [1.0, -5.0], [-1.0, 2.0]])
    f = Until((0, 2), Atom(0), Atom(1))
    np.testing.assert_array_equal(robustness(compile_formula(f, 3, "clip"), z), [-1.0, -1.0, -1.0])
    np.testing.assert_array_equal(robustness(compile_formula(f, 3), z), [-1.0])
    # Witness at offset 0 alone: min(left, right) at t, not right alone.
    g = Until((0, 0), Atom(0), Atom(1))
    np.testing.assert_array_equal(robustness(compile_formula(g, 3), z), [-5.0, -5.0, -1.0])


def test_strict_boundary_lengths_and_last_sample():
    z = jnp.arange(10.0)[:, None] * jnp.array([1.0, -1.0])
    f = Eventually((2, 4), Atom(0))
    rho = robustness(compile_formula(f, 10), z)
    assert rho.shape == (6,)
    np.testing.assert_array_equal(rho, jnp.arange(6.0) + 4)  # t = 5 reads the last sample, 9
    with pytest.raises(ValueError):
        compile_formula(Always((0, 3), Eventually((0, 6), Atom(0))), 9)
    assert compile_formula(Always((0, 3), Eventually((0, 6), Atom(0))), 10).steps[-1].length == 1


def test_clip_boundary_partial_and_empty_windows():
    z = jnp.arange(5.0)[:, None] * jnp.array([1.0, -1.0])
    G = robustness(compile_formula(Always((1, 3), Atom(1)), 5, "clip"), z)
    F = robustness(compile_formula(Eventually((1, 3), Atom(0)), 5, "clip"), z)
    U = robustness(compile_formula(Until((2, 3), Atom(0), Atom(0)), 5, "clip"), z)
    R = robustness(compile_formula(Release((2, 3), Atom(0), Atom(0)), 5, "clip"), z)
    np.testing.assert_array_equal(G, [-3.0, -4.0, -4.0, -4.0, np.inf])
    np.testing.assert_array_equal(F, [3.0, 4.0, 4.0, 4.0, -np.inf])
    np.testing.assert_array_equal(U, [0.0, 1.0, 2.0, -np.inf, -np.inf])
    np.testing.assert_array_equal(R, [2.0, 3.0, 4.0, np.inf, np.inf])
    with pytest.raises(ValueError):
        robustness(compile_formula(Always((1, 3), Atom(1)), 5, "clip"), z, "lse", 5.0)


@pytest.mark.parametrize("beta", [1.0, 10.0])
def test_lse_is_lower_bound_within_budget(beta):
    rng = np.random.default_rng(int(beta))
    tol = 1e-12 if jax.config.jax_enable_x64 else 2e-5
    for _ in range(25):  # formula count
        f = oracle.random_formula(rng, 3, int(rng.integers(1, 4)))
        T = stl.horizon(f) + int(rng.integers(1, 5))
        z = jnp.asarray(scores(rng, T), float)  # float64 under JAX_ENABLE_X64
        prog = compile_formula(f, T)
        gap = np.asarray(robustness(prog, z), np.float64) - np.asarray(robustness(prog, z, "lse", beta), np.float64)
        B = stl_jax.budget(prog, "lse", beta)
        assert np.all(gap >= -tol) and np.all(gap <= B + tol)


def test_budget_closed_forms():
    beta = 4.0
    b = stl_jax.budget(compile_formula(Always((2, 6), Atom(0)), 12), "lse", beta)
    np.testing.assert_allclose(b, np.log(5) / beta)
    # Until [1, 3]: outer max over 3 witnesses plus the widest inner min, 3 + 2 values.
    b = stl_jax.budget(compile_formula(Until((1, 3), Atom(0), Eventually((0, 1), Atom(1))), 12), "lse", beta)
    np.testing.assert_allclose(b, (np.log(3) + np.log(5) + np.log(2)) / beta)


def test_direct_and_recursive_grouping():
    # The lower lse maximum carries a log(m)/beta offset, so regrouping changes
    # it; the lower lse minimum has no offset and is exactly associative.
    z = jnp.array([[0.3, 0.1, -0.2]])
    beta = 5.0
    for op in (Or, And):
        direct = compile_formula(op(Atom(0), Atom(1), Atom(2)), 1)
        nested = compile_formula(op(Atom(0), op(Atom(1), Atom(2))), 1)
        np.testing.assert_array_equal(robustness(direct, z), robustness(nested, z))
        a, b = robustness(direct, z, "lse", beta)[0], robustness(nested, z, "lse", beta)[0]
        e = np.exp(beta * np.array([0.3, 0.1, -0.2], np.float64))
        if op is Or:
            np.testing.assert_allclose(a, np.log(e.sum() / 3) / beta, rtol=1e-6)
            np.testing.assert_allclose(b, np.log((e[0] + (e[1] + e[2]) / 2) / 2) / beta, rtol=1e-6)
            assert abs(float(a - b)) > 1e-2
        else:
            np.testing.assert_allclose(a, b, rtol=1e-6)


def test_shared_subformulas_share_steps():
    f = And(Eventually((0, 2), Atom(0)), Or(Eventually((0, 2), Atom(0)), Atom(1)))
    prog = compile_formula(f, 6)
    assert sum(s.label == "F[0,2]" for s in prog.steps) == 1


def test_predicate_dependencies():
    X = jax.random.uniform(jax.random.key(3), (64, 2), minval=-3, maxval=3)
    assert float(stl_jax.undeclared_gradient(twolink.PREDICATES, X)) == 0.0
    np.testing.assert_array_equal(stl_jax.dependency_matrix(twolink.PREDICATES, 2), [[1, 1], [1, 1], [0, 1]])
    g, grad, hess = twolink.analytic(np.asarray(X))
    Z = stl_jax.score_traces(twolink.PREDICATES, X)
    np.testing.assert_allclose(Z, g, rtol=1e-5, atol=1e-5)


def test_pluggable_reduction_pairs():
    # Any pair reduce(z, param, mask) plugs in; lower_max and lower_min already
    # have that signature. The built-in names are the same pairs.
    from sparsemax_dstl.jax.operators import lower_max, lower_min
    rng = np.random.default_rng(9)
    f = Or(Until((0, 3), Atom(0), Atom(1)), Always((1, 2), Implies(Atom(2), Atom(0))))
    z = jnp.asarray(scores(rng, 10), float)
    prog = compile_formula(f, 10)
    np.testing.assert_array_equal(robustness(prog, z, (stl_jax.exact_max, stl_jax.exact_min)), robustness(prog, z))
    np.testing.assert_array_equal(robustness(prog, z, (stl_jax.lse_max, stl_jax.lse_min), 3.0),
                                  robustness(prog, z, "lse", 3.0))
    gamma = 0.5
    sm = robustness(prog, z, (lower_max, lower_min), gamma)
    B = stl_jax.budget(prog, lambda m, g: g / 2 * (1 - 1 / m), gamma)
    gap = np.asarray(robustness(prog, z), np.float64) - np.asarray(sm, np.float64)
    tol = 1e-12 if jax.config.jax_enable_x64 else 2e-5
    assert np.all(gap >= -tol) and np.all(gap <= B + tol)
    with pytest.raises(ValueError):
        robustness(compile_formula(f, 10, "clip"), z, (lower_max, lower_min), gamma)
