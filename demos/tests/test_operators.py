import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from sparsemax_diffstl.operators import lower_max, lower_min, sparsemax_weights


def test_batched_masked_bands_and_gradients():
    z = jax.random.normal(jax.random.key(17), (2048, 31))
    mask = jax.random.uniform(jax.random.key(18), z.shape) > 0.35
    mask = mask.at[:, 0].set(True)
    gamma = jnp.array(0.7)
    p = jax.jit(sparsemax_weights)(z, gamma, mask)
    hi = jax.jit(lower_max)(z, gamma, mask)
    lo = jax.jit(lower_min)(z, gamma, mask)
    ref_hi = jnp.max(jnp.where(mask, z, -jnp.inf), axis=-1)
    ref_lo = jnp.min(jnp.where(mask, z, jnp.inf), axis=-1)
    band = gamma / 2 * (1 - 1 / mask.sum(-1))
    assert bool(jnp.all((ref_hi - hi >= -2e-6) & (ref_hi - hi <= band + 2e-6)))
    assert bool(jnp.all((ref_lo - lo >= -2e-6) & (ref_lo - lo <= band + 2e-6)))
    np.testing.assert_allclose(p.sum(-1), 1, atol=2e-6)
    assert bool(jnp.all(p >= 0))
    assert bool(jnp.all(jnp.where(mask, 0, p) == 0))
    # A winner can lie exactly gamma above the threshold: non-strict bound.
    assert bool(jnp.all((p == 0) | (z >= ref_hi[:, None] - gamma - 2e-6)))
    grad = jax.jit(jax.vmap(jax.grad(lower_max), in_axes=(0, None, 0)))(z, gamma, mask)
    np.testing.assert_allclose(grad, p, atol=2e-6)


def test_finite_difference_and_gamma_derivative():
    z = jnp.array([0.12, 0.07, -0.3, -2.0])
    gamma = jnp.array(0.8)
    eps = 0.001
    eye = jnp.eye(4)
    fd = jax.vmap(lambda e: (lower_max(z + eps * e, gamma) - lower_max(z - eps * e, gamma)) / (2 * eps))(eye)
    np.testing.assert_allclose(jax.grad(lower_max)(z, gamma), fd, atol=5e-5)
    p = sparsemax_weights(z, gamma)
    expected = -0.5 * jnp.sum(p * p) + 1 / 8
    np.testing.assert_allclose(jax.grad(lower_max, argnums=1)(z, gamma), expected, atol=1e-6)
    hess = jax.jacfwd(jax.grad(lower_max))(z, gamma)
    active = (p > 0).astype(z.dtype)
    expected_hess = (jnp.diag(active) - jnp.outer(active, active) / active.sum()) / gamma
    np.testing.assert_allclose(hess, expected_hess, atol=2e-6)


def test_ties_masking_singleton_and_empty():
    z = jnp.ones(8)
    np.testing.assert_allclose(sparsemax_weights(z, 0.4), jnp.ones(8) / 8)
    np.testing.assert_allclose(jax.grad(jnp.max)(z), jnp.ones(8) / 8)
    np.testing.assert_allclose(lower_max(z, 0.4), 1)
    np.testing.assert_allclose(lower_min(z, 0.4), 1 - 0.2 * 7 / 8)
    base = jnp.array([0.0, -0.1, -0.5])
    padded = jnp.concatenate([base, jnp.array([jnp.inf, -jnp.inf])])
    mask = jnp.array([True, True, True, False, False])
    np.testing.assert_allclose(lower_max(padded, 0.3, mask), lower_max(base, 0.3))
    np.testing.assert_allclose(jax.grad(lower_max)(padded, 0.3, mask)[3:], 0)
    np.testing.assert_allclose(lower_max(jnp.array([2.0]), 0.7), 2)
    np.testing.assert_allclose(lower_min(jnp.array([2.0]), 0.7), 2)
    assert bool(jnp.isnan(lower_max(base, 0.3, jnp.zeros(3, dtype=bool))))


@pytest.mark.parametrize("z,gamma", [
    ([0.2, 0.1, -0.4], 0.7),
    ([1.0, 1.0, 1.0], 0.4),
    ([2.0, -3.0, 0.1], 0.2),
    ([-0.2, -0.1, -0.05, -1.0], 1.5),
])
def test_independent_constrained_qp_oracle(z, gamma):
    # Independent small optimization oracle, not the production projection.
    z = np.asarray(z)
    ref = minimize(lambda p: gamma / 2 * np.dot(p, p) - p @ z,
                   np.ones(len(z)) / len(z), jac=lambda p: gamma * p - z,
                   bounds=[(0, 1)] * len(z),
                   constraints={"type": "eq", "fun": lambda p: p.sum() - 1,
                                "jac": lambda p: np.ones_like(p)},
                   method="SLSQP", options={"ftol": 1e-12})
    assert ref.success
    p = np.asarray(sparsemax_weights(jnp.asarray(z), gamma))
    np.testing.assert_allclose(p, ref.x, atol=2e-6)
    value = -ref.fun + gamma / (2 * len(z))
    np.testing.assert_allclose(lower_max(jnp.asarray(z), gamma), value, atol=2e-6)


def test_binary_polynomial_identity():
    a = jnp.linspace(-1.5, 1.5, 2001)
    b = jnp.zeros_like(a)
    gamma = 0.7
    h = jnp.clip(0.5 + (b - a) / (2 * gamma), 0, 1)
    polynomial = b * (1 - h) + a * h - gamma * h * (1 - h)
    np.testing.assert_allclose(lower_min(jnp.stack([a, b], -1), gamma), polynomial, atol=2e-6)


def test_nested_error_budget():
    z = jax.random.normal(jax.random.key(29), (128, 16, 31))
    gamma = 0.4
    exact = jnp.max(jnp.min(z, axis=-1), axis=-1)
    smooth = lower_max(lower_min(z, gamma), gamma)
    budget = gamma / 2 * ((1 - 1 / 16) + (1 - 1 / 31))
    assert bool(jnp.all(exact - smooth >= -2e-6))
    assert bool(jnp.all(exact - smooth <= budget + 2e-6))


def test_long_window_small_gamma_and_shift():
    z = jnp.full(8192, -1e9).at[0].set(0.0).at[1].set(-0.004)
    p = jax.jit(sparsemax_weights)(z, 0.01)
    np.testing.assert_allclose(p[:2], [0.7, 0.3], atol=1e-6)
    np.testing.assert_allclose(p.sum(), 1, atol=1e-6)
    assert bool(jnp.all(p[2:] == 0))
    base = jnp.array([0.0, -0.125, -2.0])
    np.testing.assert_allclose(sparsemax_weights(base + 1e6, 0.5),
                               sparsemax_weights(base, 0.5), atol=1e-6)
