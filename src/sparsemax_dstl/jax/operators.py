"""Quadratically regularized extrema over the last axis.

Inputs must be finite on a nonempty boolean mask; gamma is a positive scalar.
Empty rows return NaN deliberately. This module does not define STL boundary
semantics. Accumulation uses the input dtype. Float64 is available for audits.
"""
import jax
import jax.numpy as jnp


def _projection(z, gamma, mask):
    valid = jnp.any(mask, axis=-1)
    center = jnp.max(jnp.where(mask, z, -jnp.inf), axis=-1)
    center = jnp.where(valid, center, 0)
    s = jnp.where(mask, (z - center[..., None]) / gamma, -jnp.inf)
    u = jnp.sort(s, axis=-1, descending=True)
    total = jnp.cumsum(u, axis=-1)
    rank = jnp.arange(1, z.shape[-1] + 1, dtype=z.dtype)
    active = jnp.isfinite(u) & (1 + rank * u > total)
    k = jnp.maximum(jnp.sum(active, axis=-1), 1)
    threshold = (jnp.take_along_axis(total, (k - 1)[..., None], axis=-1)[..., 0] - 1) / k
    # A select rather than jnp.maximum: at an exact support switch (s == threshold)
    # maximum's derivative is 1/2, which gave a non-symmetric Hessian (E006).
    p = jnp.where(mask & (s > threshold[..., None]), s - threshold[..., None], 0)
    return p, center, threshold, valid


def sparsemax_weights(z, gamma, mask=None):
    z = jnp.asarray(z)
    mask = jnp.ones_like(z, dtype=bool) if mask is None else jnp.broadcast_to(mask, z.shape)
    p, _, _, valid = _projection(z, gamma, mask)
    return jnp.where(valid[..., None], p, jnp.nan)


@jax.custom_jvp
def _quadratic_max(z, gamma, mask):
    p, center, threshold, valid = _projection(z, gamma, mask)
    value = center + gamma * (threshold + 0.5 * jnp.sum(p * p, axis=-1))
    return jnp.where(valid, value, jnp.nan)


@_quadratic_max.defjvp
def _quadratic_max_jvp(primals, tangents):
    z, gamma, mask = primals
    dz, dgamma, _ = tangents
    p, _, _, _ = _projection(z, gamma, mask)
    value = _quadratic_max(z, gamma, mask)
    tangent = jnp.sum(p * jnp.where(mask, dz, 0), axis=-1)
    tangent -= 0.5 * jnp.sum(p * p, axis=-1) * dgamma
    return value, tangent


def lower_max(z, gamma, mask=None):
    """C1 under-approximation of max, exact at a full tie."""
    z = jnp.asarray(z)
    mask = jnp.ones_like(z, dtype=bool) if mask is None else jnp.broadcast_to(mask, z.shape)
    m = jnp.sum(mask, axis=-1)
    return _quadratic_max(z, gamma, mask) + gamma / (2 * jnp.maximum(m, 1))


def lower_min(z, gamma, mask=None):
    """C1 under-approximation of min, exact beyond the bottom-gap band."""
    z = jnp.asarray(z)
    mask = jnp.ones_like(z, dtype=bool) if mask is None else jnp.broadcast_to(mask, z.shape)
    return -_quadratic_max(-z, gamma, mask) - gamma / 2
