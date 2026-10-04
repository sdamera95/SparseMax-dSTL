"""Sparsemax weights and the sparsemax lower maximum and minimum over the last axis
(Eqs. (5) and (7) of the paper). A row with no valid entry in mask returns NaN."""
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
    # a select: jnp.maximum would have the derivative 1/2 at s == threshold
    p = jnp.where(mask & (s > threshold[..., None]), s - threshold[..., None], 0)
    return p, center, threshold, valid


def sparsemax_weights(z, gamma, mask=None):
    """Sparsemax weights p_gamma(z): z / gamma projected onto the simplex (Eq. (5))."""
    z = jnp.asarray(z)
    mask = jnp.ones_like(z, dtype=bool) if mask is None else jnp.broadcast_to(mask, z.shape)
    p, _, _, valid = _projection(z, gamma, mask)
    return jnp.where(valid[..., None], p, jnp.nan)


@jax.custom_jvp
def _quadratic_max(z, gamma, mask):
    """M_gamma(z) = max over p in the simplex of p.z - gamma/2 |p|^2 (Eq. (5))."""
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
    """Sparsemax lower maximum M_gamma(z) + gamma / (2m) over m valid entries (Eq. (7))."""
    z = jnp.asarray(z)
    mask = jnp.ones_like(z, dtype=bool) if mask is None else jnp.broadcast_to(mask, z.shape)
    m = jnp.sum(mask, axis=-1)
    return _quadratic_max(z, gamma, mask) + gamma / (2 * jnp.maximum(m, 1))


def lower_min(z, gamma, mask=None):
    """Sparsemax lower minimum -M_gamma(-z) - gamma / 2 (Eq. (7))."""
    z = jnp.asarray(z)
    mask = jnp.ones_like(z, dtype=bool) if mask is None else jnp.broadcast_to(mask, z.shape)
    return -_quadratic_max(-z, gamma, mask) - gamma / 2
