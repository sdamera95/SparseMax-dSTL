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


# ------------------------------------------------------------------
# the same operators with the threshold found by recounting instead of a sort

def _recount_threshold(s, mask):
    """The threshold of s (valid entries in [-1, 0] with maximum 0, masked entries -inf) by Michelot's recount: from
    tau = -1, tau <- max(tau, (sum of s over {s > tau} - 1) / |{s > tau}|) until no row's set changes."""
    sv = jax.lax.stop_gradient(s)
    n = s.shape[-1]

    def body(c):
        tau, prev, _, it = c
        on = sv > tau[..., None]
        k = jnp.sum(on, axis=-1, dtype=jnp.int32)
        theta = (jnp.sum(jnp.where(on, sv, 0), axis=-1) - 1) / jnp.maximum(k, 1)
        return jnp.maximum(tau, theta), k, jnp.any(k != prev), it + 1

    init = (jnp.full(s.shape[:-1], -1, s.dtype), jnp.full(s.shape[:-1], -1, jnp.int32), jnp.array(True), jnp.array(0))
    tau, _, _, _ = jax.lax.while_loop(lambda c: c[2] & (c[3] < n + 2), body, init)
    support = mask & (sv > tau[..., None])
    return (jnp.sum(jnp.where(support, s, 0), axis=-1) - 1) / jnp.maximum(jnp.sum(support, axis=-1), 1)


def _projection_recount(z, gamma, mask):
    valid = jnp.any(mask, axis=-1)
    center = jnp.max(jnp.where(mask, z, -jnp.inf), axis=-1)
    center = jnp.where(valid, center, 0)
    s = jnp.where(mask, (z - center[..., None]) / gamma, -jnp.inf)
    threshold = _recount_threshold(s, mask)
    p = jnp.where(mask & (s > threshold[..., None]), s - threshold[..., None], 0)
    return p, center, threshold, valid


@jax.custom_jvp
def _quadratic_max_recount(z, gamma, mask):
    p, center, threshold, valid = _projection_recount(z, gamma, mask)
    value = center + gamma * (threshold + 0.5 * jnp.sum(p * p, axis=-1))
    return jnp.where(valid, value, jnp.nan)


@_quadratic_max_recount.defjvp
def _quadratic_max_recount_jvp(primals, tangents):
    z, gamma, mask = primals
    dz, dgamma, _ = tangents
    p, _, _, _ = _projection_recount(z, gamma, mask)
    value = _quadratic_max_recount(z, gamma, mask)
    tangent = jnp.sum(p * jnp.where(mask, dz, 0), axis=-1)
    tangent -= 0.5 * jnp.sum(p * p, axis=-1) * dgamma
    return value, tangent


def _full_mask(z, mask):
    return jnp.ones_like(z, dtype=bool) if mask is None else jnp.broadcast_to(mask, z.shape)


def lower_max_recount(z, gamma, mask=None):
    """lower_max with the threshold found by recounting."""
    z = jnp.asarray(z)
    mask = _full_mask(z, mask)
    return _quadratic_max_recount(z, gamma, mask) + gamma / (2 * jnp.maximum(jnp.sum(mask, axis=-1), 1))


def lower_min_recount(z, gamma, mask=None):
    """lower_min with the threshold found by recounting."""
    z = jnp.asarray(z)
    return -_quadratic_max_recount(-z, gamma, _full_mask(z, mask)) - gamma / 2


def extremum_and_weights(z, gamma, mask, kind):
    """lower_max (kind "max") or lower_min ("min") over the last axis and the sparsemax weights of z (of -z for "min"),
    from one recount; not for differentiation."""
    mask = _full_mask(z, mask)
    y = z if kind == "max" else -z
    p, center, threshold, valid = _projection_recount(y, gamma, mask)
    q = jnp.where(valid, center + gamma * (threshold + 0.5 * jnp.sum(p * p, axis=-1)), jnp.nan)
    value = q + gamma / (2 * jnp.maximum(jnp.sum(mask, axis=-1), 1)) if kind == "max" else -q - gamma / 2
    return value, jnp.where(valid[..., None], p, jnp.nan)
