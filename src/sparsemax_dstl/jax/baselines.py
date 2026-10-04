"""Reductions reduce(z, param, mask) over the last axis of the softmax-mean measure and of the continuously
differentiable generalized-mean measure ([7] and [12] in the paper). A row with no valid entry returns NaN."""
import jax
import jax.numpy as jnp


def _valid(z, mask):
    return jnp.ones(z.shape, bool) if mask is None else jnp.broadcast_to(jnp.asarray(mask, bool), z.shape)


def softmax_mean_min(z, k1, mask=None):
    """Minimum of the softmax-mean measure, -(1/k1) log sum_i exp(-k1 z_i)."""
    z = jnp.asarray(z)
    valid = _valid(z, mask)
    k = jnp.asarray(k1, z.dtype)
    nonempty = jnp.any(valid, axis=-1)
    zm = jnp.where(valid, z, jnp.inf)
    # the value does not depend on the shift m
    m = jax.lax.stop_gradient(jnp.where(nonempty, jnp.min(zm, axis=-1), 0))[..., None]
    d = jnp.where(valid, z, m) - m
    # tail = sum_i exp(-k d_i) - 1, the 1 taken from one minimizing entry through expm1
    at_min = jnp.arange(z.shape[-1]) == jnp.argmin(zm, axis=-1)[..., None]
    tail = jnp.sum(jnp.where(valid, jnp.where(at_min, jnp.expm1(-k * d), jnp.exp(-k * d)), 0), axis=-1)
    value = m[..., 0] - jnp.log1p(tail) / k
    return jnp.where(nonempty, value, jnp.nan)


def softmax_mean_max(z, k2, mask=None):
    """Maximum of the softmax-mean measure, sum_i z_i exp(k2 z_i) / sum_i exp(k2 z_i)."""
    z = jnp.asarray(z)
    valid = _valid(z, mask)
    k = jnp.asarray(k2, z.dtype)
    nonempty = jnp.any(valid, axis=-1)
    m = jnp.max(jnp.where(valid, z, -jnp.inf), axis=-1)
    # the value does not depend on the shift m
    m = jax.lax.stop_gradient(jnp.where(nonempty, m, 0))
    d = jnp.where(valid, z, m[..., None]) - m[..., None]
    e = jnp.where(valid, jnp.exp(k * d), 0)
    total = jnp.where(nonempty, jnp.sum(e, axis=-1), 1)
    value = m + jnp.sum(e * d, axis=-1) / total
    return jnp.where(nonempty, value, jnp.nan)


def _gap(log_eps, s):
    # exp(log_eps/2 + s) - exp(log_eps/2), written without the subtraction
    return -jnp.exp(log_eps / 2 + s) * jnp.expm1(-s)


def smooth_gm_and(z, param, mask=None):
    """Conjunction of [12], sqrt(M0(z_+^2)) - sqrt(Mp(z_-^2)), param = (eps, p) or (eps, p, w), W = sum_i w_i,
    M0(x) = (eps^W + prod_i x_i^w_i)^(1/W) and Mp(x) = (eps^p + (1/W) sum_i w_i x_i^p)^(1/p)."""
    eps, p = param[0], param[1]
    w = param[2] if len(param) > 2 else None
    y = jnp.asarray(z)
    dt = y.dtype
    valid = _valid(y, mask)
    w = jnp.ones(y.shape, dt) if w is None else jnp.broadcast_to(jnp.asarray(w, dt), y.shape)
    w = jnp.where(valid, w, 0)
    nonempty = jnp.any(valid, axis=-1)
    W = jnp.where(nonempty, jnp.sum(w, axis=-1), 1)
    log_eps = jnp.log(jnp.asarray(eps, dt))
    p = jnp.asarray(p, dt)
    pos = valid & (y > 0)
    neg = valid & (y < 0)
    all_pos = nonempty & jnp.all(pos | ~valid, axis=-1)
    any_neg = jnp.any(neg, axis=-1)

    # u = log(prod_i x_i^w_i / eps^W) with x_i = y_i^2; softplus(u) / (2W) = (log M0 - log eps)/2
    log_pos = jnp.log(jnp.where(pos, y, 1))
    u = jnp.sum(jnp.where(pos, w * (2 * log_pos - log_eps), 0), axis=-1)
    first = jnp.where(all_pos, _gap(log_eps, jax.nn.softplus(u) / (2 * W)), 0)

    # v = log((1/W) sum_i w_i (x_i / eps)^p) over negative entries; softplus(v) / (2p) = (log Mp - log eps)/2
    log_neg = jnp.log(jnp.where(neg, -y, 1))
    t = jnp.where(neg, jnp.log(jnp.where(neg, w, 1)) + p * (2 * log_neg - log_eps), -jnp.inf)
    t = jnp.where(any_neg[..., None], t, 0)  # a row without negatives must not reach logsumexp as all -inf
    v = jax.nn.logsumexp(t, axis=-1) - jnp.log(W)
    second = jnp.where(any_neg, _gap(log_eps, jax.nn.softplus(v) / (2 * p)), 0)

    return jnp.where(nonempty, first - second, jnp.nan)


def smooth_gm_or(z, param, mask=None):
    """Disjunction of [12], -smooth_gm_and(-z)."""
    return -smooth_gm_and(-jnp.asarray(z), param, mask)


# (max_reduce, min_reduce) pairs for the evaluator: param is (k1, k2) for the softmax-mean measure
# and (eps, p) or (eps, p, w) with a scalar w for the generalized-mean measure of [12]
SOFTMAX_MEAN = (lambda z, k, mask=None: softmax_mean_max(z, k[1], mask),
                lambda z, k, mask=None: softmax_mean_min(z, k[0], mask))
SMOOTH_GM = (smooth_gm_or, smooth_gm_and)
