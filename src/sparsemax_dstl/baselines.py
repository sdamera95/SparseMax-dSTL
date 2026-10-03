"""Published smooth-robustness reductions over the last axis, as baselines.

Each function has the form reduce(z, param, mask) of operators.lower_max and
operators.lower_min. mask is None or a boolean array broadcastable to z;
masked entries are excluded exactly and may hold any padding, including
infinities. Valid entries must be finite. A row with no valid entry returns
NaN deliberately, with a zero gradient.

Gilpin, Kurtz and Lin, "A smooth robustness measure of signal temporal logic
for symbolic control", IEEE Control Systems Letters 2020, arXiv:2006.05239:

    gilpin_min, eq. (9):  -(1/k1) log sum_i exp(-k1 a_i)
    gilpin_max, eq. (11): sum_i a_i exp(k2 a_i) / sum_i exp(k2 a_i)

Uzun, Elango, Garoche and Acikmese, D-GMSR, arXiv:2405.10996, eqs. (3), (4a)
and (4b), with W = sum_i w_i over the valid entries:

    dgmsr_and: M0(|y|_+^2)^(1/2) - Mp(|y|_-^2)^(1/2)
    M0(z) = (eps^W + prod_i z_i^w_i)^(1/W)
    Mp(z) = (eps^p + (1/W) sum_i w_i z_i^p)^(1/p)
    dgmsr_or(y) = -dgmsr_and(-y)

Every value is computed in a rearranged but mathematically identical form.
The shifts and log-domain means keep exp, eps^W, the product, eps^p and z^p
from overflowing or underflowing. The D-GMSR brackets avoid cancelling
sqrt(M0) against sqrt(Mp). Implementation log E008 lists each rearrangement.
Derivatives come from JAX automatic differentiation; no custom rule is used.
"""
import jax
import jax.numpy as jnp


def _valid(z, mask):
    return jnp.ones(z.shape, bool) if mask is None else jnp.broadcast_to(jnp.asarray(mask, bool), z.shape)


def gilpin_min(z, k1, mask=None):
    """Gilpin's smooth minimum, eq. (9); never above the exact minimum."""
    z = jnp.asarray(z)
    valid = _valid(z, mask)
    k = jnp.asarray(k1, z.dtype)
    nonempty = jnp.any(valid, axis=-1)
    zm = jnp.where(valid, z, jnp.inf)
    # The value does not depend on the shift m, so it carries no derivative.
    m = jax.lax.stop_gradient(jnp.where(nonempty, jnp.min(zm, axis=-1), 0))[..., None]
    d = jnp.where(valid, z, m) - m
    # tail = sum_i exp(-k d_i) - 1, taking the 1 from one minimizing entry through
    # expm1, which is exactly 0 there; log1p(tail) keeps the gap to the minimum
    # accurate when the other entries are far above it.
    at_min = jnp.arange(z.shape[-1]) == jnp.argmin(zm, axis=-1)[..., None]
    tail = jnp.sum(jnp.where(valid, jnp.where(at_min, jnp.expm1(-k * d), jnp.exp(-k * d)), 0), axis=-1)
    value = m[..., 0] - jnp.log1p(tail) / k
    return jnp.where(nonempty, value, jnp.nan)


def gilpin_max(z, k2, mask=None):
    """Gilpin's smooth maximum, eq. (11), a softmax-weighted mean; never above the exact maximum."""
    z = jnp.asarray(z)
    valid = _valid(z, mask)
    k = jnp.asarray(k2, z.dtype)
    nonempty = jnp.any(valid, axis=-1)
    m = jnp.max(jnp.where(valid, z, -jnp.inf), axis=-1)
    # The value does not depend on the shift m, so it carries no derivative.
    m = jax.lax.stop_gradient(jnp.where(nonempty, m, 0))
    d = jnp.where(valid, z, m[..., None]) - m[..., None]
    e = jnp.where(valid, jnp.exp(k * d), 0)
    total = jnp.where(nonempty, jnp.sum(e, axis=-1), 1)
    value = m + jnp.sum(e * d, axis=-1) / total
    return jnp.where(nonempty, value, jnp.nan)


def _gap(log_eps, s):
    # exp(log_eps/2 + s) - exp(log_eps/2) for s >= 0, without cancellation or overflow.
    return -jnp.exp(log_eps / 2 + s) * jnp.expm1(-s)


def dgmsr_and(z, param, mask=None):
    """D-GMSR conjunction, eq. (3), with param = (eps, p) or (eps, p, w).

    Evaluated as [M0^(1/2) - eps^(1/2)] - [Mp^(1/2) - eps^(1/2)]. The first
    bracket is nonzero only if every valid entry is positive, the second only
    if some valid entry is negative (Remark 2 of the paper). Each bracket is
    computed from the log of its mean relative to eps.
    """
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

    # log(prod_i z_i^w_i / eps^W) with z_i = y_i^2, summed entrywise; (log M0 - log eps)/2 = softplus(u)/(2W).
    log_pos = jnp.log(jnp.where(pos, y, 1))
    u = jnp.sum(jnp.where(pos, w * (2 * log_pos - log_eps), 0), axis=-1)
    first = jnp.where(all_pos, _gap(log_eps, jax.nn.softplus(u) / (2 * W)), 0)

    # log((1/W) sum_i w_i (z_i/eps)^p) over negative entries; (log Mp - log eps)/2 = softplus(v)/(2p).
    log_neg = jnp.log(jnp.where(neg, -y, 1))
    t = jnp.where(neg, jnp.log(jnp.where(neg, w, 1)) + p * (2 * log_neg - log_eps), -jnp.inf)
    t = jnp.where(any_neg[..., None], t, 0)  # a row without negatives must not reach logsumexp as all -inf
    v = jax.nn.logsumexp(t, axis=-1) - jnp.log(W)
    second = jnp.where(any_neg, _gap(log_eps, jax.nn.softplus(v) / (2 * p)), 0)

    return jnp.where(nonempty, first - second, jnp.nan)


def dgmsr_or(z, param, mask=None):
    """D-GMSR disjunction, -dgmsr_and(-z) (paper's definition after eq. (4b))."""
    return -dgmsr_and(-jnp.asarray(z), param, mask)


# (max_reduce, min_reduce) pairs for the STL evaluator's semantics argument.
# Gilpin's pair takes param = (k1, k2); D-GMSR's takes (eps, p) or (eps, p, w)
# with w a scalar, since one param reaches every node.
GILPIN = (lambda z, k, mask=None: gilpin_max(z, k[1], mask),
          lambda z, k, mask=None: gilpin_min(z, k[0], mask))
DGMSR = (dgmsr_or, dgmsr_and)
