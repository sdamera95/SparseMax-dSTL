"""Method registry for the core study (E009): extremum reductions and their parameters.

Every method is a semantics for E005's evaluator, a pair (max_reduce, min_reduce) of
functions reduce(z, param, mask) over the last axis, or the name "exact".

Matched graph budgets (core draft smd:sec:experiments, D006 item 14). Every nontrivial
extremum node v, with valid arity m_v > 1, receives the same allocated error eps. The
smooth parameter follows from the node's own arity,

    lse:        beta_v  = log(m_v) / eps,        local bound log(m_v) / beta_v = eps,
    sparsemax:  gamma_v = 2 eps / (1 - 1/m_v),   local bound gamma_v / 2 (1 - 1/m_v) = eps,

and unary nodes (m_v = 1) are evaluated exactly. The wrappers below compute the parameter
from the arity inside reduce(z, eps, mask), so E005's evaluator is unchanged. With a mask,
the arity varies by row; the wrappers then use the positive homogeneity of both operators,

    Q_gamma(z) = gamma Q_1(z / gamma),   L_beta(z) = L_1(beta z) / beta,

which holds exactly in real arithmetic for the lower sparsemax extrema and for both lower
log-sum-exp extrema, so one call serves every row. Without a mask the arity is static and
the operator is called with its scalar parameter directly.

The root budget of a matched method is eps times D, the largest number of nontrivial nodes
on a root-to-leaf path (path_depth). The study fixes a target graph budget B and uses
eps = B / D for both smooth methods.

Gilpin's and D-GMSR's reductions (E008) are used unchanged at every node, unary nodes
included, with their own parameters (k1, k2) and (eps, p). They have no matched budget:
Gilpin's maximum has only a data-dependent error bound, and D-GMSR is not a lower bound.
They are tuned independently (D006 item 14).
"""
import math

import jax.numpy as jnp
import numpy as np

from .baselines import DGMSR, GILPIN
from .evaluator import budget, exact_max, exact_min, gm_exp_min, gm_pm01_max, gm_pm01_min, gm_pm10_max, gm_pm10_min, lse_max, lse_min, lse_plain_max
from .operators import lower_max, lower_min


def _rows(z, mask):
    """Valid arity per row, as an array of z's dtype."""
    return jnp.sum(jnp.broadcast_to(mask, z.shape), axis=-1).astype(z.dtype)


def _gamma(m, eps):
    return 2 * eps / (1 - 1 / m)


def _beta(m, eps):
    return jnp.log(m) / eps


def sparsemax_max(z, eps, mask=None):
    """Lower sparsemax maximum with gamma = 2 eps / (1 - 1/m) at arity m; exact at m = 1."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else lower_max(z, _gamma(m, eps))
    m = _rows(z, mask)
    g = _gamma(jnp.maximum(m, 2), eps)
    smooth = g * lower_max(z / g[..., None], 1.0, mask)
    return jnp.where(m > 1, smooth, exact_max(z, None, mask))


def sparsemax_min(z, eps, mask=None):
    """Lower sparsemax minimum with gamma = 2 eps / (1 - 1/m) at arity m; exact at m = 1."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else lower_min(z, _gamma(m, eps))
    m = _rows(z, mask)
    g = _gamma(jnp.maximum(m, 2), eps)
    smooth = g * lower_min(z / g[..., None], 1.0, mask)
    return jnp.where(m > 1, smooth, exact_min(z, None, mask))


def lse_max_matched(z, eps, mask=None):
    """Lower log-sum-exp maximum with beta = log(m) / eps at arity m; exact at m = 1."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else lse_max(z, math.log(m) / eps)
    m = _rows(z, mask)
    b = _beta(jnp.maximum(m, 2), eps)
    smooth = lse_max(z * b[..., None], 1.0, mask) / b
    return jnp.where(m > 1, smooth, exact_max(z, None, mask))


def lse_min_matched(z, eps, mask=None):
    """Lower log-sum-exp minimum with beta = log(m) / eps at arity m; exact at m = 1."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else lse_min(z, math.log(m) / eps)
    m = _rows(z, mask)
    b = _beta(jnp.maximum(m, 2), eps)
    smooth = lse_min(z * b[..., None], 1.0, mask) / b
    return jnp.where(m > 1, smooth, exact_min(z, None, mask))


def lse_plain_max_matched(z, eps, mask=None):
    """Plain log-sum-exp maximum (no -log(m)/beta shift) with beta = log(m) / eps at arity m;
    exact at m = 1. Not a lower bound: it lies between the exact maximum and the exact
    maximum plus eps."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else lse_plain_max(z, math.log(m) / eps)
    m = _rows(z, mask)
    b = _beta(jnp.maximum(m, 2), eps)
    smooth = lse_plain_max(z * b[..., None], 1.0, mask) / b
    return jnp.where(m > 1, smooth, exact_max(z, None, mask))


def gm_exp_min_matched(z, eps, mask=None):
    """Conjunction of the generalized mean robustness's exponential member (stl.semantics.gm_exp_min) with beta = log(m) / eps
    at arity m (E036, M003 note 2026-10-01 06:25Z); the entry itself at m = 1. Not a lower bound and
    not matched in error (the measure has no per-node error bound): beta is set by the rule the
    log-sum-exp arms use. Inside an Until (eq. 15, nested) the prefix gets m = k + 1, the pair m = 2."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else gm_exp_min(z, math.log(m) / eps)
    m = _rows(z, mask)
    smooth = gm_exp_min(z, _beta(jnp.maximum(m, 2), eps), mask)
    return jnp.where(m > 1, smooth, exact_min(z, None, mask))


def gm_exp_max_matched(z, eps, mask=None):
    """Disjunction of the generalized mean robustness's exponential member with beta = log(m) / eps, -conj(-z) (De Morgan)."""
    return -gm_exp_min_matched(-jnp.asarray(z), eps, mask)


gm_exp_min_matched.nested_until = gm_exp_max_matched.nested_until = True


SEMANTICS = {
    "exact": "exact",
    "sparsemax": (sparsemax_max, sparsemax_min),
    "lse": (lse_max_matched, lse_min_matched),
    "gilpin": GILPIN,
    "dgmsr": DGMSR,
    "lse_plain": (lse_plain_max_matched, lse_min_matched),
    "gm_pm01": (gm_pm01_max, gm_pm01_min),
    "gm_pm10": (gm_pm10_max, gm_pm10_min),
    "gm_exp": (gm_exp_max_matched, gm_exp_min_matched),
}
# the methods older experiments iterate over; lse_plain (E034), gm_pm01, gm_pm10 and gm_exp (E036, the
# generalized mean robustness of orders (0, 1) and (-10, 10) and its exponential member) are reached by name only
METHODS = ("exact", "sparsemax", "lse", "gilpin", "dgmsr")
MATCHED = ("sparsemax", "lse")


# ------------------------------------------------------------------
# budgets, on the host in float64

def local_error(name, m, eps):
    """Published worst-case local error of a matched method at arity m, at the parameter
    its wrapper uses: gamma_m / 2 (1 - 1/m) for sparsemax and log(m) / beta_m for lse,
    zero at unary nodes. Both equal eps at every nontrivial node in exact arithmetic."""
    m = np.asarray(m, np.float64)
    safe = np.maximum(m, 2)
    if name == "sparsemax":
        b = 2 * eps / (1 - 1 / safe) / 2 * (1 - 1 / safe)
    elif name == "lse":
        b = np.log(safe) / (np.log(safe) / eps)
    else:
        raise ValueError("no matched budget for " + repr(name))
    return np.where(m > 1, b, 0.0)


def path_depth(program):
    """Largest number of nontrivial nodes (arity > 1) on a root-to-leaf path, per root entry."""
    return budget(program, lambda m, _: (np.asarray(m) > 1).astype(np.float64)).astype(int)


def graph_budget(program, name, eps):
    """Root budget of a matched method from E005's budget function and local_error."""
    return budget(program, lambda m, e: local_error(name, m, e), eps)


def node_error(program, target):
    """Per-node allocated error eps = target / D for the root entry at t = 0."""
    D = int(path_depth(program)[0])
    if D == 0:
        raise ValueError("the formula has no nontrivial node, so no budget can be allocated")
    return target / D
