"""Semantics by method name. The reductions here take an error eps per node as their parameter and set
beta = log(m) / eps or gamma = 2 eps / (1 - 1/m) at a node of m valid entries (Eq. (15) of the paper)."""
import math

import jax.numpy as jnp
import numpy as np

from .baselines import SMOOTH_GM, SOFTMAX_MEAN
from .evaluator import budget, exact_max, exact_min, gm_exp_min, gm_pm01_max, gm_pm01_min, gm_pm10_max, gm_pm10_min, lse_max, lse_min, lse_plain_max
from .operators import lower_max, lower_min


def _rows(z, mask):
    """Number of valid entries per row, in z's dtype."""
    return jnp.sum(jnp.broadcast_to(mask, z.shape), axis=-1).astype(z.dtype)


def _gamma(m, eps):
    return 2 * eps / (1 - 1 / m)


def _beta(m, eps):
    return jnp.log(m) / eps


def sparsemax_max(z, eps, mask=None):
    """Sparsemax lower maximum with gamma = 2 eps / (1 - 1/m) at m valid entries; the entry itself at m = 1."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else lower_max(z, _gamma(m, eps))
    m = _rows(z, mask)
    g = _gamma(jnp.maximum(m, 2), eps)
    smooth = g * lower_max(z / g[..., None], 1.0, mask)
    return jnp.where(m > 1, smooth, exact_max(z, None, mask))


def sparsemax_min(z, eps, mask=None):
    """Sparsemax lower minimum with gamma = 2 eps / (1 - 1/m) at m valid entries; the entry itself at m = 1."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else lower_min(z, _gamma(m, eps))
    m = _rows(z, mask)
    g = _gamma(jnp.maximum(m, 2), eps)
    smooth = g * lower_min(z / g[..., None], 1.0, mask)
    return jnp.where(m > 1, smooth, exact_min(z, None, mask))


def lse_max_matched(z, eps, mask=None):
    """Sound log-sum-exp maximum with beta = log(m) / eps at m valid entries; the entry itself at m = 1."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else lse_max(z, math.log(m) / eps)
    m = _rows(z, mask)
    b = _beta(jnp.maximum(m, 2), eps)
    smooth = lse_max(z * b[..., None], 1.0, mask) / b
    return jnp.where(m > 1, smooth, exact_max(z, None, mask))


def lse_min_matched(z, eps, mask=None):
    """Log-sum-exp minimum with beta = log(m) / eps at m valid entries; the entry itself at m = 1."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else lse_min(z, math.log(m) / eps)
    m = _rows(z, mask)
    b = _beta(jnp.maximum(m, 2), eps)
    smooth = lse_min(z * b[..., None], 1.0, mask) / b
    return jnp.where(m > 1, smooth, exact_min(z, None, mask))


def lse_plain_max_matched(z, eps, mask=None):
    """Plain log-sum-exp maximum with beta = log(m) / eps at m valid entries; the entry itself at m = 1."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else lse_plain_max(z, math.log(m) / eps)
    m = _rows(z, mask)
    b = _beta(jnp.maximum(m, 2), eps)
    smooth = lse_plain_max(z * b[..., None], 1.0, mask) / b
    return jnp.where(m > 1, smooth, exact_max(z, None, mask))


def gm_exp_min_matched(z, eps, mask=None):
    """Conjunction of the exponential member of the generalized-mean robustness with beta = log(m) / eps
    at m valid entries; the entry itself at m = 1."""
    z = jnp.asarray(z)
    if mask is None:
        m = z.shape[-1]
        return z[..., 0] if m == 1 else gm_exp_min(z, math.log(m) / eps)
    m = _rows(z, mask)
    smooth = gm_exp_min(z, _beta(jnp.maximum(m, 2), eps), mask)
    return jnp.where(m > 1, smooth, exact_min(z, None, mask))


def gm_exp_max_matched(z, eps, mask=None):
    """Disjunction of the exponential member with beta = log(m) / eps, -gm_exp_min_matched(-z)."""
    return -gm_exp_min_matched(-jnp.asarray(z), eps, mask)


gm_exp_min_matched.nested_until = gm_exp_max_matched.nested_until = True


SEMANTICS = {
    "exact": "exact",
    "sparsemax": (sparsemax_max, sparsemax_min),
    "lse": (lse_max_matched, lse_min_matched),
    "softmax_mean": SOFTMAX_MEAN,
    "smooth_gm": SMOOTH_GM,
    "lse_plain": (lse_plain_max_matched, lse_min_matched),
    "gm_pm01": (gm_pm01_max, gm_pm01_min),
    "gm_pm10": (gm_pm10_max, gm_pm10_min),
    "gm_exp": (gm_exp_max_matched, gm_exp_min_matched),
}
METHODS = ("exact", "sparsemax", "lse", "softmax_mean", "smooth_gm")
MATCHED = ("sparsemax", "lse")


# ------------------------------------------------------------------
# budgets, on the host in float64

def local_error(name, m, eps):
    """Error band of "sparsemax" or "lse" at m valid entries with the parameter its reduction uses:
    gamma_m / 2 (1 - 1/m) or log(m) / beta_m, and 0 at m = 1."""
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
    """Largest number of nodes with more than one valid entry on a root-to-leaf path, per root entry."""
    return budget(program, lambda m, _: (np.asarray(m) > 1).astype(np.float64)).astype(int)


def graph_budget(program, name, eps):
    """Budget of every root entry (evaluator.budget) with the bands of local_error."""
    return budget(program, lambda m, e: local_error(name, m, e), eps)


def node_error(program, target):
    """Error per node eps = target / D, with D the path depth of the root entry at t = 0."""
    D = int(path_depth(program)[0])
    if D == 0:
        raise ValueError("the formula has no nontrivial node, so no budget can be allocated")
    return target / D
