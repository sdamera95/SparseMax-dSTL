"""STL robustness in JAX with pluggable extremum reductions.

scores has shape (..., T, P): leaf-score traces with time on the second-to-last
axis and predicates on the last. Leading axes are batch axes. Each step
gathers its window entries with static index arrays (an XLA gather, which
asdex traces; it has no reduce_window handler) and reduces the last axis.

A semantics is a pair (max_reduce, min_reduce) of functions

    reduce(z, param, mask) -> reduction of z over its last axis,

the signature of operators.lower_max and operators.lower_min. mask is None
when every entry is valid, otherwise a static boolean array broadcastable to
z; masked entries hold finite padding values and must not affect the result.
The names "exact" and "lse" select the built-in pairs below. With beta > 0 and
m valid entries,

    lse max: (1/beta) log sum exp(beta z) - log(m)/beta
    lse min: -(1/beta) log sum exp(-beta z)

Both lie between the exact extremum minus log(m)/beta and the exact extremum.
"lse_plain" is the plain log-sum-exp in common use: lse min, and at a maximum

    lse_plain max: (1/beta) log sum exp(beta z)

without the shift. It lies between the exact maximum and the exact maximum plus
log(m)/beta, so it is not a lower bound: it can report a violated
specification as satisfied. local_error and budget have no entry for it.
"sparsemax" selects operators.lower_max and operators.lower_min with parameter
gamma > 0; each lies between the exact extremum minus gamma/2 (1 - 1/m) and the
exact extremum.

"gm_pm01" and "gm_exp" (E036) are the generalized mean robustness of
Mehdipour, Vasile and Belta (IEEE TAC 70(3), 2025, eq. 12, 13, 15, 17). With
m valid entries, [x]_- = min(x, 0) and [x]_+ = max(x, 0), the conjunction is

    conj(x) = F_c(x)          if min(x) > 0                 (eq. 12, 17)
            = -F_g(-[x]_-)    otherwise,

and the disjunction is De Morgan's dual disj(x) = -conj(-x) (the paper's
definition; it gives disj(x) = -F_c(-x) if max(x) < 0 and F_g([x]_+)
otherwise, so disj = 0 at max(x) = 0, where printed eq. 13 puts that point in
the first branch). "gm_pm01" is the power mean robustness of order (p, q) =
(0, 1) (eq. 3, Definition 4): F_c is the geometric mean exp(mean log x), F_g
the arithmetic mean, so the conjunction of a violated vector is the mean of
its negative parts over all m entries; "gm_pm10" is the order (-10, 10), and
gm_power(p, q) builds any order (p real, q >= 1), with the power mean
M_p(x) = ((1/m) sum x^p)^(1/p) evaluated as exp((logsumexp(p log x) - log m)/p)
and M_q(v) as max(v) (mean((v / max v)^q))^(1/q). "gm_exp" is the exponential member
c(x) = -exp(-beta x), g(x) = exp(beta x) (Section IV-C):

    F_c(x) = -(1/beta) log((1/m) sum exp(-beta x)),
    F_g(y) = (1/beta) log((1/m) sum exp(beta y)).

The derivative of [x]_- at x = 0 is taken as 0. The sign of either measure
equals the sign of the exact robustness (their Theorem 2), but neither is a
lower bound: conj(x) >= min(x) and disj(x) <= max(x) (their Remark 1).
Numerics: F_c is evaluated shifted by min(x) (every term at most 1, so the
first branch stays positive in floating point); -F_g(-[x]_-) is evaluated as
-log1p(mean(expm1(beta v)))/beta, v = -[x]_-, when beta max(v) <= 1, which keeps
the sign of violations far below the floating-point resolution of 1, and
shifted by max(v) otherwise.

Until under these two semantics follows eq. 15 with the closed prefix nested:

    eta(phi U_[a,b] psi, t) = disj_k conj2(eta(psi, t+k), conj_{s in [t, t+k]} eta(phi, s)),

conj2 the conjunction of two entries. The program flattens the inner step into
one row psi(t+k), phi(t), ..., phi(t+k); evaluate() nests it (a reduction whose
nested_until attribute is true is applied to phi(t..t+k) first and then to the
pair (psi(t+k), that value)). Release is nested the same way with the
disjunction. Every other And, Or, Always and Eventually node is one m-ary
conjunction or disjunction over its entries, as the formula groups them.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ..operators import lower_max, lower_min


def exact_max(z, param=None, mask=None):
    return jnp.max(z if mask is None else jnp.where(mask, z, -jnp.inf), axis=-1)


def exact_min(z, param=None, mask=None):
    return jnp.min(z if mask is None else jnp.where(mask, z, jnp.inf), axis=-1)


def _count(z, mask):
    if mask is None:
        return jnp.asarray(z.shape[-1], z.dtype)
    return jnp.sum(jnp.broadcast_to(mask, z.shape), axis=-1).astype(z.dtype)


def lse_max(z, beta, mask=None):
    beta = jnp.asarray(beta, z.dtype)
    return jax.nn.logsumexp(beta * z, axis=-1, where=mask) / beta - jnp.log(_count(z, mask)) / beta


def lse_plain_max(z, beta, mask=None):
    """Plain log-sum-exp maximum (1/beta) log sum exp(beta z), without the -log(m)/beta shift;
    an upper smoothing, between the exact maximum and the exact maximum plus log(m)/beta."""
    beta = jnp.asarray(beta, z.dtype)
    return jax.nn.logsumexp(beta * z, axis=-1, where=mask) / beta


def lse_min(z, beta, mask=None):
    beta = jnp.asarray(beta, z.dtype)
    return -jax.nn.logsumexp(-beta * z, axis=-1, where=mask) / beta


# ------------------------------------------------------------------
# generalized mean robustness (E036)

def _valid(z, mask):
    return jnp.ones(z.shape, bool) if mask is None else jnp.broadcast_to(mask, z.shape)


def gm_conj(u, mask=None, p=0.0, q=1.0, beta=None):
    """The conjunction of eq. 12 over the last axis, masked entries not counted: the power mean
    robustness of order (p, q) (Definition 4; p any real, p = 0 the geometric mean; q >= 1), or,
    with beta (a number or one value per row), the exponential member of eq. 17."""
    valid = _valid(u, mask)
    m = jnp.sum(valid, -1).astype(u.dtype)
    lo = jnp.min(jnp.where(valid, u, jnp.inf), -1)
    pos = lo > 0
    neg = jnp.where(valid & (u < 0), u, 0.0)  # [u]_-, derivative 0 at u = 0
    v = -neg
    if beta is None:
        # first branch M_p(u) = exp(log M_p), log M_p = (logsumexp(p log u) - log m) / p, the mean of log u at p = 0
        L = jnp.log(jnp.where(valid & pos[..., None], u, 1.0))
        if p == 0:
            logM = jnp.sum(jnp.where(valid, L, 0.0), -1) / m
        else:
            logM = (jax.nn.logsumexp(p * L, axis=-1, where=valid) - jnp.log(m)) / p
        first = jnp.exp(logM)
        # second branch -M_q(v), v = -[u]_- >= 0: the mean at q = 1, else max(v) (mean((v / max v)^q))^(1/q)
        if q == 1:
            second = -jnp.sum(v, -1) / m
        else:
            c = jnp.max(v, -1)
            hit = c > 0
            cs = jnp.where(hit, c, 1.0)
            r = jnp.where(v > 0, jnp.exp(q * jnp.log(jnp.where(v > 0, v, 1.0) / cs[..., None])), 0.0)
            second = -jnp.where(hit, cs * jnp.where(hit, jnp.sum(r, -1) / m, 1.0) ** (1.0 / q), 0.0)
        return jnp.where(pos, first, second)
    beta = jnp.asarray(beta, u.dtype)
    bb = beta[..., None]
    lo_s = jnp.where(pos, lo, 0.0)
    first = lo_s - (jax.nn.logsumexp(-bb * (u - lo_s[..., None]), axis=-1, where=valid) - jnp.log(m)) / beta
    c = jnp.max(v, -1)
    small = beta * c <= 1
    vs = jnp.where(small[..., None], v, 0.0)
    near = jnp.log1p(jnp.sum(jnp.where(valid, jnp.expm1(bb * vs), 0.0), -1) / m) / beta
    cl = jnp.where(small, 0.0, c)
    far = cl + (jax.nn.logsumexp(bb * (v - cl[..., None]), axis=-1, where=valid) - jnp.log(m)) / beta
    return jnp.where(pos, first, -jnp.where(small, near, far))


def gm_power(p, q):
    """(max_reduce, min_reduce) of the power mean robustness of order (p, q); the parameter is ignored."""
    def conj(z, param=None, mask=None):
        return gm_conj(jnp.asarray(z), mask, p, q)

    def disj(z, param=None, mask=None):
        return -gm_conj(-jnp.asarray(z), mask, p, q)
    conj.nested_until = disj.nested_until = True
    conj.order = disj.order = (p, q)
    return disj, conj


gm_pm01_max, gm_pm01_min = gm_power(0.0, 1.0)  # order (0, 1), as the paper reports
gm_pm10_max, gm_pm10_min = gm_power(-10.0, 10.0)  # order (-10, 10), close to the exact min and max


def gm_exp_min(z, beta, mask=None):
    """Conjunction of the exponential member of eq. 17, c(x) = -exp(-beta x), g(x) = exp(beta x)."""
    return gm_conj(jnp.asarray(z), mask, beta=beta)


def gm_exp_max(z, beta, mask=None):
    """Disjunction of the exponential member, -conj(-z) (De Morgan)."""
    return -gm_conj(-jnp.asarray(z), mask, beta=beta)


gm_exp_min.nested_until = gm_exp_max.nested_until = True


REDUCTIONS = {"exact": (exact_max, exact_min), "lse": (lse_max, lse_min), "sparsemax": (lower_max, lower_min),
              "lse_plain": (lse_plain_max, lse_min), "gm_pm01": (gm_pm01_max, gm_pm01_min), "gm_pm10": (gm_pm10_max, gm_pm10_min),
              "gm_exp": (gm_exp_max, gm_exp_min)}


def reductions(semantics):
    """(max_reduce, min_reduce) for a built-in name or a user pair."""
    if isinstance(semantics, str):
        if semantics not in REDUCTIONS:
            raise ValueError("unknown semantics " + repr(semantics))
        return REDUCTIONS[semantics]
    max_reduce, min_reduce = semantics
    return max_reduce, min_reduce


def _mask(step):
    return None if step.full else np.arange(step.index.shape[1])[None, :] < step.count[:, None]


def evaluate(program, scores, semantics="exact", param=None):
    """All step outputs, in program order; the last one is the robustness trace.

    Only "exact" accepts programs with empty windows (clip boundary); every
    other semantics needs each reduction to have at least one valid entry.
    """
    max_reduce, min_reduce = reductions(semantics)
    if semantics != "exact" and any(s.kind != "atom" and np.any(s.count == 0) for s in program.steps):
        raise ValueError("this semantics needs nonempty windows; compile with boundary='strict'")
    scores = jnp.asarray(scores)
    if scores.shape[-2] != program.T or scores.shape[-1] < program.n_predicates:
        raise ValueError("scores must have shape (..., T, P) with T = " + str(program.T))
    out = []
    for step in program.steps:
        if step.kind == "atom":
            out.append(step.sign * scores[..., step.atom])
            continue
        src = out[step.sources[0]] if len(step.sources) == 1 else jnp.concatenate([out[s] for s in step.sources], -1)
        reduce = max_reduce if step.kind == "max" else min_reduce
        z, mask = src[..., step.index], _mask(step)
        if getattr(reduce, "nested_until", False) and step.label.endswith(".inner"):
            # eq. 15 of Mehdipour et al.: the prefix phi(t..t+k) first, then the pair (psi(t+k), prefix)
            prefix = reduce(z[..., 1:], param, None if mask is None else mask[:, 1:])
            out.append(reduce(jnp.stack([z[..., 0], prefix], -1), param, None))
            continue
        out.append(reduce(z, param, mask))
    return out


def robustness(program, scores, semantics="exact", param=None):
    """Robustness trace of the formula: entry t is its value at time t."""
    return evaluate(program, scores, semantics, param)[-1]


def local_error(m, semantics, param=None):
    """Largest gap between an exact extremum of m values and its built-in lower reduction."""
    m = np.asarray(m, dtype=np.float64)
    if semantics == "exact":
        return np.zeros_like(m)
    if semantics == "lse":
        return np.log(np.maximum(m, 1)) / float(param)
    if semantics == "sparsemax":
        return float(param) / 2 * (1 - 1 / np.maximum(m, 1))
    raise ValueError("no built-in error bound for " + repr(semantics))


def budget(program, semantics, param=None):
    """Per-entry budget B of the root, from B_v = b(m_v) + max over children B_c.

    semantics is a built-in name, or a function b(m, param) giving the local
    error bound of a reduction with m valid entries. For a lower reduction,
    0 <= exact - smoothed <= B holds entrywise in exact arithmetic (core draft,
    graph-level lower bound). Shared nodes are counted once.
    """
    error = semantics if callable(semantics) else (lambda m, param: local_error(m, semantics, param))
    out = []
    for step in program.steps:
        if step.kind == "atom":
            out.append(np.zeros(step.length))
            continue
        src = np.concatenate([out[s] for s in step.sources])
        valid = np.arange(step.index.shape[1])[None, :] < step.count[:, None]
        child = np.max(np.where(valid, src[step.index], 0.0), axis=-1)
        out.append(np.asarray(error(step.count, param), np.float64) + child)
    return out[-1]
