"""STL robustness of a compiled program in JAX. scores has shape (..., T, P): batch axes, then time, then predicates.
A semantics is a name in REDUCTIONS or a pair (max_reduce, min_reduce) of functions reduce(z, param, mask)."""
import jax
import jax.numpy as jnp
import numpy as np

from .operators import lower_max, lower_min


def exact_max(z, param=None, mask=None):
    return jnp.max(z if mask is None else jnp.where(mask, z, -jnp.inf), axis=-1)


def exact_min(z, param=None, mask=None):
    return jnp.min(z if mask is None else jnp.where(mask, z, jnp.inf), axis=-1)


def _count(z, mask):
    if mask is None:
        return jnp.asarray(z.shape[-1], z.dtype)
    return jnp.sum(jnp.broadcast_to(mask, z.shape), axis=-1).astype(z.dtype)


def lse_max(z, beta, mask=None):
    """Sound log-sum-exp maximum (1/beta) log sum exp(beta z) - log(m)/beta over the m valid entries (Eq. (14))."""
    beta = jnp.asarray(beta, z.dtype)
    return jax.nn.logsumexp(beta * z, axis=-1, where=mask) / beta - jnp.log(_count(z, mask)) / beta


def lse_plain_max(z, beta, mask=None):
    """Plain log-sum-exp maximum (1/beta) log sum exp(beta z) (Eq. (12))."""
    beta = jnp.asarray(beta, z.dtype)
    return jax.nn.logsumexp(beta * z, axis=-1, where=mask) / beta


def lse_min(z, beta, mask=None):
    """Log-sum-exp minimum -(1/beta) log sum exp(-beta z)."""
    beta = jnp.asarray(beta, z.dtype)
    return -jax.nn.logsumexp(-beta * z, axis=-1, where=mask) / beta


# ------------------------------------------------------------------
# generalized-mean robustness

def _valid(z, mask):
    return jnp.ones(z.shape, bool) if mask is None else jnp.broadcast_to(mask, z.shape)


def gm_conj(u, mask=None, p=0.0, q=1.0, beta=None):
    """Conjunction of the generalized-mean robustness over the m valid entries (Appendix I-A of the paper): M_p(u) if
    every entry is positive, else -M_q(-min(u, 0)), M_p the power mean; with beta, the exponential member (gm_exp_min)."""
    valid = _valid(u, mask)
    m = jnp.sum(valid, -1).astype(u.dtype)
    lo = jnp.min(jnp.where(valid, u, jnp.inf), -1)
    pos = lo > 0
    neg = jnp.where(valid & (u < 0), u, 0.0)  # [u]_-, derivative 0 at u = 0
    v = -neg
    if beta is None:
        L = jnp.log(jnp.where(valid & pos[..., None], u, 1.0))
        if p == 0:
            logM = jnp.sum(jnp.where(valid, L, 0.0), -1) / m
        else:
            logM = (jax.nn.logsumexp(p * L, axis=-1, where=valid) - jnp.log(m)) / p
        first = jnp.exp(logM)
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
    """(max_reduce, min_reduce) of the generalized-mean robustness with power means of order (p, q); param is ignored."""
    def conj(z, param=None, mask=None):
        return gm_conj(jnp.asarray(z), mask, p, q)

    def disj(z, param=None, mask=None):
        return -gm_conj(-jnp.asarray(z), mask, p, q)
    conj.nested_until = disj.nested_until = True
    conj.order = disj.order = (p, q)
    return disj, conj


gm_pm01_max, gm_pm01_min = gm_power(0.0, 1.0)
gm_pm10_max, gm_pm10_min = gm_power(-10.0, 10.0)


def gm_exp_min(z, beta, mask=None):
    """Conjunction of the exponential member of the generalized-mean robustness: -(1/beta) log mean exp(-beta z) if
    every valid entry is positive, else -(1/beta) log mean exp(-beta min(z, 0))."""
    return gm_conj(jnp.asarray(z), mask, beta=beta)


def gm_exp_max(z, beta, mask=None):
    """Disjunction of the exponential member, -gm_exp_min(-z)."""
    return -gm_conj(-jnp.asarray(z), mask, beta=beta)


gm_exp_min.nested_until = gm_exp_max.nested_until = True


REDUCTIONS = {"exact": (exact_max, exact_min), "lse": (lse_max, lse_min), "sparsemax": (lower_max, lower_min),
              "lse_plain": (lse_plain_max, lse_min), "gm_pm01": (gm_pm01_max, gm_pm01_min), "gm_pm10": (gm_pm10_max, gm_pm10_min),
              "gm_exp": (gm_exp_max, gm_exp_min)}


def reductions(semantics):
    """(max_reduce, min_reduce) for a built-in name or a pair of functions reduce(z, param, mask) over the last axis,
    mask None or a boolean array that marks the valid entries."""
    if isinstance(semantics, str):
        if semantics not in REDUCTIONS:
            raise ValueError("unknown semantics " + repr(semantics))
        return REDUCTIONS[semantics]
    max_reduce, min_reduce = semantics
    return max_reduce, min_reduce


def _mask(step):
    return None if step.full else np.arange(step.index.shape[1])[None, :] < step.count[:, None]


def evaluate(program, scores, semantics="exact", param=None):
    """Outputs of every step, in program order; the last one is the robustness trace.
    Only "exact" accepts a program with empty windows (boundary "clip")."""
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
            # row right(t+k), left(t), ..., left(t+k): reduce left(t), ..., left(t+k), then the pair with right(t+k)
            prefix = reduce(z[..., 1:], param, None if mask is None else mask[:, 1:])
            out.append(reduce(jnp.stack([z[..., 0], prefix], -1), param, None))
            continue
        out.append(reduce(z, param, mask))
    return out


def robustness(program, scores, semantics="exact", param=None):
    """Robustness trace of the formula: entry t is its value at time t."""
    return evaluate(program, scores, semantics, param)[-1]


def local_error(m, semantics, param=None):
    """Error band of one reduction over m entries: 0 for "exact", log(m)/beta for "lse" and
    gamma/2 (1 - 1/m) for "sparsemax" (Proposition 1)."""
    m = np.asarray(m, dtype=np.float64)
    if semantics == "exact":
        return np.zeros_like(m)
    if semantics == "lse":
        return np.log(np.maximum(m, 1)) / float(param)
    if semantics == "sparsemax":
        return float(param) / 2 * (1 - 1 / np.maximum(m, 1))
    raise ValueError("no built-in error bound for " + repr(semantics))


def budget(program, semantics, param=None):
    """Budget of every root entry from B_v = b(m_v) + max over children B_c with B = 0 at the atoms (Eq. (9)).
    semantics is a built-in name or a function b(m, param), the band of a reduction with m valid entries."""
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


def read(program, values):
    """The values of a pruned program's reads, concatenated in read order along the last axis.
    values is the list evaluate() returns."""
    if program.outputs is None:
        raise ValueError("read() needs a program compiled with reads")
    return jnp.concatenate([values[s][..., rows] for s, rows in program.outputs], -1)
