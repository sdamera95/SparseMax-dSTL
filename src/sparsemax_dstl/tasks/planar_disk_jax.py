"""The planar unicycle with disk regions in JAX: the predicate values, the matched smoothings with the sound
log-sum-exp, and the values of the specification and of its conjuncts."""
import jax.numpy as jnp

from ..jax.evaluator import evaluate, lse_max, lse_min, read
from ..stl import compile_formula
from . import planar_jax as P0
from .planar_disk import specification


def scores(xy, regions):
    """Leaf scores (..., T, 8) from positions (..., T, 2) and regions (4, 3)."""
    c = jnp.asarray(regions[:, :2])
    r = jnp.asarray(regions[:, 2])
    d = jnp.sqrt(jnp.sum((xy[..., None, :] - c) ** 2, -1))  # (..., T, 4)
    x, y = xy[..., 0:1], xy[..., 1:2]
    return jnp.concatenate([r - d, x, 10.0 - x, y, 10.0 - y], -1)


def matched(name, eps):
    """(max_reduce, min_reduce): planar_jax.matched, and for 'lse' the sound log-sum-exp with
    beta = log(m) / eps at a node of m valid entries (Equation (15) of the paper)."""
    if name != "lse":
        return P0.matched(name, eps)

    def mx(z, param=None, mask=None):
        return P0._per_row(lse_max, z, jnp.log(P0._count(z, mask)) / eps, mask)

    def mn(z, param=None, mask=None):
        return P0._per_row(lse_min, z, jnp.log(P0._count(z, mask)) / eps, mask)
    return mx, mn


def values(S, timing, sem):
    """(specification at t = 0, its four conjuncts at t = 0) under the reduction pair sem; S (T, 8)."""
    spec, conj, _ = specification(timing["a1"], timing["b1"], timing["a2"], timing["b2"], timing["T"])
    prog = compile_formula(spec, timing["T"], reads=[(spec, [0])] + [(c, [0]) for c in conj])
    out = read(prog, evaluate(prog, S, sem))
    return out[..., 0], out[..., 1:]
