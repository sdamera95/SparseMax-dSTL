"""The planar unicycle with box regions in JAX: the predicate values, the rollout, the smoothings matched at one
worst-case error per node, and the value and the weights of the until."""
import jax
import jax.numpy as jnp
import numpy as np

from ..jax.evaluator import evaluate, exact_max, exact_min, gm_power, lse_min, lse_plain_max, read
from ..jax.operators import lower_max, lower_min
from ..stl import Atom, Not, Until, compile_formula
from .planar import H, REGIONS, box, specification


# ------------------------------------------------------------------
# predicates

def scores(xy):
    """Leaf-score array (..., T, 20) of the four linear predicates of every region."""
    x, y = xy[..., 0:1], xy[..., 1:2]
    lo = np.array([r[0] for r in REGIONS]), np.array([r[1] for r in REGIONS])
    yl = np.array([r[2] for r in REGIONS]), np.array([r[3] for r in REGIONS])
    s = jnp.stack([x - lo[0], lo[1] - x, y - yl[0], yl[1] - y], -1)  # (..., T, 5, 4)
    return s.reshape(s.shape[:-2] + (20,))


# ------------------------------------------------------------------
# dynamics

def rollout(z0, u):
    """States (T+1, 3) from z0 under inputs u (T, 2), unicycle with period H."""
    def step(z, ui):
        x, y, th = z
        z1 = jnp.stack([x + H * ui[0] * jnp.cos(th), y + H * ui[0] * jnp.sin(th), th + H * ui[1]])
        return z1, z1
    z0 = jnp.asarray(z0, jnp.result_type(float))
    _, zs = jax.lax.scan(step, z0, jnp.asarray(u, z0.dtype))
    return jnp.concatenate([z0[None], zs], 0)


# ------------------------------------------------------------------
# matched smoothings

def _per_row(fn, z, param, mask):
    """Apply fn(z_row, param_row, mask_row) to every row of the last two axes."""
    valid = jnp.broadcast_to(jnp.ones((), bool) if mask is None else mask, z.shape)
    zr, vr = z.reshape(-1, z.shape[-1]), valid.reshape(-1, z.shape[-1])
    pr = jnp.broadcast_to(param, z.shape[:-1]).reshape(-1)
    return jax.vmap(fn)(zr, pr, vr).reshape(z.shape[:-1])


def _count(z, mask):
    valid = jnp.broadcast_to(jnp.ones((), bool) if mask is None else mask, z.shape)
    return jnp.maximum(jnp.sum(valid, -1), 2).astype(z.dtype)  # one entry: exact for any parameter


def matched(name, eps):
    """(max_reduce, min_reduce) at worst-case error eps per node, for name in
    'exact', 'lse_plain', 'sparsemax', 'gm01', 'gm10'."""
    if name == "exact":
        return exact_max, exact_min
    if name == "gm01":
        return gm_power(0.0, 1.0)
    if name == "gm10":
        return gm_power(-10.0, 10.0)
    if name == "lse_plain":
        def mx(z, param=None, mask=None):
            return _per_row(lse_plain_max, z, jnp.log(_count(z, mask)) / eps, mask)

        def mn(z, param=None, mask=None):
            return _per_row(lse_min, z, jnp.log(_count(z, mask)) / eps, mask)
        return mx, mn
    if name == "sparsemax":
        def mx(z, param=None, mask=None):
            return _per_row(lower_max, z, 2 * eps / (1 - 1 / _count(z, mask)), mask)

        def mn(z, param=None, mask=None):
            return _per_row(lower_min, z, 2 * eps / (1 - 1 / _count(z, mask)), mask)
        return mx, mn
    raise ValueError("unknown smoothing " + repr(name))


def until_on_operands(phi, psi, a1, b1, sem):
    """Value at t = 0 of phi U_[a1, b1] psi for operand traces phi, psi (T,) under the pair sem."""
    T = phi.shape[-1]
    prog = compile_formula(Until((a1, b1), Atom(0), Atom(1)), T, reads=[(Until((a1, b1), Atom(0), Atom(1)), [0])])
    return read(prog, evaluate(prog, jnp.stack([phi, psi], -1), sem))[..., 0]


def operand_traces(S, sem):
    """Traces (T,) of not Red and Green under the pair sem, from leaf scores S (T, 20)."""
    T = S.shape[-2]
    p1 = compile_formula(Not(box(0)), T)
    p2 = compile_formula(box(1), T)
    return evaluate(p1, S, sem)[-1], evaluate(p2, S, sem)[-1]


def spec_values(S, timing, sem):
    """(specification at t = 0, until at t = 0) under the pair sem."""
    spec, until = specification(timing["a1"], timing["b1"], timing["a2"], timing["b2"], timing["T"])
    prog = compile_formula(spec, timing["T"], reads=[(spec, [0]), (until, [0])])
    out = read(prog, evaluate(prog, S, sem))
    return out[..., 0], out[..., 1]


def until_weight(S, timing, sem, viol):
    """Sum over the samples in viol (bool, T) of the derivative of the until value at t = 0 with
    respect to the value of not Red at that sample, by reverse-mode AD; also the until value."""
    phi, psi = operand_traces(S, sem)
    f = lambda ph: until_on_operands(ph, psi, timing["a1"], timing["b1"], sem)  # noqa: E731
    val, g = jax.value_and_grad(f)(phi)
    return val, jnp.sum(jnp.where(viol, g, 0.0)), g
