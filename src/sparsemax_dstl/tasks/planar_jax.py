"""The planar unicycle in JAX: the rollout, the smoothings matched at one worst-case error per node, and the value of
an until on operand traces."""
import jax
import jax.numpy as jnp

from ..jax.evaluator import evaluate, exact_max, exact_min, gm_power, lse_min, lse_plain_max, read
from ..jax.operators import lower_max, lower_min
from ..stl import Atom, Until, compile_formula
from .planar import H


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
    return jnp.maximum(jnp.sum(valid, -1), 2).astype(z.dtype)  # at least 2: log(m) and 1 - 1/m are nonzero


def matched(name, eps):
    """(max_reduce, min_reduce) for name in 'exact', 'lse_plain', 'sparsemax', 'gm_pm01', 'gm_pm10'; at a node of m
    valid entries beta = log(m) / eps and gamma = 2 eps / (1 - 1/m) (Equation (15) of the paper)."""
    if name == "exact":
        return exact_max, exact_min
    if name == "gm_pm01":
        return gm_power(0.0, 1.0)
    if name == "gm_pm10":
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
