"""asdex's colorings of declared patterns as seed matrices and gathers, and dense automatic differentiation of a whole
program (every constraint's reverse product, every variable's forward-over-reverse product)."""
import warnings

import asdex
import jax
import jax.numpy as jnp
import numpy as np

from .patterns import positions

CHUNK = 32  # automatic-differentiation products per batch of jax.lax.map


def jacobian_plan(pattern, shape, mode, dtype):
    """asdex's coloring of a Jacobian pattern (rows, cols) in mode "rev" (rows) or "fwd" (columns): seeds S (colors, rows
    or cols), the gather (color, index) per entry and the number of colors."""
    sp = asdex.SparsityPattern.from_coo(pattern[0].astype(np.int32), pattern[1].astype(np.int32), shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asdex.DenseColoringWarning)
        col = asdex.jacobian_coloring_from_sparsity(sp, mode=mode)
    c = np.asarray(col.colors)
    nc = int(col.num_colors)
    S = jnp.asarray(c[None, :] == np.arange(nc)[:, None], dtype)
    if mode == "rev":
        return {"S": S, "gc": jnp.asarray(c[pattern[0]]), "ge": jnp.asarray(pattern[1]), "colors": nc}
    return {"S": S, "gc": jnp.asarray(c[pattern[1]]), "ge": jnp.asarray(pattern[0]), "colors": nc}


def hessian_plan(up, n, dtype):
    """asdex's star coloring of the symmetric pattern with upper triangle up: seeds S (colors, n) and per upper entry its
    (color, row) in the compressed matrix (asdex's decompression read as a gather)."""
    r = np.concatenate([up[0], up[1][up[0] != up[1]]])
    c = np.concatenate([up[1], up[0][up[0] != up[1]]])
    sp = asdex.SparsityPattern.from_coo(r.astype(np.int32), c.astype(np.int32), (n, n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asdex.DenseColoringWarning)
        ch = asdex.hessian_coloring_from_sparsity(sp, mode="fwd_over_rev", symmetric=True)
    nh = int(ch.num_colors)
    code = np.arange(nh * n, dtype=np.float64).reshape(nh, n)
    flat = np.asarray(asdex.decompress_data(jnp.asarray(code), ch))
    if not np.array_equal(flat, np.round(flat)):
        raise ValueError("asdex's star decompression is not a gather")
    s = ch.sparsity
    sr, sc = np.asarray(s.rows), np.asarray(s.cols)
    keep = sr <= sc
    pos = positions(up, sr[keep], sc[keep], n)
    src = np.zeros(up[0].size, np.int64)
    src[pos] = flat[keep].astype(np.int64)
    hc = np.asarray(ch.colors)
    return {"S": jnp.asarray(hc[None, :] == np.arange(nh)[:, None], dtype), "gc": jnp.asarray(src // n), "ge": jnp.asarray(src % n), "colors": nh}


def dense_jacobian(program, semantics="sparsemax", chunk=CHUNK):
    """w -> the whole constraint Jacobian (m, n) of an mpc.program.PeriodProgram from m reverse products."""
    m = program.m

    def jac(w):
        _, vjp = jax.vjp(lambda v: program.constraints(v, semantics), w)
        return jax.lax.map(lambda i: vjp(jax.nn.one_hot(i, m, dtype=w.dtype))[0], jnp.arange(m), batch_size=chunk)
    return jax.jit(jac)


def dense_hessian(program, semantics="sparsemax", chunk=CHUNK):
    """(w, sigma, lam) -> the whole Lagrangian Hessian (n, n) from n forward-over-reverse products."""
    n = program.n

    def hess(w, s, lam):
        _, hvp = jax.linearize(jax.grad(lambda v: program.lagrangian(v, s, lam, semantics)), w)
        return jax.lax.map(lambda i: hvp(jax.nn.one_hot(i, n, dtype=w.dtype)), jnp.arange(n), batch_size=chunk)
    return jax.jit(hess)


def dense_derivatives(pb, semantics="sparsemax", chunk=CHUNK):
    """(dense_jacobian, dense_hessian) of a PeriodProgram pb, compiled."""
    return dense_jacobian(pb, semantics, chunk), dense_hessian(pb, semantics, chunk)
