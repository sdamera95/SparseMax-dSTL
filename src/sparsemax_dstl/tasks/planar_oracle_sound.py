"""NumPy evaluation of the smoothed robustness under the sound log-sum-exp from its definition, without JAX."""
import numpy as np

from ..stl.formula import Always, And, Atom, Eventually, Not, Or, Until
from .planar_oracle import _lse


def reduce(z, valid, kind, eps):
    """Sound log-sum-exp reduction of each row of z (R, W) over its m valid entries with beta = log(m) / eps: the
    plain log-sum-exp minimum, and the plain maximum minus log(m) / beta; kind 'max' or 'min'."""
    m = valid.sum(1)
    beta = np.log(np.maximum(m, 2)) / eps
    if kind == "min":
        return -_lse(-beta[:, None] * z, valid) / beta
    return _lse(beta[:, None] * z, valid) / beta - np.log(m) / beta


def _gather(f, S, ts, neg, eps, offsets):
    idx = ts[:, None] + offsets[None, :]
    need = np.unique(idx)
    return ev(f, S, need, neg, eps)[np.searchsorted(need, idx)]


def until_rows(psi, phi, ks, b):
    """Flat until rows for every evaluation time and witness: psi (n_t, n_w), phi (n_t, b + 1)."""
    n_t, n_w = psi.shape
    pre = np.broadcast_to(phi[:, None, :], (n_t, n_w, b + 1)).reshape(-1, b + 1)
    pv = np.broadcast_to(np.arange(b + 1)[None, None, :] <= ks[None, :, None], (n_t, n_w, b + 1)).reshape(-1, b + 1)
    row = np.concatenate([psi.reshape(-1, 1), pre], 1)
    return row, np.concatenate([np.ones((n_t * n_w, 1), bool), pv], 1)


def ev(f, S, ts, neg=False, eps=0.1):
    """Values of the STL specification f under the sound log-sum-exp at sample times ts on leaf scores S (T, P)."""
    ts = np.asarray(ts)
    if isinstance(f, Atom):
        return S[ts, f.index] * (-1.0 if f.negated != neg else 1.0)
    if isinstance(f, Not):
        return ev(f.child, S, ts, not neg, eps)
    if isinstance(f, (And, Or)):
        z = np.stack([ev(c, S, ts, neg, eps) for c in f.children], 1)
        return reduce(z, np.ones(z.shape, bool), "min" if isinstance(f, And) != neg else "max", eps)
    if isinstance(f, (Always, Eventually)):
        a, b = f.interval
        z = _gather(f.child, S, ts, neg, eps, np.arange(a, b + 1))
        return reduce(z, np.ones(z.shape, bool), "min" if isinstance(f, Always) != neg else "max", eps)
    if isinstance(f, Until) and not neg:
        a, b = f.interval
        ks = np.arange(a, b + 1)
        psi = _gather(f.right, S, ts, False, eps, ks)
        phi = _gather(f.left, S, ts, False, eps, np.arange(b + 1))
        row, valid = until_rows(psi, phi, ks, b)
        inner = reduce(row, valid, "min", eps).reshape(psi.shape)
        return reduce(inner, np.ones(inner.shape, bool), "max", eps)
    raise TypeError("the oracle covers the NNF of this example only: " + repr(f))
