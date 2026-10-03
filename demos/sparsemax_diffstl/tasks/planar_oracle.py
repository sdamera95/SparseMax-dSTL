"""Independent NumPy evaluation of the E049 smoothings from their definitions (no JAX; it shares
only the formula classes of sparsemax_diffstl.stl.formula), used to check the JAX values.

Reductions over the rows of z (R, W) with a validity mask, at worst-case error eps per node
($m$ the number of valid entries of the row; a row with one entry is exact):

- plain log-sum-exp: $\\max \\approx \\frac1\\beta \\log \\sum_i e^{\\beta z_i}$,
  $\\min \\approx -\\frac1\\beta \\log \\sum_i e^{-\\beta z_i}$, $\\beta = \\log m / \\varepsilon$;
- sparsemax: $Q(z) = \\max_{p \\in \\Delta} \\langle p, z\\rangle - \\frac\\gamma2 \\lVert p\\rVert^2$, the lower
  maximum $Q(z) + \\frac{\\gamma}{2m}$ and the lower minimum $-Q(-z) - \\frac\\gamma2$,
  $\\gamma = 2\\varepsilon / (1 - 1/m)$; the maximizer $p$ is the Euclidean projection of
  $z / \\gamma$ onto the simplex, computed by sorting;
- generalized-mean robustness of order $(p, q)$ (Mehdipour, Vasile, Belta, eq. 12 and 15):
  conjunction $M_p(x)$ if $\\min x > 0$, else $-M_q(-[x]_-)$ over all $d$ entries, with
  $M_r(x) = (\\frac1d \\sum x_i^r)^{1/r}$ and $M_0$ the geometric mean; disjunction
  $-\\mathrm{conj}(-x)$; the until nests the prefix conjunction inside a two-entry conjunction.

The formula is walked recursively over its structure (negation carried down as a flag); every
node is evaluated only at the sample times its parent reads, with whole-array NumPy.
"""
import numpy as np

from ..stl.formula import Always, And, Atom, Eventually, Not, Or, Until


def _m(valid):
    return np.maximum(valid.sum(1), 2).astype(float)  # one entry: exact for any parameter


def _lse(a, valid):
    a = np.where(valid, a, -np.inf)
    c = a.max(1)
    return c + np.log(np.sum(np.where(valid, np.exp(a - c[:, None]), 0.0), 1))


def project(w, valid):
    """Projection of each row of w onto the simplex over its valid entries; returns p (R, W)."""
    w = np.where(valid, w, -np.inf)
    u = -np.sort(-w, 1)
    fin = np.isfinite(u)
    cs = np.cumsum(np.where(fin, u, 0.0), 1)
    r = np.arange(1, w.shape[1] + 1)
    k = np.sum(fin & (1 + r * np.where(fin, u, 0.0) > cs), 1)
    tau = (cs[np.arange(len(k)), k - 1] - 1) / k
    return np.where(valid, np.maximum(w - tau[:, None], 0.0), 0.0)


def qmax(z, gamma, valid):
    """(Q(z), p) per row."""
    p = project(z / gamma[:, None], valid)
    zz = np.where(valid, z, 0.0)
    return np.sum(p * zz, 1) - gamma / 2 * np.sum(p * p, 1), p


def _power(x, r, valid):
    d = valid.sum(1)
    if r == 0:
        return np.exp(np.sum(np.where(valid, np.log(np.where(valid, x, 1.0)), 0.0), 1) / d)
    return (np.sum(np.where(valid, np.where(valid, x, 1.0) ** r, 0.0), 1) / d) ** (1 / r)


def gm_conj(x, valid, p, q):
    lo = np.min(np.where(valid, x, np.inf), 1)
    pos = lo > 0
    first = _power(np.where(valid & pos[:, None], x, 1.0), p, valid)
    v = np.where(valid, -np.minimum(x, 0.0), 0.0)
    second = -_power(v, q, valid) if q != 1 else -np.sum(v, 1) / valid.sum(1)
    return np.where(pos, first, second)


def reduce(z, valid, kind, sem, eps):
    """Reduction of each row of z over its valid entries; kind is 'max' or 'min'."""
    if sem == "exact":
        return np.max(np.where(valid, z, -np.inf), 1) if kind == "max" else np.min(np.where(valid, z, np.inf), 1)
    if sem == "lse_plain":
        beta = np.log(_m(valid)) / eps
        return _lse(beta[:, None] * z, valid) / beta if kind == "max" else -_lse(-beta[:, None] * z, valid) / beta
    if sem == "sparsemax":
        gamma = 2 * eps / (1 - 1 / _m(valid))
        if kind == "max":
            return qmax(z, gamma, valid)[0] + gamma / (2 * valid.sum(1))
        return -qmax(-z, gamma, valid)[0] - gamma / 2
    p, q = {"gm01": (0.0, 1.0), "gm10": (-10.0, 10.0)}[sem]
    return gm_conj(z, valid, p, q) if kind == "min" else -gm_conj(-z, valid, p, q)


def _gather(f, S, ts, neg, sem, eps, offsets):
    idx = ts[:, None] + offsets[None, :]
    need = np.unique(idx)
    return ev(f, S, need, neg, sem, eps)[np.searchsorted(need, idx)]


def ev(f, S, ts, neg=False, sem="exact", eps=0.2):
    """Values of formula f at sample times ts (int array) on leaf scores S (T, P)."""
    ts = np.asarray(ts)
    if isinstance(f, Atom):
        return S[ts, f.index] * (-1.0 if f.negated != neg else 1.0)
    if isinstance(f, Not):
        return ev(f.child, S, ts, not neg, sem, eps)
    if isinstance(f, (And, Or)):
        z = np.stack([ev(c, S, ts, neg, sem, eps) for c in f.children], 1)  # formula structure
        kind = "min" if isinstance(f, And) != neg else "max"
        return reduce(z, np.ones(z.shape, bool), kind, sem, eps)
    if isinstance(f, (Always, Eventually)):
        a, b = f.interval
        z = _gather(f.child, S, ts, neg, sem, eps, np.arange(a, b + 1))
        kind = "min" if isinstance(f, Always) != neg else "max"
        return reduce(z, np.ones(z.shape, bool), kind, sem, eps)
    if isinstance(f, Until) and not neg:
        return until(f, S, ts, sem, eps)
    raise TypeError("the oracle covers the NNF of this example only: " + repr(f))


def until(f, S, ts, sem, eps):
    a, b = f.interval
    ks = np.arange(a, b + 1)
    psi = _gather(f.right, S, ts, False, sem, eps, ks)  # (n_t, n_w)
    phi = _gather(f.left, S, ts, False, sem, eps, np.arange(b + 1))  # (n_t, b + 1)
    n_t, n_w = psi.shape
    pre = np.broadcast_to(phi[:, None, :], (n_t, n_w, b + 1)).reshape(-1, b + 1)
    pv = np.broadcast_to(np.arange(b + 1)[None, None, :] <= ks[None, :, None], (n_t, n_w, b + 1)).reshape(-1, b + 1)
    ps = psi.reshape(-1)
    if sem in ("gm01", "gm10"):
        prefix = reduce(pre, pv, "min", sem, eps)
        inner = reduce(np.stack([ps, prefix], 1), np.ones((len(ps), 2), bool), "min", sem, eps)
    else:
        row = np.concatenate([ps[:, None], pre], 1)
        inner = reduce(row, np.concatenate([np.ones((len(ps), 1), bool), pv], 1), "min", sem, eps)
    inner = inner.reshape(n_t, n_w)
    return reduce(inner, np.ones(inner.shape, bool), "max", sem, eps)


def until_rows(phi, psi, a, b):
    """Rows of the flat until at t = 0: entries psi(k), phi(0..k) for k in [a, b], and validity."""
    ks = np.arange(a, b + 1)
    row = np.concatenate([psi[ks][:, None], np.broadcast_to(phi[:b + 1], (len(ks), b + 1))], 1)
    valid = np.concatenate([np.ones((len(ks), 1), bool), np.arange(b + 1)[None, :] <= ks[:, None]], 1)
    return row, valid


def until_grad_closed(phi, psi, a, b, sem, eps):
    """Derivative of the flat until at t = 0 with respect to phi (T,), from the closed forms of the
    reductions' derivatives: softmax and softmin weights for the plain log-sum-exp, the sparsemax
    projections for sparsemax (the outer weights times the inner weights, summed over witnesses)."""
    row, valid = until_rows(phi, psi, a, b)
    if sem == "lse_plain":
        beta = np.log(_m(valid)) / eps
        lq = np.where(valid, -beta[:, None] * row, -np.inf)
        qin = np.exp(lq - _lse(-beta[:, None] * row, valid)[:, None])
        inner = -_lse(-beta[:, None] * row, valid) / beta
        bo = np.log(max(len(inner), 2)) / eps
        u = np.exp(bo * inner - _lse((bo * inner)[None], np.ones((1, len(inner)), bool))[0])
    elif sem == "sparsemax":
        gamma = 2 * eps / (1 - 1 / _m(valid))
        _, qin = qmax(-row, gamma, valid)
        inner = -qmax(-row, gamma, valid)[0] - gamma / 2
        go = 2 * eps / (1 - 1 / max(len(inner), 2))
        _, u = qmax(inner[None], np.array([go]), np.ones((1, len(inner)), bool))
        u = u[0]
    else:
        raise ValueError(sem)
    g = np.zeros(len(phi))
    w = (u[:, None] * qin)[:, 1:]  # (n_w, b + 1), weight of phi(s) in row k
    g[:b + 1] = w.sum(0)
    return g
