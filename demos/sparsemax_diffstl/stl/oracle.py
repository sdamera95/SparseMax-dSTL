"""Brute-force test oracles, analytic references and random formulas.

This module is for tests and gates only. rho, trace, smooth_rho, the rational
sparsemax functions and path_budget are labeled brute-force oracles: they loop
over times and window entries in plain Python on purpose and never build a
program or call JAX. rho reads a formula as written, with negation,
implication and Release taken straight from their definitions and no negation
normal form. analytic_derivatives is a vectorized analytic reference that
reuses a compiled program's time expansion but no automatic differentiation.
"""
from fractions import Fraction

import numpy as np

from .formula import Always, And, Atom, Eventually, Implies, Not, Or, Release, Until, horizon


def _window(t, a, b, T, boundary):
    if boundary == "strict":
        return range(t + a, t + b + 1)
    return range(t + a, min(t + b, T - 1) + 1)


def rho(f, z, t, boundary="strict"):
    """Exact robustness of f at time t for scores z of shape (T, P)."""
    T = z.shape[0]
    if isinstance(f, Atom):
        if not 0 <= t < T:
            raise IndexError("sample " + str(t) + " does not exist")
        v = float(z[t, f.index])
        return -v if f.negated else v
    if isinstance(f, Not):
        return -rho(f.child, z, t, boundary)
    if isinstance(f, Implies):
        return max(-rho(f.left, z, t, boundary), rho(f.right, z, t, boundary))
    if isinstance(f, And):
        return min(rho(c, z, t, boundary) for c in f.children)
    if isinstance(f, Or):
        return max(rho(c, z, t, boundary) for c in f.children)
    a, b = f.interval
    taus = _window(t, a, b, T, boundary)
    if isinstance(f, Always):
        return min((rho(f.child, z, tau, boundary) for tau in taus), default=np.inf)
    if isinstance(f, Eventually):
        return max((rho(f.child, z, tau, boundary) for tau in taus), default=-np.inf)
    if isinstance(f, Until):
        best = -np.inf
        for tau in taus:
            v = rho(f.right, z, tau, boundary)
            for s in range(t, tau + 1):
                v = min(v, rho(f.left, z, s, boundary))
            best = max(best, v)
        return best
    if isinstance(f, Release):
        best = np.inf
        for tau in taus:
            v = rho(f.right, z, tau, boundary)
            for s in range(t, tau + 1):
                v = max(v, rho(f.left, z, s, boundary))
            best = min(best, v)
        return best
    raise TypeError("not an STL formula: " + repr(f))


def trace(f, z, boundary="strict"):
    """Robustness at every time the boundary mode defines."""
    T = z.shape[0]
    n = T - horizon(f) if boundary == "strict" else T
    return np.array([rho(f, z, t, boundary) for t in range(n)])


def _local(Z, valid, kind, semantics, beta):
    """Closed-form value, weights and Hessian of one reduction row by row."""
    sign = 1.0 if kind == "max" else -1.0
    S = np.where(valid, sign * Z, -np.inf)
    top = np.max(S, axis=-1, keepdims=True)
    if semantics == "exact":
        hit = valid & (S == top)
        w = hit / np.sum(hit, axis=-1, keepdims=True)
        return sign * top[:, 0], w, np.zeros(Z.shape + Z.shape[-1:])
    if semantics == "sparsemax":
        return _local_sparsemax(S, valid, kind, beta)
    e = np.where(valid, np.exp(beta * (S - top)), 0.0)
    total = np.sum(e, axis=-1)
    w = e / total[:, None]
    value = top[:, 0] + np.log(total) / beta
    if kind == "max":
        value = value - np.log(np.sum(valid, axis=-1)) / beta
    else:
        value = -value
    H = sign * beta * (w[:, :, None] * np.eye(Z.shape[-1]) - w[:, :, None] * w[:, None, :])
    return value, w, H


def _local_sparsemax(S, valid, kind, gamma):
    """Rows of y = +-z (masked entries -inf): threshold, weights and Hessian in closed form.

    p_i = (y_i - theta)_+ / gamma with sum_i (y_i - theta)_+ = gamma; the support
    size k is the largest rank j with gamma + j u_j > u_1 + ... + u_j for the
    sorted u. M = p.y - gamma/2 |p|^2; the lower maximum adds gamma/(2m), the
    lower minimum is -M(-z) - gamma/2. Hessian +-(Diag(s) - s s^T/k)/gamma on the
    support indicator s, for a maximum and a minimum respectively.
    """
    sign = 1.0 if kind == "max" else -1.0
    u = -np.sort(-S, axis=-1)
    total = np.cumsum(np.where(np.isfinite(u), u, 0.0), axis=-1)
    rank = np.arange(1, S.shape[-1] + 1)
    k = np.sum(np.isfinite(u) & (gamma + rank * u > total), axis=-1)
    theta = (np.take_along_axis(total, (k - 1)[:, None], axis=-1)[:, 0] - gamma) / k
    w = np.where(valid, np.maximum(np.where(valid, S, 0.0) - theta[:, None], 0.0), 0.0) / gamma
    M = np.sum(w * np.where(valid, S, 0.0), axis=-1) - gamma / 2 * np.sum(w * w, axis=-1)
    value = M + gamma / (2 * np.sum(valid, axis=-1)) if kind == "max" else -M - gamma / 2
    on = (w > 0).astype(np.float64)
    H = sign / gamma * (on[:, :, None] * np.eye(S.shape[-1]) - on[:, :, None] * on[:, None, :] / k[:, None, None])
    return value, w, H


def analytic_derivatives(program, z, semantics="exact", beta=None):
    """Root values, gradients and Hessians with respect to the flattened scores.

    An analytic reference, not automatic differentiation: every reduction
    applies its closed-form weights w and Hessian (zero for exact; for lse
    beta (Diag(w) - w w^T) at a maximum and its negative at a minimum; for
    sparsemax, with beta read as gamma, see _local_sparsemax), and the
    chain rule composes them, value by value, in float64 NumPy. z has shape
    (T, P); the outputs have shapes (L,), (L, T*P) and (L, T*P, T*P). Exact
    semantics split the weight equally among tied entries, as JAX does.
    """
    z = np.asarray(z, dtype=np.float64)
    T, P = z.shape
    n = T * P
    vals, grads, hess = [], [], []
    for step in program.steps:
        if step.kind == "atom":
            g = np.zeros((T, n))
            g[np.arange(T), np.arange(T) * P + step.atom] = step.sign
            vals.append(step.sign * z[:, step.atom])
            grads.append(g)
            hess.append(np.zeros((T, n, n)))
            continue
        V = np.concatenate([vals[s] for s in step.sources])
        G = np.concatenate([grads[s] for s in step.sources])[step.index]
        H = np.concatenate([hess[s] for s in step.sources])[step.index]
        valid = np.arange(step.index.shape[1])[None, :] < step.count[:, None]
        value, w, Hloc = _local(V[step.index], valid, step.kind, semantics, beta)
        vals.append(value)
        grads.append(np.einsum("rm,rmn->rn", w, G))
        hess.append(np.einsum("rab,ran,rbk->rnk", Hloc, G, G) + np.einsum("rm,rmnk->rnk", w, H))
    return vals[-1], grads[-1], hess[-1]


def smooth_rho(f, z, t, max_reduce, min_reduce):
    """Value at t of an NNF formula under reductions on Python lists, strict boundary.

    A brute-force oracle for smooth semantics that never builds a program: it
    recurses on the formula, with Until and Release grouped as in program.py
    (an inner reduction over psi(t+k), phi(t), ..., phi(t+k) for each witness
    offset k, then an outer reduction over k). With Fraction scores and the
    rational reductions below it is exact.
    """
    if isinstance(f, Atom):
        v = z[t][f.index]
        return -v if f.negated else v
    if isinstance(f, (And, Or)):
        vals = [smooth_rho(c, z, t, max_reduce, min_reduce) for c in f.children]
        return min_reduce(vals) if isinstance(f, And) else max_reduce(vals)
    a, b = f.interval
    if isinstance(f, (Always, Eventually)):
        vals = [smooth_rho(f.child, z, tau, max_reduce, min_reduce) for tau in range(t + a, t + b + 1)]
        return min_reduce(vals) if isinstance(f, Always) else max_reduce(vals)
    if isinstance(f, (Until, Release)):
        inner, outer = (min_reduce, max_reduce) if isinstance(f, Until) else (max_reduce, min_reduce)
        rows = []
        for k in range(a, b + 1):
            row = [smooth_rho(f.right, z, t + k, max_reduce, min_reduce)]
            row += [smooth_rho(f.left, z, t + s, max_reduce, min_reduce) for s in range(k + 1)]
            rows.append(inner(row))
        return outer(rows)
    raise TypeError("smooth_rho needs a formula in negation normal form: " + repr(f))


def rational_sparsemax(z, gamma):
    """Exact M_gamma(z), weights p and threshold theta for a list of Fractions."""
    u = sorted(z, reverse=True)
    k = max(j for j in range(1, len(u) + 1) if gamma + j * u[j - 1] > sum(u[:j]))
    theta = (sum(u[:k]) - gamma) / k
    p = [max(v - theta, Fraction(0)) / gamma for v in z]
    return sum(pi * v for pi, v in zip(p, z)) - gamma / 2 * sum(pi * pi for pi in p), p, theta


def rational_lower_max(z, gamma):
    return rational_sparsemax(z, gamma)[0] + gamma / (2 * len(z))


def rational_lower_min(z, gamma):
    return -rational_sparsemax([-v for v in z], gamma)[0] - gamma / 2


def path_budget(f, t, local):
    """Largest sum of local errors local(m) along any root-to-leaf path of the
    time-expanded graph of an NNF formula at time t, by enumeration."""
    if isinstance(f, Atom):
        return 0.0
    if isinstance(f, (And, Or)):
        return local(len(f.children)) + max(path_budget(c, t, local) for c in f.children)
    a, b = f.interval
    if isinstance(f, (Always, Eventually)):
        return local(b - a + 1) + max(path_budget(f.child, tau, local) for tau in range(t + a, t + b + 1))
    best = 0.0
    for k in range(a, b + 1):
        below = [path_budget(f.right, t + k, local)] + [path_budget(f.left, t + s, local) for s in range(k + 1)]
        best = max(best, local(k + 2) + max(below))
    return local(b - a + 1) + best


def random_formula(rng, n_pred, depth, max_bound=3, ops=("not", "implies", "and", "or", "G", "F", "U", "R"),
                   zero_start=False):
    """Random formula of the given nesting depth over n_pred predicates."""
    if depth == 0:
        return Atom(int(rng.integers(n_pred)), bool(rng.random() < 0.3))
    op = ops[int(rng.integers(len(ops)))]

    def sub():
        return random_formula(rng, n_pred, depth - 1, max_bound, ops, zero_start)

    def interval():
        b = int(rng.integers(max_bound + 1))
        return (0 if zero_start else int(rng.integers(b + 1)), b)

    if op == "not":
        return Not(sub())
    if op == "implies":
        return Implies(sub(), sub())
    if op in ("and", "or"):
        kids = [sub() for _ in range(int(rng.integers(1, 4)))]
        return And(*kids) if op == "and" else Or(*kids)
    if op == "G":
        return Always(interval(), sub())
    if op == "F":
        return Eventually(interval(), sub())
    if op == "U":
        return Until(interval(), sub(), sub())
    return Release(interval(), sub(), sub())
