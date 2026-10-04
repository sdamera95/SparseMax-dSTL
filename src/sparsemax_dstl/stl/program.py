"""Expansion of a formula over T samples into a list of steps. Boundary "strict" keeps a node only at the times
where every sample it reads exists; "clip" keeps every time and drops the window entries past the last sample."""
from dataclasses import dataclass

import numpy as np

from .formula import Always, And, Atom, Eventually, Or, Release, Until, atoms, horizon, to_nnf


@dataclass(frozen=True, eq=False)
class Step:
    """An atom step is sign * scores[..., atom]; a reduction step is out[r] = max or min over k < count[r] of
    src[index[r, k]], with src the concatenated outputs of sources and the entries at k >= count[r] padding."""
    kind: str  # "atom", "max" or "min"
    length: int
    label: str
    sources: tuple = ()
    index: np.ndarray = None  # (length, width) int32, local positions into concat(sources)
    count: np.ndarray = None  # (length,) int32, number of valid entries per row
    atom: int = -1
    sign: float = 1.0
    times: np.ndarray = None  # (length,) sample time of each row; None when row r is time r

    @property
    def full(self):
        return self.count is not None and bool(np.all(self.count == self.index.shape[1]))


@dataclass(frozen=True, eq=False)
class Program:
    formula: object  # the NNF formula
    steps: tuple
    T: int
    boundary: str
    n_predicates: int
    nodes: dict = None  # NNF subformula -> step (the outer step of an Until or Release)
    inner: dict = None  # Until or Release subformula -> its inner step
    outputs: tuple = None  # per read, (step, rows); None for an unpruned program

    @property
    def root(self):
        return len(self.steps) - 1


def compile_formula(formula, T, boundary="strict", reads=None):
    """Expand a formula over T samples into a Program. reads keeps only the rows its entries need: (g, times),
    the subformula g at those times, or (g, times, witnesses), rows (t, k) of an Until's or Release's inner step."""
    if boundary not in ("strict", "clip"):
        raise ValueError("boundary must be 'strict' or 'clip'")
    nnf = to_nnf(formula)
    if boundary == "strict" and horizon(nnf) > T - 1:
        raise ValueError("formula horizon " + str(horizon(nnf)) + " needs more than T = " + str(T) + " samples")
    order = _postorder(nnf, [], set())
    need = None
    if reads is not None:
        reads = [_read(r, order, T, boundary) for r in reads]
        need = _needed(order, reads, T, boundary)
    steps, nodes, inner = [], {}, {}
    for f in order:
        _expand(f, T, boundary, steps, nodes, inner, need)
    program = Program(nnf, tuple(steps), T, boundary, max(atoms(nnf)) + 1, nodes, inner)
    if reads is None:
        return program
    outputs = tuple(locate(program, g, t, k) for g, t, k in reads)
    return Program(nnf, tuple(steps), T, boundary, max(atoms(nnf)) + 1, nodes, inner, outputs)


def _length(T, h, boundary):
    return T - h if boundary == "strict" else T


def _children(f):
    if isinstance(f, Atom):
        return ()
    if isinstance(f, (And, Or)):
        return f.children
    if isinstance(f, (Always, Eventually)):
        return (f.child,)
    if isinstance(f, (Until, Release)):
        return (f.left, f.right)
    raise TypeError("formula is not in negation normal form: " + repr(f))


def _postorder(f, out, seen):
    """Distinct NNF subformulas, children before parents, in the order the expansion adds them."""
    if f in seen:
        return out
    for c in _children(f):
        _postorder(c, out, seen)
    seen.add(f)
    out.append(f)
    return out


def _read(r, order, T, boundary):
    """A read as (NNF subformula, times, witnesses or None), checked against the formula."""
    g = to_nnf(r[0])
    if g not in set(order):
        raise ValueError("read of a subformula that the formula does not contain: " + repr(g))
    t = np.atleast_1d(np.asarray(r[1], np.int64))
    L = _length(T, horizon(g), boundary)
    if t.size and (t.min() < 0 or t.max() >= L):
        raise ValueError("read times must lie in [0, " + str(L) + ") for " + repr(g))
    if len(r) < 3 or r[2] is None:
        return g, t, None
    if not isinstance(g, (Until, Release)):
        raise ValueError("witness reads need an Until or Release")
    k = np.broadcast_to(np.atleast_1d(np.asarray(r[2], np.int64)), t.shape)
    a, b = g.interval
    if k.size and (k.min() < a or k.max() > b):
        raise ValueError("witness offsets must lie in [" + str(a) + ", " + str(b) + "]")
    return g, t, np.array(k)


def _intervals(starts, ends, L):
    """Boolean mask over [0, L) of the union of the closed intervals [starts_i, ends_i]."""
    d = np.zeros(L + 1, np.int64)
    np.add.at(d, starts, 1)
    np.add.at(d, ends + 1, -1)
    return np.cumsum(d)[:L] > 0


def _needed(order, reads, T, boundary):
    """Per NNF subformula, the Boolean mask of the times its (outer) step must hold, and per
    Until or Release the mask of the times whose witness rows its inner step must hold."""
    L = {f: _length(T, horizon(f), boundary) for f in order}
    out = {f: np.zeros(L[f], bool) for f in order}
    inn = {f: np.zeros(L[f], bool) for f in order if isinstance(f, (Until, Release))}
    for g, t, k in reads:
        (out if k is None else inn)[g][t] = True
    referenced = set()
    for f in reversed(order):  # parents before children
        M = out[f]
        if f in referenced and not M.any() and not (f in inn and inn[f].any()):
            M[0] = True  # clip mode: a parent reads only padding here, which points at row 0
        if isinstance(f, (And, Or)):
            for c in f.children:
                out[c][:L[f]] |= M
        elif isinstance(f, (Always, Eventually)):
            a, b = f.interval
            t = np.nonzero(M)[0]
            n = np.clip(L[f.child] - (t + a), 0, b - a + 1)
            t, n = t[n > 0], n[n > 0]
            out[f.child] |= _intervals(t + a, t + a + n - 1, L[f.child])
        elif isinstance(f, (Until, Release)):
            a, b = f.interval
            inn[f] |= M
            t = np.nonzero(inn[f])[0]
            n = np.clip(min(L[f.left], L[f.right]) - (t + a), 0, b - a + 1)
            t, n = t[n > 0], n[n > 0]
            out[f.right] |= _intervals(t + a, t + a + n - 1, L[f.right])
            out[f.left] |= _intervals(t, t + a + n - 1, L[f.left])
        if M.any() or (f in inn and inn[f].any()):
            referenced.update(_children(f))
    return out, inn


def _add(steps, step):
    steps.append(step)
    return len(steps) - 1


def _positions(step, n):
    """Row of each time 0..n-1 in a step, -1 where the step holds no row for it; n is at
    least the step's largest time plus one (the node's unpruned length)."""
    if step.times is None:
        return np.concatenate([np.arange(min(n, step.length)), np.full(max(n - step.length, 0), -1)])
    pos = np.full(n, -1)
    pos[step.times] = np.arange(step.length)
    return pos


def _expand(f, T, boundary, steps, nodes, inner, need):
    L = _length(T, horizon(f), boundary)
    pruned = need is not None
    times = np.nonzero(need[0][f])[0] if pruned else np.arange(L)
    row_times = times if pruned else None
    if isinstance(f, Atom):
        if not pruned or times.size:
            nodes[f] = _add(steps, Step("atom", T, ("~p" if f.negated else "p") + str(f.index),
                                        atom=f.index, sign=-1.0 if f.negated else 1.0))
        return
    if isinstance(f, (Until, Release)):
        _expand_until(f, L, T, boundary, steps, nodes, inner, need)
        return
    if not times.size:
        return
    if isinstance(f, (And, Or)):
        kids = tuple(nodes[c] for c in f.children)
        offsets = np.cumsum([0] + [steps[k].length for k in kids[:-1]])
        index = np.stack([_positions(steps[k], _length(T, horizon(c), boundary))[times]
                          for k, c in zip(kids, f.children)], 1) + offsets[None, :]
        count = np.full(times.size, len(kids))
        kind = "min" if isinstance(f, And) else "max"
        nodes[f] = _add(steps, Step(kind, times.size, "and" if kind == "min" else "or", kids, index.astype(np.int32),
                                    count.astype(np.int32), times=row_times))
    elif isinstance(f, (Always, Eventually)):
        a, b = f.interval
        kid = nodes[f.child]
        Lc = _length(T, horizon(f.child), boundary)
        width = b - a + 1
        count = np.clip(Lc - (times + a), 0, width)
        valid = np.arange(width)[None, :] < count[:, None]
        index = np.where(valid, times[:, None] + a + np.arange(width)[None, :], 0)
        index = np.where(valid, _positions(steps[kid], Lc)[index], 0)
        kind = "min" if isinstance(f, Always) else "max"
        name = ("G" if kind == "min" else "F") + "[" + str(a) + "," + str(b) + "]"
        nodes[f] = _add(steps, Step(kind, times.size, name, (kid,), index.astype(np.int32), count.astype(np.int32),
                                    times=row_times))
    else:
        raise TypeError("formula is not in negation normal form: " + repr(f))


def _expand_until(f, L, T, boundary, steps, nodes, inner, need):
    """Two steps for an Until: an inner minimum over right(t+k), left(t), ..., left(t+k) for every time t and offset
    k in [a, b], and an outer maximum over k. Release swaps minimum and maximum."""
    a, b = f.interval
    pruned = need is not None
    t_out = np.nonzero(need[0][f])[0] if pruned else np.arange(L)
    t_in = np.nonzero(need[1][f])[0] if pruned else np.arange(L)
    if not t_in.size:
        return
    left, right = nodes[f.left], nodes[f.right]
    L_left = _length(T, horizon(f.left), boundary)
    L_right = _length(T, horizon(f.right), boundary)
    n_w = b - a + 1
    t = t_in[:, None, None]
    k = a + np.arange(n_w)[None, :, None]  # witness offset
    s = np.arange(b + 1)[None, None, :]  # prefix offset
    # offset k is valid when sample t+k lies in both operand traces
    n_valid = np.clip(min(L_left, L_right) - (t_in + a), 0, n_w)
    witness_ok = np.arange(n_w)[None, :] < n_valid[:, None]
    count = np.where(witness_ok, k[..., 0] + 2, 0)
    valid = np.arange(b + 2)[None, None, :] < count[..., None]
    # row (t, k) holds right(t+k), then left(t), ..., left(t+k); src = concat(right, left)
    witness = np.broadcast_to(t + k, (t_in.size, n_w, 1))
    prefix = np.broadcast_to(t + s, (t_in.size, n_w, b + 1))
    times = np.where(valid, np.concatenate([witness, prefix], axis=-1), 0)
    pos_right = _positions(steps[right], L_right)
    pos_left = _positions(steps[left], L_left)
    index = np.concatenate([pos_right[times[..., :1]], steps[right].length + pos_left[times[..., 1:]]], axis=-1)
    index = np.where(valid, index, 0)
    inner_kind, outer_kind = ("min", "max") if isinstance(f, Until) else ("max", "min")
    name = ("U" if isinstance(f, Until) else "R") + "[" + str(a) + "," + str(b) + "]"
    inner[f] = _add(steps, Step(inner_kind, t_in.size * n_w, name + ".inner", (right, left),
                                index.reshape(t_in.size * n_w, b + 2).astype(np.int32),
                                count.reshape(t_in.size * n_w).astype(np.int32),
                                times=np.repeat(t_in, n_w) if pruned else None))
    if not t_out.size:
        return
    block = np.full(L, -1)
    block[t_in] = np.arange(t_in.size)
    outer_index = block[t_out][:, None] * n_w + np.arange(n_w)[None, :]
    count = np.clip(min(L_left, L_right) - (t_out + a), 0, n_w)
    nodes[f] = _add(steps, Step(outer_kind, t_out.size, name, (inner[f],), outer_index.astype(np.int32),
                                count.astype(np.int32), times=t_out if pruned else None))


def locate(program, g, times, witnesses=None):
    """(step, rows) holding the subformula g at the given times, or with witnesses the rows
    (t, k) of the inner step of the Until or Release g."""
    g = to_nnf(g)
    t = np.atleast_1d(np.asarray(times, np.int64))
    if witnesses is None:
        if g not in program.nodes:
            raise ValueError("the program holds no step for " + repr(g))
        sid = program.nodes[g]
        step = program.steps[sid]
        full = step.length if step.times is None else int(step.times.max()) + 1
        pos = _positions(step, max(full, int(t.max(initial=0)) + 1))
        rows = pos[t]
    else:
        if g not in program.inner:
            raise ValueError("the program holds no inner step for " + repr(g))
        sid = program.inner[g]
        step = program.steps[sid]
        a, b = g.interval
        n_w = b - a + 1
        first = np.arange(step.length // n_w) if step.times is None else step.times[::n_w]
        block = np.full(max(int(first.max(initial=0)), int(t.max(initial=0))) + 1, -1)
        block[first] = np.arange(first.size)
        k = np.broadcast_to(np.atleast_1d(np.asarray(witnesses, np.int64)), t.shape)
        rows = np.where(block[t] >= 0, block[t] * n_w + (k - a), -1)
    if np.any(rows < 0):
        raise ValueError("the program holds no row for some of the requested times")
    return sid, rows.astype(np.int64)
