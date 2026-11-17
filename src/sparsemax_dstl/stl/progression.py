"""Progression of a compiled specification at a margin delta over a recorded prefix: which rows the record decides, and
the residual operands of the rows it leaves open.

At control period k the samples 0..k are recorded and the later ones predicted. A row is decided at k when its
verdict (exact value at least delta) depends on recorded samples only: an atom row t is decided at t; a minimum is
decided by its first child below delta, or by its last child; a maximum by its first child at or above delta, or by
its last child. The residual of an open row keeps its open operands; the decided ones it drops cannot change its
verdict. NumPy; readers() and the incremental states also need SciPy. Loops run over the program's steps only.
"""
import copy

import numpy as np

INF = np.iinfo(np.int64).max // 4


def valid(step):
    """(rows, width) Boolean: the window entries of a reduction step that are not padding."""
    return np.arange(step.index.shape[1])[None, :] < step.count[:, None]


def _gather(step, per_step):
    src = np.concatenate([per_step[s] for s in step.sources], -1)
    return src[..., step.index]


def decisions(program, record, delta):
    """Per step, (d, ok): the first period at which each row is decided by the record and its verdict there.
    record (..., T, P) holds the exact atom scores; delta is a number or broadcasts against the batch axes."""
    record = np.asarray(record, np.float64)
    T = program.T
    delta = np.asarray(delta, np.float64)[..., None]
    out = []
    for step in program.steps:
        if step.kind == "atom":
            ok = step.sign * record[..., :, step.atom] >= delta
            out.append((np.broadcast_to(np.arange(T, dtype=np.int64), ok.shape), ok))
            continue
        v = valid(step)
        d = _gather(step, [x[0] for x in out])
        ok = _gather(step, [x[1] for x in out])
        decisive = v & (ok if step.kind == "max" else ~ok)
        first = np.min(np.where(decisive, d, INF), axis=-1)
        last = np.max(np.where(v, d, -1), axis=-1)
        hit = first < INF
        verdict = hit if step.kind == "max" else ~hit
        out.append((np.where(hit, first, last), verdict))
    return out


def residual(program, dec, k):
    """Per step at period k (an int or an array broadcasting against the batch axes): None for atoms, else (open rows
    (..., rows), operand masks (..., rows, width))."""
    k = np.asarray(k, np.int64)[..., None]
    out = []
    for step, (d, _) in zip(program.steps, dec):
        if step.kind == "atom":
            out.append(None)
            continue
        dc = np.concatenate([dec[s][0] for s in step.sources], -1)[..., step.index]
        out.append((d > k, valid(step) & (dc > k[..., None])))
    return out


def evaluation_masks(program, res):
    """Per step the masks to evaluate under: the residual operands on open rows, every valid entry on decided rows."""
    out = []
    for step, r in zip(program.steps, res):
        if r is None:
            out.append(None)
            continue
        is_open, m = r
        out.append(np.where(is_open[..., None], m, valid(step)))
    return out


def reach(program, root_step, roots, follow):
    """Per step, the rows reached from root rows through the followed entries: roots (B, R) rows of root_step, -1 for
    an empty slot; follow per step None (atoms) or a Boolean (B, rows, width). Returns per step a Boolean (B, R, rows)
    array, None for steps that no root reaches."""
    roots = np.asarray(roots, np.int64)
    B, R = roots.shape
    steps = program.steps
    out = [None] * len(steps)
    first = np.zeros((B, R, steps[root_step].length), bool)
    b, r = np.nonzero(roots >= 0)
    first[b, r, roots[b, r]] = True
    out[root_step] = first
    for i in range(len(steps) - 1, -1, -1):
        s = steps[i]
        if s.kind == "atom" or out[i] is None:
            continue
        b, r, p = np.nonzero(out[i])
        live = np.asarray(follow[i])[b, p]
        e_n, e_w = np.nonzero(live)
        target = s.index[p[e_n], e_w]
        lengths = [steps[k].length for k in s.sources]
        offsets = np.cumsum([0] + lengths)
        for j, k in enumerate(s.sources):
            sel = (target >= offsets[j]) & (target < offsets[j + 1])
            if out[k] is None:
                out[k] = np.zeros((B, R, lengths[j]), bool)
            out[k][b[e_n[sel]], r[e_n[sel]], target[sel] - offsets[j]] = True
    return out


def readers(program):
    """Per step, (ptr, rows): rows[ptr[t]:ptr[t + 1]] are the sorted rows of that step that read sample t."""
    import scipy.sparse as sps  # imported here so that the module imports without SciPy

    T = program.T
    reads, out = [], []
    for step in program.steps:
        if step.kind == "atom":
            m = sps.identity(T, dtype=np.int64, format="csr")
        else:
            src = sps.vstack([reads[s] for s in step.sources], format="csr")
            r, w = np.nonzero(valid(step))
            E = sps.csr_matrix((np.ones(r.size, np.int64), (r, step.index[r, w].astype(np.int64))), shape=(step.length, src.shape[0]))
            m = (E @ src).tocsr()
            m.data[:] = 1
        reads.append(m)
        c = m.tocsc()
        c.sort_indices()
        out.append((c.indptr.astype(np.int64), c.indices.astype(np.int64)))
    return out


class Progression:
    """The decisions and the residual of one program for G members that share it (members None: one member, no leading
    axis), updated one recorded sample at a time; equal to decisions() and residual() on the record so far.

    dec: per step (d, ok); residual: per step None or (open, masks); k: the last recorded sample (-1 before the first).
    The arrays are views that change at the next update(); copy() gives an independent state."""

    def __init__(self, program, delta, members=None):
        self.program, self.members = program, members
        G = 1 if members is None else int(members)
        self.delta = np.broadcast_to(np.asarray(delta, np.float64), (G,)).copy()
        steps = program.steps
        lengths = np.array([s.length for s in steps], np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(lengths)])
        self.gidx, self.valid = [], []
        for step in steps:
            if step.kind == "atom":
                self.gidx.append(None)
                self.valid.append(None)
                continue
            pos = np.concatenate([self.offsets[s] + np.arange(steps[s].length) for s in step.sources])
            self.gidx.append(pos[step.index])
            self.valid.append(valid(step))
        self.cones = readers(program)
        nan = np.full((G, program.T, program.n_predicates), np.nan)
        dec = decisions(program, nan, self.delta)
        res = residual(program, dec, np.full((G,), -1, np.int64))
        self.k = -1
        self.d = np.concatenate([x[0] for x in dec], -1)
        self.ok = np.concatenate([x[1] for x in dec], -1)
        self.open = np.concatenate([np.zeros((G, s.length), bool) if r is None else r[0] for s, r in zip(steps, res)], -1)
        self.masks = [None if r is None else r[1].copy() for r in res]
        self._views()

    def _views(self):
        o = self.offsets
        steps = self.program.steps
        dec = [(self.d[:, o[i]:o[i + 1]], self.ok[:, o[i]:o[i + 1]]) for i in range(len(steps))]
        res = [None if s.kind == "atom" else (self.open[:, o[i]:o[i + 1]], self.masks[i]) for i, s in enumerate(steps)]
        if self.members is None:
            dec = [(d[0], ok[0]) for d, ok in dec]
            res = [None if r is None else (r[0][0], r[1][0]) for r in res]
        self.dec, self.residual = dec, res

    def member(self, g):
        """(dec, residual) of member g, shaped as for a single program."""
        return ([(d[g], ok[g]) for d, ok in self.dec], [None if r is None else (r[0][g], r[1][g]) for r in self.residual])

    def copy(self):
        new = copy.copy(self)
        new.delta, new.d, new.ok, new.open = self.delta.copy(), self.d.copy(), self.ok.copy(), self.open.copy()
        new.masks = [None if m is None else m.copy() for m in self.masks]
        new._views()
        return new

    def update(self, values):
        """Record sample k + 1: values (P,) or (G, P), its exact atom scores."""
        k = self.k + 1
        if k >= self.program.T:
            raise ValueError("the record already holds all " + str(self.program.T) + " samples")
        values = np.asarray(values, np.float64).reshape(self.delta.size, -1)
        d, ok, o = self.d, self.ok, self.offsets
        for i, step in enumerate(self.program.steps):
            if step.kind == "atom":
                ok[:, o[i] + k] = step.sign * values[:, step.atom] >= self.delta
                continue
            ptr, rows = self.cones[i]
            R = rows[ptr[k]:ptr[k + 1]]
            if R.size == 0:
                continue
            gi, v = self.gidx[i][R], self.valid[i][R]
            dc, okc = d[:, gi], ok[:, gi]
            decisive = v & (okc if step.kind == "max" else ~okc)
            first = np.min(np.where(decisive, dc, INF), axis=-1)
            last = np.max(np.where(v, dc, -1), axis=-1)
            hit = first < INF
            new = np.where(hit, first, last)
            d[:, o[i] + R] = new
            ok[:, o[i] + R] = hit if step.kind == "max" else ~hit
            self.open[:, o[i] + R] = new > k
            self.masks[i][:, R] = v & (dc > k)
        self.k = k

    def update_block(self, values):
        """Record samples k + 1 .. k + B at once: values (B, P) or (B, G, P); the state equals B calls of update()."""
        values = np.asarray(values, np.float64)
        B = values.shape[0]
        k0, K = self.k + 1, self.k + B
        if K >= self.program.T:
            raise ValueError("the record holds " + str(self.program.T) + " samples; the block ends at sample " + str(K))
        if B == 0:
            return
        values = values.reshape(B, self.delta.size, -1)
        d, ok, o = self.d, self.ok, self.offsets
        for i, step in enumerate(self.program.steps):
            if step.kind == "atom":
                ok[:, o[i] + k0:o[i] + K + 1] = (step.sign * values[:, :, step.atom] >= self.delta).T
                continue
            ptr, rows = self.cones[i]
            R = np.unique(rows[ptr[k0]:ptr[K + 1]])
            if R.size == 0:
                continue
            gi, v = self.gidx[i][R], self.valid[i][R]
            dc, okc = d[:, gi], ok[:, gi]
            decisive = v & (okc if step.kind == "max" else ~okc)
            first = np.min(np.where(decisive, dc, INF), axis=-1)
            last = np.max(np.where(v, dc, -1), axis=-1)
            hit = first < INF
            new = np.where(hit, first, last)
            d[:, o[i] + R] = new
            ok[:, o[i] + R] = hit if step.kind == "max" else ~hit
            self.open[:, o[i] + R] = new > K
            self.masks[i][:, R] = v & (dc > K)
        self.k = K

    def holds(self, k):
        """Whether residual is the residual at period k (k = self.k, or any k >= T - 1 once every sample is recorded)."""
        return k == self.k or (self.k == self.program.T - 1 and k >= self.k)


class Progressions:
    """Progression states of many members over one shared record row, members with the same program object grouped.

    programs: one per member; columns: per member the record's columns of its atoms; delta: a number or one per member.
    update(row) records the next sample from the whole row; states() gives per member {"dec", "residual"}."""

    def __init__(self, programs, columns, delta):
        delta = np.broadcast_to(np.asarray(delta, np.float64), (len(programs),))
        ids = {}
        for i, p in enumerate(programs):
            ids.setdefault(id(p), []).append(i)
        self.groups, self.cols, self.where = [], [], [None] * len(programs)
        for j, idx in enumerate(ids.values()):
            self.groups.append(Progression(programs[idx[0]], delta[idx], len(idx)))
            self.cols.append(np.stack([np.asarray(columns[i], np.int64) for i in idx]))
            for g, i in enumerate(idx):
                self.where[i] = (j, g)
        self._cache()

    def _cache(self):
        views = [g.member(m) for g in self.groups for m in range(g.delta.size)]
        start = np.cumsum([0] + [g.delta.size for g in self.groups])
        self._states = [dict(zip(("dec", "residual"), views[start[j] + g])) for j, g in self.where]

    @property
    def k(self):
        return self.groups[0].k

    def update(self, row):
        row = np.asarray(row, np.float64)
        for g, c in zip(self.groups, self.cols):
            g.update(row[c])

    def update_block(self, rows):
        """Record the next B samples from rows (B, columns); the state equals B calls of update()."""
        rows = np.asarray(rows, np.float64)
        for g, c in zip(self.groups, self.cols):
            g.update_block(rows[:, c])

    def copy(self):
        new = copy.copy(self)
        new.groups = [g.copy() for g in self.groups]
        new._cache()
        return new

    def states(self, k=None):
        """Per member {"dec": per step (d, ok), "residual": per step None or (open, masks)} at period k (default the
        last recorded sample); views that change at the next update()."""
        if k is not None and not self.groups[0].holds(k):
            raise ValueError("the state holds the residual at period " + str(self.k) + ", not " + str(k))
        return [dict(s) for s in self._states]
