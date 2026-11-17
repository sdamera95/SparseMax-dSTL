"""Sparsity patterns of one control period's program prescribed by the dynamics and the specification.

A row's pattern is stored as prefix lengths: at horizon sample j and arm b, the number L of leading joints whose positions
the row reads (no predicate reads a velocity). The defects' pattern comes from the interval map's detected Jacobian, the
cost's Hessian from its position diagonal and its acceleration band. An instance row reads the per-sample nodes its program
reaches from its root; three reaches give three patterns: through the sparsemax supports (weights above zero), through the
entries within a margin kappa of their node's threshold, and through every residual operand (the prescribed pattern, which a
structural detector gives for any smoothing). A row's Hessian pattern is the clique of its columns.
"""
import asdex
import jax
import jax.numpy as jnp
import numpy as np

from ..jax.evaluator import lse_min_matched
from ..jax.operators import lower_min, sparsemax_weights
from ..jax import progression as jpg
from ..stl import progression as pg
from ..tasks import dual_arm as D

NX, NU, NQ, NJ = D.NX, D.NU, D.NQ, D.NJ
KAPPAS = (0.5, 1.0)  # near-support margins, in units of the conjunct's gamma


def upper(rows, cols, n):
    """Sorted, duplicate-free upper triangle (rows, cols) of a symmetric pattern given by any of its entries."""
    a, b = np.minimum(rows, cols).astype(np.int64), np.maximum(rows, cols).astype(np.int64)
    key = np.unique(a * n + b)
    return key // n, key % n


def positions(pattern, rows, cols, n):
    """Positions of (rows, cols) in a sorted row-major pattern; raises if one is missing."""
    keys = pattern[0] * n + pattern[1]
    k = np.asarray(rows, np.int64) * n + np.asarray(cols, np.int64)
    pos = np.searchsorted(keys, k)
    if np.any(pos >= keys.size) or np.any(keys[np.minimum(pos, keys.size - 1)] != k):
        raise ValueError("an entry is not in the pattern")
    return pos


def columns(prefix):
    """(rows, cols) of rows given as prefix lengths (R, T, 2): the positions of joints 1..L of arm b at horizon sample j."""
    i = np.arange(NJ)
    r, j, arm, ii = np.nonzero(i[None, None, None, :] < np.asarray(prefix)[..., None])
    return r, j * NX + arm * NJ + ii


def hessian_union(rows, cols, n_rows, n_cols):
    """The union of the rows' cliques as a symmetric SciPy CSR Boolean pattern (n_cols, n_cols)."""
    import scipy.sparse as sps
    B = sps.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(n_rows, n_cols))
    H = (B.T @ B).tocsr()
    H.data[:] = 1
    return H


def interval_block(dtype=jnp.float64):
    """(28, 42) Boolean pattern of one interval map's Jacobian [D_x f, D_u f], detected by asdex."""
    f = D.interval_map(D.H_STEP)
    with jax.enable_x64(True):
        sp = asdex.jacobian_sparsity(lambda z: f(z[:NX], z[NX:]), jnp.zeros(NX + NU, dtype))
    B = np.zeros((NX, NX + NU), bool)
    B[np.asarray(sp.rows), np.asarray(sp.cols)] = True
    return B


def defect_pattern(T):
    """(rows, cols) of the 28 T defect rows in the 42 T variables (states first, then accelerations)."""
    n = NX * T
    i, c = np.nonzero(interval_block())
    j = np.arange(T)[:, None]
    xs, us = c < NX, c >= NX
    r = np.concatenate([(j * NX + np.arange(NX)[None]).ravel(), np.broadcast_to(j * NX + i[us][None], (T, int(us.sum()))).ravel(),
                        (j * NX + i[xs][None])[1:].ravel()])
    cc = np.concatenate([(j * NX + np.arange(NX)[None]).ravel(), (n + j * NU + (c[us] - NX)[None]).ravel(), ((j - 1) * NX + c[xs][None])[1:].ravel()])
    return r, cc


def control_band(T):
    """(rows, cols) of each acceleration with itself and with the same joint's acceleration of the next interval."""
    u = NX * T + np.arange(T * NU)
    nxt = u[:-NU] if T > 1 else u[:0]
    return np.concatenate([u, nxt]), np.concatenate([u, nxt + NU])


def cost_pattern(T):
    """Upper (rows, cols) of the cost's Hessian: every position's diagonal and the acceleration band."""
    pos = (np.arange(T)[:, None] * NX + np.arange(NQ)[None]).ravel()
    br, bc = control_band(T)
    return upper(np.concatenate([pos, br]), np.concatenate([pos, bc]), (NX + NU) * T)


def jacobian_pattern(T, r, c):
    """The constraint Jacobian's (rows, cols), row-major: the defects and the instance rows' entries (r, c)."""
    n, n_def = (NX + NU) * T, NX * T
    dr, dc = defect_pattern(T)
    key = np.unique(np.concatenate([dr * n + dc, (n_def + np.asarray(r)) * n + np.asarray(c)]))
    return key // n, key % n


def hessian_pattern(T, n_rows, r, c, with_cost=True):
    """Upper (rows, cols) of the Lagrangian Hessian: the union of the rows' cliques over their entries (r, c), and the cost's."""
    n = (NX + NU) * T
    H = hessian_union(r, c, max(n_rows, 1), n)
    hr, hc = H.nonzero()
    cr, cc = cost_pattern(T) if with_cost else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    return upper(np.concatenate([hr, cr]), np.concatenate([hc, cc]), n)


# ------------------------------------------------------------------
# the instances' prefix rows over control periods

@jax.jit
def _and_node(z, gamma, kap, dp):
    w = sparsemax_weights(-z, gamma)
    on = w > 0
    theta = (jnp.sum(jnp.where(on, -z, 0), -1) - gamma) / jnp.sum(on, -1)
    near = (-z[..., None, :] > theta[..., None, None] - kap[:, None]) | on[..., None, :]
    sel = jnp.concatenate([on[..., None, :], near, jnp.ones_like(on)[..., None, :]], -2)
    return {"sm": lower_min(z, gamma), "lse": lse_min_matched(z, gamma), "support": jnp.sum(on, -1).astype(jnp.int32),
            "operands": jnp.full(z.shape[:-1], z.shape[-1], jnp.int32), "prefix": jnp.max(jnp.where(sel[..., None], dp, 0), -2).astype(jnp.int8)}


def node_table(cj, P, joints, kappas=KAPPAS):
    """Per per-sample node of a conjunct, over predicate vectors P (..., n_pred): its sparsemax and matched log-sum-exp
    values, its support size and operand count, and its prefix lengths (..., modes, 2) through its supports (mode 0), its
    entries within each kappa gamma of its threshold, and every operand (last mode)."""
    kap = jnp.asarray([k * cj.gamma for k in kappas], jnp.float64)
    out = []
    for nd in cj.nodes:
        idx = np.atleast_1d(nd["preds"])
        dp = jnp.asarray(joints[idx])
        z = jnp.asarray(P)[..., idx]
        if nd["kind"] == "leaf":
            v = z[..., 0]
            out.append({"sm": v, "lse": v, "support": jnp.ones(v.shape, jnp.int32), "operands": jnp.ones(v.shape, jnp.int32),
                        "prefix": jnp.broadcast_to(dp[0], v.shape + (kap.size + 2, 2)).astype(jnp.int8)})
            continue
        out.append(_and_node(z, cj.gamma, kap, dp))
    return [{k: np.asarray(v) for k, v in x.items()} for x in out]


_FNS = {}


def _program_fns(prog):
    """The compiled evaluations of one program: every step under sparsemax, the root step under matched log-sum-exp, the
    sparsemax weights."""
    if id(prog) not in _FNS:
        root = prog.outputs[0][0]
        _FNS[id(prog)] = (prog, jax.jit(lambda sc, m, g: jpg.evaluate(prog, sc, m, "sparsemax", g)),
                          jax.jit(lambda sc, m, g: jpg.evaluate(prog, sc, m, "lse_matched", g)[root]),
                          jax.jit(lambda vals, m, g: jpg.weights(prog, vals, m, g)))
    return _FNS[id(prog)][1:]


class Reference:
    """The nodes of a scene's conjuncts along its states (L samples, the record's) and at the held states of control
    periods: the values from which the patterns at the reference motion are read."""

    def __init__(self, scene, conjs, kappas=KAPPAS):
        self.scene, self.conjs, self.kappas = scene, conjs, kappas
        with jax.enable_x64(True):
            L = conjs[0].program.T
            self._pred = jax.jit(jax.vmap(scene.predicates))
            P = np.asarray(self._pred(jnp.asarray(scene.states[:L], jnp.float64)))
            self.nodes = [node_table(cj, P, scene.joints, kappas) for cj in conjs]

    def held(self, ks):
        """Per conjunct, the node tables (K, ...) at the held states of periods ks: the state at k + T at zero velocity."""
        x = np.array(self.scene.states[np.asarray(ks) + self.scene.T])
        x[:, NQ:] = 0.0
        with jax.enable_x64(True):
            P = np.asarray(self._pred(jnp.asarray(x, jnp.float64)))
            return [node_table(cj, P, self.scene.joints, self.kappas) for cj in self.conjs]


def _expand(m, like):
    return m.reshape(m.shape + (1,) * (like.ndim - m.ndim))


def traces(ref, hold, ks, T):
    """Per atom, the node quantities (K, L, ...) at periods ks: along the reference, and at the held state beyond k + T."""
    L = next(iter(ref[0].values())).shape[0]
    beyond = np.arange(L)[None] > np.asarray(ks)[:, None] + T
    out = []
    for r, h in zip(ref, hold):
        x = {}
        for key in ("sm", "lse", "prefix", "support", "operands"):
            base = np.broadcast_to(r[key][None], (len(ks),) + r[key].shape)
            x[key] = np.where(_expand(beyond, base), h[key][:, None], base)
        out.append(x)
    return out


def to_horizon(acc, ks, T):
    """Prefix rows over absolute samples (K, n, L, 2) mapped to the horizon's samples j = 0..T-1 (sample k + 1 + j); every
    sample beyond k + T is the held state, horizon sample T - 1; a horizon sample after the record's last receives nothing."""
    L = acc.shape[2]
    s = np.asarray(ks)[:, None] + 1 + np.arange(T)[None]
    out = np.take_along_axis(acc, np.minimum(s, L - 1)[:, None, :, None], 2)
    out[np.broadcast_to((s > L - 1)[:, None], out.shape[:3])] = 0
    tail = np.where((np.arange(L)[None] > np.asarray(ks)[:, None] + T)[:, None, :, None], acc, 0).max(2)
    out[:, :, T - 1] = np.maximum(out[:, :, T - 1], tail)
    return out


def period_rows(cj, ref, hold, ks, T, n_slots=None):
    """One conjunct's imposed instances at periods ks (K,): per slot (K, n) its instance index (-1 empty), root values under
    sparsemax and matched log-sum-exp, prefix rows (K, n, modes, T, 2) and reached leaf counts (K, n, modes) per reach mode
    (supports, each near margin, every residual operand), whether the supports reach a node that the record decides at the
    next period, whether they reach the held state, and the root's support size and operand count."""
    ks = np.asarray(ks)
    prog = cj.program
    tr = traces(ref, hold, ks, T)
    sm = np.stack([x["sm"] for x in tr], -1)
    lse = np.stack([x["lse"] for x in tr], -1)
    res = pg.residual(prog, cj.decisions, ks)
    masks = [None if m is None else jnp.asarray(m) for m in pg.evaluation_masks(prog, res)]
    ev, ev_lse, weights = _program_fns(prog)
    with jax.enable_x64(True):
        vals = ev(jnp.asarray(sm), masks, cj.gamma)
        root_lse = np.asarray(ev_lse(jnp.asarray(lse), masks, cj.gamma))
        wts = weights(vals, masks, cj.gamma)
    wts = [None if w is None else {k: np.asarray(v) for k, v in w.items()} for w in wts]
    masks = [None if m is None else np.asarray(m) for m in masks]
    rows, inst = cj.imposed(ks, T, n_slots)
    follows = [[None if w is None else w["p"] > 0 for w in wts]]
    for kap in (k * cj.gamma for k in KAPPAS):
        follows.append([None if w is None else (w["zeta"] > w["theta"][..., None] - kap) & m | (w["p"] > 0) for w, m in zip(wts, masks)])
    follows.append([None if r is None else (r[1] & r[0][..., None]) for r in res])
    n_modes = len(follows)
    K, n = rows.shape
    L = prog.T
    prefix = np.zeros((K, n, n_modes, T, 2), np.int8)
    leaves = np.zeros((K, n, n_modes), np.int64)
    root_vals = np.asarray(vals[cj.root_step])
    atom_steps = [i for i, s in enumerate(prog.steps) if s.kind == "atom"]
    for m, follow in enumerate(follows):
        reached = pg.reach(prog, cj.root_step, rows, follow)
        acc = np.zeros((K, n, L, 2), np.int8)
        cnt = np.zeros((K, n), np.int64)
        for i in atom_steps:
            if reached[i] is None:
                continue
            a = prog.steps[i].atom
            pre = tr[a]["prefix"][:, :, m]
            acc = np.maximum(acc, np.where(reached[i][..., None], pre[:, None], 0).astype(np.int8))
            cnt += np.sum(np.where(reached[i], (tr[a]["support"] if m == 0 else tr[a]["operands"])[:, None], 0), -1)
        prefix[:, :, m] = to_horizon(acc, ks, T)
        leaves[:, :, m] = cnt
        if m == 0:
            nxt = np.zeros((K, n), bool)
            for i, r in enumerate(reached):
                if r is not None:
                    nxt |= np.any(r & (cj.decisions[i][0][None, None, :] == ks[:, None, None] + 1), -1)
            held = np.any(acc.max(-1) * (np.arange(L)[None, None] >= ks[:, None, None] + T), -1)
    rr = np.maximum(rows, 0)
    return {"inst": inst, "sm": np.where(rows >= 0, np.take_along_axis(root_vals, rr, 1), np.nan),
            "lse": np.where(rows >= 0, np.take_along_axis(root_lse, rr, 1), np.nan), "prefix": prefix, "leaves": leaves,
            "next_decided": nxt, "reaches_end": held,
            "root_support": np.where(rows >= 0, np.take_along_axis((wts[cj.root_step]["p"] > 0).sum(-1), rr, 1), -1),
            "root_operands": np.where(rows >= 0, np.take_along_axis(res[cj.root_step][1].sum(-1), rr, 1), -1)}


def scene_rows(reference, ks, n_slots=None):
    """period_rows of every conjunct at periods ks, concatenated over the slots in row order; "read" is each slot's
    instance index and "inst" is 0 for an imposed slot, -1 for an empty one. n_slots: per conjunct, or None."""
    hold = reference.held(ks)
    T = reference.scene.T
    outs = [period_rows(cj, ref, h, ks, T, None if n_slots is None else n_slots[i])
            for i, (cj, ref, h) in enumerate(zip(reference.conjs, reference.nodes, hold))]
    parts = {k: np.concatenate([o[k] for o in outs], 1) for k in outs[0]}
    parts["read"] = parts.pop("inst")
    parts["inst"] = np.where(parts["read"] >= 0, 0, -1)
    return parts


LEVELS = (("prescribed", -1), ("supports", 0))  # (pattern, reach mode)


def prescribed(pb, reference=None):
    """The prescribed patterns of a PeriodProgram at its period: the constraint Jacobian's (rows, cols) (the defects and
    every residual operand's predicate dependencies) and the Lagrangian Hessian's upper (rows, cols) (the rows' cliques, the
    cost's pattern)."""
    reference = Reference(pb.scene, pb.conjs) if reference is None else reference
    parts = scene_rows(reference, np.array([pb.k]))
    on = parts["inst"][0] >= 0
    r, c = columns(parts["prefix"][0][on][:, -1])
    J = jacobian_pattern(pb.T, r, c)
    H = hessian_pattern(pb.T, int(on.sum()), r, c)
    br, bc = control_band(pb.T)
    return J, upper(np.concatenate([H[0], br]), np.concatenate([H[1], bc]), pb.n)
