"""One nonlinear program per horizon, compiled once: each control period's measured state, reference and imposed instances
are parameters of fixed-size slots, and the derivative calls are compiled at a few padded lengths before the first period."""
import time

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

from ..jax.evaluator import REDUCTIONS
from ..jax.operators import extremum_and_weights, lower_min_recount
from ..sparse import patterns as SP
from ..sparse.assembly import bucket, group_pairs, pad
from ..sparse.coloring import CHUNK, hessian_plan, jacobian_plan
from ..sparse.supports import forward, size, skeleton
from ..stl import progression as pg
from ..tasks import dual_arm as D
from . import conjuncts as MC
from .ipopt import ACCEPT_STATUS, ACCEPT_VIOLATION, FUNCS, Function, Hessian, Iteration, ccs, ipopt_options, violation
from .program import W_DU, W_Q, W_U, sub_values, subprogram

NX, NU, NQ, NJ = D.NX, D.NU, D.NQ, D.NJ
FAMILIES = ("reach", "until")  # the slot vector: reach slots, then until slots
OFF_VALUE = 1.0  # the constraint value of a switched-off or empty slot
LENGTHS = {"jac": ("ng", "m", "nl", "nz", "nj"), "hess": ("ng", "m", "nl", "np", "nk", "M", "nh", "ec")}


def node_key(nd, gamma):
    return (nd["kind"], tuple(np.atleast_1d(nd["preds"]).tolist()), float(gamma) if nd["kind"] == "and" else 0.0)


def full_masks(prog):
    """Every valid entry of every reduction step."""
    return [None if s.kind == "atom" else pg.valid(s) for s in prog.steps]


def pad_slot(sub, lay, k):
    """A cut program at period k in a family's padded layout: per atom step the relative samples s - k - 1, per reduction
    step the operand indices into the padded sources and the masks; a padding row reads entry 0."""
    if len(sub) != len(lay["steps"]) or any(d["kind"] != s[0] for d, s in zip(sub, lay["steps"])):
        raise ValueError("an instance's cut program differs from its family's layout")
    rel, idx, msk = [], [], []
    for d, n, w in zip(sub, lay["rows"], lay["width"]):
        if d["kind"] == "atom":
            r = np.zeros(n, np.int64)
            r[:d["rows"].size] = d["rows"] - k - 1
            rel.append(r)
            idx.append(None)
            msk.append(None)
            continue
        off_n = np.cumsum([0] + [sub[s]["rows"].size for s in d["sources"]])
        off_N = np.cumsum([0] + [lay["rows"][s] for s in d["sources"]])
        x, m = np.asarray(d["index"], np.int64), np.asarray(d["mask"], bool)
        j = np.clip(np.searchsorted(off_n, x, side="right") - 1, 0, len(d["sources"]) - 1)
        I, M = np.zeros((n, w), np.int64), np.zeros((n, w), bool)
        I[:x.shape[0], :x.shape[1]] = np.where(m, x - off_n[j] + off_N[j], 0)
        M[:x.shape[0], :x.shape[1]] = m
        M[x.shape[0]:, 0] = True
        rel.append(None)
        idx.append(I)
        msk.append(M)
    return rel, idx, msk


def empty_slot(lay):
    """The arrays of an empty or switched-off slot: every row reads entry 0."""
    rel = [np.zeros(n, np.int64) if s[0] == "atom" else None for s, n in zip(lay["steps"], lay["rows"])]
    idx = [None if s[0] == "atom" else np.zeros((n, w), np.int64) for s, n, w in zip(lay["steps"], lay["rows"], lay["width"])]
    msk = [None if s[0] == "atom" else np.zeros((n, w), bool) for s, n, w in zip(lay["steps"], lay["rows"], lay["width"])]
    for m in msk:
        if m is not None:
            m[:, 0] = True
    return rel, idx, msk


# ------------------------------------------------------------------
# the rows, slots and record of one horizon

class Horizon:
    """The conjuncts, per-sample nodes, slot layouts and record of a scene at its horizon; exact_length is the outside
    predicates' smoothing length (m) of the record and of the exact rows (0 the exact norm)."""

    def __init__(self, scene, exact_length=0.0):
        self.scene, self.T, self.exact_length, self.delta = scene, int(scene.T), exact_length, scene.margin
        X = np.asarray(scene.states, np.float64)
        self.L = X.shape[0]
        with jax.enable_x64(True):
            cells = jnp.asarray(scene.cells, jnp.float64)
            one = lambda x: jnp.concatenate([D.at_waypoint(x, scene.waypoints, scene.waypoint_arms, 0.0), D.outside(x, cells, scene.cell_radius, exact_length)])  # noqa: E731
            self._record = jax.jit(jax.vmap(one))
        self.conjs = MC.conjuncts(scene, self.predicates(X), self.delta)
        nodes = {}
        self.cnodes, self.family, self.span, self.anchors = [], [], [], []
        for cj in self.conjs:
            if any(nd["kind"] not in ("and", "leaf") for nd in cj.nodes):
                raise ValueError("the per-sample nodes are conjunctions or leaves")
            self.cnodes.append(np.array([nodes.setdefault(node_key(nd, cj.gamma), len(nodes)) for nd in cj.nodes], np.int64))
            self.family.append(cj.kind)
            self.span.append(int(cj.window))
            self.anchors.append(np.asarray(cj.anchors, np.int64))
        if self.family != sorted(self.family, key=FAMILIES.index):
            raise ValueError("the conjuncts are not reach conjuncts first, then untils")
        self.node_keys = sorted(nodes, key=nodes.get)
        S = max(len(kk[1]) for kk in self.node_keys)
        self.pidx = np.array([list(kk[1]) + [kk[1][0]] * (S - len(kk[1])) for kk in self.node_keys], np.int64)
        self.gamma, self.layout = {}, {}
        for f in FAMILIES:
            g = sorted(set(float(cj.gamma) for cj, ff in zip(self.conjs, self.family) if ff == f))
            if len(g) > 1:
                raise ValueError("the conjuncts of a family do not share one smoothing parameter")
            self.gamma[f] = g[0] if g else 1.0
            self.layout[f] = self._layout(f)
        counts = self.candidate_counts()
        self.n_slots = {f: int(counts[f].max(initial=0)) for f in FAMILIES}
        self.progs = None

    def _layout(self, f):
        """Per step of the family's cut program (kind, atom, sign, sources) with its largest rows and operands."""
        skel, rows, width, root = None, None, None, None
        for ci, cj in enumerate(self.conjs):
            if self.family[ci] != f:
                continue
            sub, rt, rrow = subprogram(cj.program, cj.root_step, int(cj.root_rows[0]), full_masks(cj.program))
            s = [(d["kind"], d.get("atom", -1), d.get("sign", 0.0), tuple(d.get("sources", ()))) for d in sub]
            if skel is None:
                skel, root = s, rt
            elif s != skel or rt != root or rrow != 0:
                raise ValueError("the cut programs of a family differ in their steps")
            r = np.array([d["rows"].size for d in sub])
            w = np.array([0 if d["kind"] == "atom" else d["index"].shape[1] for d in sub])
            rows = r if rows is None else np.maximum(rows, r)
            width = w if width is None else np.maximum(width, w)
        if skel is None:
            return None
        return {"steps": skel, "root": int(root), "rows": [int(x) for x in rows], "width": [int(x) for x in width],
                "atoms": [i for i, s in enumerate(skel) if s[0] == "atom"]}

    def candidate_counts(self):
        """Per family (K,) the row candidates at every period k = 0 .. L - T - 1."""
        K = self.L - self.T
        out = {f: np.zeros(K, np.int64) for f in FAMILIES}
        for ci in range(len(self.conjs)):
            s = self.anchors[ci]
            last = s + self.span[ci] - 1
            lo, hi = (last - self.T, last - 1) if self.family[ci] == "until" else (s, last - 1)
            lo, hi = np.clip(lo, 0, K), np.clip(hi + 1, 0, K)
            dlt = np.zeros(K + 1, np.int64)
            np.add.at(dlt, lo, 1)
            np.add.at(dlt, hi, -1)
            out[self.family[ci]] += np.cumsum(dlt)[:K]
        return out

    # ------------------------------------------------------------------
    # the record

    def predicates(self, X):
        with jax.enable_x64(True):
            return np.asarray(self._record(jnp.asarray(np.asarray(X, np.float64).reshape(-1, NX))))

    def node_exact(self, X):
        """(n, n_nodes) the record's node values at states X (n, 28)."""
        return self.predicates(X)[:, self.pidx].min(-1)

    def record(self, X, k):
        """Start the record on the states X with samples 0..k recorded."""
        self.progs = pg.Progressions([cj.program for cj in self.conjs], self.cnodes, self.delta)
        V = self.node_exact(X[:k + 1])
        for g, cols in zip(self.progs.groups, self.progs.cols):
            prog, G = g.program, cols.shape[0]
            rec = np.full((G, prog.T, prog.n_predicates), np.nan)
            n = min(k + 1, prog.T)
            rec[:, :n] = np.moveaxis(V[:n][:, cols], 1, 0)
            dec = pg.decisions(prog, rec, g.delta)
            res = pg.residual(prog, dec, np.full((G,), k, np.int64))
            g.d[:] = np.concatenate([x[0] for x in dec], -1)
            g.ok[:] = np.concatenate([x[1] for x in dec], -1)
            g.open[:] = np.concatenate([np.zeros((G, s.length), bool) if r is None else r[0] for s, r in zip(prog.steps, res)], -1)
            for i, r in enumerate(res):
                if r is not None:
                    g.masks[i][:] = r[1]
            g.k = k

    def update(self, X, t0):
        """Record the samples t0, t0 + 1, ... at states X (n, 28)."""
        if self.progs.k != t0 - 1:
            raise ValueError("the record holds samples up to " + str(self.progs.k) + "; the next is not " + str(t0))
        self.progs.update_block(self.node_exact(X))

    # ------------------------------------------------------------------
    # one control period

    def candidates(self, k):
        """Per row candidate at period k, in row order: (conjunct index, instance index, root row, family, anchor)."""
        out = []
        for ci, cj in enumerate(self.conjs):
            s = self.anchors[ci]
            last = s + self.span[ci] - 1
            cand = ((last >= k + 1) & (last <= k + self.T)) if self.family[ci] == "until" else ((s <= k) & (last >= k + 1))
            out += [(ci, int(r), int(cj.root_rows[r]), self.family[ci], int(s[r])) for r in np.flatnonzero(cand)]
        return out

    def rows(self, k):
        """candidates(k), each with its switch: on when the record has not decided it at k."""
        st = self.progs.states(k)
        return [(ci, r, root, f, s, bool(st[ci]["dec"][self.conjs[ci].root_step][0][root] > k)) for ci, r, root, f, s in self.candidates(k)]

    def period(self, k, X):
        """The data of period k on the states X (L, 28), the record holding samples 0..k: x0, qref, the cold start w_ref, the
        rows, their identities (conjunct, root row), perm (each row's slot) and per family the slot arrays."""
        T = self.T
        rows = self.rows(k)
        st = self.progs.states(k)
        masks, slots, used = {}, {}, {}
        for f in FAMILIES:
            lay, N = self.layout[f], self.n_slots[f]
            if lay is None:
                slots[f] = None
                continue
            rel0, idx0, msk0 = empty_slot(lay)
            slots[f] = {"rel": [None if r is None else np.tile(r, (N, 1)) for r in rel0], "idx": [None if x is None else np.tile(x, (N, 1, 1)) for x in idx0],
                        "msk": [None if m is None else np.tile(m, (N, 1, 1)) for m in msk0], "nid": np.zeros((N, 2), np.int64),
                        "on": np.zeros(N, bool), "beyond": np.zeros(N, bool), "real": [None if r is None else np.zeros((N, r.size), bool) for r in rel0]}
            used[f] = 0
        perm = []
        base = {"reach": 0, "until": self.n_slots["reach"]}
        for ci, r, root, f, s, on in rows:
            j = used[f]
            if j >= self.n_slots[f]:
                raise ValueError("more rows than slots at period " + str(k))
            used[f] += 1
            perm.append(base[f] + j)
            sl = slots[f]
            nd = self.cnodes[ci]
            sl["nid"][j] = (list(nd) + list(nd))[:2]
            if not on:
                continue
            cj = self.conjs[ci]
            if ci not in masks:
                masks[ci] = pg.evaluation_masks(cj.program, st[ci]["residual"])
            sub, rt, rrow = subprogram(cj.program, cj.root_step, root, masks[ci])
            if rt != self.layout[f]["root"] or rrow != 0:
                raise ValueError("an instance's root differs from its family's layout")
            rel, idx, msk = pad_slot(sub, self.layout[f], k)
            for i in range(len(rel)):
                if rel[i] is not None:
                    sl["rel"][i][j] = rel[i]
                    sl["real"][i][j, :sub[i]["rows"].size] = True
                else:
                    sl["idx"][i][j], sl["msk"][i][j] = idx[i], msk[i]
            sl["on"][j] = True
            sl["beyond"][j] = bool(np.any(np.concatenate([d["rows"] for d in sub if d["kind"] == "atom"]) > k + T))
        Xs = np.asarray(X, np.float64)
        Us = np.diff(Xs[k:k + 1 + T, NQ:], axis=0) / D.H_STEP
        return {"k": int(k), "x0": Xs[k].copy(), "qref": Xs[k + 1:k + 1 + T, :NQ].copy(), "w_ref": np.concatenate([Xs[k + 1:k + 1 + T].ravel(), Us.ravel()]),
                "rows": rows, "ids": [(ci, root) for ci, _, root, _, _, _ in rows], "perm": np.array(perm, np.int64), "slots": slots}

    def instances(self, pd):
        """[(conjunct name, anchor, on)] per row of a period's data."""
        return [(self.conjs[ci].name, s, on) for ci, _, _, _, s, on in pd["rows"]]


# ------------------------------------------------------------------
# the program with the period as parameters

class HorizonProgram:
    """The values and derivative parts of a Horizon's program, every function taking the period's parameters th."""

    def __init__(self, hz):
        scene, T = hz.scene, hz.T
        self.hz, self.scene, self.T = hz, scene, T
        self.n, self.n_def = (NX + NU) * T, NX * T
        joints = np.asarray(scene.joints)
        self.link, self.arm = joints.max(-1), joints.argmax(-1)
        keys = hz.node_keys
        self.n_nodes = len(keys)
        self.is_leaf = np.array([kk[0] == "leaf" for kk in keys])
        S = max([len(kk[1]) for kk in keys if kk[0] == "and"] or [1])
        ga = sorted(set(kk[2] for kk in keys if kk[0] == "and"))
        if len(ga) > 1 or any(len(kk[1]) != S for kk in keys if kk[0] == "and"):
            raise ValueError("the conjunction nodes share one smoothing parameter and one number of predicates")
        self.gamma_and = ga[0] if ga else 1.0
        self.pidx = np.array([list(kk[1]) if kk[0] == "and" else [kk[1][0]] * S for kk in keys], np.int64)
        self.node_arm = self.arm[self.pidx[:, 0]]
        n_wp, nd_ = len(scene.waypoints), D.DISTAL.size
        rel = self.pidx - n_wp
        cell = np.where(self.is_leaf, 0, rel[:, 0] // nd_)
        if np.any(~self.is_leaf[:, None] & ((rel < 0) | (rel // nd_ != cell[:, None]))):
            raise ValueError("a conjunction node reads predicates of more than one cell")
        self.node_presc = np.where(self.is_leaf, self.link[self.pidx[:, 0]], self.link[self.pidx].max(-1))
        self.lay, self.N = hz.layout, hz.n_slots
        self.fams = [f for f in FAMILIES if self.lay[f] is not None and self.N[f] > 0]
        self.n_slots = sum(self.N[f] for f in self.fams)
        self.base = {"reach": 0, "until": self.N["reach"] if "reach" in self.fams else 0}
        self.ent = {f: sum(self.lay[f]["rows"][i] for i in self.lay[f]["atoms"]) for f in self.fams}
        spans = [s for s, f in zip(hz.span, hz.family) if f == "reach"]
        self.full_reach = bool(spans) and max(spans) - 1 > T  # a reach window can extend past k + T
        with jax.enable_x64(True):
            self._sph = jnp.asarray(np.where(self.is_leaf[:, None], 0, D.DISTAL[np.where(rel >= 0, rel, 0) % nd_]))
            self._cell = jnp.asarray(cell)
            self._arm = jnp.asarray(self.node_arm)
            self._leaf_pred = jnp.asarray(self.pidx[:, 0])
            self._link = jnp.asarray(self.link[self.pidx])
            self._leaf_link = jnp.asarray(self.link[self.pidx[:, 0]])
            self._is_leaf = jnp.asarray(self.is_leaf)
            self.cells = jnp.asarray(scene.cells, jnp.float64)
        self._step = jax.vmap(D.interval_map(D.H_STEP))

    # ------------------------------------------------------------------
    # the period's parameters

    def theta(self, pd):
        th = {"x0": jnp.asarray(pd["x0"]), "qref": jnp.asarray(pd["qref"])}
        for f in self.fams:
            sl = pd["slots"][f]
            th[f] = {"rel": [None if r is None else jnp.asarray(r, jnp.int32) for r in sl["rel"]],
                     "idx": [None if x is None else jnp.asarray(x, jnp.int32) for x in sl["idx"]],
                     "msk": [None if m is None else jnp.asarray(m) for m in sl["msk"]],
                     "nid": jnp.asarray(sl["nid"], jnp.int32), "on": jnp.asarray(sl["on"]), "beyond": jnp.asarray(sl["beyond"])}
        return th

    def nid_host(self, pd):
        return np.concatenate([pd["slots"][f]["nid"] for f in self.fams]).astype(np.int64)

    def on_host(self, pd):
        return np.concatenate([pd["slots"][f]["on"] for f in self.fams])

    def nid_all(self, th):
        return jnp.concatenate([th[f]["nid"] for f in self.fams])

    def on_all(self, th):
        return jnp.concatenate([th[f]["on"] for f in self.fams])

    def prescribed_prefix(self, pd):
        """(n_slots, T, 2) per slot the prescribed pattern's joint prefix lengths (every operand of its cut program)."""
        T = self.T
        out = np.zeros((self.n_slots, T, 2), np.int64)
        for f in self.fams:
            sl, lay, b = pd["slots"][f], self.lay[f], self.base[f]
            for i in lay["atoms"]:
                r, e = np.nonzero(sl["real"][i] & sl["on"][:, None])
                n = sl["nid"][r, lay["steps"][i][1]]
                t = np.clip(sl["rel"][i][r, e], 0, T - 1)
                np.maximum.at(out, (b + r, t, self.node_arm[n]), self.node_presc[n])
        return out

    # ------------------------------------------------------------------
    # values

    def split(self, w):
        T = self.T
        return w[:NX * T].reshape(T, NX), w[NX * T:].reshape(T, NU)

    def objective(self, w, th):
        Xs, Us = self.split(w)
        E = Xs[:, :NQ] - th["qref"]
        return 0.5 * (W_Q * jnp.sum(E * E) + W_U * jnp.sum(Us * Us) + W_DU * jnp.sum((Us[1:] - Us[:-1]) ** 2))

    def defects(self, w, th):
        Xs, Us = self.split(w)
        prev = jnp.concatenate([th["x0"][None], Xs[:-1]])
        return (Xs - self._step(prev, Us)).ravel()

    def predicates(self, Xs, exact=False):
        """(P (T, n_pred), Ph (n_pred,)) at the horizon's samples and at the held state."""
        sc = self.scene
        held = jnp.concatenate([Xs[-1, :NQ], jnp.zeros(NQ, Xs.dtype)])[None]
        if exact:
            cells = self.cells.astype(Xs.dtype)
            f = jax.vmap(lambda x: jnp.concatenate([D.at_waypoint(x, sc.waypoints, sc.waypoint_arms, 0.0), D.outside(x, cells, sc.cell_radius, self.hz.exact_length)]))
        else:
            f = jax.vmap(sc.predicates)
        return f(Xs), f(held)[0]

    def nodes(self, P):
        """(values, support prefixes) of every per-sample node over predicate vectors P."""
        z = P[..., self.pidx]
        v, wt = extremum_and_weights(z, self.gamma_and, None, "min")
        pre = jnp.max(jnp.where(wt > 0, self._link, 0), -1)
        return jnp.where(self._is_leaf, z[..., 0], v), jnp.where(self._is_leaf, self._leaf_link, pre)

    def _sub(self, f, idx, msk):
        lay, sub = self.lay[f], []
        for i, (kind, atom, sign, sources) in enumerate(lay["steps"]):
            d = {"kind": kind, "rows": np.zeros(lay["rows"][i])}
            if kind == "atom":
                d.update(atom=atom, sign=sign)
            else:
                d.update(sources=list(sources), index=idx[i], mask=msk[i])
            sub.append(d)
        return sub

    def _traces(self, f, N, Nh, th):
        """Per atom step of family f, (slots, rows) the node values at its samples (the held state's after k + T)."""
        T, sl, lay = self.T, th[f], self.lay[f]
        out = []
        for i in lay["atoms"]:
            rel, nid = sl["rel"][i], sl["nid"][:, lay["steps"][i][1]][:, None]
            out.append(jnp.where(rel > T - 1, Nh[nid], N[jnp.clip(rel, 0, T - 1), nid]))
        return out

    def _slot_values(self, f, N, Nh, th, red):
        lay = self.lay[f]

        def one(trs, idx, msk):
            sub, tr = self._sub(f, idx, msk), dict(zip(lay["atoms"], trs))
            if red is None:
                return forward(sub, tr, self.hz.gamma[f])[0][lay["root"]][0]
            return sub_values(sub, tr, self.hz.gamma[f], red)[lay["root"]][0]
        return jax.vmap(one)(self._traces(f, N, Nh, th), th[f]["idx"], th[f]["msk"])

    def rows(self, w, th, exact=False):
        """(n_slots,) per slot the sparsemax robustness of its cut program, or with exact its exact robustness."""
        Xs, _ = self.split(w)
        P, Ph = self.predicates(Xs, exact)
        if exact:
            z, zh = P[..., self.pidx], Ph[..., self.pidx]
            N, Nh = jnp.where(self._is_leaf, z[..., 0], jnp.min(z, -1)), jnp.where(self._is_leaf, zh[..., 0], jnp.min(zh, -1))
            red = REDUCTIONS["exact"]
        else:
            N, Nh, red = self.nodes(P)[0], self.nodes(Ph)[0], None
        return jnp.concatenate([self._slot_values(f, N, Nh, th, red) for f in self.fams])

    def constraints(self, w, th):
        """The defects, then per slot its row less the margin, or OFF_VALUE for a switched-off or empty slot."""
        return jnp.concatenate([self.defects(w, th), jnp.where(self.on_all(th), self.rows(w, th) - self.hz.delta, OFF_VALUE)])

    def exact_rows(self, Xs, th):
        return self.rows(jnp.concatenate([jnp.asarray(Xs).ravel(), jnp.zeros(self.T * NU, Xs.dtype)]), th, True)

    # ------------------------------------------------------------------
    # part A: supports, path weights and compact skeleton Hessians

    def skeletons(self, w, th, hessian=True, ec=16):
        """(V, Pf (n_slots, T, 2), HZ or None, pn (T, n_nodes), bad) as sparse.supports.Supports.skeletons, on the slots."""
        T = self.T
        Xs, _ = self.split(w)
        P, Ph = self.predicates(Xs)
        N, pn = self.nodes(P)
        Nh = self.nodes(Ph)[0]
        V, Pf, HZ, bad = [], [], [], []
        for f in self.fams:
            lay, sl = self.lay[f], th[f]
            full = f == "reach" and self.full_reach
            ecg = self.ent[f] if full else ec

            def one(trs, idx, msk, f=f, lay=lay, ecg=ecg):
                return skeleton(self._sub(f, idx, msk), lay["root"], 0, dict(zip(lay["atoms"], trs)), self.hz.gamma[f], hessian, ecg)
            v, Pa, Hz, idx_c, over = jax.vmap(one)(self._traces(f, N, Nh, th), sl["idx"], sl["msk"])
            on = sl["on"]
            Pa = jnp.where(on[:, None], Pa, 0)
            rel = jnp.concatenate([sl["rel"][i] for i in lay["atoms"]], 1)
            a_all = np.concatenate([np.full(lay["rows"][i], lay["steps"][i][1]) for i in lay["atoms"]])
            fold = 2 * jnp.clip(rel, 0, T - 1) + a_all[None]
            G_ = Pa.shape[0]
            Pf.append(jnp.zeros((G_, 2 * T), w.dtype).at[np.arange(G_)[:, None], fold].add(Pa).reshape(G_, T, 2))
            bad.append(jnp.any((Pa != 0) & (rel < 0)))
            V.append(v)
            if hessian:
                HZ.append((jnp.where(on[:, None, None], Hz, 0), jnp.take_along_axis(fold, idx_c, 1)))
                if full:
                    bad.append(jnp.any(on & ~sl["beyond"] & (jnp.sum(jnp.abs(Pa) > 0, -1) > ec)))
                else:
                    bad.append(jnp.any(over & on))
        V, Pf = jnp.concatenate(V), jnp.concatenate(Pf)
        if hessian:
            ecm = max([h[0].shape[1] for h in HZ] + [ec])
            Hz = jnp.concatenate([jnp.pad(h[0], ((0, 0), (0, ecm - h[0].shape[1]), (0, ecm - h[0].shape[1]))) for h in HZ])
            fi = jnp.concatenate([jnp.pad(h[1], ((0, 0), (0, ecm - h[1].shape[1])), constant_values=-1) for h in HZ])
            HZ = (Hz, fi)
        else:
            HZ = None
        return V, Pf, HZ, pn, jnp.any(jnp.stack(bad))

    def row_prefix(self, Pf, pn, th):
        """(n_slots, T, 2) the supports' joint prefix lengths per slot, sample and arm."""
        nid = self.nid_all(th)
        val = jnp.where(jnp.abs(Pf) > 0, jnp.moveaxis(pn[:, nid], 0, 1), 0)
        arm = self._arm[nid]
        return jnp.stack([jnp.max(jnp.where(arm[:, None, :] == b, val, 0), -1) for b in (0, 1)], -1)

    def support_prefix(self, w, th):
        _, Pf, _, pn, _ = self.skeletons(w, th, hessian=False)
        return self.row_prefix(Pf, pn, th)

    def part_a(self, w, th, declared, ec):
        """Pf, HZ and whether the supports leave the declared prefix."""
        _, Pf, HZ, pn, bad = self.skeletons(w, th, True, ec)
        return Pf, HZ, jnp.any(self.row_prefix(Pf, pn, th) > declared) | bad

    # ------------------------------------------------------------------
    # part B: the nodes' derivatives at the active (sample, node) pairs

    def _conj(self, q, a, nodes, v):
        _, C = D.points(jax.lax.dynamic_update_slice(q, v, (a * NJ,)))
        sph = self._sph[nodes]
        R = self.scene.cell_radius
        dist = D._norm_minus(C[sph] - self.cells.astype(q.dtype)[self._cell[nodes]][:, None], D.EPS_LENGTH)
        z = (dist - jnp.asarray(D.RADIUS, q.dtype)[sph] - R) / R
        return lower_min_recount(z, self.gamma_and)

    def _leaf(self, q, n, v):
        x = jnp.concatenate([jax.lax.dynamic_update_slice(q, v, (self._arm[n] * NJ,)), jnp.zeros(NQ, q.dtype)])
        return self.scene.predicates(x)[self._leaf_pred[n]]

    def leaf_block(self, Q, N_, c=None):
        """The leaf nodes' gradients (pairs, 14) at positions Q and, with weights c, their weighted 7 x 7 Hessians."""
        qa = lambda q, n: jax.lax.dynamic_slice(q, (self._arm[n] * NJ,), (NJ,))  # noqa: E731
        G7 = jax.vmap(lambda q, n: jax.jacfwd(lambda v: self._leaf(q, n, v))(qa(q, n)))(Q, N_)
        col = self._arm[N_][:, None] * NJ + np.arange(NJ)[None]
        G = jnp.zeros((N_.size, NQ), Q.dtype).at[np.arange(N_.size)[:, None], col].set(G7)
        if c is None:
            return G, None, col
        Hl = jax.vmap(lambda q, n, ci: jax.hessian(lambda v: ci * self._leaf(q, n, v))(qa(q, n)))(Q, N_, c)
        return G, Hl, col

    def conj_block(self, w, cf, pr):
        """The conjunction pairs' gradients (ng m, 14) and, with cf (T, n_nodes), their weighted per-sample Hessians K."""
        Xs, _ = self.split(w)
        Q = Xs[pr["tc"], :NQ]
        ng, m = pr["nc"].shape
        col = pr["ac"][:, None] * NJ + np.arange(NJ)[None]
        qa = lambda q, a: jax.lax.dynamic_slice(q, (a * NJ,), (NJ,))  # noqa: E731
        if cf is None:
            G7 = jax.vmap(lambda q, a, nd: jax.jacfwd(lambda v: self._conj(q, a, nd, v))(qa(q, a)))(Q, pr["ac"], pr["nc"])
        else:
            c = jnp.where(pr["vc"], cf[pr["tc"][:, None], pr["nc"]], 0)

            def both(q, a, nd, ci):
                f = lambda v: self._conj(q, a, nd, v)  # noqa: E731
                return jax.jacfwd(lambda v: (f(v), jax.grad(lambda u: jnp.dot(ci, f(u)))(v)))(qa(q, a))
            G7, H7 = jax.vmap(both)(Q, pr["ac"], pr["nc"], c)
        G = jnp.zeros((ng, m, NQ), w.dtype).at[np.arange(ng)[:, None, None], np.arange(m)[None, :, None], col[:, None, :]].set(G7)
        if cf is None:
            return G.reshape(ng * m, NQ)
        K = jnp.zeros((self.T, NQ, NQ), w.dtype).at[pr["tc"][:, None, None], col[:, :, None], col[:, None, :]].add(H7)
        return G.reshape(ng * m, NQ), K

    def part_b(self, w, th, cf, pr):
        """Gc (pairs + 1, 14), the last row zero, and with cf the per-sample Hessians K (T, 14, 14)."""
        out = self.conj_block(w, cf, pr)
        G, K = (out, None) if cf is None else out
        parts = [G]
        if "tl" in pr:
            Xs, _ = self.split(w)
            if cf is None:
                Gl, _, _ = self.leaf_block(Xs[pr["tl"], :NQ], pr["nl"])
            else:
                Gl, Kl, coll = self.leaf_block(Xs[pr["tl"], :NQ], pr["nl"], jnp.where(pr["vl"], cf[pr["tl"], pr["nl"]], 0))
                K = K.at[pr["tl"][:, None, None], coll[:, :, None], coll[:, None, :]].add(Kl)
            parts.append(Gl)
        Gc = jnp.concatenate(parts + [jnp.zeros((1, NQ), w.dtype)])
        return Gc if cf is None else (Gc, K)

    def weights(self, th, mu, Pf):
        """(T, n_nodes) the multiplier-weighted path weights of every node."""
        return jnp.zeros((self.T, self.n_nodes), Pf.dtype).at[:, self.nid_all(th)].add(jnp.moveaxis(Pf * mu[:, None, None], 0, 1))

    # ------------------------------------------------------------------
    # part C: the assembly on the declared structures

    def part_c_jac(self, w, th, Pf, Gc, comp, jp, dpos, dp, njb):
        nid = self.nid_all(th)
        r, t, p = jp["r"], jp["t"], jp["p"]
        vs = Pf[r, t, 0] * Gc[comp[t, nid[r, 0]], p] + Pf[r, t, 1] * Gc[comp[t, nid[r, 1]], p]
        _, lin = jax.linearize(lambda v: self.defects(v, th), w)
        B = jax.lax.map(lin, dp["S"], batch_size=CHUNK)
        out = jnp.zeros(njb, w.dtype).at[dpos].set(B[dp["gc"], dp["ge"]])
        return out.at[jp["pos"]].set(vs)

    def part_c_hess(self, w, th, sigma, mu, HZ, Gc, comp, K, hp, cost, nb):
        Hz, fold = HZ
        nid = self.nid_all(th)
        node = jnp.take_along_axis(nid, jnp.maximum(fold, 0) % 2, 1)
        colT, colP = hp["colT"], hp["colP"]
        g = Gc[comp[colT[:, None, :], node[:, :, None]], colP[:, None, :]]
        on = (fold[:, :, None] >= 0) & (fold[:, :, None] // 2 == colT[:, None, :]) & hp["valid"][:, None, :]
        Gm = jnp.where(on, g, 0)
        Hr = jnp.einsum("rem,ref,rfn->rmn", Gm, Hz, Gm) * mu[:, None, None]
        out = jnp.zeros(nb + 1, w.dtype).at[hp["pos"]].add(Hr[hp["r"], hp["li"], hp["lj"]])
        out = out.at[hp["kpos"]].add(K[hp["kt"], hp["kp"], hp["kq"]])
        _, hvp = jax.linearize(jax.grad(lambda v: self.objective(v, th)), w)
        B = jax.lax.map(hvp, cost["S"], batch_size=CHUNK)
        return out.at[cost["pos"]].add(sigma * B[cost["gc"], cost["ge"]])

    def bc_jac(self, w, th, Pf, pr, comp, jp, dpos, dp, njb):
        return self.part_c_jac(w, th, Pf, self.part_b(w, th, None, pr), comp, jp, dpos, dp, njb)

    def bc_hess(self, w, th, sigma, mu, Pf, HZ, pr, comp, hp, cost, nb):
        Gc, K = self.part_b(w, th, self.weights(th, mu, Pf), pr)
        return self.part_c_hess(w, th, sigma, mu, HZ, Gc, comp, K, hp, cost, nb)

    def bounds(self, m):
        """(lbx, ubx, lbg, ubg) as mpc.program.PeriodProgram.bounds with m constraints."""
        T = self.T
        lo, hi = D.joint_bounds()
        v, a = D.VELOCITY_LIMIT, D.ACCELERATION_LIMIT
        lbx = np.concatenate([np.tile(np.concatenate([lo, np.full(NQ, -v)]), T), np.full(T * NU, -a)])
        ubx = np.concatenate([np.tile(np.concatenate([hi, np.full(NQ, v)]), T), np.full(T * NU, a)])
        term = (T - 1) * NX + NQ + np.arange(NQ)
        lbx[term], ubx[term] = 0.0, 0.0
        return lbx, ubx, np.zeros(m), np.concatenate([np.zeros(self.n_def), np.full(m - self.n_def, np.inf)])


# ------------------------------------------------------------------
# the lengths of the derivative calls: bounds over the horizon and levels

def rounded_lengths(need):
    """Each needed length rounded up on its own (powers of two)."""
    out = {}
    for a, v in need.items():
        if a == "ng":
            out[a] = max(8, 1 << int(np.ceil(np.log2(max(int(v), 1)))))
        elif a == "m":
            out[a] = max(1, 1 << int(np.ceil(np.log2(max(int(v), 1)))))
        elif a == "nl":
            out[a] = 0 if v == 0 else max(8, 1 << int(np.ceil(np.log2(max(int(v), 1)))))
        elif a == "ec":
            out[a] = v
        else:
            out[a] = bucket(v)
    return out


def cap(a, v, leaf):
    """A level's capacity for a needed length v of length a."""
    if a in ("m", "ec"):
        return max(int(np.ceil(v)), 1)
    if a == "nl" and not leaf:
        return 0
    return int(size(np.ceil(v)))


def sizes(lo, hi):
    """The padded lengths of sparse.supports.size from lo to the first at or above hi."""
    out, v = [], int(size(lo))
    while v < hi:
        out.append(v)
        v = int(size(v + 1))
    return out + [int(size(hi))]


def instance_rows(hz):
    """Per instance the first and last period at which it is a row candidate, and per atom row of its cut through every
    valid entry its instance, sample and node."""
    lo, hi, ri, rs, rn, base = [], [], [], [], [], 0
    for ci, cj in enumerate(hz.conjs):
        s = hz.anchors[ci]
        last = s + hz.span[ci] - 1
        lo.append(last - hz.T if hz.family[ci] == "until" else s)
        hi.append(last - 1)
        roots = np.asarray(cj.root_rows, np.int64)[None]
        reached = pg.reach(cj.program, cj.root_step, roots, [None if m is None else m[None] for m in full_masks(cj.program)])
        nd = list(hz.cnodes[ci]) * 2
        for i, st in enumerate(cj.program.steps):
            if st.kind != "atom" or reached[i] is None:
                continue
            r, smp = np.nonzero(reached[i][0])
            ri.append(base + r)
            rs.append(smp)
            rn.append(np.full(r.size, nd[st.atom], np.int64))
        base += s.size
    return tuple(np.concatenate(x).astype(np.int64) for x in (lo, hi, ri, rs, rn))


def period_lengths(hz, pb, ks=None):
    """Per period (default every period 0 .. L - T - 1) the lengths of the structure with every candidate switched on and
    cut through every valid entry: nz, np, M, e (prescribed columns) and ng, m, nl (part B's pairs)."""
    T, K = hz.T, hz.L - hz.T
    lo, hi, ri, rs, rn = instance_rows(hz)
    ks = np.arange(K) if ks is None else np.asarray(ks, np.int64)
    sel = np.zeros(K, bool)
    sel[ks] = True
    lo, hi = np.clip(lo, 0, K), np.clip(hi, -1, K - 1)
    nk = np.maximum(hi - lo + 1, 0)
    pi = np.repeat(np.arange(lo.size), nk)
    pk = lo[pi] + np.arange(pi.size) - np.repeat(np.cumsum(nk) - nk, nk)
    keep = sel[pk]
    pi, pk = pi[keep], pk[keep]
    order = np.argsort(ri, kind="stable")
    rs, rn = rs[order], rn[order]
    cnt = np.bincount(ri, minlength=lo.size)
    start = np.cumsum(cnt) - cnt
    mr = cnt[pi]
    ep_ = np.repeat(np.arange(pi.size), mr)
    er = start[pi][ep_] + np.arange(ep_.size) - np.repeat(np.cumsum(mr) - mr, mr)
    t = np.clip(rs[er] - pk[ep_] - 1, 0, T - 1)
    n = rn[er]
    arm, presc = pb.node_arm[n], pb.node_presc[n]
    u1, inv1 = np.unique((ep_ * T + t) * 2 + arm, return_inverse=True)
    pre = np.zeros(u1.size, np.int64)
    np.maximum.at(pre, inv1, presc)
    pair = u1 // (2 * T)
    c = np.bincount(pair, weights=pre, minlength=pi.size).astype(np.int64)
    e = np.bincount(pair, weights=pre > 0, minlength=pi.size).astype(np.int64)
    out = {"nz": np.bincount(pk, weights=c, minlength=K).astype(np.int64), "np": np.bincount(pk, weights=c * (c + 1) // 2, minlength=K).astype(np.int64),
           "M": np.zeros(K, np.int64), "e": np.zeros(K, np.int64)}
    np.maximum.at(out["M"], pk, c)
    np.maximum.at(out["e"], pk, e)
    u2 = np.unique((pk[ep_] * T + t) * pb.n_nodes + n)
    k2, t2, n2 = u2 // (T * pb.n_nodes), (u2 // pb.n_nodes) % T, u2 % pb.n_nodes
    lf = pb.is_leaf[n2]
    g, gc = np.unique((k2[~lf] * T + t2[~lf]) * 2 + pb.node_arm[n2[~lf]], return_counts=True)
    out["ng"] = np.bincount(g // (2 * T), minlength=K).astype(np.int64)
    out["m"] = np.zeros(K, np.int64)
    np.maximum.at(out["m"], g // (2 * T), gc)
    out["nl"] = np.bincount(k2[lf], minlength=K).astype(np.int64)
    return {a: v[ks] for a, v in out.items()}


def fixed_parts(T):
    """The defects' entries plus one, the cost's upper entries plus one."""
    return {"nj": SP.defect_pattern(T)[0].size + 1, "nh": SP.cost_pattern(T)[0].size + 1}


def horizon_bounds(hz, pb, ks=None):
    """The bound of every length of the Jacobian and Hessian calls over the horizon's periods, part A's compact entry
    counts and the widths part A gives."""
    L = period_lengths(hz, pb, ks)
    big = {a: int(max(v.max(initial=0), 0)) for a, v in L.items()}
    ec_top = bucket(max(big["e"], 1))
    ecs = [8 << i for i in range(int(np.log2(ec_top // 8)) + 1)]
    E = pb.ent["reach"] if ("reach" in pb.fams and pb.full_reach) else 0
    widths = sorted({max(E, ec) for ec in ecs})
    fixed = fixed_parts(hz.T)
    pairs = {"ng": max(big["ng"], 1), "m": max(big["m"], 1), "nl": big["nl"]}
    jac = dict(pairs, nz=big["nz"], nj=fixed["nj"] + big["nz"])
    hess = dict(pairs, np=big["np"], nk=NQ * (NQ + 1) // 2 * hz.T, M=max(big["M"], 1), nh=fixed["nh"] + big["np"], ec=widths[-1])
    return {"jac": jac, "hess": hess}, ecs, widths


def levels(B, leaf, widths, fixed, n_slots, obs=None):
    """The levels of part B's conjunction groups (ng), its leaf pairs (nl) and part C's declaration (decl), each ascending
    with the last at the bounds B; obs, the needed lengths recorded before, places the declaration's smallest level."""
    H = B["hess"]
    hess_obs = [o for o in (obs or {}).get("hess", []) if o["M"] > 1]
    kappa = n_slots if not hess_obs else min(n_slots, 2 * max(o["np"] / (o["M"] * (o["M"] + 1) / 2) for o in hess_obs))
    floor_M = int(max(8, np.ceil(H["M"] / 64), min([o["M"] // 2 for o in hess_obs], default=8)))
    ng = [8 << i for i in range(int(np.log2(max(H["ng"], 8) / 8)) + 1) if 8 << i < H["ng"]] + [cap("ng", H["ng"], leaf)]
    nl = sizes(8, H["nl"]) if leaf else [0]
    npb = cap("np", H["np"], leaf)
    decl = []
    for M in sizes(min(floor_M, H["M"]), H["M"]):
        e = min(w for w in widths if w >= min(widths[-1], bucket(int(np.ceil(M / NJ)))))
        npr = min(npb, cap("np", kappa * M * (M + 1) / 2, leaf))
        decl.append({"M": M, "ec": e, "np": npr, "nh": fixed["nh"] + npr})
    decl[-1].update(ec=widths[-1], np=npb, nh=fixed["nh"] + npb)
    nz = cap("nz", B["jac"]["nz"], leaf)
    return {"ng": sorted(set(ng)), "nl": nl, "decl": decl, "m": H["m"], "nk": H["nk"], "nz": nz, "nj": fixed["nj"] + nz}


def point_need(pb, key):
    """The needed lengths of part B's pairs for the (sample, node) keys."""
    tn, nn = key // pb.n_nodes, key % pb.n_nodes
    lf = pb.is_leaf[nn]
    _, cnt = np.unique(tn[~lf] * 2 + pb.node_arm[nn[~lf]], return_counts=True)
    return {"ng": max(int(cnt.size), 1), "m": int(max(cnt.max(initial=0), 1)), "nl": int(lf.sum())}


def group_pairs_cap(pb, key, ng, m, nl, leaf, rows_c, rows_l):
    """sparse.assembly.group_pairs at the given lengths: ng groups of m nodes and nl leaf pairs; the leaf pairs' rows from
    rows_c on, the zero row at rows_c + rows_l."""
    tn, nn = key // pb.n_nodes, key % pb.n_nodes
    lf = pb.is_leaf[nn]
    tc, nc = tn[~lf], nn[~lf]
    ta = tc * 2 + pb.node_arm[nc]
    order = np.lexsort((nc, ta))
    tc, nc, ta = tc[order], nc[order], ta[order]
    u, start, cnt = np.unique(ta, return_index=True, return_counts=True)
    if u.size == 0:  # no conjunction pair: one padded group reading a conjunction node, with weight zero
        u, start, cnt = np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64)
        nc = np.array([int(np.flatnonzero(~pb.is_leaf)[0]) if (~pb.is_leaf).any() else 0])
        ta = np.array([2 * int(pb.node_arm[nc[0]])])
    tl, nn_l = tn[lf], nn[lf]
    if u.size > ng or cnt.max() > m or tl.size > nl or (tl.size and not leaf):
        raise ValueError("the pairs exceed the capacities")
    g = np.repeat(np.arange(u.size), cnt)
    j = np.arange(g.size) - start[g]
    nodes = np.zeros((ng, m), np.int64)
    nodes[:u.size] = nc[start][:, None]
    nodes[u.size:] = nodes[0]
    nodes[g, j] = nc[:g.size]
    valid = np.zeros((ng, m), bool)
    valid[g, j] = True
    tg = np.zeros(ng, np.int64)
    tg[:u.size], tg[u.size:] = u // 2, u[0] // 2
    ag = np.zeros(ng, np.int64)
    ag[:u.size], ag[u.size:] = u % 2, u[0] % 2
    nlp = nl if leaf else 0
    comp = np.full((pb.T, pb.n_nodes), rows_c + rows_l, np.int32)
    comp[tc[:g.size], nc[:g.size]] = g * m + j
    comp[tl, nn_l] = rows_c + np.arange(tl.size)
    prc = {"tc": jnp.asarray(tg.astype(np.int32)), "ac": jnp.asarray(ag.astype(np.int32)), "nc": jnp.asarray(nodes.astype(np.int32)), "vc": jnp.asarray(valid)}
    prl = {}
    if leaf:
        first = int(nn_l[0]) if tl.size else int(np.flatnonzero(pb.is_leaf)[0])
        prl = {"tl": pad(tl, nlp), "nl": pad(nn_l, nlp, first), "vl": jnp.asarray(np.arange(nlp) < tl.size)}
    return prc, prl, jnp.asarray(comp)


def pad_hz(HZ, e):
    """The compact skeleton Hessians padded to width e."""
    Hz, fold = HZ
    k = e - Hz.shape[1]
    return jnp.pad(Hz, ((0, 0), (0, k), (0, k))), jnp.pad(fold, ((0, 0), (0, k)), constant_values=-1)


# ------------------------------------------------------------------
# IPOPT on the horizon's program

class HorizonSolver:
    """IPOPT on a Horizon's program with the supports' Hessian structure; set_period installs a period, solve solves it;
    after set_bounds, set_levels and compile_levels no call compiles; options: further IPOPT options."""

    def __init__(self, hz, max_segments=50, **options):
        self.hz, self.T, self.max_segments, self.options = hz, hz.T, max_segments, options
        self.lv, self.bounds, self.obs = None, None, {"jac": [], "hess": []}
        self.ecs, self.widths = [8, 16, 32, 64], None
        self.overflow = {"jac": 0, "hess": 0}
        self.version = 0
        with jax.enable_x64(True):
            self.pb = pb = HorizonProgram(hz)
            self._f = jax.jit(pb.objective)
            self._grad = jax.jit(jax.grad(pb.objective))
            self._g = jax.jit(pb.constraints)
            self._prefix = jax.jit(pb.support_prefix)
            self._exact = jax.jit(pb.exact_rows)
            self.a0 = jax.jit(pb.part_a, static_argnums=(3,))
            self.bc_jac_call = jax.jit(pb.bc_jac, static_argnums=(8,))
            self.bc_hess_call = jax.jit(pb.bc_hess, static_argnums=(10,))
            self.conj_call = jax.jit(self._conj_part, static_argnums=(5,))
            self.leaf_call = jax.jit(self._leaf_part, static_argnums=(5,))
            self.cj_call = jax.jit(self._cj, static_argnums=(9,))
            self.ch_call = jax.jit(self._ch, static_argnums=(12,))
            self._padhz = jax.jit(pad_hz, static_argnums=(1,))
            self.defects = SP.defect_pattern(self.T)
            self.cost_up = SP.cost_pattern(self.T)
            dp = jacobian_plan(self.defects, (pb.n_def, pb.n), "fwd", jnp.float64)
            self.dp = {a: dp[a] for a in ("S", "gc", "ge")}
            cp = hessian_plan(self.cost_up, pb.n, jnp.float64)
            self._cplan = {a: cp[a] for a in ("S", "gc", "ge")}
            self._gl0 = jnp.zeros((0, NQ))
            self._k0 = jnp.zeros((self.T, NQ, NQ))
        self.fixed = fixed_parts(hz.T)
        self.leaf = bool(pb.is_leaf.any())

    # ------------------------------------------------------------------
    # the three parts of the calls at levels

    def _conj_part(self, w, th, mu, Pf, pr, rows):
        cf = None if mu is None else self.pb.weights(th, mu, Pf)
        out = self.pb.conj_block(w, cf, pr)
        G = out if cf is None else out[0]
        G = jnp.pad(G, ((0, rows - G.shape[0]), (0, 0)))
        return G if cf is None else (G, out[1])

    def _leaf_part(self, w, th, mu, Pf, pr, rows):
        pb = self.pb
        Xs, _ = pb.split(w)
        Xl = Xs[pr["tl"], :NQ]
        if mu is None:
            Gl, _, _ = pb.leaf_block(Xl, pr["nl"])
            return jnp.pad(Gl, ((0, rows - Gl.shape[0]), (0, 0)))
        cf = pb.weights(th, mu, Pf)
        Gl, Kl, coll = pb.leaf_block(Xl, pr["nl"], jnp.where(pr["vl"], cf[pr["tl"], pr["nl"]], 0))
        K = jnp.zeros((self.T, NQ, NQ), w.dtype).at[pr["tl"][:, None, None], coll[:, :, None], coll[:, None, :]].add(Kl)
        return jnp.pad(Gl, ((0, rows - Gl.shape[0]), (0, 0))), K

    def _cj(self, w, th, Pf, G, Gl, comp, jp, dpos, dp, njb):
        Gc = jnp.concatenate([G, Gl, jnp.zeros((1, NQ), w.dtype)])
        return self.pb.part_c_jac(w, th, Pf, Gc, comp, jp, dpos, dp, njb)

    def _ch(self, w, th, sigma, mu, HZ, G, Gl, comp, K1, K2, hp, cost, nb):
        Gc = jnp.concatenate([G, Gl, jnp.zeros((1, NQ), w.dtype)])
        return self.pb.part_c_hess(w, th, sigma, mu, HZ, Gc, comp, K1 + K2, hp, cost, nb)

    # ------------------------------------------------------------------
    # bounds and levels

    def set_bounds(self, ks=None):
        """The bounds of every length over the periods ks (default every period of the episode)."""
        self.bounds, self.ecs, self.widths = horizon_bounds(self.hz, self.pb, ks)

    def set_levels(self, lv=None):
        """Fix the levels: from the bounds and the needed lengths recorded so far, or as given."""
        if lv is None:
            lv = levels(self.bounds, self.leaf, self.widths, self.fixed, self.pb.n_slots, self.obs)
        self.lv = lv
        self.rows_c = max(lv["ng"]) * lv["m"]
        self.rows_l = max(lv["nl"]) if self.leaf else 0
        return lv

    def _caps(self, kind, need):
        """The level index of each part for a call, or None for lengths rounded up on their own."""
        if self.lv is None:
            self.obs[kind].append(need)
            return None
        lv = self.lv
        i = next((x for x, v in enumerate(lv["ng"]) if need["ng"] <= v), None)
        j = next((x for x, v in enumerate(lv["nl"]) if need["nl"] <= v), None)
        if kind == "jac":
            d, ok = -1, need["m"] <= lv["m"] and need["nz"] <= lv["nz"] and need["nj"] <= lv["nj"]
        else:
            d = next((x for x, v in enumerate(lv["decl"]) if all(need[a] <= v[a] for a in ("M", "ec", "np", "nh"))), None)
            ok = need["m"] <= lv["m"] and need["nk"] <= lv["nk"]
        if i is None or j is None or d is None or not ok:
            self.overflow[kind] += 1
            return None
        return i, j, d

    def compile_levels(self, w):
        """Compile part A at every compact entry count and parts B and C at every level, at w on the installed period."""
        with jax.enable_x64(True):
            pb, lv = self.pb, self.lv
            w = jnp.asarray(w)
            z = jnp.zeros((pb.n_slots, self.T, 2), jnp.int32)
            ecs = sorted(set(self.ecs) | {8 << i for i in range(int(np.log2(max(64, max(d["ec"] for d in lv["decl"])) // 8)) + 1)})
            widths, Pf = {}, None
            for ec in ecs:
                Pf, HZ, _ = jax.block_until_ready(self.a0(w, self.th, z, ec))
                widths[int(HZ[0].shape[1])] = HZ
            for e0, HZ in widths.items():
                for e in sorted({d["ec"] for d in lv["decl"] if d["ec"] > e0}):
                    jax.block_until_ready(self._padhz(HZ, e))
            none = np.zeros(0, np.int64)
            mu, sigma = jnp.asarray(np.zeros(pb.n_slots)), jnp.asarray(1.0, w.dtype)
            for ng in lv["ng"]:
                prc, _, comp = group_pairs_cap(pb, none, ng, lv["m"], lv["nl"][0], self.leaf, self.rows_c, self.rows_l)
                G = jax.block_until_ready(self.conj_call(w, self.th, None, None, prc, self.rows_c))
                G, K1 = jax.block_until_ready(self.conj_call(w, self.th, mu, Pf, prc, self.rows_c))
            Gl, K2 = self._gl0, self._k0
            for nl in (lv["nl"] if self.leaf else []):
                _, prl, comp = group_pairs_cap(pb, none, lv["ng"][0], lv["m"], nl, self.leaf, self.rows_c, self.rows_l)
                Gl = jax.block_until_ready(self.leaf_call(w, self.th, None, None, prl, self.rows_l))
                Gl, K2 = jax.block_until_ready(self.leaf_call(w, self.th, mu, Pf, prl, self.rows_l))
            jp = self._jac_arrays({"nz": lv["nz"]}, {a: none for a in ("r", "t", "p", "pos")}, 0)
            jax.block_until_ready(self.cj_call(w, self.th, Pf, G, Gl, comp, jp, self._dpos, self.dp, lv["nj"]))
            HZ = next(iter(widths.values()))
            for d in lv["decl"]:
                raw = {"hp": {a: none for a in ("r", "ti", "pi", "tj", "pj", "pos", "li", "lj")}, "k": {a: none for a in ("kpos", "kt", "kp", "kq")},
                       "r2": none, "c2": none, "loc": none}
                hp = self._hess_arrays({"np": d["np"], "nk": lv["nk"], "M": d["M"]}, raw, 0)
                Z = (jnp.zeros((pb.n_slots, d["ec"], d["ec"]), HZ[0].dtype), jnp.full((pb.n_slots, d["ec"]), -1, HZ[1].dtype))
                jax.block_until_ready(self.ch_call(w, self.th, sigma, mu, Z, G, Gl, comp, K1, K2, hp, self.cost, d["nh"]))

    # ------------------------------------------------------------------
    # the control period and the declared structures

    def set_period(self, pd):
        """Install period pd (Horizon.period): its parameters, its rows and the prescribed Jacobian structure."""
        pb = self.pb
        with jax.enable_x64(True):
            self.pd, self.th = pd, pb.theta(pd)
        self.perm = np.asarray(pd["perm"], np.int64)
        self.m = pb.n_def + self.perm.size
        self.inv = np.full(pb.n_slots, -1, np.int64)
        self.inv[self.perm] = np.arange(self.perm.size)
        self.keys = list(pd["ids"])
        self.nid = pb.nid_host(pd)
        self.on = pb.on_host(pd)
        self.cap = pb.prescribed_prefix(pd)
        r, c = SP.columns(self.cap)
        dr, dc = self.defects
        key = np.unique(np.concatenate([dr * pb.n + dc, (pb.n_def + self.inv[r]) * pb.n + c]))
        self.J_pre = (key // pb.n, key % pb.n)
        self.cache = (None, None)

    def ipopt_rows(self, g_slots):
        """IPOPT's constraint vector (defects, then the period's rows) from the slot vector."""
        g = np.asarray(g_slots)
        return np.concatenate([g[:self.pb.n_def], g[self.pb.n_def + self.perm]])

    def slot_multipliers(self, lam_g):
        mu = np.zeros(self.pb.n_slots)
        mu[self.perm] = np.asarray(lam_g, np.float64).ravel()[self.pb.n_def:]
        return jnp.asarray(mu)

    def declare(self, pre):
        """Declare the supports' prefix pre (n_slots, T, 2): the Hessian structure (the slots' cliques and the cost's) and
        the index arrays of part C at their own lengths; the Jacobian structure is the period's prescribed one."""
        pb, n, nd = self.pb, self.pb.n, self.pb.n_def
        self.version += 1
        pre = np.asarray(pre, np.int64)
        r, c = SP.columns(pre)
        Hu = SP.hessian_union(r, c, pb.n_slots, n)
        hr, hc = Hu.nonzero()
        cr, cc = self.cost_up
        H = SP.upper(np.concatenate([hr, cr]), np.concatenate([hc, cc]), n)
        J = self.J_pre
        self.declared = jnp.asarray(pre, jnp.int32)
        self.declared_np = pre
        self.ec = bucket(int((pre > 0).sum((1, 2)).max(initial=1)))
        self.jac_structure, self.hess_structure = J, H
        dr, dc = self.defects
        self._dpos = jnp.asarray(SP.positions(J, dr, dc, n))
        spec = J[0] >= nd
        rj, cj = J[0][spec] - nd, J[1][spec]
        self.nj = int(J[0].size)
        jp = {"r": self.perm[rj], "t": cj // NX, "p": cj % NX, "pos": SP.positions(J, nd + rj, cj, n)}
        order = np.argsort(r, kind="stable")
        rs, cs = r[order], c[order]
        start = np.searchsorted(rs, rs, "left")
        cnt = np.searchsorted(rs, rs, "right") - start
        qr, qa = np.repeat(rs, cnt), np.repeat(cs, cnt)
        off = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        qb = cs[np.repeat(start, cnt) + off]
        keep = qa <= qb
        qr, qa, qb = qr[keep], qa[keep], qb[keep]
        self.nnz = nnz = H[0].size
        same = (H[0] < nd) & (H[1] < nd) & (H[0] // NX == H[1] // NX) & (H[0] % NX < NQ) & (H[1] % NX < NQ)
        ks = np.flatnonzero(same)
        o2 = np.lexsort((c, r))
        r2, c2 = r[o2], c[o2]
        st2 = np.searchsorted(r2, np.arange(pb.n_slots))
        loc = np.arange(r2.size) - st2[r2]
        key = r2 * n + c2
        hp = {"r": qr, "ti": qa // NX, "pi": qa % NX, "tj": qb // NX, "pj": qb % NX, "pos": SP.positions(H, qa, qb, n),
              "li": np.searchsorted(key, qr * n + qa) - st2[qr], "lj": np.searchsorted(key, qr * n + qb) - st2[qr]}
        self.raw = {"jp": jp, "hp": hp, "k": {"kpos": ks, "kt": H[0][ks] // NX, "kp": H[0][ks] % NX, "kq": H[1][ks] % NX}, "r2": r2, "c2": c2, "loc": loc}
        self.need = {"nz": int(rj.size), "nj": self.nj + 1, "np": int(qr.size), "nk": int(ks.size), "M": int(loc.max(initial=0)) + 1, "nh": nnz + 1}
        self.arrays = {}
        self.cost = dict(self._cplan, pos=jnp.asarray(SP.positions(H, cr, cc, n)))

    def _jac_arrays(self, caps, jp, nj):
        return {a: pad(jp[a], caps["nz"], nj if a == "pos" else 0) for a in ("r", "t", "p", "pos")}

    def _hess_arrays(self, caps, raw, nnz):
        hp = {a: pad(raw["hp"][a], caps["np"], nnz if a == "pos" else 0) for a in ("r", "ti", "pi", "tj", "pj", "pos", "li", "lj")}
        hp.update({a: pad(raw["k"][a], caps["nk"], nnz if a == "kpos" else 0) for a in ("kpos", "kt", "kp", "kq")})
        M, ns = caps["M"], self.pb.n_slots
        colT, colP, valid = np.zeros((ns, M), np.int32), np.zeros((ns, M), np.int32), np.zeros((ns, M), bool)
        colT[raw["r2"], raw["loc"]], colP[raw["r2"], raw["loc"]], valid[raw["r2"], raw["loc"]] = raw["c2"] // NX, raw["c2"] % NX, True
        hp.update({"colT": jnp.asarray(colT), "colP": jnp.asarray(colP), "valid": jnp.asarray(valid)})
        return hp

    def _arrays(self, kind, caps):
        k = (kind,) + tuple(sorted(caps.items()))
        if k not in self.arrays:
            self.arrays[k] = self._jac_arrays(caps, self.raw["jp"], self.nj) if kind == "jac" else self._hess_arrays(caps, self.raw, self.nnz)
        return self.arrays[k]

    # ------------------------------------------------------------------
    # the derivatives at a point

    def _point(self, w, xkey):
        """Part A at w and the point's active (sample, node) keys, once per point and declaration."""
        if self.cache[0] is not None and self.cache[0] == (xkey, self.version):
            return self.cache[1]
        Pf, HZ, esc = jax.block_until_ready(self.a0(w, self.th, self.declared, self.ec))
        r, tt, a = np.nonzero(np.abs(np.asarray(Pf)) > 0)
        key = np.unique(tt * self.pb.n_nodes + self.nid[r, a])
        out = (Pf, HZ, bool(esc), {"key": key, "need": point_need(self.pb, key), "pairs": {}, "hz": {}})
        self.cache = ((xkey, self.version), out)
        return out

    def _pairs(self, pt, k):
        if k not in pt["pairs"]:
            if k is None:
                pt["pairs"][k] = group_pairs(self.pb, pt["key"])
            else:
                pt["pairs"][k] = group_pairs_cap(self.pb, pt["key"], k[0], self.lv["m"], k[1], self.leaf, self.rows_c, self.rows_l)
        return pt["pairs"][k]

    def jacobian_at(self, w, xkey):
        Pf, _, esc, pt = self._point(w, xkey)
        need = dict(pt["need"], nz=self.need["nz"], nj=self.need["nj"])
        idx = self._caps("jac", need)
        if idx is None:
            caps = rounded_lengths(need)
            pr, comp = self._pairs(pt, None)
            v = self.bc_jac_call(w, self.th, Pf, pr, comp, self._arrays("jac", caps), self._dpos, self.dp, caps["nj"])
        else:
            lv = self.lv
            prc, prl, comp = self._pairs(pt, (lv["ng"][idx[0]], lv["nl"][idx[1]]))
            G = self.conj_call(w, self.th, None, None, prc, self.rows_c)
            Gl = self.leaf_call(w, self.th, None, None, prl, self.rows_l) if self.leaf else self._gl0
            v = self.cj_call(w, self.th, Pf, G, Gl, comp, self._arrays("jac", {"nz": lv["nz"]}), self._dpos, self.dp, lv["nj"])
        return np.asarray(jax.block_until_ready(v))[:self.nj], esc

    def hessian_at(self, w, xkey, sigma, mu):
        Pf, HZ, esc, pt = self._point(w, xkey)
        need = dict(pt["need"], ec=int(HZ[0].shape[1]), **{a: self.need[a] for a in ("np", "nk", "M", "nh")})
        idx = self._caps("hess", need)
        s = jnp.asarray(sigma, w.dtype)
        if idx is None:
            caps = rounded_lengths(need)
            pr, comp = self._pairs(pt, None)
            v = self.bc_hess_call(w, self.th, s, mu, Pf, HZ, pr, comp, self._arrays("hess", caps), self.cost, caps["nh"])
        else:
            lv = self.lv
            d = lv["decl"][idx[2]]
            prc, prl, comp = self._pairs(pt, (lv["ng"][idx[0]], lv["nl"][idx[1]]))
            hp = self._arrays("hess", {"np": d["np"], "nk": lv["nk"], "M": d["M"]})
            if d["ec"] > need["ec"]:
                if d["ec"] not in pt["hz"]:
                    pt["hz"][d["ec"]] = self._padhz(HZ, d["ec"])
                HZ = pt["hz"][d["ec"]]
            G, K1 = self.conj_call(w, self.th, mu, Pf, prc, self.rows_c)
            Gl, K2 = self.leaf_call(w, self.th, mu, Pf, prl, self.rows_l) if self.leaf else (self._gl0, self._k0)
            v = self.ch_call(w, self.th, s, mu, HZ, G, Gl, comp, K1, K2, hp, self.cost, d["nh"])
        return np.asarray(jax.block_until_ready(v))[:self.nnz], esc

    # ------------------------------------------------------------------
    # IPOPT's callbacks

    def _x(self, x):
        xn = np.asarray(x, np.float64).ravel()
        return xn, xn.tobytes()

    def _f_cb(self, x):
        return [float(self._f(jnp.asarray(self._x(x)[0]), self.th))]

    def _g_cb(self, x):
        xn, key = self._x(x)
        v = self.ipopt_rows(self._g(jnp.asarray(xn), self.th))
        self.last_g = (key, v)
        return [v]

    def _grad_cb(self, x):
        w = jnp.asarray(self._x(x)[0])
        return [float(self._f(w, self.th)), np.asarray(self._grad(w, self.th))]

    def _jac_cb(self, x):
        xn, key = self._x(x)
        w = jnp.asarray(xn)
        g = self.last_g[1] if self.last_g[0] == key else self.ipopt_rows(self._g(w, self.th))
        v, esc = self.jacobian_at(w, key)
        self.last = (key, esc)
        return [g, v[self.obj["J_order"]]]

    def _hess_cb(self, x, lam_f, lam_g):
        xn, key = self._x(x)
        v, _ = self.hessian_at(jnp.asarray(xn), key, float(lam_f), self.slot_multipliers(lam_g))
        return v[self.obj["H_order"]]

    def _stop(self, x):
        """At an accepted iterate: whether its supports leave the declared prefix (IPOPT then stops there)."""
        xn, key = self._x(x)
        if self.last[0] == key:
            return bool(self.last[1])
        return bool(np.any(np.asarray(self._prefix(jnp.asarray(xn), self.th)) > self.declared_np))

    def _object(self, warm, mu):
        pb = self.pb
        n, m = pb.n, self.m
        J_sp, J_order = ccs(self.J_pre[0], self.J_pre[1], m, n)
        x_sp, p_sp = ca.Sparsity.dense(n, 1), ca.Sparsity.dense(0, 1)
        f_sp, g_sp = ca.Sparsity.dense(1, 1), ca.Sparsity.dense(m, 1)
        fns = {"F": Function("F", [x_sp], [f_sp], self._f_cb), "G": Function("G", [x_sp], [g_sp], self._g_cb, J_sp),
               "grad_f": Function("grad_f", [x_sp, p_sp], [f_sp, ca.Sparsity.dense(n, 1)], self._grad_cb),
               "jac_g": Function("jac_g", [x_sp, p_sp], [g_sp, J_sp], self._jac_cb), "iteration": Iteration(n, m, self._stop)}
        extra = dict(self.options) if mu is None else dict(self.options, mu_init=mu)
        opts = ipopt_options("exact", warm, **extra)
        opts.update({"grad_f": fns["grad_f"], "jac_g": fns["jac_g"], "iteration_callback": fns["iteration"], "no_nlp_grad": True})
        H = self.hess_structure
        H_sp, H_order = ccs(H[0], H[1], n, n)
        fns["hess_lag"] = Hessian("hess_lag", [x_sp, p_sp, f_sp, g_sp], [H_sp], self._hess_cb)
        opts["hess_lag"] = fns["hess_lag"]
        xs = ca.MX.sym("x", n)
        S = ca.nlpsol("horizon", "ipopt", {"x": xs, "f": fns["F"](xs), "g": fns["G"](xs)}, opts)
        self.obj = {"S": S, "fns": fns, "J_order": J_order, "H": H, "H_order": H_order}

    def solve(self, w0, lam_g0=None, lam_x0=None, warm=False, carry=None):
        """One solve of the installed period from w0 (cold, or IPOPT's warm start from the multipliers); carry, a declared
        prefix per slot joined with the supports at w0. Returns a summary; the solution is kept as x, lam_g, lam_x."""
        with jax.enable_x64(True):
            return self._solve(w0, lam_g0, lam_x0, warm, carry)

    def _solve(self, w0, lam_g0, lam_x0, warm, carry):
        pb = self.pb
        self.last, self.last_g = (None, None), (None, None)
        lbx, ubx, lbg, ubg = pb.bounds(self.m)
        x = np.asarray(w0, np.float64)
        t0 = time.perf_counter()
        sup = np.asarray(self._prefix(jnp.asarray(x), self.th))
        self.declare(sup if carry is None else np.maximum(np.minimum(np.asarray(carry), self.cap), sup))
        out = {"segments": [], "updates": 0, "hessian_upper_start": int(self.hess_structure[0].size)}
        lam_g = np.zeros(self.m) if lam_g0 is None else np.asarray(lam_g0, np.float64)
        lam_x = np.zeros(pb.n) if lam_x0 is None else np.asarray(lam_x0, np.float64)
        mu = None
        for seg in range(self.max_segments):  # one IPOPT run per declared Hessian structure
            self._object(warm or seg > 0, mu)
            r = self.obj["S"](x0=x, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, lam_g0=lam_g, lam_x0=lam_x)
            st = self.obj["S"].stats()
            x, lam_g, lam_x = np.asarray(r["x"]).ravel(), np.asarray(r["lam_g"]).ravel(), np.asarray(r["lam_x"]).ravel()
            it = st.get("iterations", {})
            mu = float(it["mu"][-1]) if it and len(it.get("mu", [])) else None
            out["updates"] += int(st["iter_count"])
            out["segments"].append({"status": st["return_status"], "updates": int(st["iter_count"]), "hess_upper": int(self.obj["H"][0].size)})
            if st["return_status"] != "User_Requested_Stop":
                break
            self.declare(np.maximum(self.declared_np, np.asarray(self._prefix(jnp.asarray(x), self.th))))
        period_s = time.perf_counter() - t0
        g = self.ipopt_rows(self._g(jnp.asarray(x), self.th))
        ex_slot = np.asarray(self._exact(jnp.asarray(x[:NX * self.T].reshape(self.T, NX)), self.th))
        ex = np.where(self.on[self.perm], ex_slot[self.perm], np.nan)
        v = violation(x, g, lbx, ubx, lbg, ubg)
        out.update({"status": out["segments"][-1]["status"], "violation": v, "objective": float(self._f(jnp.asarray(x), self.th)),
                    "accepted": bool(out["segments"][-1]["status"] in ACCEPT_STATUS and v <= ACCEPT_VIOLATION),
                    "rows": g[pb.n_def:], "multipliers": lam_g[pb.n_def:], "exact_per_row": ex, "solver_objects": len(out["segments"]),
                    "hessian_upper_end": int(self.hess_structure[0].size), "period_s": period_s})
        self.x, self.lam_g, self.lam_x = x, lam_g, lam_x
        return out

    def snapshot(self):
        """What the next period's start reads from this solve."""
        return {"T": self.T, "n_def": self.pb.n_def, "x": self.x.copy(), "lam_g": self.lam_g.copy(), "lam_x": self.lam_x.copy(),
                "keys": list(self.keys), "perm": self.perm.copy(), "declared": self.declared_np.copy()}


def shift_start(prev, nxt, d=1):
    """The start of the next period from prev (HorizonSolver.snapshot) shifted by d samples, for the solver nxt with that
    period installed: (w, lam_g, lam_x, the carried declared prefix per slot)."""
    T, nd = prev["T"], prev["n_def"]
    X, U = prev["x"][:NX * T].reshape(T, NX), prev["x"][NX * T:].reshape(T, NU)
    last = np.concatenate([X[-1, :NQ], np.zeros(NQ)])
    w = np.concatenate([np.concatenate([X[d:], np.tile(last, (d, 1))]).ravel(), np.concatenate([U[d:], np.zeros((d, NU))]).ravel()])
    ld = prev["lam_g"][:nd].reshape(T, NX)
    lam_g = np.zeros(nxt.m)
    lam_g[:nd] = np.concatenate([ld[d:], np.zeros((d, NX))]).ravel()
    old = {key: i for i, key in enumerate(prev["keys"])}
    carry = np.zeros((nxt.pb.n_slots, T, 2), np.int64)
    for i, key in enumerate(nxt.keys):
        if key in old:
            j = old[key]
            lam_g[nd + i] = prev["lam_g"][nd + j]
            carry[nxt.perm[i], :-d] = prev["declared"][prev["perm"][j], d:]
    lx, lu = prev["lam_x"][:NX * T].reshape(T, NX), prev["lam_x"][NX * T:].reshape(T, NU)
    lam_x = np.concatenate([np.concatenate([lx[d:], np.tile(lx[-1:], (d, 1))]).ravel(), np.concatenate([lu[d:], np.zeros((d, NU))]).ravel()])
    return w, lam_g, lam_x, carry
