"""The nonlinear program of one control period k with horizon T.

Variables w = (x_0, ..., x_{T-1}, u_0, ..., u_{T-1}), n = 42 T: x_j the state of sample k + 1 + j, u_j the joint
accelerations of the interval from sample k + j to k + j + 1. Constraints: the 28 T defects of the exact double
integrator, x_j - f(x_{j-1}, u_j) = 0 with x_{-1} the state at k, then one row per imposed conjunct instance, its smoothed
robustness less the margin, at least 0. Cost: 0.5 sum_j (W_Q |q_j - qref_j|^2 + W_U |u_j|^2) + 0.5 W_DU sum_j
|u_{j+1} - u_j|^2, qref the scene's positions at samples k + 1 .. k + T. Bounds: the joint limits less their margin, the
velocity and acceleration limits, zero velocity at the last sample.

An instance's program is cut to what its root reaches through the period's residual masks (`subprogram`); its atoms then
lie at predicted samples only: inside the horizon they read the per-sample node at that sample, beyond it the node at the
held state (the last sample's positions at zero velocity). Rows whose cut programs have the same structure and parameter
are evaluated together under jax.vmap.
"""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from ..jax.evaluator import REDUCTIONS
from ..stl import progression as pg
from ..tasks import dual_arm as D

NX, NU, NQ = D.NX, D.NU, D.NQ
W_Q, W_U, W_DU = 1.0, 1e-3, 1e-2  # the cost's weights on the position error, the accelerations and their change


def subprogram(prog, root_step, root_row, masks):
    """The part of prog that root_row reaches through the entries the masks keep: (steps, root position, root row), steps
    in program order as dicts with kind and rows (the step's reached rows), an atom's atom and sign, a reduction's sources
    (positions in the list), index (into its reached sources' concatenation) and mask."""
    steps = prog.steps
    follow = [None if m is None else np.asarray(m)[None] for m in masks]
    reached = pg.reach(prog, root_step, np.array([[root_row]]), follow)
    keep = [None if r is None else np.flatnonzero(r[0, 0]) for r in reached]
    out, pos = [], {}
    for i, st in enumerate(steps):
        if keep[i] is None or keep[i].size == 0:
            continue
        rows = keep[i]
        d = {"kind": st.kind, "rows": rows}
        if st.kind == "atom":
            d.update(atom=int(st.atom), sign=float(st.sign))
        else:
            lengths = [steps[s].length for s in st.sources]
            offs = np.cumsum([0] + lengths)
            idx, m = st.index[rows], np.asarray(masks[i])[rows]
            new = np.zeros_like(idx)
            found = np.zeros(idx.shape, bool)
            srcs, base = [], 0
            for j, s in enumerate(st.sources):
                kj = keep[s]
                if kj is None or kj.size == 0:
                    continue
                loc = idx - offs[j]
                p = np.searchsorted(kj, loc)
                ok = (loc >= 0) & (loc < lengths[j]) & (p < kj.size) & (kj[np.minimum(p, kj.size - 1)] == loc)
                new = np.where(ok, base + p, new)
                found |= ok
                srcs.append(pos[s])
                base += kj.size
            if np.any(m & ~found):
                raise ValueError("a kept operand is not among its source's reached rows")
            d.update(sources=srcs, index=new, mask=m)
        pos[i] = len(out)
        out.append(d)
    return out, pos[root_step], int(np.flatnonzero(keep[root_step] == root_row)[0])


def structure_key(sub, gamma):
    """A digest of a cut program's structure and parameter: equal digests evaluate as one vmapped group."""
    h = hashlib.sha1(repr(float(gamma)).encode())
    for d in sub:
        h.update((d["kind"] + str(d["rows"].size)).encode())
        if d["kind"] == "atom":
            h.update((str(d["atom"]) + str(d["sign"])).encode())
        else:
            h.update(np.asarray(d["sources"]).tobytes() + d["index"].tobytes() + d["mask"].tobytes())
    return h.hexdigest()


def sub_values(sub, tr, gamma, red):
    """Step outputs of a cut program; tr maps each atom step's position to its values at the step's rows; red is a pair
    (max_reduce, min_reduce)."""
    out = []
    for i, d in enumerate(sub):
        if d["kind"] == "atom":
            out.append(d["sign"] * tr[i])
            continue
        src = jnp.concatenate([out[s] for s in d["sources"]], -1)
        r = red[0] if d["kind"] == "max" else red[1]
        out.append(r(src[..., d["index"]], gamma, jnp.asarray(d["mask"])))
    return out


class PeriodProgram:
    """The program of control period k (module docstring) for a scene and its conjuncts (mpc.conjuncts). exact_length is the
    outside predicates' smoothing length of the exact semantics (m; 0 the exact norm)."""

    def __init__(self, scene, conjs, k, exact_length=0.0):
        T = scene.T
        self.scene, self.conjs, self.k, self.T, self.exact_length = scene, conjs, int(k), T, exact_length
        self.n, self.n_def = (NX + NU) * T, NX * T
        self.delta = scene.margin
        X = np.asarray(scene.states, np.float64)
        self.x0, self.qref = X[k], X[k + 1:k + 1 + T, :NQ]
        self._step = jax.vmap(D.interval_map(D.H_STEP))
        joints = np.asarray(scene.joints)
        self.link, self.arm = joints.max(-1), joints.argmax(-1)
        nodes, rows = {}, []
        for cj in conjs:
            roots, _ = cj.imposed([k], T)
            rr = roots[0][roots[0] >= 0]
            if rr.size == 0:
                continue
            masks = pg.evaluation_masks(cj.program, pg.residual(cj.program, cj.decisions, k))
            nid = []
            for nd in cj.nodes:
                key = (nd["kind"], tuple(np.atleast_1d(nd["preds"]).tolist()), cj.gamma if nd["kind"] == "and" else 0.0)
                nid.append(nodes.setdefault(key, len(nodes)))
            for r in rr:
                sub, root, rrow = subprogram(cj.program, cj.root_step, int(r), masks)
                rows.append({"cj": cj, "sub": sub, "root": root, "rrow": rrow, "gamma": cj.gamma, "nid": nid,
                             "key": structure_key(sub, cj.gamma) + str(len(nid))})
        self.rowinfo, self.n_rows = rows, len(rows)
        self.m = self.n_def + self.n_rows
        keys = sorted(nodes, key=nodes.get)
        self.n_nodes = len(keys)
        self.is_leaf = np.array([kk[0] == "leaf" for kk in keys])
        S = max([len(kk[1]) for kk in keys if kk[0] == "and"] or [1])
        ga = sorted(set(kk[2] for kk in keys if kk[0] == "and"))
        if len(ga) > 1 or any(len(kk[1]) != S for kk in keys if kk[0] == "and"):
            raise ValueError("the conjunction nodes share one smoothing parameter and one number of predicates")
        self.gamma_and = ga[0] if ga else 1.0
        self.pidx = np.array([list(kk[1]) if kk[0] == "and" else [kk[1][0]] * S for kk in keys], np.int64).reshape(-1, S)
        self.node_arm = self.arm[self.pidx[:, 0]]
        n_wp, nd_ = len(scene.waypoints), D.DISTAL.size
        rel = self.pidx - n_wp
        self.node_sphere = np.where(self.is_leaf[:, None], 0, D.DISTAL[np.where(rel >= 0, rel, 0) % nd_])
        cell = np.where(self.is_leaf, 0, rel[:, 0] // nd_)
        if np.any(~self.is_leaf[:, None] & ((rel < 0) | (rel // nd_ != cell[:, None]))):
            raise ValueError("a conjunction node reads predicates of more than one cell")
        self.node_cell = cell
        self.leaf_pred = self.pidx[:, 0]
        self.nid = np.array([(r["nid"] + r["nid"])[:2] for r in rows], np.int64).reshape(-1, 2)
        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault(r["key"], []).append(i)
        self.groups = []
        for members in groups.values():
            r0 = rows[members[0]]
            atoms = [d for d in r0["sub"] if d["kind"] == "atom"]
            s_abs = [np.stack([[d for d in rows[i]["sub"] if d["kind"] == "atom"][a]["rows"] for i in members]) for a in range(len(atoms))]
            nid_at = [np.stack([np.full(atoms[a]["rows"].size, rows[i]["nid"][atoms[a]["atom"]]) for i in members]) for a in range(len(atoms))]
            if any(np.any(s <= k) for s in s_abs):
                raise ValueError("a cut program reads a recorded sample")
            self.groups.append({"members": np.array(members), "sub": r0["sub"], "root": r0["root"], "rrow": r0["rrow"], "gamma": r0["gamma"],
                                "s": s_abs, "nid": nid_at, "atom_of": [d["atom"] for d in atoms]})
        order = np.concatenate([g["members"] for g in self.groups]) if self.groups else np.zeros(0, np.int64)
        self.perm = np.argsort(order)

    # ------------------------------------------------------------------
    # values

    def split(self, w):
        T = self.T
        return w[:NX * T].reshape(T, NX), w[NX * T:].reshape(T, NU)

    def predicates(self, Xs, exact=False):
        """(P (T, n_pred) at the horizon's samples, Ph (n_pred,) at the held state)."""
        held = jnp.concatenate([Xs[-1, :NQ], jnp.zeros(NQ, Xs.dtype)])[None]
        sc = self.scene
        if exact:
            cells = jnp.asarray(sc.cells, Xs.dtype)
            f = jax.vmap(lambda x: jnp.concatenate([D.at_waypoint(x, sc.waypoints, sc.waypoint_arms, 0.0),
                                                    D.outside(x, cells, sc.cell_radius, self.exact_length)]))
        else:
            f = jax.vmap(sc.predicates)
        return f(Xs), f(held)[0]

    def node_values(self, P, semantics):
        """(..., n_nodes): every per-sample node over predicate vectors P (..., n_pred)."""
        z = P[..., self.pidx]
        a = jnp.min(z, -1) if semantics == "exact" else REDUCTIONS[semantics][1](z, self.gamma_and)
        return jnp.where(jnp.asarray(self.is_leaf), z[..., 0], a)

    def traces(self, g, N, Nh):
        """Per atom step of a group's cut program, the node values (members, rows) at its samples."""
        k, T = self.k, self.T
        return [jnp.where(s > k + T, Nh[nid], N[np.clip(s - k - 1, 0, T - 1), nid]) for s, nid in zip(g["s"], g["nid"])]

    def rows(self, w, semantics="sparsemax"):
        """The instances' smoothed robustness (margin not subtracted) under "sparsemax", "lse_matched" or "exact"."""
        Xs, _ = self.split(w)
        P, Ph = self.predicates(Xs, semantics == "exact")
        N, Nh = self.node_values(P, semantics), self.node_values(Ph, semantics)
        red = REDUCTIONS[semantics]
        out = []
        for g in self.groups:
            f = lambda *t, g=g: sub_values(g["sub"], _atom_map(g["sub"], t), g["gamma"], red)[g["root"]][g["rrow"]]  # noqa: E731
            out.append(jax.vmap(f)(*self.traces(g, N, Nh)))
        if not out:
            return jnp.zeros(0, w.dtype)
        return jnp.concatenate(out)[self.perm]

    def defects(self, Xs, Us):
        prev = jnp.concatenate([jnp.asarray(self.x0, Xs.dtype)[None], Xs[:-1]])
        return (Xs - self._step(prev, Us)).ravel()

    def constraints(self, w, semantics="sparsemax"):
        Xs, Us = self.split(w)
        return jnp.concatenate([self.defects(Xs, Us), self.rows(w, semantics) - self.delta])

    def objective(self, w):
        Xs, Us = self.split(w)
        E = Xs[:, :NQ] - jnp.asarray(self.qref, Xs.dtype)
        return 0.5 * (W_Q * jnp.sum(E * E) + W_U * jnp.sum(Us * Us) + W_DU * jnp.sum((Us[1:] - Us[:-1]) ** 2))

    def lagrangian(self, w, sigma, lam, semantics="sparsemax"):
        return sigma * self.objective(w) + jnp.dot(lam, self.constraints(w, semantics))

    def bounds(self):
        """(lbx, ubx, lbg, ubg): the joint limits less their margin, |qd| and |u| within their limits, qd = 0 at the last
        sample; defects 0; instance rows at least 0."""
        T = self.T
        lo, hi = D.joint_bounds()
        v, a = D.VELOCITY_LIMIT, D.ACCELERATION_LIMIT
        lbx = np.concatenate([np.tile(np.concatenate([lo, np.full(NQ, -v)]), T), np.full(T * NU, -a)])
        ubx = np.concatenate([np.tile(np.concatenate([hi, np.full(NQ, v)]), T), np.full(T * NU, a)])
        term = (T - 1) * NX + NQ + np.arange(NQ)
        lbx[term], ubx[term] = 0.0, 0.0
        lbg = np.zeros(self.m)
        ubg = np.concatenate([np.zeros(self.n_def), np.full(self.m - self.n_def, np.inf)])
        return lbx, ubx, lbg, ubg

    def reference(self):
        """The scene's states at samples k + 1 .. k + T and the accelerations of its velocities: the start point."""
        k, T = self.k, self.T
        X = np.asarray(self.scene.states)
        Us = np.diff(X[k:k + 1 + T, NQ:], axis=0) / D.H_STEP
        return np.concatenate([X[k + 1:k + 1 + T].ravel(), Us.ravel()])

    def exact_rows(self, Xs):
        """The instances' exact robustness along planned states Xs (T, 28)."""
        return self.rows(jnp.concatenate([jnp.asarray(Xs).ravel(), jnp.zeros(self.T * NU)]), "exact")

    def instances(self):
        """[(conjunct name, anchor)] per instance row, in row order."""
        out = []
        for r in self.rowinfo:
            cj = r["cj"]
            out.append((cj.name, int(cj.anchors[int(np.flatnonzero(cj.root_rows == r["sub"][r["root"]]["rows"][r["rrow"]])[0])])))
        return out


def _atom_map(sub, t):
    """{position of each atom step: its values}, the atoms in sub's order."""
    atoms = [i for i, d in enumerate(sub) if d["kind"] == "atom"]
    return dict(zip(atoms, t))
