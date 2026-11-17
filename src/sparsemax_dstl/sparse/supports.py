"""The sparsemax supports of one control period's program at a point: the forward evaluation with each reduction's weights,
the path weights of every instance row's atom entries, its compact skeleton Hessian, and the supports' pattern.

With rho_r = S_r(a) a row's cut program on its atom trace a (a_{t,i} = N_i(q_t), the per-sample node values), the path
weight P_r[t, i] is d rho_r / d a_{t,i}: the product of the sparsemax weights along each path from the root to the atom
entry, summed over paths. The skeleton Hessian is d^2 rho_r / da^2, composed through the program's steps from each node's
Hessian +-(1/gamma)(Diag(1_S) - 1_S 1_S^T / |S|) on its support S, kept on the ec atom entries of largest |P_r|. The
supports' pattern of a row: at sample t and arm b, the joints 1..L whose positions move a sphere of a node that the row
reaches at t through edges of positive weight. Thresholds are found by recounting (`jax.operators.extremum_and_weights`).
"""
import jax
import jax.numpy as jnp
import numpy as np

from ..jax.operators import extremum_and_weights


def size(n):
    """A padded length for n entries: at least 8, rounded up to three significant bits (8, 10, 12, 14, 16, 20, 24, ...)."""
    n = max(int(n), 8)
    step = 1 << (int(np.floor(np.log2(n))) - 2)
    return -(-n // step) * step


def forward(sub, tr, gamma):
    """The step outputs of a cut program and each reduction step's sparsemax weights (None for an atom step)."""
    vals, W = [], [None] * len(sub)
    for i, d in enumerate(sub):
        if d["kind"] == "atom":
            vals.append(d["sign"] * tr[i])
            continue
        src = jnp.concatenate([vals[s] for s in d["sources"]], -1)
        v, W[i] = extremum_and_weights(src[..., d["index"]], gamma, jnp.asarray(d["mask"]), d["kind"])
        vals.append(v)
    return vals, W


def skeleton(sub, root, root_row, tr, gamma, hessian, ec):
    """A row's value, the path weights of its atom entries (in sub's atom order, concatenated) and, with hessian, its skeleton
    Hessian on the ec entries of largest |P| with their indices and whether more than ec entries have a nonzero weight."""
    vals, W = forward(sub, tr, gamma)
    dt = vals[root].dtype
    P = [jnp.zeros(d["rows"].size, dt) for d in sub]
    P[root] = P[root].at[root_row].set(1)
    for i in range(len(sub) - 1, -1, -1):
        d = sub[i]
        if d["kind"] == "atom":
            continue
        lengths = [sub[s]["rows"].size for s in d["sources"]]
        acc = jnp.zeros(sum(lengths), dt).at[d["index"]].add(P[i][:, None] * W[i])
        offs = np.cumsum([0] + lengths)
        for j, s in enumerate(d["sources"]):
            P[s] = P[s] + acc[offs[j]:offs[j + 1]]
    atoms = [i for i, d in enumerate(sub) if d["kind"] == "atom"]
    Pa = jnp.concatenate([sub[i]["sign"] * P[i] for i in atoms])
    if not hessian:
        return vals[root][root_row], Pa, None, None, None
    E = Pa.size
    ec = min(ec, E)
    top, idx = jax.lax.top_k(jnp.abs(Pa), ec)
    over = jnp.sum(jnp.abs(Pa) > 0) > ec
    offs = np.cumsum([0] + [sub[i]["rows"].size for i in atoms])
    G = [None] * len(sub)
    Hz = jnp.zeros((ec, ec), dt)
    for i, d in enumerate(sub):
        if d["kind"] == "atom":
            a = atoms.index(i)
            here = (idx >= offs[a]) & (idx < offs[a + 1]) & (top > 0)
            G[i] = jnp.zeros((d["rows"].size + 1, ec), dt).at[jnp.where(here, idx - offs[a], d["rows"].size), np.arange(ec)].set(jnp.where(here, d["sign"], 0))[:-1]
            continue
        Gs = jnp.concatenate([G[s] for s in d["sources"]], 0)
        n_src = Gs.shape[0]
        rows = np.broadcast_to(np.arange(d["index"].shape[0])[:, None], d["index"].shape)
        G[i] = jnp.zeros((d["index"].shape[0], n_src), dt).at[rows, d["index"]].add(W[i]) @ Gs
        on = W[i] > 0
        k = jnp.sum(on, -1)
        c = jnp.where(k >= 2, (1.0 if d["kind"] == "max" else -1.0) * P[i] / gamma, 0)
        dd = jnp.zeros(n_src, dt).at[d["index"]].add(jnp.where(on, c[:, None], 0))
        Hz = Hz + (Gs * dd[:, None]).T @ Gs
        Bm = jnp.zeros((d["index"].shape[0], n_src), dt).at[rows, d["index"]].add(on.astype(dt)) @ Gs
        Hz = Hz - (Bm * (c / jnp.maximum(k, 1))[:, None]).T @ Bm
    return vals[root][root_row], Pa, Hz, idx, over


class Supports:
    """The forward evaluation of a PeriodProgram pb with the supports."""

    def __init__(self, pb):
        self.pb = pb
        self.link = jnp.asarray(pb.link[pb.pidx])
        self.leaf_link = jnp.asarray(pb.link[pb.pidx[:, 0]])
        self.leaf = jnp.asarray(pb.is_leaf)

    def nodes(self, P):
        """(values, prefixes) (..., n_nodes) of every per-sample node over predicate vectors P (..., n_pred)."""
        pb = self.pb
        z = P[..., pb.pidx]
        v, wt = extremum_and_weights(z, pb.gamma_and, None, "min")
        pre = jnp.max(jnp.where(wt > 0, self.link, 0), -1)
        return jnp.where(self.leaf, z[..., 0], v), jnp.where(self.leaf, self.leaf_link, pre)

    def nodes_at(self, w):
        Xs, _ = self.pb.split(w)
        P, Ph = self.pb.predicates(Xs)
        N, pn = self.nodes(P)
        return N, self.nodes(Ph)[0], pn

    def rows(self, w):
        """The instances' sparsemax robustness (margin not subtracted) from the forward pass; not for differentiation."""
        pb = self.pb
        N, Nh, _ = self.nodes_at(w)
        out = []
        for g in pb.groups:
            atoms = [i for i, d in enumerate(g["sub"]) if d["kind"] == "atom"]
            f = lambda *t, g=g, atoms=atoms: forward(g["sub"], dict(zip(atoms, t)), g["gamma"])[0][g["root"]][g["rrow"]]  # noqa: E731
            out.append(jax.vmap(f)(*pb.traces(g, N, Nh)))
        if not out:
            return jnp.zeros(0, w.dtype)
        return jnp.concatenate(out)[pb.perm]

    def skeletons(self, w, hessian=True, ec=16):
        """(V, Pf, HZ, pn, bad): the rows' values, their path weights folded onto (n_rows, T, 2) (sample, atom), with hessian
        the compact skeleton Hessians HZ = (Hz (n_rows, ec, ec), fold (n_rows, ec): each compact entry's 2 t + atom, -1 for
        padding), the nodes' support prefixes pn (T, n_nodes), and whether a row has more than ec active entries."""
        pb = self.pb
        N, Nh, pn = self.nodes_at(w)
        T, k = pb.T, pb.k
        V, Pf, Hf, bad = [], [], [], []
        ecs = [int(sum(a.shape[1] for a in g["s"])) if np.any(np.concatenate(g["s"], 1) > k + T) else ec for g in pb.groups]
        ecm = max(ecs + [ec])
        for g, ecg in zip(pb.groups, ecs):
            atoms = [i for i, d in enumerate(g["sub"]) if d["kind"] == "atom"]
            f = lambda *t, g=g, atoms=atoms, ecg=ecg: skeleton(g["sub"], g["root"], g["rrow"], dict(zip(atoms, t)), g["gamma"], hessian, ecg)  # noqa: E731
            v, Pa, Hz, idx, over = jax.vmap(f)(*pb.traces(g, N, Nh))
            s_all = np.concatenate(g["s"], 1)
            a_all = np.concatenate([np.full(s.shape, g["atom_of"][a]) for a, s in enumerate(g["s"])], 1)
            fold = 2 * np.clip(s_all - k - 1, 0, T - 1) + a_all
            G_ = s_all.shape[0]
            rr = np.arange(G_)[:, None]
            Pf.append(jnp.zeros((G_, 2 * T), w.dtype).at[rr, fold].add(Pa).reshape(G_, T, 2))
            bad.append(jnp.any((Pa != 0) & jnp.asarray(s_all <= k)))
            V.append(v)
            if hessian:
                fi = jnp.take_along_axis(jnp.asarray(fold), idx, 1)
                e = Hz.shape[1]
                Hf.append((jnp.pad(Hz, ((0, 0), (0, ecm - e), (0, ecm - e))), jnp.pad(fi, ((0, 0), (0, ecm - e)), constant_values=-1)))
                bad.append(jnp.any(over))
        V, Pf = (jnp.concatenate(x)[pb.perm] for x in (V, Pf))
        HZ = (jnp.concatenate([h[0] for h in Hf])[pb.perm], jnp.concatenate([h[1] for h in Hf])[pb.perm]) if hessian else None
        return V, Pf, HZ, pn, jnp.any(jnp.stack(bad))

    def row_prefix(self, Pf, pn):
        """(n_rows, T, 2) per row, sample and arm, the supports' joint prefix length."""
        pb = self.pb
        val = jnp.where(jnp.abs(Pf) > 0, jnp.moveaxis(pn[:, pb.nid], 0, 1), 0)
        arm = jnp.asarray(pb.node_arm[pb.nid])
        return jnp.stack([jnp.max(jnp.where(arm[:, None, :] == b, val, 0), -1) for b in (0, 1)], -1)

    def support_prefix(self, w):
        _, Pf, _, pn, _ = self.skeletons(w, hessian=False)
        return self.row_prefix(Pf, pn)
