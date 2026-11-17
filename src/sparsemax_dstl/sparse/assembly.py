"""The constraint Jacobian and the Lagrangian Hessian of one control period's program on a declared structure, by the
composition of the program's sparsemax structure with the per-sample nodes' derivatives from automatic differentiation.

With rho_r = S_r(a) an instance row on its atom trace a_{t,i} = N_i(q_t):
    grad rho_r = sum_{t,i} P_r[t,i] grad N_i(q_t),
    Hess rho_r = sum_{(t,i),(s,j)} H_r[(t,i),(s,j)] grad N_i(q_t) grad N_j(q_s)^T + sum_{t,i} P_r[t,i] Hess N_i(q_t),
P_r and H_r the path weights and the skeleton Hessian (`sparse.supports`). Three parts: A, the forward evaluation with the
supports and the test that they lie inside the declared ones; B, the nodes' gradients and the per-sample Hessians of
sum_n c_{t,n} N_n(q_t) at the active (sample, node) pairs (forward products of one sample's conjunction nodes of one arm, as
one vector function of that arm's 7 joint positions; a reach target's predicate likewise); C, the assembly on the declared
structure. The defects' Jacobian comes from forward products over a column coloring of its pattern, the cost's Hessian from
forward-over-reverse products over a star coloring of its pattern. Part A is compiled apart from parts B and C, so a new
declaration recompiles B and C only; part A at a point is evaluated once for that point's Jacobian and Hessian.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ..jax.operators import lower_min_recount
from ..tasks import dual_arm as D
from . import patterns as SP
from .coloring import CHUNK, hessian_plan, jacobian_plan
from .supports import Supports

NX, NU, NQ, NJ = D.NX, D.NU, D.NQ, D.NJ


def bucket(n):
    """A power of two, at least 8 and at least n."""
    return max(8, 1 << int(np.ceil(np.log2(max(int(n), 1)))))


def pad(x, n, v=0):
    x = np.asarray(x)
    return jnp.asarray(np.pad(x, (0, n - x.size), constant_values=v).astype(np.int32))


def _pow2(n, low):
    return max(low, 1 << int(np.ceil(np.log2(max(int(n), 1)))))


def active_keys(pb, Pf):
    """The (sample, node) pairs t n_nodes + n of nonzero path weight in Pf (n_rows, T, 2)."""
    r, t, a = np.nonzero(np.abs(np.asarray(Pf)) > 0)
    return np.unique(t * pb.n_nodes + pb.nid[r, a])


def group_pairs(pb, key):
    """Part B's layout of the pairs `key` (sorted, unique): conjunction pairs grouped by (sample, arm), each group padded to
    the largest group's count (a power of two) and the number of groups to a power of two (at least 8); leaf pairs padded
    likewise (at least 8). Returns (index arrays, comp (T, n_nodes) each pair's row in part B's output, a zero row for the
    others)."""
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
    g = np.repeat(np.arange(u.size), cnt)
    j = np.arange(g.size) - start[g]
    m, ng = _pow2(int(max(cnt.max(), 1)), 1), _pow2(u.size, 8)
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
    tl, nl = tn[lf], nn[lf]
    nlp = 0 if tl.size == 0 else _pow2(tl.size, 8)
    comp = np.full((pb.T, pb.n_nodes), ng * m + nlp, np.int32)
    comp[tc[:g.size], nc[:g.size]] = g * m + j
    comp[tl, nl] = ng * m + np.arange(tl.size)
    pr = {"tc": jnp.asarray(tg.astype(np.int32)), "ac": jnp.asarray(ag.astype(np.int32)), "nc": jnp.asarray(nodes.astype(np.int32)), "vc": jnp.asarray(valid)}
    if tl.size:
        pr.update(tl=pad(tl, nlp), nl=pad(nl, nlp, int(nl[0])), vl=jnp.asarray(np.arange(nlp) < tl.size))
    return pr, jnp.asarray(comp)


class StructuralDerivatives:
    """The structural Jacobian and Hessian of a PeriodProgram pb on a declared supports' prefix (module docstring).

    declare(pre, jac) fixes the structures: jac_structure (rows, cols) row-major, the given one or the prefix's with the
    defects; hess_structure the upper triangle (rows <= cols) row-major. jacobian_at / hessian_at return (values on the
    structure, escape), escape True when the supports at the point leave the declared prefix."""

    def __init__(self, pb):
        self.pb = pb
        self.supports = Supports(pb)
        self.prefix = jax.jit(self.supports.support_prefix)
        self.rows = jax.jit(self.supports.rows)
        self.a0 = jax.jit(self._a0, static_argnums=(2,))
        self.bc_jac = jax.jit(self._bc_jac, static_argnums=(6,))
        self.bc_hess = jax.jit(self._bc_hess, static_argnums=(9,))
        self.cache = (None, None)
        self.version, self.reused, self.escapes = 0, 0, 0
        self.node_sphere = jnp.asarray(pb.node_sphere)
        self.node_cell = jnp.asarray(pb.node_cell)
        self.node_arm = jnp.asarray(pb.node_arm)
        self.leaf_pred = jnp.asarray(pb.leaf_pred)
        self.cells = jnp.asarray(pb.scene.cells)
        self.cell_radius = pb.scene.cell_radius

    # ------------------------------------------------------------------
    # declaration

    def declare(self, pre, dtype, jac=None):
        """Declare the supports' prefix pre (n_rows, T, 2): the Hessian's structure (the rows' cliques and the cost's) and
        the index arrays of part C; the Jacobian's structure is the prefix's with the defects (jac None) or jac (rows, cols),
        which must hold the defects' pattern and the prefix's entries."""
        pb = self.pb
        T, n, n_def = pb.T, pb.n, pb.n_def
        self.version += 1
        r, c = SP.columns(pre)
        J, H = SP.jacobian_pattern(T, r, c), SP.hessian_pattern(T, pb.n_rows, r, c)
        if jac is not None:
            J = (np.asarray(jac[0], np.int64), np.asarray(jac[1], np.int64))
            spec = J[0] >= n_def
            r, c = J[0][spec] - n_def, J[1][spec]
        self.declared = jnp.asarray(pre, jnp.int32)
        self.ec = bucket(int((np.asarray(pre) > 0).sum((1, 2)).max(initial=1)))
        self.jac_structure, self.hess_structure = J, H
        if not hasattr(self, "defect_plan"):
            self.defect_plan = jacobian_plan(SP.defect_pattern(T), (n_def, n), "fwd", dtype)
        dr, dc = SP.defect_pattern(T)
        self.dpos = jnp.asarray(SP.positions(J, dr, dc, n))
        nz = bucket(r.size)
        self.nj = J[0].size
        self.jp = {"r": pad(r, nz), "t": pad(c // NX, nz), "p": pad(c % NX, nz), "pos": pad(SP.positions(J, n_def + r, c, n), nz, self.nj)}
        # the clique pairs (i <= j) of every row over the prefix's entries, and their positions in H
        r, c = SP.columns(pre)
        order = np.argsort(r, kind="stable")
        rs, cs = r[order], c[order]
        start = np.searchsorted(rs, rs, "left")
        cnt = np.searchsorted(rs, rs, "right") - start
        pr, pa = np.repeat(rs, cnt), np.repeat(cs, cnt)
        off = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        pb_ = cs[np.repeat(start, cnt) + off]
        keep = pa <= pb_
        pr, pa, pb_ = pr[keep], pa[keep], pb_[keep]
        nnz = H[0].size
        npairs = bucket(pr.size)
        hp = {"r": pad(pr, npairs), "ti": pad(pa // NX, npairs), "pi": pad(pa % NX, npairs), "tj": pad(pb_ // NX, npairs),
              "pj": pad(pb_ % NX, npairs), "pos": pad(SP.positions(H, pa, pb_, n), npairs, nnz)}
        self.nnz, self.nb = nnz, bucket(nnz + 1)
        same = (H[0] < n_def) & (H[1] < n_def) & (H[0] // NX == H[1] // NX) & (H[0] % NX < NQ) & (H[1] % NX < NQ)
        ks = np.flatnonzero(same)
        nk = bucket(ks.size)
        hp.update({"kpos": pad(ks, nk, nnz), "kt": pad(H[0][ks] // NX, nk), "kp": pad(H[0][ks] % NX, nk), "kq": pad(H[1][ks] % NX, nk)})
        cu = SP.cost_pattern(T)
        if not hasattr(self, "cost_plan"):
            self.cost_plan = hessian_plan(cu, n, dtype)
        self.cost = dict(self.cost_plan, pos=jnp.asarray(SP.positions(H, cu[0], cu[1], n)))
        # per row its clique's columns (sample, position) padded to M, and each clique pair's local indices
        r, c = SP.columns(pre)
        order = np.lexsort((c, r))
        r, c = r[order], c[order]
        start = np.searchsorted(r, np.arange(pb.n_rows))
        loc = np.arange(r.size) - start[r]
        M = bucket(int(loc.max(initial=0)) + 1)
        colT, colP, valid = np.zeros((pb.n_rows, M), np.int32), np.zeros((pb.n_rows, M), np.int32), np.zeros((pb.n_rows, M), bool)
        colT[r, loc], colP[r, loc], valid[r, loc] = c // NX, c % NX, True
        key = r * n + c
        prr = np.asarray(hp["r"]).astype(np.int64)
        ti, pi, tj, pj = (np.asarray(hp[a]).astype(np.int64) for a in ("ti", "pi", "tj", "pj"))
        li = np.searchsorted(key, prr * n + ti * NX + pi) - start[prr]
        lj = np.searchsorted(key, prr * n + tj * NX + pj) - start[prr]
        real = np.asarray(hp["pos"]) < nnz
        li, lj = np.where(real, np.clip(li, 0, M - 1), 0), np.where(real, np.clip(lj, 0, M - 1), 0)
        hp.update({"li": jnp.asarray(li.astype(np.int32)), "lj": jnp.asarray(lj.astype(np.int32)), "colT": jnp.asarray(colT),
                   "colP": jnp.asarray(colP), "valid": jnp.asarray(valid)})
        self.hp = hp
        self.products = {"defect_colors": self.defect_plan["colors"], "cost_colors": self.cost_plan["colors"], "compact_entries": self.ec}

    # ------------------------------------------------------------------
    # part B: the nodes' derivatives

    def _conj(self, q, a, nodes, v):
        """The conjunction nodes `nodes` (m,) at one sample's positions q with arm a's 7 joint positions replaced by v, from
        one evaluation of the kinematics."""
        pb = self.pb
        _, C = D.points(jax.lax.dynamic_update_slice(q, v, (a * NJ,)))
        sph = self.node_sphere[nodes]
        d = C[sph] - self.cells[self.node_cell[nodes]][:, None]
        eps, R = D.EPS_LENGTH, self.cell_radius
        dist = jnp.sqrt(jnp.sum(d * d, -1) + eps * eps) - eps
        z = (dist - jnp.asarray(D.RADIUS, q.dtype)[sph] - R) / R
        return lower_min_recount(z, pb.gamma_and)

    def _leaf(self, q, t, n, v):
        x = jnp.concatenate([jax.lax.dynamic_update_slice(q, v, (self.node_arm[n] * NJ,)), jnp.zeros(NQ, q.dtype)])
        return self.pb.scene.predicates(x)[self.leaf_pred[n]]

    def _leaf_block(self, Q, T_, N_, c=None):
        def g(q, t, n):
            qa = jax.lax.dynamic_slice(q, (self.node_arm[n] * NJ,), (NJ,))
            return jax.jacfwd(lambda v: self._leaf(q, t, n, v))(qa)
        G7 = jax.vmap(g)(Q, T_, N_)
        col = self.node_arm[N_][:, None] * NJ + np.arange(NJ)[None]
        G = jnp.zeros((N_.size, NQ), Q.dtype).at[np.arange(N_.size)[:, None], col].set(G7)
        if c is None:
            return G, None, col

        def h(q, t, n, ci):
            qa = jax.lax.dynamic_slice(q, (self.node_arm[n] * NJ,), (NJ,))
            return jax.hessian(lambda v: ci * self._leaf(q, t, n, v))(qa)
        return G, jax.vmap(h)(Q, T_, N_, c), col

    def _fb(self, w, cf, pr):
        """Part B at the pairs pr: Gc (pairs + 1, NQ) each pair's node gradient in the 14 positions (a zero last row) and, with
        cf (T, n_nodes), K (T, NQ, NQ) the per-sample Hessians of sum_n cf[t, n] N_n(q_t)."""
        pb = self.pb
        Xs, _ = pb.split(w)
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
        out = [G.reshape(ng * m, NQ)]
        leaf = "tl" in pr
        if leaf:
            Xl = Xs[pr["tl"], :NQ]
            if cf is None:
                Gl, _, _ = self._leaf_block(Xl, pr["tl"], pr["nl"])
            else:
                Gl, Kl, coll = self._leaf_block(Xl, pr["tl"], pr["nl"], jnp.where(pr["vl"], cf[pr["tl"], pr["nl"]], 0))
            out.append(Gl)
        Gc = jnp.concatenate(out + [jnp.zeros((1, NQ), w.dtype)])
        if cf is None:
            return Gc
        K = jnp.zeros((pb.T, NQ, NQ), w.dtype).at[pr["tc"][:, None, None], col[:, :, None], col[:, None, :]].add(H7)
        if leaf:
            K = K.at[pr["tl"][:, None, None], coll[:, :, None], coll[:, None, :]].add(Kl)
        return Gc, K

    # ------------------------------------------------------------------
    # parts A and C

    def _a0(self, w, declared, ec):
        """Part A: the path weights Pf, the compact skeleton Hessians HZ and the escape flag."""
        _, Pf, HZ, pn, bad = self.supports.skeletons(w, True, ec)
        return Pf, HZ, jnp.any(self.supports.row_prefix(Pf, pn) > declared) | bad

    def _c_jac(self, w, Pf, Gc, comp, jp, dpos, nj):
        """The Jacobian: the rows' entries sum_a Pf[r, t, a] grad N_{nid[r, a]}(q_t)[p]; the defects' by forward products."""
        pb = self.pb
        nid = jnp.asarray(pb.nid)
        r, t, p = jp["r"], jp["t"], jp["p"]
        vs = Pf[r, t, 0] * Gc[comp[t, nid[r, 0]], p] + Pf[r, t, 1] * Gc[comp[t, nid[r, 1]], p]
        _, lin = jax.linearize(lambda v: pb.defects(*pb.split(v)), w)
        B = jax.lax.map(lin, self.defect_plan["S"], batch_size=CHUNK)
        out = jnp.zeros(nj + 1, w.dtype).at[dpos].set(B[self.defect_plan["gc"], self.defect_plan["ge"]])
        return out.at[jp["pos"]].set(vs)[:nj]

    def _c_hess(self, w, sigma, mu, HZ, Gc, comp, K, hp, cost, nb):
        """The Hessian: per row Gm_r^T Hz_r Gm_r on its clique columns (Gm_r[e, m] the gradient of compact entry e's node at
        column m's sample and position) weighted by mu_r, the per-sample Hessians, the cost's Hessian."""
        pb = self.pb
        Hz, fold = HZ
        nid = jnp.asarray(pb.nid)
        node = jnp.take_along_axis(nid, jnp.maximum(fold, 0) % 2, 1)
        colT, colP = hp["colT"], hp["colP"]
        g = Gc[comp[colT[:, None, :], node[:, :, None]], colP[:, None, :]]
        on = (fold[:, :, None] >= 0) & (fold[:, :, None] // 2 == colT[:, None, :]) & hp["valid"][:, None, :]
        Gm = jnp.where(on, g, 0)
        Hr = jnp.einsum("rem,ref,rfn->rmn", Gm, Hz, Gm) * mu[:, None, None]
        out = jnp.zeros(nb + 1, w.dtype).at[hp["pos"]].add(Hr[hp["r"], hp["li"], hp["lj"]])
        out = out.at[hp["kpos"]].add(K[hp["kt"], hp["kp"], hp["kq"]])
        _, hvp = jax.linearize(jax.grad(pb.objective), w)
        B = jax.lax.map(hvp, cost["S"], batch_size=CHUNK)
        return out.at[cost["pos"]].add(sigma * B[cost["gc"], cost["ge"]])

    def _bc_jac(self, w, Pf, pr, comp, jp, dpos, nj):
        return self._c_jac(w, Pf, self._fb(w, None, pr), comp, jp, dpos, nj)

    def _bc_hess(self, w, sigma, mu, Pf, HZ, pr, comp, hp, cost, nb):
        pb = self.pb
        cf = jnp.zeros((pb.T, pb.n_nodes), w.dtype).at[:, jnp.asarray(pb.nid)].add(jnp.moveaxis(Pf * mu[:, None, None], 0, 1))
        Gc, K = self._fb(w, cf, pr)
        return self._c_hess(w, sigma, mu, HZ, Gc, comp, K, hp, cost, nb)

    # ------------------------------------------------------------------
    # evaluations at a point

    def point(self, w, xkey):
        """Part A at w and the active pairs there, evaluated once per point and declaration."""
        if self.cache[0] is not None and self.cache[0] == (xkey, self.version):
            self.reused += 1
            return self.cache[1]
        Pf, HZ, esc = jax.block_until_ready(self.a0(w, self.declared, self.ec))
        esc = bool(esc)
        self.escapes += int(esc)
        out = (Pf, HZ, esc) + group_pairs(self.pb, active_keys(self.pb, Pf))
        self.cache = ((xkey, self.version), out)
        return out

    def jacobian_at(self, w, xkey):
        Pf, _, esc, pr, comp = self.point(w, xkey)
        return jax.block_until_ready(self.bc_jac(w, Pf, pr, comp, self.jp, self.dpos, self.nj)), esc

    def hessian_at(self, w, xkey, sigma, lam):
        mu = lam[self.pb.n_def:]
        Pf, HZ, esc, pr, comp = self.point(w, xkey)
        v = self.bc_hess(w, jnp.asarray(sigma, w.dtype), mu, Pf, HZ, pr, comp, self.hp, self.cost, self.nb)[:self.nnz]
        return jax.block_until_ready(v), esc

    def jacobian(self, w):
        return self.jacobian_at(w, np.asarray(w).tobytes())

    def hessian(self, w, sigma, lam):
        return self.hessian_at(w, np.asarray(w).tobytes(), sigma, lam)
