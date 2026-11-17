"""The baselines of the receding-horizon programs: the program of one control period in CasADi SX with the sound
log-sum-exp, the sparsemax operators or the generalized-mean robustness (GMR) at every node, whose sparsity patterns,
colorings and derivatives are CasADi's own, and dense automatic differentiation in JAX.

The GMR is the power mean robustness of order (0, 1) of Mehdipour, Vasile and Belta (IEEE TAC 70(3), 2025). With m
operands x and [x]_- = min(x, 0) its conjunction (their eq. (12)) is the geometric mean of x if every operand is positive
and (1/m) sum_i [x_i]_- otherwise; the disjunction is -conj(-x) and the until nests its closed prefix (their eq. (15)).

The expression graphs are built operand by operand and sample by sample (construction, not numerical work).
"""
import math

import casadi as ca
import numpy as np

from ..sparse.coloring import dense_hessian, dense_jacobian  # noqa: F401 (re-exported)
from ..stl import progression as pg
from ..tasks import dual_arm as D
from .ipopt import ipopt_options  # noqa: F401 (re-exported)
from .program import W_DU, W_Q, W_U

STEPS = 40  # bisection steps of the sparsemax threshold on [-1, 0]; the value is exact once the support is found


# ------------------------------------------------------------------
# the plant and the predicates in CasADi

def _rot(axis, angle):
    """Rotation by angle about a constant unit axis: I + sin(angle) K + (1 - cos(angle)) K^2."""
    x, y, z = (float(v) for v in axis)
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return ca.DM.eye(3) + ca.sin(angle) * ca.DM(K) + (1 - ca.cos(angle)) * ca.DM(K @ K)


def frames(q):
    """Per arm, the rotations and origins of links 1 to 7 and of the hand at q (SX, 14)."""
    Rs, ps = [], []
    for a in range(2):  # the two arms
        R, p = ca.DM.eye(3), ca.DM(D.PLACEMENT[a])
        Ra, pa = [], []
        for k in range(D.NJ):  # the joints of the kinematic chain
            p = p + ca.mtimes(R, ca.DM(D.ORIGIN[a, k]))
            R = ca.mtimes(R, _rot(D.AXIS[a, k], q[a * D.NJ + k]))
            Ra.append(R)
            pa.append(p)
        pa.append(p + ca.mtimes(R, ca.DM(D.MOUNT_POS)))
        Ra.append(ca.mtimes(R, ca.DM(D.MOUNT_ROT)))
        Rs.append(Ra)
        ps.append(pa)
    return Rs, ps


def points(q):
    """Hand points (two SX 3-vectors) and sphere centres (a list of 52 SX 3-vectors) at q."""
    Rs, ps = frames(q)
    centres = []
    for i in range(D.N_SPHERES):  # the spheres
        a, k = int(D.SPHERES[i, 0]), int(D.SPHERES[i, 1]) - 1
        centres.append(ps[a][k] + ca.mtimes(Rs[a][k], ca.DM(D.SPHERES[i, 2:5])))
    hands = [ps[a][D.NJ] + ca.mtimes(Rs[a][D.NJ], ca.DM(D.HAND_POINT[a])) for a in range(2)]
    return hands, centres


def interval(x, u, h):
    """x+ = f_h(x, u), the exact double integrator over h."""
    q, qd = x[:D.NQ], x[D.NQ:]
    return ca.vertcat(q + h * qd + 0.5 * h * h * u, qd + h * u)


def _norm_minus(e, eps):
    return ca.sqrt(ca.sumsqr(e) + eps * eps) - eps


def _norm_plus(e, eps):
    return ca.sqrt(ca.sumsqr(e) + eps * eps)


class Predicates:
    """The predicates of one state x (SX, 28) in a scene's order (waypoint predicates, then outside predicates), each built
    when first asked for."""

    def __init__(self, x, scene, eps=D.EPS_LENGTH):
        self.hands, self.C = points(x[:D.NQ])
        self.n_wp = len(scene.waypoints)
        self.waypoints, self.arms = np.asarray(scene.waypoints, np.float64), np.asarray(scene.waypoint_arms)
        self.cells, self.radius = np.asarray(scene.cells, np.float64), float(scene.cell_radius)
        self.eps, self.cache = eps, {}

    def __call__(self, i):
        i = int(i)
        if i not in self.cache:
            self.cache[i] = self._pred(i)
        return self.cache[i]

    def _pred(self, i):
        eps = self.eps
        if i >= self.n_wp:
            z, d = divmod(i - self.n_wp, len(D.DISTAL))
            s, rz = int(D.DISTAL[d]), self.radius
            return (_norm_minus(self.C[s] - ca.DM(self.cells[z]), eps) - D.RADIUS[s] - rz) / rz
        return (D.R_GOAL - _norm_plus(self.hands[int(self.arms[i])] - ca.DM(self.waypoints[i]), eps)) / D.R_GOAL


# ------------------------------------------------------------------
# the reductions over the remaining operands (a list of SX scalars)

def lse_max(z, gamma):
    """Sound log-sum-exp maximum (logsumexp(beta z) - log m) / beta, beta matched to sparsemax's gamma; exact for m = 1."""
    m = len(z)
    if m == 1:
        return z[0]
    b = math.log(m) / (gamma / 2 * (1 - 1 / m))
    return (ca.logsumexp(b * ca.vertcat(*z)) - math.log(m)) / b


def lse_min(z, gamma):
    """Log-sum-exp minimum -logsumexp(-beta z) / beta, beta matched to sparsemax's gamma; exact for m = 1."""
    m = len(z)
    if m == 1:
        return z[0]
    b = math.log(m) / (gamma / 2 * (1 - 1 / m))
    return -ca.logsumexp(-b * ca.vertcat(*z)) / b


def _quadratic_max(v, gamma, steps):
    c = ca.mmax(v)
    s = (v - c) / gamma
    lo, hi = ca.SX(-1.0), ca.SX(0.0)
    for _ in range(steps):  # bisection steps
        mid = (lo + hi) / 2
        big = ca.sum1(ca.fmax(s - mid, 0)) > 1
        lo, hi = ca.if_else(big, mid, lo), ca.if_else(big, hi, mid)
    ind = s > (lo + hi) / 2
    th = (ca.sum1(ind * s) - 1) / ca.sum1(ind)
    p = ca.fmax(s - th, 0)
    return c + gamma * (th + 0.5 * ca.sumsqr(p))


def sparsemax_max(z, gamma, steps=STEPS):
    """Sparsemax lower maximum M_gamma(z) + gamma / (2m); exact for m = 1."""
    m = len(z)
    if m == 1:
        return z[0]
    return _quadratic_max(ca.vertcat(*z), gamma, steps) + gamma / (2 * m)


def sparsemax_min(z, gamma, steps=STEPS):
    """Sparsemax lower minimum -M_gamma(-z) - gamma / 2; exact for m = 1."""
    m = len(z)
    if m == 1:
        return z[0]
    return -_quadratic_max(-ca.vertcat(*z), gamma, steps) - gamma / 2


def gmr_and(z):
    """GMR conjunction of order (0, 1): the geometric mean if every operand is positive, else the mean of min(z, 0)."""
    m = len(z)
    u = ca.vertcat(*z)
    pos = ca.mmin(u) > 0
    v = ca.if_else(u < 0, -u, 0)  # -[u]_-, derivative 0 at u = 0
    first = ca.exp(ca.sum1(ca.log(ca.if_else(pos, u, 1))) / m)  # the argument is 1 where the branch is not taken
    second = -ca.sum1(v) / m
    return ca.if_else(pos, first, second)


def gmr_or(z):
    """GMR disjunction of order (0, 1), -gmr_and(-z)."""
    return -gmr_and([-e for e in z])


def reductions(smoothing, gamma=None, steps=STEPS):
    """(max_reduce, min_reduce) over a list of SX operands for "lse", "sparsemax" (both at gamma) or "gmr"."""
    if smoothing == "lse":
        return (lambda z: lse_max(z, gamma)), (lambda z: lse_min(z, gamma))
    if smoothing == "sparsemax":
        return (lambda z: sparsemax_max(z, gamma, steps)), (lambda z: sparsemax_min(z, gamma, steps))
    if smoothing == "gmr":
        return gmr_or, gmr_and
    raise ValueError("smoothing is 'lse', 'sparsemax' or 'gmr', got " + repr(smoothing))


# ------------------------------------------------------------------
# one root row of a compiled specification

def evaluate_row(program, root_step, root_row, masks, atom, reduce, memo, opened=None, nested=False):
    """The SX value of one root row of a compiled program, reduction by reduction through the entries masks keep (per
    step a boolean (rows, width) array; memo caches (step, row)). atom(a, t) is atom a at time t, reduce the pair
    (max_reduce, min_reduce); opened (per step a boolean row array) marks the rows that may be reached. With nested, the
    rows of an until's inner step are reduced as (witness, reduction of the prefix)."""
    lens = [s.length for s in program.steps]
    offs = {i: np.cumsum([0] + [lens[j] for j in s.sources]) for i, s in enumerate(program.steps) if s.kind != "atom"}
    inner = set(program.inner.values()) if nested and program.inner else set()

    def val(i, r):
        key = (i, r)
        if key in memo:
            return memo[key]
        st = program.steps[i]
        if st.kind == "atom":
            v = st.sign * atom(st.atom, r)
        else:
            if opened is not None and not opened[i][r]:
                raise ValueError("a decided row is reached at step " + str(i) + " row " + str(r))
            keep = np.flatnonzero(masks[i][r])
            z = []
            for e in keep:  # the remaining operands
                gi = int(st.index[r, e])
                j = int(np.searchsorted(offs[i], gi, side="right")) - 1
                z.append(val(st.sources[j], gi - int(offs[i][j])))
            f = reduce[0] if st.kind == "max" else reduce[1]
            if i in inner:
                if keep.size < 2 or keep[0] != 0:
                    raise ValueError("an open row of an until's inner step keeps its witness and a prefix entry (step " + str(i)
                                     + " row " + str(r) + ")")
                v = f([z[0], f(z[1:])])
            else:
                v = f(z)
        memo[key] = v
        return v

    return val(root_step, int(root_row))


def node_value(node, sample, reduce):
    """A per-sample node at a Predicates sample: "leaf" the predicate, "and" the min_reduce over its predicates."""
    idx = np.atleast_1d(np.asarray(node["preds"], np.int64))
    if node["kind"] == "leaf":
        return sample(idx[0])
    if node["kind"] != "and":
        raise ValueError("a per-sample node is 'and' or 'leaf', got " + node["kind"])
    return reduce[1]([sample(i) for i in idx])


# ------------------------------------------------------------------
# the program of one control period in CasADi

FORMS = ("rows", "authors")
GMR_WEIGHT = 100.0  # the weight of the robustness in the cost of the authors' form (their Example 4)


class Period:
    """The program of control period k (mpc.program's variables, defects, cost, bounds, start and imposed instances) in
    CasADi SX with smoothing "lse", "sparsemax" or "gmr" at every node. Form "rows": one row per imposed instance, its
    robustness less the margin, at least 0. Form "authors" (GMR only, the formulation of its authors' eq. (11)): the cost
    less GMR_WEIGHT times the GMR conjunction eta of the conjuncts' conjunctions of their instances, and the one row
    eta - margin >= 0."""

    def __init__(self, scene, conjs, k, smoothing="lse", form="rows", steps=STEPS, weight=GMR_WEIGHT):
        if form not in FORMS or (form == "authors" and smoothing != "gmr"):
            raise ValueError("form is 'rows', or 'authors' with the GMR")
        T, nx, nu, h = scene.T, D.NX, D.NU, D.H_STEP
        self.scene, self.conjs, self.k, self.T, self.smoothing, self.form = scene, conjs, int(k), T, smoothing, form
        self.delta = scene.margin
        self.n, self.n_def = T * (nx + nu), T * nx
        w = ca.SX.sym("w", self.n)
        self.w = w
        X = [w[nx * j:nx * (j + 1)] for j in range(T)]  # samples
        U = [w[nx * T + nu * j:nx * T + nu * (j + 1)] for j in range(T)]  # intervals
        states = np.asarray(scene.states)
        x_meas = ca.DM(states[k])
        d = [X[j] - interval(X[j - 1] if j else x_meas, U[j], h) for j in range(T)]  # intervals
        samples = [Predicates(X[j], scene) for j in range(T)]  # samples
        node_memo = {}

        def node_at(nd, j, reduce, gamma):
            key = (nd["kind"], tuple(np.atleast_1d(nd["preds"]).tolist()), gamma, j)
            if key not in node_memo:
                node_memo[key] = node_value(nd, samples[j], reduce)
            return node_memo[key]

        vals, self.instances, self.conj = [], [], []
        for ci, cj in enumerate(conjs):  # conjuncts, in row order
            gamma = None if smoothing == "gmr" else cj.gamma
            reduce = reductions(smoothing, gamma, steps)
            res = pg.residual(cj.program, cj.decisions, k)
            masks = pg.evaluation_masks(cj.program, res)
            opened = [None if r_ is None else r_[0] for r_ in res]
            roots, inst = cj.imposed([k], T)
            memo = {}

            def atom(a, s, cj=cj, reduce=reduce, gamma=gamma):
                if s <= k:
                    raise ValueError("a recorded sample is reached: " + cj.name + " at " + str(s))
                return node_at(cj.nodes[a], min(s - k - 1, T - 1), reduce, gamma)

            for rt, i in zip(roots[0], inst[0]):  # the conjunct's imposed instances
                if rt < 0:
                    continue
                vals.append(evaluate_row(cj.program, cj.root_step, rt, masks, atom, reduce, memo, opened, smoothing == "gmr"))
                self.instances.append((cj.name, int(cj.anchors[i])))
                self.conj.append(ci)
        if not vals and smoothing == "gmr":
            raise ValueError("no instance is imposed at period " + str(k))
        self.conj = np.array(self.conj, np.int64)
        self.values = ca.vertcat(*vals)
        qref = states[k + 1:k + 1 + T, :D.NQ]
        E = ca.vertcat(*[X[j][:D.NQ] - ca.DM(qref[j]) for j in range(T)])  # samples
        dU = ca.vertcat(*[U[j + 1] - U[j] for j in range(T - 1)])  # intervals
        self.J = 0.5 * (W_Q * ca.sumsqr(E) + W_U * ca.sumsqr(w[nx * T:]) + W_DU * ca.sumsqr(dU))
        if form == "rows":
            self.g, self.f = ca.vertcat(*d, *([self.values - self.delta] if vals else [])), self.J
        else:
            self.eta = gmr_and([gmr_and([v for v, c in zip(vals, self.conj) if c == ci]) for ci in np.unique(self.conj)])
            self.g, self.f = ca.vertcat(*d, self.eta - self.delta), self.J - weight * self.eta
        self.m = self.g.numel()

    def bounds(self):
        """(lbx, ubx, lbg, ubg): the joint limits less their margin, |qd| and |u| within their limits, qd = 0 at the last
        sample; defects 0; rows at least 0."""
        T, NQ = self.T, D.NQ
        lo, hi = D.joint_bounds()
        v, a = D.VELOCITY_LIMIT, D.ACCELERATION_LIMIT
        lbx = np.concatenate([np.tile(np.concatenate([lo, np.full(NQ, -v)]), T), np.full(T * D.NU, -a)])
        ubx = np.concatenate([np.tile(np.concatenate([hi, np.full(NQ, v)]), T), np.full(T * D.NU, a)])
        term = (T - 1) * D.NX + NQ + np.arange(NQ)
        lbx[term], ubx[term] = 0.0, 0.0
        lbg = np.zeros(self.m)
        ubg = np.concatenate([np.zeros(self.n_def), np.full(self.m - self.n_def, np.inf)])
        return lbx, ubx, lbg, ubg

    def reference(self):
        """The scene's states at samples k + 1 .. k + T and the accelerations of its velocities: the start point."""
        k, T = self.k, self.T
        X = np.asarray(self.scene.states)
        Us = np.diff(X[k:k + 1 + T, D.NQ:], axis=0) / D.H_STEP
        return np.concatenate([X[k + 1:k + 1 + T].ravel(), Us.ravel()])

    def nlp(self):
        return {"x": self.w, "f": self.f, "g": self.g}


def shift_start(sol, keys_old, keys_new, m_new, T, d=1):
    """The start of a later period from sol (x, lam_g, lam_x of a solve): states, controls and their bound multipliers d
    intervals earlier (the last state repeated, the last controls and multipliers zero, the multipliers of the former
    terminal velocities zero), the defects' multipliers likewise, each row's multiplier carried by its instance key."""
    NX, NU, NQ = D.NX, D.NU, D.NQ
    nd = NX * T
    x, lg, lx = sol["x"], sol["lam_g"], sol["lam_x"]
    Xs, Us = x[:nd].reshape(T, NX), x[nd:].reshape(T, NU)
    Lx, Lu = lx[:nd].reshape(T, NX), lx[nd:].reshape(T, NU)
    Xn, Un = np.concatenate([Xs[d:], np.tile(Xs[-1:], (d, 1))]), np.concatenate([Us[d:], np.zeros((d, NU))])
    Lxn, Lun = np.concatenate([Lx[d:], np.zeros((d, NX))]), np.concatenate([Lu[d:], np.zeros((d, NU))])
    Lxn[T - 1 - d, NQ:] = 0.0
    lgn = np.zeros(m_new)
    lgn[:nd] = np.concatenate([lg[:nd].reshape(T, NX)[d:], np.zeros((d, NX))]).ravel()
    old = dict(zip(keys_old, lg[nd:]))
    lgn[nd:] = [old.get(key, 0.0) for key in keys_new]
    return np.concatenate([Xn.ravel(), Un.ravel()]), lgn, np.concatenate([Lxn.ravel(), Lun.ravel()])


# ------------------------------------------------------------------
# CasADi's own sparsity detection and colorings

AD_WEIGHT = 0.33  # CasADi's default weight between forward and reverse mode when it picks a Jacobian coloring


def _triplets(sp):
    r, c = sp.get_triplet()
    return np.asarray(r, np.int64), np.asarray(c, np.int64)


def _sparsity(rows, cols, shape):
    return ca.Sparsity.triplet(int(shape[0]), int(shape[1]), np.asarray(rows).tolist(), np.asarray(cols).tolist())


def detected_patterns(period):
    """CasADi's detected patterns of a Period: the constraint Jacobian and the upper triangle of the Lagrangian Hessian
    (the Jacobian pattern of the Lagrangian's gradient, symmetrized), as Sparsity objects."""
    x, f, g = period.w, period.f, period.g
    J = ca.jacobian_sparsity(g, x)
    lam, sig = ca.SX.sym("lam", g.numel()), ca.SX.sym("sigma")
    Hs = ca.jacobian_sparsity(ca.gradient(sig * f + ca.dot(lam, g), x), x)
    r, c = _triplets(Hs * Hs.T)
    return J, _sparsity(r[r <= c], c[r <= c], Hs.shape)


def colorings(J, H_upper):
    """CasADi's colorings of the patterns: the Hessian's star colors, the Jacobian's forward (columns) and reverse (rows)
    colors, and the mode CasADi keeps with its default weight."""
    r, c = _triplets(H_upper)
    Hf = _sparsity(np.concatenate([r, c]), np.concatenate([c, r]), H_upper.shape)
    nf, na = J.uni_coloring(J.T).size2(), J.T.uni_coloring(J).size2()
    m, n = J.shape
    w = AD_WEIGHT
    if w * n <= (1 - w) * m:
        mode = "reverse" if na <= np.floor(w * nf / (1 - w)) else "forward"
    else:
        mode = "forward" if nf <= np.floor((1 - w) * na / w) else "reverse"
    return {"hessian_star": int(Hf.star_coloring().size2()), "jacobian_forward": int(nf), "jacobian_reverse": int(na),
            "jacobian_mode": mode, "jacobian_products": int(nf if mode == "forward" else na)}
