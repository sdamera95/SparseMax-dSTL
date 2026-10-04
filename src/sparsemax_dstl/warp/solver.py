"""First-order augmented Lagrangian solver under single shooting with one constraint on the smoothed
robustness of the whole specification (the single-constraint formulation of Appendix I-C)."""
import os
import time

import mujoco
import numpy as np
import warp as wp
from mujoco import rollout as mj_rollout

from .evaluator import Evaluator, evaluate_warp, matched_param
from .plant import Plant
from .predicates import Predicates

ARMIJO_C = 1e-4
TRIALS = 6
EVERY = 10
MU0 = 1.0
MU_MAX = 64.0
ALPHA_MIN = 1e-6
VIOL_TOL = 1e-3
# one trace row per run and iterate, at the iterate before the step; "exact" is the float64 replay's value
TRACE_KEYS = ("L", "rho", "exact", "effort", "grad_norm", "w", "mu", "nu", "step", "trial", "accepted", "L_next", "rho_next", "rho_ref")
GRADIENT = ("rollout", "stl", "plant_backward")
TRIAL = ("trial_rollout", "trial_stl")
REFEREE = ("referee_mujoco", "referee_stl")


# ------------------------------------------------------------------
# the solver's arithmetic

def effort(V):
    """sum(V^2) / N per run, V (..., N, m)."""
    return np.sum(V * V, (-2, -1)) / V.shape[-2]


def al_value(r, E, nu, mu, lam, delta):
    """L = lam E + (mu / 2) max(0, delta - r + nu / mu)^2."""
    return lam * E + 0.5 * mu * np.maximum(0.0, delta - r + nu / mu) ** 2


def al_weight(r, nu, mu, delta):
    """w = max(0, nu + mu (delta - r)) = -d L / d r."""
    return np.maximum(0.0, nu + mu * (delta - r))


def multiplier(nu, mu, r, delta):
    return np.maximum(0.0, nu + mu * (delta - r))


def penalty(mu, viol_start, viol_end, mu_max=MU_MAX, tol=VIOL_TOL):
    """mu doubled, up to mu_max, where the violation at the block's end exceeds tol and half of its
    value at the block's start."""
    return np.where((viol_end > 0.5 * viol_start) & (viol_end > tol), np.minimum(2.0 * mu, mu_max), mu)


def al_change(V, Vt, r, rt, nu, mu, lam, delta):
    """L(Vt) - L(V) per trial, from differences: V (n, N, m), Vt (n, trials, N, m), r (n,), rt (n, trials)."""
    dE = np.sum((Vt - V[:, None]) * (Vt + V[:, None]), (-2, -1)) / V.shape[-2]
    s0 = np.maximum(0.0, delta - r + nu / mu)[:, None]
    st = np.maximum(0.0, delta - rt + (nu / mu)[:, None])
    return lam * dE + 0.5 * mu[:, None] * (st - s0) * (st + s0)


def select_step(V, alpha, dL, g, a, Vt, c=ARMIJO_C):
    """The first trial j with dL <= c g . (Vt - V) per run. Returns the new V, the new step scale, the
    accepted index, the acceptance flags and the picking function."""
    n, trials = a.shape
    drop = np.sum(g[:, None] * (Vt - V[:, None]), (-2, -1))
    ok = np.isfinite(dL) & (dL <= c * drop)
    hit = np.any(ok, 1)
    j = np.argmax(ok, 1)
    pick = lambda x: np.take_along_axis(x, j.reshape((n, 1) + (1,) * (x.ndim - 2)), 1)[:, 0]  # noqa: E731
    Vn = np.where(hit[:, None, None], pick(Vt), V)
    an = np.maximum(np.where(hit, pick(a), alpha * 0.5 ** trials), ALPHA_MIN)
    return Vn, an, j, hit, pick


def certification(ref, delta=0.0):
    """From exact robustness values ref (n, K) at iterates 0..K-1: the first iterate with ref >= delta
    (-1 if none) and whether every later value stays >= delta."""
    ok = np.asarray(ref) >= delta
    first = np.where(ok.any(1), np.argmax(ok, 1), -1)
    tail_ok = np.flip(np.logical_and.accumulate(np.flip(ok, 1), 1), 1)
    stays = np.where(first >= 0, np.take_along_axis(tail_ok, np.maximum(first, 0)[:, None], 1)[:, 0], False)
    return first, stays


# ------------------------------------------------------------------
# the plant-and-specification function

class Chain:
    """Controls to the smoothed robustness of n runs, its gradient and the trial values of the line search.
    groups ((method, eps, a, b[, program]), ...): runs a..b-1 use method at node error eps; x0 is (n, 14)."""

    def __init__(self, spec_plant, sc, prog, x0, pick, handover, hc, hr, groups, trials=TRIALS, device="cuda:0"):
        self.device = wp.get_device(device)
        self.n, self.T, self.lanes = len(x0), prog.T, trials + 1
        n, L = self.n, self.lanes
        rep = lambda x: np.repeat(np.asarray(x), L, axis=0)  # noqa: E731
        self.x0, self.x0_t = np.asarray(x0, np.float32), rep(np.asarray(x0, np.float32))
        self.plant = Plant(n, self.T - 1, device=device)
        self.plant_t = Plant(n * L, self.T - 1, device=device)
        self.pred = Predicates(spec_plant, sc, self.T, nworld=n, device=device)
        self.pred.set_instance(pick, handover, hc, hr)
        self.pred_t = Predicates(spec_plant, sc, self.T, nworld=n * L, device=device)
        self.pred_t.set_instance(rep(pick), rep(handover), rep(hc), hr)
        self.groups = tuple(groups)
        bounds = sorted((g[2], g[3]) for g in self.groups)
        if bounds[0][0] != 0 or bounds[-1][1] != n or any(b != a2 for (_, b), (a2, _) in zip(bounds, bounds[1:])):
            raise ValueError("groups must cover the runs 0..n-1 in contiguous blocks")
        self.ev, self.ev_t = [], []
        for g in self.groups:  # over (method, eps) groups (a handful)
            name, eps, a, b = g[:4]
            pg = g[4] if len(g) > 4 else prog
            if pg.T != self.T:
                raise ValueError("every group's program must cover prog.T samples")
            p = matched_param(pg, name, float(eps))
            self.ev.append((a, b, Evaluator(pg, name, p, b - a, wp.float32, device)))
            self.ev_t.append((a * L, b * L, Evaluator(pg, name, p, (b - a) * L, wp.float32, device)))
        with wp.ScopedDevice(self.device):
            self.gZ = wp.zeros((n, self.T, self.pred.P), dtype=float)
        self.C = None
        self.times = {}

    def _mark(self, key, t0):
        wp.synchronize_device(self.device)
        t1 = time.perf_counter()
        self.times[key] = self.times.get(key, 0.0) + t1 - t0
        return t1

    def _states(self, X, grad):
        nq = X.shape[-1] // 2
        q = wp.array(np.ascontiguousarray(X[..., :nq].reshape(-1, nq)), dtype=float, device=self.device, requires_grad=grad)
        v = wp.array(np.ascontiguousarray(X[..., nq:].reshape(-1, nq)), dtype=float, device=self.device, requires_grad=grad)
        return q, v

    def forward(self, V):
        """rho (n,) at V (n, T - 1, 7), keeping the gradient of rho with respect to the sampled
        states for pullback()."""
        t0 = time.perf_counter()
        X = self.plant.rollout(self.x0, V)
        t0 = self._mark("rollout", t0)
        q, v = self._states(X, True)
        tape = wp.Tape()
        Z = self.pred.scores(q, v, tape)
        rho = np.empty(self.n, np.float32)
        for a, b, ev in self.ev:  # over (method, eps) groups
            r, G = ev.gradient(Z[a:b])
            wp.copy(self.gZ[a:b], G)
            rho[a:b] = r.numpy()[:, 0]
        tape.backward(grads={Z: self.gZ})
        self.C = np.concatenate([q.grad.numpy(), v.grad.numpy()], -1).reshape(self.n, self.T, -1)
        self._mark("stl", t0)
        return rho

    def pullback(self, w):
        """w_r d rho_r / d V_r for every run r, (n, T - 1, 7), at the last forward()."""
        t0 = time.perf_counter()
        gV, _ = self.plant.vjp(self.C * np.asarray(w, np.float32)[:, None, None])
        self._mark("plant_backward", t0)
        return gV

    def values(self, Va):
        """rho (n, lanes) at the points Va (n, lanes, T - 1, 7)."""
        t0 = time.perf_counter()
        X = self.plant_t.rollout(self.x0_t, Va.reshape((self.n * self.lanes,) + Va.shape[2:]))
        t0 = self._mark("trial_rollout", t0)
        q, v = self._states(X, False)
        Z = self.pred_t.scores(q, v)
        out = np.empty(self.n * self.lanes, np.float32)
        for a, b, ev in self.ev_t:  # over (method, eps) groups
            out[a:b] = ev.value(Z[a:b]).numpy()[:, 0]
        self._mark("trial_stl", t0)
        return out.reshape(self.n, self.lanes)


class Referee:
    """The exact robustness of n runs on a float64 MuJoCo replay: __call__(V (n, T - 1, 7)) returns it
    (n,) and the top node's children (n, c); blocks ((a, b, program), ...) gives runs a..b-1 a program."""

    def __init__(self, spec_plant, sc, prog, x0, pick, handover, hc, hr, device="cuda:0", mjm=None, n_sub=10, times=None, blocks=None):
        from ..tasks import panda
        self.mjm = panda.model() if mjm is None else mjm
        self.nq, self.n_sub = self.mjm.nq, n_sub
        if mujoco.mj_stateSize(self.mjm, mujoco.mjtState.mjSTATE_FULLPHYSICS) != 1 + 2 * self.nq:
            raise NotImplementedError("the full physics state must be time, qpos and qvel")
        self.n, self.T = len(x0), prog.T
        self.x0 = np.asarray(x0, np.float64)
        self.state0 = np.concatenate([np.zeros((self.n, 1)), self.x0], -1)
        self.datas = [mujoco.MjData(self.mjm) for _ in range(min(self.n, len(os.sched_getaffinity(0))))]
        self.umax = np.asarray(panda.torque_limit(), np.float64)
        self.device = wp.get_device(device)
        self.pred = Predicates(spec_plant, sc, self.T, nworld=self.n, dtype=wp.float64, device=device)
        self.pred.consts[-2:] = [wp.float64(0.0), wp.float64(0.0)]  # eps_length, eps_speed: unsmoothed norms
        self.pred.set_instance(np.asarray(pick, np.float64), np.asarray(handover, np.float64), np.asarray(hc, np.float64), np.asarray(hr, np.float64))
        self.prog = prog
        root = prog.steps[prog.root]
        self.sources, self.r_idx = list(root.sources), np.asarray(root.index[0, :root.count[0]])
        self.blocks = ((0, self.n, prog),) if blocks is None else tuple(sorted(blocks, key=lambda x: x[0]))
        if self.blocks[0][0] != 0 or self.blocks[-1][1] != self.n or any(x[1] != y[0] for x, y in zip(self.blocks, self.blocks[1:])):
            raise ValueError("blocks must cover the runs 0..n-1 in contiguous blocks")
        self.times = {} if times is None else times

    def states(self, V):
        """The float64 MuJoCo states (n, T, 14) at the interval boundaries."""
        U = np.repeat(np.asarray(V, np.float64) * self.umax, self.n_sub, axis=1)
        st, _ = mj_rollout.rollout(self.mjm, self.datas, self.state0, U)
        return np.concatenate([self.x0[:, None], st[:, self.n_sub - 1::self.n_sub, 1:]], 1)

    def __call__(self, V):
        t0 = time.perf_counter()
        X = self.states(V)
        t1 = time.perf_counter()
        self.times["referee_mujoco"] = self.times.get("referee_mujoco", 0.0) + t1 - t0
        nq = self.nq
        q = wp.array(np.ascontiguousarray(X[..., :nq].reshape(-1, nq)), dtype=wp.float64, device=self.device)
        v = wp.array(np.ascontiguousarray(X[..., nq:].reshape(-1, nq)), dtype=wp.float64, device=self.device)
        Z = self.pred.scores(q, v)
        rho, conj = np.empty(self.n), []
        for a, b, pg in self.blocks:  # over programs (a handful)
            vals, offsets = evaluate_warp(pg, Z if (a, b) == (0, self.n) else Z[a:b], "exact")
            vals = vals.numpy()
            root = pg.steps[pg.root]
            rho[a:b] = vals[:, offsets[pg.root]]
            r_idx = np.asarray(root.index[0, :root.count[0]])
            conj.append(np.concatenate([vals[:, offsets[j]:offsets[j] + pg.steps[j].length] for j in root.sources], -1)[:, r_idx])
        conj = np.concatenate(conj)
        self.times["referee_stl"] = self.times.get("referee_stl", 0.0) + time.perf_counter() - t1
        return rho, conj


# ------------------------------------------------------------------
# the loop

def init_state(V0, alpha0):
    n = len(V0)
    f = np.float32
    return {"V": np.asarray(V0, f), "alpha": np.full(n, alpha0, f), "nu": np.zeros(n, f), "mu": np.full(n, MU0, f), "k": 0}


def iterate(chain, state, ex, lam, delta, c=ARMIJO_C):
    """One step of normalized gradient descent on the augmented Lagrangian with an Armijo line search. Returns
    the new (V, alpha) and the trace row (n, len(TRACE_KEYS)); ex is the exact robustness at V."""
    V, alpha, nu, mu = state["V"], state["alpha"], state["nu"], state["mu"]
    n, N = V.shape[:2]
    trials = chain.lanes - 1
    r = chain.forward(V)
    s = time.perf_counter()
    E = effort(V)
    w = al_weight(r, nu, mu, delta)
    L0 = al_value(r, E, nu, mu, lam, delta)
    solver = time.perf_counter() - s
    pulled = chain.pullback(w)
    s = time.perf_counter()
    g = lam * 2.0 * V / N - pulled
    gn = np.sqrt(np.sum(g * g, (1, 2)))
    d = -g / np.maximum(gn, 1e-30)[:, None, None]
    a = 2.0 * alpha[:, None] * 0.5 ** np.arange(trials, dtype=V.dtype)
    Vt = np.clip(V[:, None] + a[..., None, None] * d[:, None], -1.0, 1.0)
    Va = np.concatenate([Vt, V[:, None]], 1)
    solver += time.perf_counter() - s
    ra = chain.values(Va)
    s = time.perf_counter()
    rt, r_ref = ra[:, :trials], ra[:, trials]
    dL = al_change(V, Vt, r_ref, rt, nu, mu, lam, delta)
    Vn, an, j, hit, pick = select_step(V, alpha, dL, g, a, Vt, c)
    rn = np.where(hit, pick(rt), r_ref)
    Ln = al_value(r_ref, E, nu, mu, lam, delta) + np.where(hit, pick(dL), 0.0)
    rec = np.stack([L0, r, ex, E, gn, w, mu, nu, np.where(hit, an, 0.0), j.astype(V.dtype), hit.astype(V.dtype), Ln, rn, r_ref], -1)
    chain.times["solver"] = chain.times.get("solver", 0.0) + solver + time.perf_counter() - s
    return Vn.astype(np.float32), an.astype(np.float32), rec.astype(np.float32)


def block_update(state, r_start, r_end, delta, tol=None):
    """The multiplier and penalty updates after a block of EVERY iterates. tol: the violation below which
    the penalty is not doubled, a number or one per run (None: VIOL_TOL)."""
    nu, mu = state["nu"], state["mu"]
    state["nu"] = multiplier(nu, mu, r_end, delta).astype(np.float32)
    state["mu"] = penalty(mu, np.maximum(0.0, delta - r_start), np.maximum(0.0, delta - r_end),
                          tol=VIOL_TOL if tol is None else tol).astype(np.float32)


def solve(chain, referee, state, iterations, lam, delta, ref=None, log=None, stop_certified=False, viol_tol=None):
    """iterations more iterates from state, with the float64 replay after each. Returns trace (n, K, 14),
    V (n, K, N, m), referee64 (n, K + 1), conjuncts64 (n, K + 1, c), seconds (K, parts), wall (K + 1,)."""
    if iterations % EVERY and not stop_certified:
        raise ValueError("iterations must be a multiple of EVERY")
    keys = GRADIENT + TRIAL + REFEREE + ("solver",)
    t_start = time.perf_counter()
    if ref is None:
        ref = referee(state["V"])
    rho64, conj64 = [ref[0]], [ref[1]]
    wall = [time.perf_counter() - t_start]
    recs, Vs, secs = [], [], []
    r_start = None
    for i in range(iterations):  # over solver iterates (sequential by definition)
        before = {k: chain.times.get(k, 0.0) + referee.times.get(k, 0.0) for k in keys}
        V, alpha, rec = iterate(chain, state, rho64[-1], lam, delta)
        state["V"], state["alpha"] = V, alpha
        if state["k"] % EVERY == 0:
            r_start = rec[:, 1]
        state["k"] += 1
        if state["k"] % EVERY == 0:
            block_update(state, r_start, rec[:, 12], delta, viol_tol)
        ref = referee(V)
        rho64.append(ref[0])
        conj64.append(ref[1])
        wall.append(time.perf_counter() - t_start)
        recs.append(rec)
        Vs.append(V)
        secs.append([chain.times.get(k, 0.0) + referee.times.get(k, 0.0) - before[k] for k in keys])
        if log is not None:
            log(state["k"], rec, ref[0], wall[-1])
        if stop_certified and np.all(np.max(rho64, 0) >= delta):
            break
    return {"trace": np.stack(recs, 1), "V": np.stack(Vs, 1), "referee64": np.stack(rho64, 1), "conjuncts64": np.stack(conj64, 1),
            "seconds": np.asarray(secs), "seconds_keys": keys, "wall": np.asarray(wall)}
