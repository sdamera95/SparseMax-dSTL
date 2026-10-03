"""E040: the single-shooting constrained solve of constrained_warp with one constraint per conjunct of the specification's
top-level conjunction, in place of one constraint on the smoothed root (M003, design proposal point 1: a conjunctive top level is
imposed exactly, as separate rows). constrained_warp.py and the evaluators are not edited; this module reuses their pieces.

The specification is And(separation, slowdown, order, handover); its robustness is at least delta exactly when each conjunct's is,
so the solver keeps K constraints r_c(V) >= delta, r_c the arm's smoothed value of conjunct c (its own program, the conjunct as the
root; the same nodes, arities and per-node parameters as inside the full program), and no smoothed root.

Merit (augmented Lagrangian, one-sided term per conjunct):
    L = lam E + sum_c (mu_c / 2) max(0, delta - r_c + nu_c / mu_c)^2,   w_c = max(0, nu_c + mu_c (delta - r_c)) = -dL/dr_c.
The descent direction is -grad L / |grad L| with grad L = lam 2 V / N - sum_c w_c dr_c/dV. The plant backward pass runs once per
iterate: each conjunct's gradient with respect to the atoms (its own Evaluator) is pulled back through the predicates to the
sampled states (C_c), and the plant's vector-Jacobian product is applied once to sum_c w_c C_c. The line search, the trial count,
the step-size state and the Armijo constant are constrained_warp's. Multiplier and penalty: E033's schedule per conjunct (choices
recorded in E040's entry): after every block of EVERY iterates nu_c <- max(0, nu_c + mu_c (delta - r_c)) and mu_c doubles (to at
most MU_MAX) when conjunct c's violation at the block's end exceeds half of its violation at the block's start and VIOL_TOL; each
conjunct has its own penalty, so a satisfied conjunct's penalty does not grow.

A claim is the arm's values of all K conjuncts at least delta; the trace's "rho" column holds min_c r_c (the claim value), "w" the
sum of the weights, "mu" and "nu" their largest values; the per-conjunct values, weights, penalties and multipliers are returned
separately. With K = 1 (the full program as the one conjunct) every operation is constrained_warp's in the same order, so the
iterates are bit-equal to constrained_warp.solve's (tests/test_e040_conj.py).
"""
import time

import numpy as np
import warp as wp

from . import constrained_warp as CW
from .stl.warp_backend import Evaluator, matched_param
from .warp_plant import Plant
from .warp_predicates import Predicates


class ConjChain:
    """constrained_warp.Chain with K programs per group (the conjuncts): forward(V) returns r (n, K) and keeps the K state
    cotangents; pullback(w (n, K)) applies the plant's backward pass once to sum_c w_c C_c; values(Va) returns (n, lanes, K).
    groups: ((method, eps, a, b, (program_1, ..., program_K)), ...), every group with the same K, every program over prog.T samples."""

    def __init__(self, spec_plant, sc, prog, x0, pick, handover, hc, hr, groups, trials=CW.TRIALS, device="cuda:0"):
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
        self.K = len(self.groups[0][4])
        bounds = sorted((g[2], g[3]) for g in self.groups)
        if bounds[0][0] != 0 or bounds[-1][1] != n or any(b != a2 for (_, b), (a2, _) in zip(bounds, bounds[1:])):
            raise ValueError("groups must cover the runs 0..n-1 in contiguous blocks")
        if any(len(g[4]) != self.K for g in self.groups):
            raise ValueError("every group needs the same number of conjunct programs")
        self.ev, self.ev_t = [], []
        for name, eps, a, b, progs in self.groups:  # over (method, eps, wait) groups (a handful)
            evs, evs_t = [], []
            for pg in progs:  # over the conjuncts (formula structure)
                if pg.T != self.T:
                    raise ValueError("every program must cover prog.T samples")
                p = matched_param(pg, name, float(eps))
                # P: every evaluator reads the full atom layout (a conjunct's program reads only its own atoms' columns)
                evs.append(Evaluator(pg, name, p, b - a, wp.float32, device, P=self.pred.P))
                evs_t.append(Evaluator(pg, name, p, (b - a) * L, wp.float32, device, P=self.pred.P))
            self.ev.append((a, b, evs))
            self.ev_t.append((a * L, b * L, evs_t))
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
        """r (n, K) at V (n, T - 1, 7); keeps each conjunct's gradient with respect to the sampled states for pullback()."""
        t0 = time.perf_counter()
        X = self.plant.rollout(self.x0, V)
        t0 = self._mark("rollout", t0)
        q, v = self._states(X, True)
        tape = wp.Tape()
        Z = self.pred.scores(q, v, tape)
        r = np.empty((self.n, self.K), np.float32)
        C = []
        for c in range(self.K):  # over the conjuncts (formula structure)
            if c:
                tape.zero()
            for a, b, evs in self.ev:  # over groups
                rr, G = evs[c].gradient(Z[a:b])
                wp.copy(self.gZ[a:b], G)
                r[a:b, c] = rr.numpy()[:, 0]
            tape.backward(grads={Z: self.gZ})
            C.append(np.concatenate([q.grad.numpy(), v.grad.numpy()], -1).reshape(self.n, self.T, -1))
        self.C = np.stack(C, 1)
        self._mark("stl", t0)
        return r

    def pullback(self, w):
        """sum over runs and conjuncts of w_rc d r_rc / d V (n, T - 1, 7) at the last forward(); w (n, K)."""
        t0 = time.perf_counter()
        w = np.asarray(w, np.float32)
        if self.K == 1:
            Cw = self.C[:, 0] * w[:, 0][:, None, None]
        else:
            Cw = np.sum(self.C * w[:, :, None, None], 1)
        gV, _ = self.plant.vjp(Cw)
        self._mark("plant_backward", t0)
        return gV

    def values(self, Va):
        """r (n, lanes, K) at the points Va (n, lanes, T - 1, 7)."""
        t0 = time.perf_counter()
        X = self.plant_t.rollout(self.x0_t, Va.reshape((self.n * self.lanes,) + Va.shape[2:]))
        t0 = self._mark("trial_rollout", t0)
        q, v = self._states(X, False)
        Z = self.pred_t.scores(q, v)
        out = np.empty((self.n * self.lanes, self.K), np.float32)
        for a, b, evs in self.ev_t:  # over groups
            for c, ev in enumerate(evs):  # over the conjuncts
                out[a:b, c] = ev.value(Z[a:b]).numpy()[:, 0]
        self._mark("trial_stl", t0)
        return out.reshape(self.n, self.lanes, self.K)


def init_state(V0, alpha0, K):
    n = len(V0)
    f = np.float32
    return {"V": np.asarray(V0, f), "alpha": np.full(n, alpha0, f), "nu": np.zeros((n, K), f), "mu": np.full((n, K), CW.MU0, f), "k": 0}


def iterate(chain, state, ex, lam, delta, c=CW.ARMIJO_C):
    """One iterate of constrained_warp.iterate with the per-conjunct merit. Returns the new (V, alpha), the trace row
    (n, len(CW.TRACE_KEYS)) and the per-conjunct record {r, w, r_next, r_ref} (n, K) each."""
    V, alpha, nu, mu = state["V"], state["alpha"], state["nu"], state["mu"]
    n, N = V.shape[:2]
    trials = chain.lanes - 1
    r = chain.forward(V)
    s = time.perf_counter()
    E = CW.effort(V)
    w = CW.al_weight(r, nu, mu, delta)
    L0 = lam * E + np.sum(0.5 * mu * np.maximum(0.0, delta - r + nu / mu) ** 2, -1)
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
    dE = np.sum((Vt - V[:, None]) * (Vt + V[:, None]), (-2, -1)) / V.shape[-2]
    s0 = np.maximum(0.0, delta - r_ref + nu / mu)[:, None]
    st = np.maximum(0.0, delta - rt + (nu / mu)[:, None])
    dL = lam * dE + np.sum(0.5 * mu[:, None] * (st - s0) * (st + s0), -1)
    Vn, an, j, hit, pick = CW.select_step(V, alpha, dL, g, a, Vt, c)
    rn = np.where(hit[:, None], pick(rt), r_ref)
    Ln = lam * E + np.sum(0.5 * mu * np.maximum(0.0, delta - r_ref + nu / mu) ** 2, -1) + np.where(hit, pick(dL), 0.0)
    rec = np.stack([L0, r.min(1), ex, E, gn, w.sum(1), mu.max(1), nu.max(1), np.where(hit, an, 0.0), j.astype(V.dtype), hit.astype(V.dtype), Ln,
                    rn.min(1), r_ref.min(1)], -1)
    chain.times["solver"] = chain.times.get("solver", 0.0) + solver + time.perf_counter() - s
    conj = {"r": r, "w": w, "r_next": rn, "r_ref": r_ref, "mu": mu.copy(), "nu": nu.copy()}
    return Vn.astype(np.float32), an.astype(np.float32), rec.astype(np.float32), conj


def block_update(state, r_start, r_end, delta):
    """constrained_warp.block_update per conjunct: r_start, r_end (n, K)."""
    nu, mu = state["nu"], state["mu"]
    state["nu"] = CW.multiplier(nu, mu, r_end, delta).astype(np.float32)
    state["mu"] = CW.penalty(mu, np.maximum(0.0, delta - r_start), np.maximum(0.0, delta - r_end)).astype(np.float32)


def solve(chain, referee, state, iterations, lam, delta, ref=None, log=None, stop_certified=False):
    """constrained_warp.solve with the per-conjunct iterate and block update; the records of constrained_warp.solve plus
    conj (dict of (n, iterates, K) arrays: r, w, r_next, r_ref, mu, nu)."""
    if iterations % CW.EVERY and not stop_certified:
        raise ValueError("iterations must be a multiple of EVERY")
    keys = CW.GRADIENT + CW.TRIAL + CW.REFEREE + ("solver",)
    t_start = time.perf_counter()
    if ref is None:
        ref = referee(state["V"])
    rho64, conj64 = [ref[0]], [ref[1]]
    wall = [time.perf_counter() - t_start]
    recs, Vs, secs, cons = [], [], [], []
    r_start = None
    for i in range(iterations):  # over solver iterates (sequential by definition)
        before = {k: chain.times.get(k, 0.0) + referee.times.get(k, 0.0) for k in keys}
        V, alpha, rec, conj = iterate(chain, state, rho64[-1], lam, delta)
        state["V"], state["alpha"] = V, alpha
        if state["k"] % CW.EVERY == 0:
            r_start = conj["r"]
        state["k"] += 1
        if state["k"] % CW.EVERY == 0:
            block_update(state, r_start, conj["r_next"], delta)
        ref = referee(V)
        rho64.append(ref[0])
        conj64.append(ref[1])
        wall.append(time.perf_counter() - t_start)
        recs.append(rec)
        Vs.append(V)
        cons.append(conj)
        secs.append([chain.times.get(k, 0.0) + referee.times.get(k, 0.0) - before[k] for k in keys])
        if log is not None:
            log(state["k"], rec, ref[0], wall[-1])
        if stop_certified and np.all(np.max(rho64, 0) >= delta):
            break
    return {"trace": np.stack(recs, 1), "V": np.stack(Vs, 1), "referee64": np.stack(rho64, 1), "conjuncts64": np.stack(conj64, 1),
            "seconds": np.asarray(secs), "seconds_keys": keys, "wall": np.asarray(wall),
            "conj": {k: np.stack([c_[k] for c_ in cons], 1) for k in cons[0]}}
