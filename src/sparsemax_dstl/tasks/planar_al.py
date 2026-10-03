"""E049 round 2: the single-shooting constrained solve of the manipulator study with the planar unicycle as
the plant, on a JAX chain (this module) and on a Warp chain (planar_warp.py), driven by the same update.

Solver (E029's merit and step, E040's one constraint per conjunct; the update below is
constrained_conj.iterate with one added term). Over normalized commands V (n, N, 2) in the box |V| <= 1,
the command U = V * U_MAX applied over each step,

    L = lam E(V) - reward r_0(V) + sum_c (mu_c / 2) max(0, delta - r_c + nu_c / mu_c)^2,
    w_c = max(0, nu_c + mu_c (delta - r_c)),   grad L = lam 2 V / N - sum_c (w_c + reward [c = 0]) d r_c / d V,

E(V) = sum(V^2) / N, r_c the smoothed robustness of conjunct c at t = 0. reward = 0 is constrained_conj's
merit (the constraint form); the authors' form of the generalized-mean robustness (its eq. (11)) is the
one-program chain (the whole STL specification as conjunct 0) with reward > 0. The step: the direction
-grad L / |grad L|, the trial steps 2 alpha 0.5^j (j < TRIALS) projected onto the box and evaluated in
parallel with the current point as a reference lane, the first trial meeting the Armijo test on the
change of L assembled from differences (constrained_warp.select_step), and every EVERY updates the
multiplier and penalty update per conjunct (constrained_conj.block_update). Constants are
constrained_warp's: TRIALS 6, EVERY 10, ARMIJO_C 1e-4, MU0 1, MU_MAX 64, ALPHA_MIN 1e-6.

Chain interface (as constrained_conj.ConjChain): forward(V) -> r (n, K) and keeps d r_c / d V;
pullback(w) -> sum_c w_c d r_c / d V (n, N, 2); values(Va) -> r (n, lanes, K) at Va (n, lanes, N, 2);
times (dict of seconds).
"""
import numpy as np

# ------------------------------------------------------------------
# the solver's arithmetic: constrained_warp.py lines 54-60 and 69-113 (at 607f883), copied unchanged,
# because importing constrained_warp loads the manipulator's MuJoCo model


class CW:
    ARMIJO_C = 1e-4
    TRIALS = 6
    EVERY = 10
    MU0 = 1.0
    MU_MAX = 64.0
    ALPHA_MIN = 1e-6
    VIOL_TOL = 1e-3

    @staticmethod
    def effort(V):
        """sum(V^2) / N per run, V (..., N, m)."""
        return np.sum(V * V, (-2, -1)) / V.shape[-2]

    @staticmethod
    def al_weight(r, nu, mu, delta):
        """w = max(0, nu + mu (delta - r)) = -d L / d r."""
        return np.maximum(0.0, nu + mu * (delta - r))

    @staticmethod
    def multiplier(nu, mu, r, delta):
        return np.maximum(0.0, nu + mu * (delta - r))

    @staticmethod
    def penalty(mu, viol_start, viol_end, mu_max=64.0, tol=1e-3):
        return np.where((viol_end > 0.5 * viol_start) & (viol_end > tol), np.minimum(2.0 * mu, mu_max), mu)

    @staticmethod
    def select_step(V, alpha, dL, g, a, Vt, c=1e-4):
        """The first trial j with dL <= c g . (Vt - V) per run (constrained.select_step)."""
        n, trials = a.shape
        drop = np.sum(g[:, None] * (Vt - V[:, None]), (-2, -1))
        ok = np.isfinite(dL) & (dL <= c * drop)
        hit = np.any(ok, 1)
        j = np.argmax(ok, 1)
        pick = lambda x: np.take_along_axis(x, j.reshape((n, 1) + (1,) * (x.ndim - 2)), 1)[:, 0]  # noqa: E731
        Vn = np.where(hit[:, None, None], pick(Vt), V)
        an = np.maximum(np.where(hit, pick(a), alpha * 0.5 ** trials), 1e-6)
        return Vn, an, j, hit, pick


def init_state(V0, alpha0, K):
    n = len(V0)
    return {"V": np.asarray(V0, np.float64), "alpha": np.full(n, alpha0), "nu": np.zeros((n, K)), "mu": np.full((n, K), CW.MU0), "k": 0}


def iterate(chain, state, lam, delta, reward=0.0, c=CW.ARMIJO_C):
    """One update; constrained_conj.iterate with the reward term. Returns V, alpha and the record
    {L, r (n, K), w (n, K), accepted (n,)}."""
    V, alpha, nu, mu = state["V"], state["alpha"], state["nu"], state["mu"]
    n, N = V.shape[:2]
    trials = chain.lanes - 1
    r = chain.forward(V)
    E = CW.effort(V)
    w = CW.al_weight(r, nu, mu, delta)
    wr = w.copy()
    wr[:, 0] += reward
    L0 = lam * E - reward * r[:, 0] + np.sum(0.5 * mu * np.maximum(0.0, delta - r + nu / mu) ** 2, -1)
    g = lam * 2.0 * V / N - chain.pullback(wr)
    gn = np.sqrt(np.sum(g * g, (1, 2)))
    d = -g / np.maximum(gn, 1e-30)[:, None, None]
    a = 2.0 * alpha[:, None] * 0.5 ** np.arange(trials)
    Vt = np.clip(V[:, None] + a[..., None, None] * d[:, None], -1.0, 1.0)
    ra = chain.values(np.concatenate([Vt, V[:, None]], 1))
    rt, r_ref = ra[:, :trials], ra[:, trials]
    dE = np.sum((Vt - V[:, None]) * (Vt + V[:, None]), (-2, -1)) / N
    s0 = np.maximum(0.0, delta - r_ref + nu / mu)[:, None]
    st = np.maximum(0.0, delta - rt + (nu / mu)[:, None])
    dL = lam * dE - reward * (rt[..., 0] - r_ref[:, None, 0]) + np.sum(0.5 * mu[:, None] * (st - s0) * (st + s0), -1)
    Vn, an, j, hit, pick = CW.select_step(V, alpha, dL, g, a, Vt, c)
    rn = np.where(hit[:, None], pick(rt), r_ref)
    return Vn, an, {"L": L0, "r": r, "w": w, "r_next": rn, "accepted": hit}


def block_update(state, r_start, r_end, delta):
    nu, mu = state["nu"], state["mu"]
    state["nu"] = CW.multiplier(nu, mu, r_end, delta)
    state["mu"] = CW.penalty(mu, np.maximum(0.0, delta - r_start), np.maximum(0.0, delta - r_end))


def solve(chain, V0, budget, lam, delta, alpha0, reward=0.0):
    """budget updates from V0 (n, N, 2), every run the same number (no early stop). Returns the final V,
    and per update the smoothed values r (n, budget, K) and the merit L (n, budget)."""
    if budget % CW.EVERY:
        raise ValueError("budget must be a multiple of EVERY")
    state = init_state(V0, alpha0, chain.K)
    rs, Ls = [], []
    r_start = None
    for _ in range(budget):  # over solver updates (sequential by definition)
        V, alpha, rec = iterate(chain, state, lam, delta, reward)
        state["V"], state["alpha"] = V, alpha
        if state["k"] % CW.EVERY == 0:
            r_start = rec["r"]
        state["k"] += 1
        if state["k"] % CW.EVERY == 0:
            block_update(state, r_start, rec["r_next"], delta)
        rs.append(rec["r"])
        Ls.append(rec["L"])
    return state["V"], np.stack(rs, 1), np.stack(Ls, 1)
