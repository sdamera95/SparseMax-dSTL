"""First-order augmented Lagrangian solver of the planar unicycle example (Section V-A of the paper) under single
shooting, with one constraint per conjunct, over a chain: planar_al_jax.JaxChain or planar_warp.WarpChain."""
import numpy as np

# ------------------------------------------------------------------
# constants and arithmetic of the update


class Update:
    """Constants and arithmetic of the update."""
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
        """nu <- max(0, nu + mu (delta - r))."""
        return np.maximum(0.0, nu + mu * (delta - r))

    @staticmethod
    def penalty(mu, viol_start, viol_end, mu_max=64.0, tol=1e-3):
        """mu doubled, up to mu_max, where the violation exceeds tol and half its value at the block's start."""
        return np.where((viol_end > 0.5 * viol_start) & (viol_end > tol), np.minimum(2.0 * mu, mu_max), mu)

    @staticmethod
    def select_step(V, alpha, dL, g, a, Vt, c=1e-4):
        """The first trial j with dL <= c g . (Vt - V) per run; without one, V is kept and the step shrinks."""
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
    return {"V": np.asarray(V0, np.float64), "alpha": np.full(n, alpha0), "nu": np.zeros((n, K)), "mu": np.full((n, K), Update.MU0), "k": 0}


def iterate(chain, state, lam, delta, reward=0.0, c=Update.ARMIJO_C):
    """One update of V along -grad L / |grad L| with trial steps 2 alpha 0.5^j clipped to the box, where
    L = lam E - reward r_0 + sum_c (mu_c / 2) max(0, delta - r_c + nu_c / mu_c)^2. Returns V, alpha and a record."""
    V, alpha, nu, mu = state["V"], state["alpha"], state["nu"], state["mu"]
    n, N = V.shape[:2]
    trials = chain.lanes - 1
    r = chain.forward(V)
    E = Update.effort(V)
    w = Update.al_weight(r, nu, mu, delta)
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
    Vn, an, j, hit, pick = Update.select_step(V, alpha, dL, g, a, Vt, c)
    rn = np.where(hit[:, None], pick(rt), r_ref)
    return Vn, an, {"L": L0, "r": r, "w": w, "r_next": rn, "accepted": hit}


def block_update(state, r_start, r_end, delta):
    """Multiplier and penalty update of every conjunct, from the values at the start and the end of a block."""
    nu, mu = state["nu"], state["mu"]
    state["nu"] = Update.multiplier(nu, mu, r_end, delta)
    state["mu"] = Update.penalty(mu, np.maximum(0.0, delta - r_start), np.maximum(0.0, delta - r_end))


def solve(chain, V0, budget, lam, delta, alpha0, reward=0.0):
    """budget updates from V0 (n, N, 2), every run the same number, with block_update every EVERY updates. Returns the
    final V and per update the conjuncts' values r (n, budget, K) and L (n, budget)."""
    if budget % Update.EVERY:
        raise ValueError("budget must be a multiple of EVERY")
    state = init_state(V0, alpha0, chain.K)
    rs, Ls = [], []
    r_start = None
    for _ in range(budget):
        V, alpha, rec = iterate(chain, state, lam, delta, reward)
        state["V"], state["alpha"] = V, alpha
        if state["k"] % Update.EVERY == 0:
            r_start = rec["r"]
        state["k"] += 1
        if state["k"] % Update.EVERY == 0:
            block_update(state, r_start, rec["r_next"], delta)
        rs.append(rec["r"])
        Ls.append(rec["L"])
    return state["V"], np.stack(rs, 1), np.stack(Ls, 1)
