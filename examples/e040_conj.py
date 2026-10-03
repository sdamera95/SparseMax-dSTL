"""E040: the constrained solve with one constraint per conjunct (sparsemax_dstl.constrained_conj) on E034's and E039's scenes.

    python -m examples.e040_conj combine <out.npz> <instance.npz>:<runs> ...
    JAX_PLATFORMS=nojax python -m examples.e040_conj run <instance.npz> <iterations> <out.npz> <runs|all> [K1] [H=20|H=40]
    JAX_PLATFORMS=cpu python -m examples.e040_conj path <run.npz> <instance.npz> <out.csv> <iterates>
    JAX_PLATFORMS=cpu python -m examples.e040_conj linesearch <run.npz> <instance.npz> <out.csv> <iterate>

combine: one instance file (e034_until_demo's layout) from rows of several (E039's person scene and E034's), kept contiguous by
(method, eps, wait); each run keeps its own person (hc).
run: e034_until_demo's run mode with constrained_conj (the four conjunct programs of each wait: separation, slowdown, order,
handover, in E022's CONJUNCTS order); K1 runs the same solver with the full program as its one conjunct (the default arithmetic).
The record adds conj_r, conj_w, conj_r_next (runs, iterates, 4) and conj_mu, conj_nu.
path: for every run of a record and iterates 0 .. <iterates>, the commands replayed in float64 MuJoCo C (the certificate's
replay): the four conjuncts' smoothed values under the run's arm and eps (float64, JAX evaluator), the exact ones, the entry
(samples inside the zone up to 2.0 s, deepest zone margin) and at the until node sparsemax's support and its hold samples.
linesearch: the solver's line search at one iterate replayed in float64 under the per-conjunct merit (ARMIJO_C, the run's nu and
mu): the accepted direction d = (V[k+1] - V[k]) / step[k], trials a_j = 2 step[k-1] 0.5^j (k = 0: 2 alpha0 0.5^j); per trial the
four conjunct values, the merit change, the descent test, the entry and the until node's support.
"""
import json
import sys
import time

import numpy as np

from examples import e034_until_demo as D

CONJ = ("separation", "slowdown", "order", "handover")


def conj_programs(sc, n_h):
    """The four conjunct programs (pruned; the conjunct at t = 0 is the root) in CONJ order."""
    from examples import e022_regime as E
    from sparsemax_dstl import stl
    from sparsemax_dstl.tasks import workspace as W
    names, rows, _ = W.specs(sc, E.N_R, n_h)
    by = dict(zip(names, rows))
    return tuple(E.prune(stl.compile_formula(by[c], sc.samples)) for c in CONJ)


def combine_main(out, items):
    keys = ("x0", "V0", "hc", "plan", "run_person", "run_eps", "run_wait", "run_method", "run_depth", "run_start")
    parts, src = [], []
    for item in items:  # over source files (a handful)
        path, runs = item.split(":")
        z = np.load(path)
        idx = np.asarray([int(i) for i in runs.split(",")])
        parts.append({k: z[k][idx] for k in keys})
        src += [(path, int(i)) for i in idx]
        base = z
    cat = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    order = np.lexsort((np.arange(len(src)), cat["run_wait"], cat["run_eps"], cat["run_method"]))
    cat = {k: v[order] for k, v in cat.items()}
    np.savez(out, pick=base["pick"], handover=base["handover"], hr=base["hr"], n_h=base["n_h"], source=json.dumps([src[i] for i in order]), **cat)
    for i in range(len(order)):  # printout, one line per run
        print(i, cat["run_person"][i], cat["run_method"][i], cat["run_eps"][i], cat["run_wait"][i], cat["run_depth"][i], cat["run_start"][i], src[order[i]])


def build(I, k1=False, device="cuda:0"):
    from examples import e022_regime as R
    from sparsemax_dstl.warp import solver as CW
    from sparsemax_dstl.warp import solver_conjuncts as CC
    n = len(I["x0"])
    waits = sorted({g[4] for g in I["groups"]})
    progs = {w: R.core_program(D.scenario(w), I["n_h"]) for w in waits}
    cps = {w: ((progs[w],) if k1 else conj_programs(D.scenario(w), I["n_h"])) for w in waits}
    sc = D.scenario(min(progs))
    groups = [(m_, e, a, b, cps[w]) for m_, e, a, b, w in I["groups"]]
    blocks = [(a, b, progs[w]) for _, _, a, b, w in I["groups"]]
    pick, hand = np.broadcast_to(I["pick"], (n, 3)), np.broadcast_to(I["handover"], (n, 3))
    chain = CC.ConjChain(R.plant, sc, progs[min(progs)], I["x0"], pick, hand, I["hc"], I["hr"], groups, device=device)
    referee = CW.Referee(R.plant, sc, progs[min(progs)], I["x0"], pick, hand, I["hc"], I["hr"], device=device, blocks=blocks)
    return chain, referee


def run_main(path, iterations, out, runs, k1=False):
    import socket

    import warp as wp

    from sparsemax_dstl.warp import solver as CW
    from sparsemax_dstl.warp import solver_conjuncts as CC
    t0 = time.perf_counter()
    I = D.load(path, runs)
    chain, referee = build(I, k1)
    setup_seconds = time.perf_counter() - t0
    state = CC.init_state(I["V0"], D.ALPHA0, chain.K)

    def log(k, rec, r64, wall):
        print("iterate", k, "seconds", round(wall, 2), "accepted", int(rec[:, 10].sum()), "of", len(rec), "rho", np.round(rec[:, 1], 4).tolist(),
              "referee64", np.round(r64, 4).tolist(), flush=True)

    res = CC.solve(chain, referee, state, iterations, D.LAM, D.DELTA, log=log)
    meta = {"host": socket.gethostname(), "device": str(chain.device), "warp": wp.config.version, "instance": path, "runs": runs, "iterations": state["k"],
            "K": chain.K, "k1": k1, "conjuncts": CONJ if not k1 else ("root",), "every": CW.EVERY, "lam": D.LAM, "delta": D.DELTA, "alpha0": D.ALPHA0,
            "trials": CW.TRIALS, "armijo_c": CW.ARMIJO_C, "mu0": CW.MU0, "mu_max": CW.MU_MAX, "viol_tol": CW.VIOL_TOL, "trace_keys": CW.TRACE_KEYS,
            "groups": I["groups"], "setup_seconds": setup_seconds, "seconds_keys": res["seconds_keys"], "resumed_from": []}
    V_all = np.concatenate([np.asarray(I["V0"], np.float32)[:, None], res["V"]], 1)
    np.savez(out, meta=D.dumps(meta), index=I["index"], trace=res["trace"], V=V_all, referee64=res["referee64"], conjuncts64=res["conjuncts64"],
             seconds=res["seconds"], wall=res["wall"], alpha=state["alpha"], nu=state["nu"], mu=state["mu"], k=state["k"],
             **{"conj_" + k: v for k, v in res["conj"].items()},
             **{k: I[k] for k in ("plan", "run_person", "run_eps", "run_wait", "run_method", "run_depth", "run_start")})
    med = np.median(res["seconds"], 0)
    print("setup seconds", round(setup_seconds, 2), "seconds per iterate (median by component)", D.dumps(dict(zip(res["seconds_keys"], np.round(med, 4)))),
          "total", round(float(np.median(res["seconds"].sum(1))), 3))


# ------------------------------------------------------------------
# float64 analysis on the CPU

def _setup64(I, r, cache):
    """Per run: (sc, inst, full program, conjunct programs, the jitted margins/scores of float64 states) for its wait and person."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from examples import e022_regime as E
    from sparsemax_dstl.tasks import workspace_mjx as Wm
    w, hc = float(I["run_wait"][r]), I["hc"][r]
    key = (w, hc.tobytes()[:4096], float(hc.sum()))
    if key not in cache:
        sc = D.scenario(w)
        inst = {"pick": jnp.asarray(I["pick"]), "handover": jnp.asarray(I["handover"]), "human_centres": jnp.asarray(np.asarray(hc, np.float64)),
                "human_radii": jnp.asarray(I["hr"])}
        mx = mjx.put_model(E.plant.model, impl="jax")
        f = jax.jit(jax.vmap(lambda Xs: (Wm.margins(mx, E.plant, sc, inst, Xs), Wm.scores(mx, E.plant, sc, inst, Xs))))
        cache[key] = (sc, f, conj_programs(sc, int(I["n_h"])))
    return cache[key]


def _conj_values(cps, Zs, arm, eps):
    import jax
    import jax.numpy as jnp

    from sparsemax_dstl import jax as stl_jax
    from sparsemax_dstl.jax import methods
    sem = "exact" if arm == "exact" else methods.SEMANTICS[arm]
    out = []
    for pg in cps:  # over the four conjuncts (formula structure)
        f = jax.jit(jax.vmap(lambda Z, pg=pg: stl_jax.evaluate(pg, Z, sem, eps)[-1][0]))
        out.append(np.asarray(f(jnp.asarray(Zs))))
    return np.stack(out, -1)


def _regime(Mg, Zs, w, eps):
    from examples import e038_horizon as X
    t = np.arange(Mg.shape[1]) * D.HS
    entry = (Mg[:, :, 2] < 0) & (t[None] <= D.T_HOLD + 1e-9)
    u = X.until_batch(Zs[:, :, 2], Zs[:, :, 0], Mg[:, :, 2], Mg[:, :, 0], entry, np.full(len(Mg), eps), w, D.T_HOLD)
    return entry.sum(1), Mg[:, t <= w + 1 + 1e-9, 2].min(1), u["support_deciding"], u["support_hold_deciding"]


def path_main(run_path, inst_path, out, iterates):
    import jax
    jax.config.update("jax_enable_x64", True)
    z, I = np.load(run_path), D.load(inst_path, "all")
    cache, rows = {}, []
    ks = np.arange(iterates + 1)
    for r in range(len(z["run_method"])):  # over runs (a handful)
        sc, f, cps = _setup64(I, r, cache)
        arm, eps, w = str(z["run_method"][r]), float(z["run_eps"][r]), float(z["run_wait"][r])
        X64 = D.replay(z["V"][r, ks], I["x0"][r].astype(np.float64))
        Mg, Zs = (np.asarray(a) for a in f(X64))
        S, Ex = _conj_values(cps, Zs, arm, eps), _conj_values(cps, Mg, "exact", None)
        inside, deepest, supp, hold = _regime(Mg, Zs, w, eps)
        for i, k in enumerate(ks):  # over the iterates (table rows)
            row = {"person": str(z["run_person"][r]), "arm": arm, "eps": eps, "wait": w, "depth": float(z["run_depth"][r]), "start": int(z["run_start"][r]),
                   "iterate": int(k), "certificate": float(z["referee64"][r, k]), "inside": int(inside[i]), "deepest_zone": float(deepest[i]),
                   "support": int(supp[i]), "hold_in_support": int(hold[i])}
            row.update({"s_" + c: float(S[i, j]) for j, c in enumerate(CONJ)})
            row.update({"x_" + c: float(Ex[i, j]) for j, c in enumerate(CONJ)})
            if k < z["trace"].shape[1]:
                row.update({"step": float(z["trace"][r, k, 8]), "trial": int(z["trace"][r, k, 9]), "accepted": int(z["trace"][r, k, 10])})
                row.update({"w_" + c: float(z["conj_w"][r, k, j]) for j, c in enumerate(CONJ)})
                row.update({"r32_" + c: float(z["conj_r"][r, k, j]) for j, c in enumerate(CONJ)})
            rows.append(row)
    D._csv(out, list(dict.fromkeys(k for r_ in rows for k in r_)), [[r_.get(k, "") for k in dict.fromkeys(k for r_ in rows for k in r_)] for r_ in rows])
    for r_ in rows:  # printout
        print(r_["person"], r_["arm"], "w", r_["wait"], "it", r_["iterate"], "cert", round(r_["certificate"], 4), "smoothed", [round(r_["s_" + c], 3) for c in CONJ],
              "weights", [round(r_.get("w_" + c, np.nan), 3) for c in CONJ], "step", r_.get("step"), "trial", r_.get("trial"), "inside", r_["inside"],
              "deepest", round(r_["deepest_zone"], 4), "support", r_["support"], "hold", r_["hold_in_support"])


def linesearch_main(run_path, inst_path, out, k):
    import jax

    from sparsemax_dstl.warp import solver as CW
    jax.config.update("jax_enable_x64", True)
    z, I = np.load(run_path), D.load(inst_path, "all")
    cache, rows = {}, []
    for r in range(len(z["run_method"])):  # over runs (a handful)
        sc, f, cps = _setup64(I, r, cache)
        arm, eps, w = str(z["run_method"][r]), float(z["run_eps"][r]), float(z["run_wait"][r])
        tr = z["trace"][r]
        step = float(tr[k, 8])
        prev = float(tr[k - 1, 8]) if k > 0 else D.ALPHA0
        d = (z["V"][r, k + 1].astype(np.float64) - z["V"][r, k].astype(np.float64)) / step
        a = np.concatenate([[0.0], 2 * prev * 0.5 ** np.arange(CW.TRIALS)])
        V = z["V"][r, k].astype(np.float64)[None] + a[:, None, None] * d[None]
        X64 = D.replay(V, I["x0"][r].astype(np.float64))
        Mg, Zs = (np.asarray(q) for q in f(X64))
        S = _conj_values(cps, Zs, arm, eps)
        inside, deepest, supp, hold = _regime(Mg, Zs, w, eps)
        mu, nu, gn = z["conj_mu"][r, k].astype(np.float64), z["conj_nu"][r, k].astype(np.float64), float(tr[k, 4])
        L = D.LAM * CW.effort(V) + np.sum(0.5 * mu * np.maximum(0.0, D.DELTA - S + nu / mu) ** 2, -1)
        for j in range(len(a)):  # over the trials (seven rows)
            row = {"person": str(z["run_person"][r]), "arm": arm, "wait": w, "iterate": k, "trial": j - 1, "a": float(a[j]), "accepted_trial": int(tr[k, 9]),
                   "dL": float(L[j] - L[0]), "passes": bool(L[j] - L[0] <= -CW.ARMIJO_C * a[j] * gn) if j else "", "inside": int(inside[j]),
                   "deepest_zone": float(deepest[j]), "support": int(supp[j]), "hold_in_support": int(hold[j])}
            row.update({"s_" + c: float(S[j, i]) for i, c in enumerate(CONJ)})
            rows.append(row)
    keys = list(rows[0])
    D._csv(out, keys, [[r_[k_] for k_ in keys] for r_ in rows])
    for r_ in rows:  # printout
        print(r_["person"], r_["arm"], "w", r_["wait"], "k", r_["iterate"], "trial", r_["trial"], "a", r_["a"], "accepted", r_["accepted_trial"], "passes", r_["passes"],
              "dL", round(r_["dL"], 6), "smoothed", [round(r_["s_" + c], 3) for c in CONJ], "inside", r_["inside"], "deepest", round(r_["deepest_zone"], 4),
              "support", r_["support"], "hold", r_["hold_in_support"])


if __name__ == "__main__":
    hs = [a_ for a_ in sys.argv if a_.startswith("H=")]
    if hs:  # E038's horizon: E034's globals H and WAITS (the longest wait) set in this process (20 and 40 s instances)
        from examples import e038_horizon
        e038_horizon.horizon(int(hs[0][2:]))
        sys.argv = [a_ for a_ in sys.argv if not a_.startswith("H=")]
    mode = sys.argv[1]
    if mode == "combine":
        combine_main(sys.argv[2], sys.argv[3:])
    elif mode == "run":
        run_main(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], len(sys.argv) > 6 and sys.argv[6] == "K1")
    elif mode == "path":
        path_main(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif mode == "linesearch":
        linesearch_main(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        raise ValueError("unknown mode " + mode)
