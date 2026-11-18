"""Helpers shared by the notebooks and scripts of this folder: the paper's values from results/, the scenes, the timed
control period, the counts' medians, one solve of each comparator and the re-planning loops after a hold-up."""
import csv
import pathlib
import time
import timeit

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

from sparsemax_dstl.mpc import baselines as B
from sparsemax_dstl.mpc import conjuncts as MC
from sparsemax_dstl.mpc import horizon as MH
from sparsemax_dstl.mpc import ipopt as MI
from sparsemax_dstl.mpc import program as MP
from sparsemax_dstl.mpc import simulation as SIM
from sparsemax_dstl.sparse import counts as SC
from sparsemax_dstl.tasks import dual_arm as D
from sparsemax_dstl.tasks import dual_arm_scenes as S

RESULTS = pathlib.Path(__file__).parent / "results"
SCENES = {"five-cell zone": S.five_cell_zone, "64-cell circle": S.circle}
JAC, HES = "constraint Jacobian", "Hessian of the Lagrangian"


def paper_rows(table, **kw):
    with open(RESULTS / (table + ".csv")) as f:
        return [r for r in csv.DictReader(f) if all(r[k] == str(v) for k, v in kw.items())]


def paper(table, **kw):
    """The value of the one row of results/<table>.csv that matches kw."""
    rows = paper_rows(table, **kw)
    assert len(rows) == 1, (table, kw, len(rows))
    return float(rows[0]["value"])


def build(scene):
    """A scene and its conjuncts, decided on the scene's states with the exact predicates."""
    return scene, MC.conjuncts(scene, MC.record_predicates(scene, scene.states))


def timed_period(scene, conjs):
    """The first control period of the scene's rule with the most until instances imposed over the rule's horizon."""
    lo, hi = scene.timed_period_rule["periods"]
    ks = np.arange(lo, hi + 1)
    n = sum((cj.imposed(ks, scene.timed_period_rule["horizon"])[0] >= 0).sum(1) for cj in conjs if cj.kind == "until")
    return int(ks[np.argmax(n)])


def timed(f):
    """Median seconds of 30 calls of f after a first call (5 calls when the first takes more than 2 s)."""
    t = time.perf_counter()
    f()
    first = time.perf_counter() - t
    return float(np.median(timeit.repeat(f, number=1, repeat=30 if first <= 2.0 else 5)))


def count_periods(name, T):
    """The control periods of the counts table: every tenth period and the next, on the circle those whose horizon ends
    by sample 999; the Hessian at every hundredth period from 0 to 900 (five-cell zone) or at 0 and 300 (circle)."""
    k0 = np.arange(0, 991, 10)
    periods = np.unique(np.concatenate([k0, k0 + 1]))
    if name == "64-cell circle":
        return periods[periods + T <= 999], np.array([0, 300])
    return periods, k0[::10]


def count_medians(A, meta):
    """Medians over the counted periods of the AD products: dense, prescribed and supports' patterns."""
    m, nv = meta["n"] + A["all_rows"], meta["nv"]
    return {JAC: [float(np.median(np.minimum(m, nv))), float(np.median(A["whole_prescribed_products"])), float(np.median(A["whole_supports_products"]))],
            HES: [float(nv), float(np.median(A["hessian_whole_prescribed_star"])), float(np.median(A["hessian_whole_supports_star"]))]}


def counts(name, T):
    """The counts at the counts table's periods and their medians beside the paper's."""
    scene, conjs = build(SCENES[name](T))
    periods, hess = count_periods(name, T)
    t = time.perf_counter()
    A, meta = SC.count(scene, conjs, periods, hess)
    print(name, "T", T, "|", periods.size, "periods,", hess.size, "Hessian periods |", round(time.perf_counter() - t, 1), "s")
    ours = count_medians(A, meta)
    for q in (JAC, HES):
        ref = [paper("counts", scene=name, horizon=T, quantity=q + " products", comparator=c) for c in ("dense AD", "prescribed pattern", "supports' pattern")]
        print("  ", q, "products, dense / prescribed / supports:", ours[q], "| paper", ref)
    return A, meta


def ours_solve(scene, conjs, k, arm="U", d=0, start="previous", reference=None, twice=False, **options):
    """Our solve of period k on the supports' structure (arm "U"), widened by d samples ("D") or prescribed ("P"), from
    the previous period's solution shifted by one sample or from the reference motion; with twice, the second solve is timed."""
    pb = MP.PeriodProgram(scene, conjs, k)
    s = MI.Solver(pb, arm, d, reference=reference, **options)
    if start == "previous":
        pp = MP.PeriodProgram(scene, conjs, k - 1)
        prev = MI.Solver(pp, arm, d, reference=reference, **options)
        prev.solve(pp.reference())
        w, lam_g, lam_x, carry = MI.shift_start(prev, pb)

        def run():
            return s.solve(w, lam_g, lam_x, warm=True, carry=carry)
    else:
        def run():
            return s.solve(pb.reference())
    if twice:
        run()
    t = time.perf_counter()
    out = run()
    out["seconds"] = time.perf_counter() - t
    return out


def casadi_solve(scene, conjs, k, smoothing="lse", start="previous", twice=False, hessian="limited-memory", **extra):
    """The baseline's solve of period k in CasADi with IPOPT, L-BFGS or CasADi's exact Hessian (smoothing "lse" or "gmr"),
    from the previous period's solution shifted by one sample or from the reference motion; with twice, the second solve is
    timed."""
    p = B.Period(scene, conjs, k, smoothing)
    x0, lam_g, lam_x = p.reference(), np.zeros(p.m), np.zeros(p.n)
    if start == "previous":
        p1 = B.Period(scene, conjs, k - 1, smoothing)
        lbx, ubx, lbg, ubg = p1.bounds()
        S1 = ca.nlpsol("previous", "ipopt", p1.nlp(), MI.ipopt_options(hessian, **extra))
        r = S1(x0=p1.reference(), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, lam_g0=np.zeros(p1.m), lam_x0=np.zeros(p1.n))
        sol = {key: np.asarray(r[key]).ravel() for key in ("x", "lam_g", "lam_x")}
        x0, lam_g, lam_x = B.shift_start(sol, p1.instances, p.instances, p.m, scene.T)
    solver = ca.nlpsol("period", "ipopt", p.nlp(), MI.ipopt_options(hessian, warm=start == "previous", **extra))
    lbx, ubx, lbg, ubg = p.bounds()

    def run():
        return solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, lam_g0=lam_g, lam_x0=lam_x)
    if twice:
        run()
    t = time.perf_counter()
    r = run()
    seconds = time.perf_counter() - t
    st = solver.stats()
    return {"status": st["return_status"], "updates": int(st["iter_count"]), "objective": float(r["f"]), "seconds": seconds,
            "x": np.asarray(r["x"]).ravel(), "lam_g": np.asarray(r["lam_g"]).ravel(), "lam_x": np.asarray(r["lam_x"]).ravel(), "period": p}


_model = {}


def advance(x, U):
    """The planning model's states after each of the accelerations U (P, 14) from x, in one compiled scan."""
    with jax.enable_x64(True):
        if not _model:
            f = D.interval_map(D.H_STEP)
            _model["scan"] = jax.jit(lambda x, U: jax.lax.scan(lambda s, u: (f(s, u), f(s, u)), x, U)[1])
        return np.asarray(_model["scan"](jnp.asarray(x), jnp.asarray(U)))


def ours_loop(delay, k0, T, N, P, simulate=False):
    """Our controller on the per-horizon program after a hold-up of the right arm by `delay` samples: three periods
    k0 - 3P, k0 - 2P, k0 - P on the plan that ignores the hold-up (they set the derivative calls' padded lengths), then the
    periods k0, k0 + P, ..., k0 + N P, each from the previous solution shifted by P samples, the state advanced by the plan's
    first P accelerations on the planning model or, with simulate, by the rigid-body simulation holding the plan's first P
    joint positions. Returns per period of the loop the summary and the whole period's seconds, and the record of states."""
    scene = S.hold_up(T, delay)
    plan = np.array(scene.states)
    hz = MH.Horizon(scene)
    hz.record(plan, k0 - 3 * P)
    sv = MH.HorizonSolver(hz, max_iter=3000)

    def run(periods, X, execute, sim=None):
        prev, out = None, []
        for j, k in enumerate(periods):  # control periods (sequential: each starts from the previous solution)
            if execute and j:
                X[k - P + 1:k + 1] = sim.run(Q) if sim is not None else advance(X[k - P], U)
            t = time.perf_counter()
            if hz.progs.k < k:
                hz.update(X[hz.progs.k + 1:k + 1], hz.progs.k + 1)
            pd = hz.period(k, X)
            sv.set_period(pd)
            if prev is None:
                s = sv.solve(pd["w_ref"])
            else:
                w, lam_g, lam_x, carry = MH.shift_start(prev, sv, P)
                s = sv.solve(w, lam_g, lam_x, warm=True, carry=carry)
            prev = sv.snapshot()
            s["seconds"] = time.perf_counter() - t
            U = sv.x[D.NX * T:].reshape(T, D.NU)[:P]
            Q = sv.x[:D.NX * T].reshape(T, D.NX)[:P, :D.NQ]
            out.append(s)
        return out, sv.x[:D.NX * T].reshape(T, D.NX), U, Q

    run([k0 - 3 * P, k0 - 2 * P, k0 - P], np.array(plan), False)
    sv.set_bounds()
    sv.set_levels()
    sv.compile_levels(sv.x)
    X = np.array(plan)
    sim = SIM.Simulation(X[k0]) if simulate else None
    out, last, U, Q = run([k0 + j * P for j in range(N + 1)], X, True, sim)
    kN = k0 + N * P
    X[kN + 1:kN + P + 1] = sim.run(Q) if simulate else advance(X[kN], U)
    X[kN + P + 1:kN + 1 + T] = last[P:]
    return out, X


def lse_loop(delay, k0, T, N, P, simulate=False):
    """The baseline (Sound-LSE in CasADi, L-BFGS, at most 3000 updates) in the same loop, without the three earlier periods;
    each period's program is built in CasADi on the record. Returns per period the status, updates, the solve's seconds and
    the seconds with the program's construction."""
    X = np.array(S.hold_up(T, delay).states)
    sim = SIM.Simulation(X[k0]) if simulate else None
    out, sol, prev = [], None, None
    for j in range(N + 1):  # control periods (sequential: each starts from the previous solution)
        k = k0 + j * P
        if j:
            Xs = sol["x"][:D.NX * T].reshape(T, D.NX)
            X[k - P + 1:k + 1] = sim.run(Xs[:P, :D.NQ]) if simulate else advance(X[k - P], sol["x"][D.NX * T:].reshape(T, D.NU)[:P])
        t = time.perf_counter()
        scene, conjs = build(S.hold_up(T, delay, states=X))
        p = B.Period(scene, conjs, k, "lse")
        if prev is None:
            x0, lam_g, lam_x = p.reference(), np.zeros(p.m), np.zeros(p.n)
        else:
            x0, lam_g, lam_x = B.shift_start(sol, prev.instances, p.instances, p.m, T, P)
        solver = ca.nlpsol("period", "ipopt", p.nlp(), MI.ipopt_options("limited-memory", warm=j > 0, max_iter=3000))
        lbx, ubx, lbg, ubg = p.bounds()
        built = time.perf_counter() - t
        t = time.perf_counter()
        r = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, lam_g0=lam_g, lam_x0=lam_x)
        seconds = time.perf_counter() - t
        st = solver.stats()
        sol = {key: np.asarray(r[key]).ravel() for key in ("x", "lam_g", "lam_x")}
        prev = p
        out.append({"status": st["return_status"], "updates": int(st["iter_count"]), "objective": float(r["f"]), "seconds": seconds, "with_construction": seconds + built})
    return out
