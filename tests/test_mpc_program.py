"""One control period's program on the five-cell zone at 100 time steps against stored values of the reference
implementation, its instance rows against the JAX evaluator's robustness of the same specification, and one IPOPT solve."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("casadi")
from sparsemax_dstl import stl  # noqa: E402
from sparsemax_dstl.jax import robustness  # noqa: E402
from sparsemax_dstl.mpc import conjuncts as MC  # noqa: E402
from sparsemax_dstl.mpc import ipopt as MI  # noqa: E402
from sparsemax_dstl.mpc import program as MP  # noqa: E402
from sparsemax_dstl.tasks import dual_arm_scenes as S  # noqa: E402

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def zone():
    f = np.load(DATA / "mpc_five_cell_zone_T100.npz")
    with jax.enable_x64(True):
        sc = S.five_cell_zone(100)
        conjs = MC.conjuncts(sc, MC.record_predicates(sc, sc.states))
        pb = MP.PeriodProgram(sc, conjs, int(f["period"]))
    return f, sc, conjs, pb


def close(a, b, tol):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)), initial=0.0) <= tol


def test_program_at_the_start_point(zone):
    f, sc, conjs, pb = zone
    assert pb.instances() == list(zip(f["instance_names"].tolist(), f["instance_anchors"].tolist()))
    assert np.array_equal(pb.reference(), f["start"])
    with jax.enable_x64(True):
        w = jnp.asarray(f["start"])
        assert np.array_equal(pb.x0, f["x0"]) and np.array_equal(pb.qref, f["qref"])
        close(pb.constraints(w), f["constraints"], 1e-12)
        close(pb.objective(w), f["objective"], 1e-12)
        close(pb.rows(w, "sparsemax"), f["rows_sparsemax"], 1e-12)
        close(pb.rows(w, "lse_matched"), f["rows_lse_matched"], 1e-12)
    lbx, ubx, lbg, ubg = pb.bounds()
    assert lbx.size == ubx.size == pb.n and lbg.size == ubg.size == pb.m
    assert np.all(ubg[:pb.n_def] == 0) and np.all(np.isinf(ubg[pb.n_def:]))


def expand(f, nodes):
    """The formula with each node atom replaced by its predicates: a conjunction node by the And of its predicates, a leaf by
    its predicate."""
    if isinstance(f, stl.Atom):
        nd = nodes[f.index]
        return stl.And(*[stl.Atom(int(p)) for p in nd["preds"]]) if nd["kind"] == "and" else stl.Atom(int(nd["preds"][0]))
    if isinstance(f, stl.Until):
        return stl.Until(f.interval, expand(f.left, nodes), expand(f.right, nodes))
    if isinstance(f, (stl.Always, stl.Eventually)):
        return type(f)(f.interval, expand(f.child, nodes))
    return type(f)(*[expand(c, nodes) for c in f.children])


def test_rows_equal_the_evaluator_robustness(zone):
    """For an instance whose window holds no recorded sample, its row is the JAX evaluator's robustness at its anchor of the
    conjunct's specification written over the predicates, on the predicate values of the start point (the held state's after
    the horizon), under sparsemax and under log-sum-exp with the matched parameter."""
    f, sc, conjs, pb = zone
    checked = 0
    with jax.enable_x64(True):
        w = jnp.asarray(f["start"])
        Xs, _ = pb.split(w)
        P, Ph = pb.predicates(Xs)
        rows = {s: pb.rows(w, s) for s in ("sparsemax", "lse_matched")}
        for i, (name, anchor) in enumerate(pb.instances()):
            if anchor <= pb.k:
                continue
            cj = next(c for c in conjs if c.name == name)
            assert cj.gamma == pb.gamma_and
            L = cj.program.T
            s = np.arange(L)
            trace = jnp.where((s > pb.k + pb.T)[:, None], Ph[None], P[np.clip(s - pb.k - 1, 0, pb.T - 1)])
            program = stl.compile_formula(expand(cj.formula, cj.nodes), L)
            for semantics in ("sparsemax", "lse_matched"):
                rho = float(robustness(program, trace, semantics, cj.gamma)[anchor])
                assert abs(float(rows[semantics][i]) - rho) <= 1e-12 * max(1.0, abs(rho))
            checked += 1
    assert checked >= 3


def test_ipopt_options():
    assert MI.ipopt_options("exact", True) == {
        "ipopt.hessian_approximation": "exact", "ipopt.max_iter": 1000, "ipopt.tol": 1e-06, "ipopt.constr_viol_tol": 1e-06,
        "ipopt.print_level": 0, "ipopt.warm_start_init_point": "yes", "ipopt.warm_start_bound_push": 1e-06,
        "ipopt.warm_start_mult_bound_push": 1e-06, "ipopt.mu_init": 1e-06, "print_time": False, "record_time": True, "error_on_fail": False}
    lm = MI.ipopt_options("limited-memory")
    assert lm["ipopt.limited_memory_max_history"] == 10 and "ipopt.mu_init" not in lm


def test_solve_from_the_previous_solution(zone):
    """Period k - 1 cold from its start point, then period k from that solution shifted by one sample, with the supports'
    Hessian structure: the stored statuses, update counts, objective and solution."""
    f, sc, conjs, pb = zone
    pp = MP.PeriodProgram(sc, conjs, pb.k - 1)
    prev = MI.Solver(pp, "U")
    p = prev.solve(pp.reference())
    assert p["updates"] == int(f["solve_previous_updates"]) and p["solver_objects"] == int(f["solve_previous_solver_objects"])
    np.testing.assert_allclose(prev.x, f["solve_x_previous"], rtol=0, atol=1e-8)
    s = MI.Solver(pb, "U")
    w, lam_g, lam_x, carry = MI.shift_start(prev, pb)
    out = s.solve(w, lam_g, lam_x, warm=True, carry=carry)
    assert out["status"] == str(f["solve_status"]) and out["updates"] == int(f["solve_updates"]) and out["accepted"]
    assert abs(out["objective"] - float(f["solve_objective"])) <= 1e-10 * max(1.0, abs(float(f["solve_objective"])))
    np.testing.assert_allclose(s.x, f["solve_x"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(s.lam_g, f["solve_lam_g"], rtol=0, atol=1e-8)
