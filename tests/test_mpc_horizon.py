"""The per-horizon program on the hold-up at 100 time steps: the slot counts and length bounds, the record in blocks, one
period against the per-period program, and the loop's first two periods against stored values of the reference code."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("casadi")
from sparsemax_dstl.mpc import conjuncts as MC  # noqa: E402
from sparsemax_dstl.mpc import horizon as MH  # noqa: E402
from sparsemax_dstl.mpc import program as MP  # noqa: E402
from sparsemax_dstl.sparse.assembly import StructuralDerivatives  # noqa: E402
from sparsemax_dstl.tasks import dual_arm as D  # noqa: E402
from sparsemax_dstl.tasks import dual_arm_scenes as S  # noqa: E402

DATA = Path(__file__).parent / "data"
T, DELAY, P = 100, 14, 25


@pytest.fixture(scope="module")
def hold_up():
    f = np.load(DATA / "mpc_horizon_hold_up_T100.npz")
    sc = S.hold_up(T, DELAY)
    return f, sc, MH.Horizon(sc, D.EPS_LENGTH)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)), initial=0.0))


def test_slots_and_bounds(hold_up):
    f, sc, hz = hold_up
    assert [hz.n_slots["reach"], hz.n_slots["until"]] == f["slots"].tolist()
    sv = MH.HorizonSolver(hz)
    sv.set_bounds()
    assert [sv.bounds["jac"][a] for a in MH.LENGTHS["jac"]] == f["jacobian_bounds"].tolist()
    assert [sv.bounds["hess"][a] for a in MH.LENGTHS["hess"]] == f["hessian_bounds"].tolist()
    lv = sv.set_levels()
    assert lv["ng"][-1] >= sv.bounds["hess"]["ng"] and lv["nl"][-1] >= sv.bounds["hess"]["nl"]
    assert all(lv["decl"][-1][a] >= sv.bounds["hess"][a] for a in ("M", "np", "nh", "ec"))


def test_record_in_blocks_equals_one_sample_at_a_time(hold_up):
    """The record updated in blocks of 25 samples to the loop's second period and in blocks of random sizes after it equals
    the record updated one sample at a time: decision periods, verdicts, open rows and operand masks."""
    f, sc, hz = hold_up
    X = np.array(sc.states)
    k0 = int(f["periods"][0])
    X[k0:k0 + P + 1] = f["states"]
    hz.record(X, k0 - 3 * P)
    one = hz.progs.copy()
    V = hz.node_exact(X)
    last = min(g.program.T for g in hz.progs.groups) - 1
    ends = list(range(k0 - 2 * P, k0 + P + 1, P))
    ends += sorted(np.random.default_rng(89).choice(np.arange(ends[-1] + 1, last), 20, replace=False).tolist()) + [last]
    for e in ends:  # blocks in order
        s = hz.progs.k + 1
        for v in V[s:e + 1]:  # brute-force reference: one sample at a time
            one.update(v)
        hz.progs.update_block(V[s:e + 1])
        assert hz.progs.k == one.k == e
        for g, h in zip(hz.progs.groups, one.groups):  # progression groups
            assert np.array_equal(g.d, h.d) and np.array_equal(g.ok, h.ok) and np.array_equal(g.open, h.open)
            assert all(m is None and n is None or np.array_equal(m, n) for m, n in zip(g.masks, h.masks))


def test_period_equals_the_per_period_program(hold_up):
    """The switched-on rows, the start point, the constraints and the supports' prefix at the first period."""
    f, sc, hz = hold_up
    k = int(f["periods"][0])
    X = np.array(sc.states)
    hz.record(X, k)
    pd = hz.period(k, X)
    sv = MH.HorizonSolver(hz)
    sv.set_period(pd)
    with jax.enable_x64(True):
        pp = MP.PeriodProgram(sc, MC.conjuncts(sc, MC.record_predicates(sc, X, D.EPS_LENGTH)), k, D.EPS_LENGTH)
        sd = StructuralDerivatives(pp)
        w = jnp.asarray(pd["w_ref"])
        on = np.array([r[5] for r in pd["rows"]])
        assert [(a, s) for a, s, o in hz.instances(pd) if o] == pp.instances()
        assert np.array_equal(pd["w_ref"], pp.reference()) and np.array_equal(pd["x0"], pp.x0)
        g = sv.ipopt_rows(sv._g(w, sv.th))
        assert rel(g[:pp.n_def], pp.defects(*pp.split(w))) <= 1e-12
        assert rel(g[pp.n_def:][on], np.asarray(sd.rows(w)) - pp.delta) <= 1e-12
        assert np.all(g[pp.n_def:][~on] == MH.OFF_VALUE)
        assert np.array_equal(np.asarray(sv._prefix(w, sv.th))[sv.perm[on]], np.asarray(sd.prefix(w)))


def test_first_two_periods_of_the_loop(hold_up):
    """The first period cold on the delayed plan, the second from its solution shifted by 25 samples on the executed
    states: the stored statuses, update counts, solver objects, objectives and planned states."""
    f, sc, hz = hold_up
    k0 = int(f["periods"][0])
    X = np.array(sc.states)
    hz.record(X, k0)
    sv = MH.HorizonSolver(hz, max_iter=3000)
    prev = None
    for j, k in enumerate(f["periods"]):  # the two control periods, in order
        if j:
            X[k0 + 1:k + 1] = f["states"][1:]
            hz.update(X[k0 + 1:k + 1], k0 + 1)
        sv.set_period(hz.period(int(k), X))
        if prev is None:
            out = sv.solve(sv.pd["w_ref"])
        else:
            w, lam_g, lam_x, carry = MH.shift_start(prev, sv, P)
            out = sv.solve(w, lam_g, lam_x, warm=True, carry=carry)
        prev = sv.snapshot()
        assert out["status"] == str(f["status"][j]) and out["accepted"]
        assert out["updates"] == int(f["updates"][j]) and out["solver_objects"] == int(f["solver_objects"][j])
        assert abs(out["objective"] - f["objective"][j]) <= 1e-10 * max(1.0, abs(f["objective"][j]))
        np.testing.assert_allclose(sv.x[:D.NX * T], f["planned_states"][j], rtol=0, atol=1e-8)
