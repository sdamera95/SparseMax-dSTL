"""The CasADi expressions of the baselines against the JAX evaluator: the reductions (sound log-sum-exp, sparsemax, GMR of
order (0, 1)) on random operands, their gradients, whole until and eventually specifications, and the plant's kinematics
and predicates."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ca = pytest.importorskip("casadi")
from sparsemax_dstl import jax as SJ  # noqa: E402
from sparsemax_dstl import stl  # noqa: E402
from sparsemax_dstl.mpc import baselines as B  # noqa: E402
from sparsemax_dstl.tasks import dual_arm as D  # noqa: E402

RTOL, ATOL = 1e-12, 1e-13
CASES = [("lse", SJ.lse_max_matched, SJ.lse_min_matched), ("sparsemax", SJ.lower_max, SJ.lower_min),
         ("gmr", lambda z, g: SJ.gm_pm01_max(z), lambda z, g: SJ.gm_pm01_min(z))]


def casadi_eval(expr, x, xs):
    f = ca.Function("f", [x], [expr, ca.gradient(expr, x)])
    v, g = f(xs)
    return float(v), np.asarray(g).ravel()


@pytest.mark.parametrize("name,jmax,jmin", CASES)
@pytest.mark.parametrize("m", [1, 2, 3, 7, 17, 40])
@pytest.mark.parametrize("shift", [-1.0, 0.0, 1.0])
def test_reductions_against_jax(name, jmax, jmin, m, shift):
    gamma = 2 / 3
    zs = np.random.default_rng(m).normal(shift, 0.4, size=(4, m))
    x = ca.SX.sym("z", m)
    cmax, cmin = B.reductions(name, gamma)
    for z in zs:  # random operand vectors
        with jax.enable_x64(True):
            z_j = jnp.asarray(z)
            ref = [(float(f(z_j, gamma)), np.asarray(jax.grad(lambda v: f(v, gamma))(z_j))) for f in (jmax, jmin)]
        got = [casadi_eval(f([x[i] for i in range(m)]), x, z) for f in (cmax, cmin)]
        for (v, g), (rv, rg) in zip(got, ref):  # maximum, then minimum
            np.testing.assert_allclose(v, rv, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(g, rg, rtol=1e-10, atol=1e-12)


def _masks(prog):
    return [None if s.kind == "atom" else np.arange(s.index.shape[1])[None, :] < s.count[:, None] for s in prog.steps]


@pytest.mark.parametrize("name,semantics", [("lse", "lse_matched"), ("sparsemax", "sparsemax"), ("gmr", "gm_pm01")])
@pytest.mark.parametrize("formula,anchors", [(stl.Until((0, 7), stl.Atom(0), stl.Atom(1)), [0, 5, 12, 20, 22]),
                                             (stl.Eventually((0, 6), stl.Atom(0)), [0, 3, 10, 23])])
@pytest.mark.parametrize("shift", [0.0, 1.5])
def test_specification_rows_against_jax(name, semantics, formula, anchors, shift):
    T, gamma = 30, 0.4
    prog = stl.compile_formula(formula, T, reads=[(formula, anchors)])
    scores = np.random.default_rng(len(anchors)).normal(shift, 0.5, size=(T, 2))
    Z = ca.SX.sym("Z", T, 2)
    reduce = B.reductions(name, gamma)
    root_step, root_rows = prog.outputs[0]
    memo = {}
    rows = ca.vertcat(*[B.evaluate_row(prog, root_step, r, _masks(prog), lambda a, t: Z[t, a], reduce, memo, nested=name == "gmr")
                        for r in root_rows])
    got = np.asarray(ca.Function("f", [Z], [rows])(scores)).ravel()
    with jax.enable_x64(True):
        ref = np.asarray(SJ.read(prog, SJ.evaluate(prog, jnp.asarray(scores), semantics, gamma)))
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_kinematics_and_predicates_against_jax():
    rng = np.random.default_rng(3)
    scene = SimpleNamespace(waypoints=np.array([[0.4, -0.3, 0.68], [0.45, 0.0, 0.72], [0.3, 0.34, 0.74]]),
                            waypoint_arms=np.array([0, 0, 1]), cells=rng.normal([0.42, 0.0, 0.74], 0.1, size=(2, 3)), cell_radius=0.03)
    x = ca.SX.sym("x", D.NX)
    hands, C = B.points(x[:D.NQ])
    pr = B.Predicates(x, scene)
    n = len(scene.waypoints) + len(scene.cells) * len(D.DISTAL)
    f = ca.Function("f", [x], [ca.horzcat(*hands).T, ca.horzcat(*C).T, ca.vertcat(*[pr(i) for i in range(n)])])
    u = ca.SX.sym("u", D.NU)
    fi = ca.Function("fi", [x, u], [B.interval(x, u, D.H_STEP)])
    for _ in range(3):  # random states
        xs = np.concatenate([D.START + rng.normal(0, 0.3, D.NQ), rng.normal(0, 1, D.NQ)])
        us = rng.normal(0, 10, D.NU)
        h, c, p = (np.asarray(v) for v in f(xs))
        with jax.enable_x64(True):
            xj = jnp.asarray(xs)
            rh, rc = D.points(xj[:D.NQ])
            rp = jnp.concatenate([D.at_waypoint(xj, scene.waypoints, scene.waypoint_arms),
                                  D.outside(xj, jnp.asarray(scene.cells), scene.cell_radius)])
            rx = D.interval_map(D.H_STEP)(xj, jnp.asarray(us))
        np.testing.assert_allclose(h, np.asarray(rh), rtol=0, atol=1e-14)
        np.testing.assert_allclose(c, np.asarray(rc), rtol=0, atol=1e-14)
        np.testing.assert_allclose(p.ravel(), np.asarray(rp), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(fi(xs, us)).ravel(), np.asarray(rx), rtol=1e-15, atol=1e-15)


def test_ipopt_options_key_by_key():
    import json
    from pathlib import Path
    f = np.load(Path(__file__).parent / "data" / "baselines_ipopt_options.npz")
    cases = {"limited_memory_cold": ("limited-memory", False, {}), "limited_memory_warm": ("limited-memory", True, {}),
             "exact_cold": ("exact", False, {}), "exact_warm": ("exact", True, {}),
             "limited_memory_warm_max_iter_3000": ("limited-memory", True, {"max_iter": 3000})}
    for key, (h, warm, extra) in cases.items():  # the stored option sets
        ref, got = json.loads(str(f[key])), B.ipopt_options(h, warm, **extra)
        assert sorted(got) == sorted(ref)
        for name in ref:  # key by key
            assert got[name] == ref[name], name


@pytest.fixture(scope="module")
def zone_period():
    from pathlib import Path
    from sparsemax_dstl.mpc import conjuncts as MC
    from sparsemax_dstl.tasks import dual_arm_scenes as S
    f = np.load(Path(__file__).parent / "data" / "baselines_five_cell_zone_T40.npz")
    scene = S.five_cell_zone(int(f["horizon"]))
    conjs = MC.conjuncts(scene, MC.record_predicates(scene, scene.states, D.EPS_LENGTH))
    return f, scene, conjs


def _values(p, points):
    fg, ff = ca.Function("g", [p.w], [p.g]), ca.Function("f", [p.w], [p.f])
    return np.stack([np.asarray(fg(w)).ravel() for w in points]), np.array([float(ff(w)) for w in points])


def test_period_programs_against_stored_values_and_ours(zone_period):
    from sparsemax_dstl.mpc import program as MP
    f, scene, conjs = zone_period
    k, pts = int(f["period"]), f["points"]
    with jax.enable_x64(True):
        ours = MP.PeriodProgram(scene, conjs, k, D.EPS_LENGTH)
        for sm, sem in (("lse", "lse_matched"), ("sparsemax", "sparsemax")):  # against the stored values and our program
            p = B.Period(scene, conjs, k, sm)
            assert [n + "@" + str(a) for n, a in p.instances] == [str(s) for s in f["instances"]]
            assert all(np.array_equal(a, b) for a, b in zip(p.bounds(), ours.bounds()))
            assert np.array_equal(p.reference(), ours.reference())
            g, c = _values(p, pts)
            np.testing.assert_allclose(g, f[sm + "_g"], rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(g, np.stack([np.asarray(ours.constraints(jnp.asarray(w), sem)) for w in pts]), rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(c, [float(ours.objective(jnp.asarray(w))) for w in pts], rtol=RTOL)
    for form in ("rows", "authors"):  # the GMR's two forms against the stored values
        g, c = _values(B.Period(scene, conjs, k, "gmr", form), pts)
        np.testing.assert_allclose(g, f["gmr_" + form + "_g"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(c, f["gmr_" + form + "_f"], rtol=RTOL)


def test_detected_patterns_hold_the_rows_dependencies(zone_period):
    f, scene, conjs = zone_period
    p = B.Period(scene, conjs, int(f["period"]), "lse")
    J, H = B.detected_patterns(p)
    Jd = np.asarray(ca.Function("J", [p.w], [ca.jacobian(p.g, p.w)])(f["points"][1]).full())
    r, c = B._triplets(J)
    mask = np.zeros(Jd.shape, bool)
    mask[r, c] = True
    assert np.all(Jd[~mask] == 0)
    col = B.colorings(J, ca.Sparsity(p.n, p.n))
    assert col["jacobian_mode"] in ("forward", "reverse") and col["jacobian_products"] <= min(p.m, p.n)
