"""Checks of E022's additions: the scripted human's pass and wait (tasks.human), the regime
instances (tasks.workspace.regime_instances), and the pieces of examples.e022_regime that
the measurement relies on (row pruning, the node weights, the blocked rollout)."""
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from examples import e022_regime as E
from sparsemax_dstl import jax as stl_jax
from sparsemax_dstl import stl
from sparsemax_dstl.jax import methods
from sparsemax_dstl.tasks import human as Hm
from sparsemax_dstl.tasks import panda as P
from sparsemax_dstl.tasks import workspace as W
from sparsemax_dstl.tasks import workspace_mjx as Wm

PLANT = W.Plant()
HAND = (Hm.NAMES.index("forearm_r"), 1)  # the hand point: forearm_r's distal endpoint


def scripted(**kw):
    base = dict(phi=0.2, stand=0.6, t_arrive=1.0, t_reach=2.0, t_release=3.4, depth=0.03)
    return Hm.Script(**(base | kw))


def test_new_fields_leave_the_e019_motion_unchanged():
    t = np.linspace(0, 16, 801)
    a = Hm.capsules(Hm.Script(), t)[0]
    b = Hm.capsules(Hm.Script(t_pass=7.0, pass_width=0.9, pass_gap=0.2, standoff=0.5, wait=9.0), t)[0]
    np.testing.assert_array_equal(a, b)


def test_pass_and_wait_attain_their_distances_at_the_stated_times():
    """1 ms sampling: the hand is pass_gap from the anchor at t_pass and nowhere closer; it is
    standoff from the anchor over the whole wait; at rest between the phases; rigid segments;
    and a batch of two scripts equals the two scripts."""
    anchor = np.array([0.42, -0.05, 0.43])
    s = scripted(anchor=anchor, t_pass=6.0, pass_width=0.6, pass_gap=0.29, standoff=0.32, wait=4.0)
    t = np.round(np.arange(0, 12.001, 0.001), 6)
    ends = Hm.capsules(s, t)[0]
    hand = ends[:, HAND[0], HAND[1]]
    d = np.linalg.norm(hand - anchor, axis=-1)
    k = int(np.argmin(d))
    assert t[k] == 6.0 and abs(d[k] - 0.29) < 1e-12
    hold = (t >= 6.3 + s.t_move) & (t <= 6.3 + s.t_move + 4.0)
    np.testing.assert_allclose(d[hold], 0.32, rtol=0, atol=1e-12)
    n, p, q = Hm.pass_points(s)
    np.testing.assert_allclose(np.linalg.norm(Hm.standing_shoulder(s) - anchor), np.linalg.norm(Hm.standing_shoulder(s) - p) + 0.29, atol=1e-12)
    rest = ends[:, 2, 0] - 0.7 * np.eye(3)[2] - 0.1 * np.array([np.cos(0.2), np.sin(0.2), 0.0])
    idle = ((t >= s.t_release + s.t_move) & (t <= 5.7)) | (t >= 6.3 + 2 * s.t_move + 4.0)
    np.testing.assert_allclose(hand[idle], rest[idle], rtol=0, atol=1e-12)
    length = np.linalg.norm(ends[:, :, 1] - ends[:, :, 0], axis=-1)
    np.testing.assert_allclose(length, np.broadcast_to(length[0], length.shape), atol=1e-12)
    two = Hm.Script(**{**s.__dict__, "anchor": np.stack([anchor, anchor + 0.02]), "phi": np.array([0.2, -0.1]),
                       "wait": np.array([4.0, 9.0]), "pass_gap": np.array([0.29, 0.3])})
    both = Hm.capsules(two, t[::100])[0]
    one = Hm.capsules(Hm.Script(**{**s.__dict__, "anchor": anchor + 0.02, "phi": -0.1, "wait": 9.0, "pass_gap": 0.3}), t[::100])[0]
    np.testing.assert_allclose(both[0], ends[::100], atol=1e-15)
    np.testing.assert_allclose(both[1], one, atol=1e-15)


def test_regime_instances_have_the_designed_margins_and_pass_the_e019_filter():
    """H = 8 s, wait 0.5 s: for two instances, recomputed here in float64 from the instance arrays,
    - the anchor pair's margin at the handover configuration is pass_margin at t_pass and
      standoff_margin over the hold, and no pair is closer at t_pass;
    - every pair clears d_min + SEP_MARGIN at the three configurations over their windows
      (every FILTER_STRIDE-th sample), and every margin outside the episode's right forearm
      lies REGIME_CLEAR above the standoff margin;
    - the candidates are those of instances()'s stream: same start, pick and handover as there."""
    H = Fraction(8)
    sc = E.scenario(8)
    rg = E.ranges(0.5)
    I = Wm.regime_instances(W.TUNING_SEED, 2, sc, rg)
    rs = W.robot_spheres(PLANT, sc.robot_spacing)
    with jax.enable_x64(True):
        mx = mjx.put_model(PLANT.model, impl="jax")
        C = np.asarray(jax.vmap(jax.vmap(lambda q: Wm.points(mx, PLANT, sc.robot_spacing, q)[1]))(
            jnp.asarray(np.stack([I["q0"], I["q_pick"], I["q_handover"]], 1))))  # (2, 3, S_r, 3)
    t = np.arange(sc.samples) * float(sc.h_s)
    hr = I["human_radii"]
    Rp = rs["radius"][:, None] + hr[None] + sc.d_min
    arm = I["human_owner"] == Hm.NAMES.index("forearm_r")
    j_hand = np.nonzero(arm)[0][-1]
    tp = int(round(6.0 / float(sc.h_s)))
    hold = (t >= 6.3 + 0.6) & (t <= 6.3 + 0.6 + 0.5)
    spans = [(0.0, 2.0), (2.0, 4.0), (4.8, 8.0)]
    for i in range(2):  # over the two instances
        d = np.linalg.norm(C[i, 2][None, :, None] - I["human_centres"][i][:, None], axis=-1)  # (T, S_r, S_h) at q_handover
        marg = (d - Rp) / Rp
        a = I["anchor"][i]
        assert abs(marg[tp, a, j_hand] - I["regime"]["pass_margin"][i]) < 1e-9
        np.testing.assert_allclose(marg[hold, a, j_hand], I["regime"]["standoff_margin"][i], atol=1e-9)
        assert marg[tp].min() >= marg[tp, a, j_hand] - 0.02  # neighbours of the anchor sphere may come within 0.02
        episode = W.regime_episode(Hm.Script(t_pass=6.0, pass_width=0.6, wait=0.5), t)
        for k, (lo, hi) in enumerate(spans):  # the three reference configurations
            sel = (t >= lo) & (t <= hi) & (np.arange(len(t)) % W.FILTER_STRIDE == 0)
            dk = np.linalg.norm(C[i, k][None, :, None] - I["human_centres"][i][sel][:, None], axis=-1)
            assert np.min(dk - Rp) >= W.SEP_MARGIN
            mk = (dk - Rp) / Rp
            other = np.where((episode[sel][:, None, None] & arm[None, None]), np.inf, mk)
            assert other.min() >= I["regime"]["standoff_margin"][i] + W.REGIME_CLEAR - 1e-12
    E019 = Wm._instances(W.TUNING_SEED, sc, PLANT)
    cand = Wm._regime_candidates(W.TUNING_SEED, sc, PLANT)
    np.testing.assert_array_equal(cand["q0"][E019["candidate"]], E019["q0"])
    np.testing.assert_array_equal(cand["Qg"][E019["candidate"], 1], E019["q_handover"])
    assert I["candidate"].tolist() == sorted(I["candidate"].tolist())


def test_pruned_program_keeps_the_root_value_and_gradient():
    """float64, coarse scenario at H = 1 s: exact, matched sparsemax and matched lse, 3 traces."""
    sc = W.Scenario(robot_spacing=0.5, human_spacing=0.6, H=Fraction(1), pick=(Fraction(1, 5), Fraction(2, 5)),
                    handover=(Fraction(1, 2), Fraction(3, 5)), dwell=Fraction(1, 10))
    n_r = len(W.robot_spheres(PLANT, sc.robot_spacing)["body"])
    n_h = 7
    core = W.specs(sc, n_r, n_h)[2]
    full = stl.compile_formula(core, sc.samples)
    pr = E.prune(full)
    assert sum(s.index.size for s in pr.steps if s.kind != "atom") < sum(s.index.size for s in full.steps if s.kind != "atom")
    rng = np.random.default_rng(0)
    with jax.enable_x64(True):
        for trial in range(3):  # random traces
            Z = jnp.asarray(rng.normal(0.5, 0.5, (sc.samples, 3 + n_r + 2 * n_r * n_h)))
            for sem in ("exact", methods.SEMANTICS["sparsemax"], methods.SEMANTICS["lse"]):  # three semantics
                f = lambda prog: jax.value_and_grad(lambda Z: stl_jax.robustness(prog, Z, sem, 0.2)[0])(Z)
                (a, ga), (b, gb) = f(pr), f(full)
                assert float(a) == float(b)
                np.testing.assert_allclose(np.asarray(ga), np.asarray(gb), rtol=0, atol=1e-14)


def test_node_weights_are_the_matched_reductions_gradients():
    """float64: node_row's sparsemax support mass and lse mass equal those of jax.grad of the
    matched wrappers of core_study.methods, at a minimum and a maximum node."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.3, 40)
    x[5] = x.min() - 0.05
    T = 40
    for kind in ("min", "max"):  # both node kinds
        for eps in E.EPS:  # the per-node errors
            row = E.node_row(x, kind, eps, T, np.arange(T) < 10, np.arange(T) >= 30)
            with jax.enable_x64(True):
                fq = methods.sparsemax_min if kind == "min" else methods.sparsemax_max
                fl = methods.lse_min_matched if kind == "min" else methods.lse_max_matched
                p = np.asarray(jax.grad(lambda z: fq(z, eps))(jnp.asarray(x)))
                q = np.asarray(jax.grad(lambda z: fl(z, eps))(jnp.asarray(x)))
            S = p > 0
            assert row["k"] == S.sum() and abs(row["W_L"] - q[S].sum()) < 1e-12
            assert abs(row["WQ_pass"] - p[:10].sum()) < 1e-12 and abs(row["WL_wait"] - q[30:].sum()) < 1e-12


def test_blocked_rollout_equals_physics_rollout():
    """float64, 2 s (100 intervals, 2 blocks): states equal to 1e-12 and the gradient of a state
    functional to 1e-9 relative."""
    I = Wm.regime_instances(W.TUNING_SEED, 1, E.scenario(8), E.ranges(0.5))
    with jax.enable_x64(True):
        mx = mjx.put_model(PLANT.model, impl="jax")
        x0 = jnp.asarray(np.concatenate([I["q0"][0], np.zeros(7)]))
        U = jnp.asarray(E.random_start(3, 100)) * jnp.asarray(P.torque_limit())
        a = E.rollout(mx, 10, x0, U)
        b = Wm.physics_rollout(mx, 10, x0, U)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)
        f = lambda roll: jax.grad(lambda U: jnp.sum(jnp.sin(roll(mx, 10, x0, U)[::10])))(U)
        ga, gb = np.asarray(f(E.rollout)), np.asarray(f(Wm.physics_rollout))
    assert np.abs(ga - gb).max() <= 1e-9 * np.abs(gb).max()
