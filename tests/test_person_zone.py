"""E039: the working person's script (tasks.workspace.until_work_script) and the pre-run check's bookkeeping
(examples.e039_person_zone)."""
import numpy as np

from examples import e034_until_demo as D
from examples import e039_person_zone as P
from sparsemax_dstl.tasks import human as Hm
from sparsemax_dstl.tasks import workspace as W

WORK = {"phi": -1.3, "stand": 0.6, "depth": -0.194, "reach_z": 0.0195, "release": 0.0, "t_move": 0.6, "hover": 0.3, "hover_z": -0.15}


def hand_point(script, t):
    ends, _ = Hm.capsules(script, t)
    return ends[:, Hm.NAMES.index("forearm_r"), 1]


def test_hand_sphere_is_the_hand_point():
    sc = D.scenario(4.0)
    I = W.until_work_instance(sc, D.DIRECTION, D.HANDOVER_RADIUS, WORK)
    t = np.arange(sc.samples) * D.HS
    assert I["human_owner"][P.HAND] == Hm.NAMES.index("forearm_r")
    assert P.HAND + 1 == len(I["human_owner"]) or I["human_owner"][P.HAND + 1] != I["human_owner"][P.HAND]
    np.testing.assert_allclose(I["human_centres"][:, P.HAND], hand_point(W.until_work_script(sc, WORK), t), atol=1e-12)


def test_hand_in_the_zone_through_the_wait_then_at_the_standoff():
    for w in D.WAITS:
        sc = D.scenario(w)
        s = W.until_work_script(sc, WORK)
        t = np.arange(sc.samples) * D.HS
        h = hand_point(s, t)
        u = np.array([np.cos(WORK["phi"]), np.sin(WORK["phi"]), 0.0])
        zone = np.asarray(sc.zone)
        target = zone - WORK["depth"] * u + np.array([0, 0, WORK["reach_z"]])
        wait = t <= w + 1e-9
        np.testing.assert_allclose(h[wait], np.broadcast_to(target, h[wait].shape), atol=1e-12)
        assert np.all(np.linalg.norm(h[wait] - zone, axis=-1) < sc.zone_radius)
        after = t >= w + WORK["release"] + WORK["t_move"] - 1e-9
        assert np.all(np.linalg.norm(h[after] - zone, axis=-1) > sc.zone_radius)
        # the standoff point lies within the arm's reach here, so the hand is at it (no clipping)
        stand = zone + (sc.zone_radius + WORK["hover"]) * u + np.array([0, 0, WORK["hover_z"]])
        np.testing.assert_allclose(h[after], np.broadcast_to(stand, h[after].shape), atol=1e-12)


def test_e034_persons_unchanged():
    sc = D.scenario(4.0)
    for person in ("zone", "out"):
        a = W.until_instance(sc, D.DIRECTION, D.HANDOVER_RADIUS, person)
        b = W.human_inputs(W.until_script(sc, person), sc)
        np.testing.assert_array_equal(a["human_centres"], b["human_centres"])
    assert W.UNTIL_PERSON == {"zone": {"stand": 0.6, "depth": -0.17, "reach_z": 0.03, "reach": True},
                              "out": {"stand": 2.0, "depth": 0.0, "reach_z": 0.10, "reach": False}}


def test_phases_partition_the_samples():
    t = np.arange(501) * D.HS
    for w in D.WAITS:
        for d in D.DEPTHS:
            m = P.phase_masks(w, d, t)
            assert np.all(m.sum(0) == 1)


def test_gaps_against_a_loop():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(2, 3, 4, 3))
    hc = rng.normal(size=(3, 20, 3))
    rr, hr = rng.uniform(0.05, 0.1, 4), rng.uniform(0.05, 0.1, 20)
    gp, gh = P.gaps(C, rr, hc, hr)
    for b in range(2):  # brute-force test oracle over plans, samples and sphere pairs
        for k in range(3):
            d = [np.linalg.norm(C[b, k, i] - hc[k, j]) - rr[i] - hr[j] for i in range(4) for j in range(20)]
            assert np.isclose(gp[b, k], min(d))
            assert np.isclose(gh[b, k], min(np.linalg.norm(C[b, k, i] - hc[k, P.HAND]) - rr[i] - hr[P.HAND] for i in range(4)))


VISIT = {"kind": "visit", "phi": 0.0, "stand": 0.9, "depth": -0.45, "reach_z": 0.10, "t_move": 0.2, "lead": 0.1, "t_rest": 2.1, "length": 10.0,
         "far": 0.6, "hover": 0.6, "hover_z": -0.15}


def test_visit_plateau_and_far_stance():
    for w in D.WAITS:
        sc = D.scenario(w)
        I = W.until_visit_inputs(sc, VISIT)
        v0, v1 = I["visit"]
        assert np.isclose(v1, w + 1 - 0.1 - 0.2) and np.isclose(v0, 2.3)
        t = np.arange(sc.samples) * D.HS
        h = I["human_centres"][:, P.HAND]
        plateau = (t >= v0 - 1e-9) & (t <= v1 + 1e-9)
        np.testing.assert_allclose(h[plateau], np.broadcast_to([0.95, 0.0, 0.10], h[plateau].shape), atol=1e-12)
        # before the step-in and after the step-back the whole body stands 0.6 m further out than in the plain standing script
        s = W.until_work_script(sc, dict(VISIT, release=v1 - w))
        ends, _ = Hm.capsules(s, t)
        out = (t <= v0 - 0.2 + 1e-9) | (t >= v1 + 0.2 - 1e-9)
        torso = I["human_ends"][:, 0]
        np.testing.assert_allclose(torso[out] - ends[out][:, 0], np.broadcast_to([0.6, 0.0, 0.0], torso[out].shape), atol=1e-12)
        # rigid: the capsule lengths do not change
        L = np.linalg.norm(I["human_ends"][:, :, 1] - I["human_ends"][:, :, 0], axis=-1)
        np.testing.assert_allclose(L, np.broadcast_to(L[0], L.shape), atol=1e-12)
