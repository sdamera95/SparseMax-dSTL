"""Tests of the person's scripted motions in the scene with the pick target just outside the zone (tasks.workspace.until_*)."""
import numpy as np

from sparsemax_dstl.tasks import human as Hm
from sparsemax_dstl.tasks import workspace as W
from sparsemax_dstl.tasks import workspace_scene as D

WAITS = (2.0, 4.0, 6.0, 7.22)
HAND = 17  # index of the hand's sphere in the person's cover, the distal sphere of forearm_r
WORK = {"phi": -1.3, "stand": 0.6, "depth": -0.194, "reach_z": 0.0195, "release": 0.0, "t_move": 0.6, "hover": 0.3, "hover_z": -0.15}


def hand_point(script, t):
    ends, _ = Hm.capsules(script, t)
    return ends[:, Hm.NAMES.index("forearm_r"), 1]


def test_hand_sphere_is_the_hand_point():
    sc = D.scenario(10, 4.0)
    I = W.until_work_instance(sc, D.DIRECTION, D.HANDOVER_RADIUS, WORK)
    t = np.arange(sc.samples) * D.HS
    assert I["human_owner"][HAND] == Hm.NAMES.index("forearm_r")
    assert HAND + 1 == len(I["human_owner"]) or I["human_owner"][HAND + 1] != I["human_owner"][HAND]
    np.testing.assert_allclose(I["human_centres"][:, HAND], hand_point(W.until_work_script(sc, WORK), t), atol=1e-12)


def test_hand_in_the_zone_through_the_wait_then_at_the_standoff():
    for w in WAITS:
        sc = D.scenario(10, w)
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


def test_standing_persons_are_the_scripted_ones():
    sc = D.scenario(10, 4.0)
    for person in ("zone", "out"):
        a = W.until_instance(sc, D.DIRECTION, D.HANDOVER_RADIUS, person)
        b = W.human_inputs(W.until_script(sc, person), sc)
        np.testing.assert_array_equal(a["human_centres"], b["human_centres"])
    assert W.UNTIL_PERSON == {"zone": {"stand": 0.6, "depth": -0.17, "reach_z": 0.03, "reach": True},
                              "out": {"stand": 2.0, "depth": 0.0, "reach_z": 0.10, "reach": False}}


VISIT = {"kind": "visit", "phi": 0.0, "stand": 0.9, "depth": -0.45, "reach_z": 0.10, "t_move": 0.2, "lead": 0.1, "t_rest": 2.1, "length": 10.0,
         "far": 0.6, "hover": 0.6, "hover_z": -0.15}


def test_visit_plateau_and_far_stance():
    for w in WAITS:
        sc = D.scenario(10, w)
        I = W.until_visit_inputs(sc, VISIT)
        v0, v1 = I["visit"]
        assert np.isclose(v1, w + 1 - 0.1 - 0.2) and np.isclose(v0, 2.3)
        t = np.arange(sc.samples) * D.HS
        h = I["human_centres"][:, HAND]
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


def test_hand_place_on_the_plateau():
    for place in (0.40, 0.45, 0.60):
        for w in (2.0, 7.22):
            I = W.until_visit_inputs(D.scenario(10, w), dict(VISIT, depth=-place))
            t = np.arange(len(I["human_centres"])) * D.HS
            v0, v1 = I["visit"]
            m = (t >= v0 + 1e-9) & (t <= v1 - 1e-9)
            h = I["human_centres"][m, HAND]
            zone = np.asarray(D.scenario(10, w).zone)
            assert np.allclose(h[:, 0] - zone[0], place, atol=1e-9) and np.allclose(h[:, 1], 0.0, atol=1e-9)
            assert np.allclose(h[:, 2], 0.1, atol=1e-9)
