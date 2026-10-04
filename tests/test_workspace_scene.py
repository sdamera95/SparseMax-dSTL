"""Tests of the manipulator's scene and its initial trajectories in examples/data (tasks.workspace_scene), on the CPU."""
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from sparsemax_dstl.tasks import panda
from sparsemax_dstl.tasks import workspace as W
from sparsemax_dstl.tasks import workspace_program as Wp
from sparsemax_dstl.tasks import workspace_scene as S
from sparsemax_dstl.tasks import workspace_warp
from sparsemax_dstl.warp import Evaluator

wp.config.log_level = wp.LOG_WARNING
DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
HAND = 17  # index of the hand's sphere in the person's cover


def exact(prog, M):
    ev = Evaluator(prog, "exact", None, len(M), wp.float64, "cpu", P=M.shape[-1])
    return ev.value(wp.array(M, dtype=wp.float64, device="cpu")).numpy()[:, 0]


@pytest.mark.parametrize("H", [10, 20, 40, 80])
def test_windows_at_the_latest_opening_time(H):
    w = S.longest_wait(H)
    sc, prog, kw = S.until_program(H, w, 27)
    assert w == round(H - 2.78, 2) and sc.samples == 50 * H + 1
    assert np.array_equal(kw, np.arange(round(50 * w), round(50 * (w + 1)) + 1)) and len(kw) == 51
    inner = [s for s in prog.steps if s.label.endswith(".inner")][0]
    assert inner.count[0] == round(50 * w) + 2  # the prefix up to the first witness and the pick sample
    assert float(sc.handover[1] * sc.H + sc.dwell * sc.H) == H


@pytest.mark.parametrize("name,H,n,wait", [("H10", 10, 4, 7.22), ("H20", 20, 4, 17.22), ("H40", 40, 4, 37.22), ("H80", 80, 4, 77.22),
                                           ("H10_w2", 10, 8, 2.0), ("H10_locations", 10, 8, 7.22)])
def test_data_files(name, H, n, wait):
    I = S.load(DATA / ("manipulator_" + name + ".npz"))
    T = 50 * H + 1
    assert I["H"] == H and I["V0"].shape == (n, T - 1, 7) and I["V0"].dtype == np.float32 and np.abs(I["V0"]).max() <= 1
    assert I["x0"].shape == (n, 14) and np.array_equal(I["x0"][0, :7], panda.model().key_qpos[0].astype(np.float32)) and not I["x0"][:, 7:].any()
    assert np.all(I["wait"] == wait) and set(I["group"]) <= {-0.05, -0.1}
    assert I["hc"].shape == (n, T, I["n_h"], 3) and I["hc"].dtype == np.float32 and I["hr"].shape == (I["n_h"],)
    sc = S.scenario(H, wait)
    if name != "H10_locations":  # the paper's one pick and handover location
        pick, hand = W.until_targets(sc, S.DIRECTION, S.HANDOVER_RADIUS)
        assert np.array_equal(I["pick"], np.broadcast_to(pick, (n, 3))) and np.array_equal(I["handover"], np.broadcast_to(hand, (n, 3)))
    zone = np.asarray(sc.zone)
    assert np.abs(np.linalg.norm(I["pick"] - zone, axis=1) - (sc.zone_radius + S.HOLD)).max() < 1e-12
    assert np.all(np.linalg.norm(I["handover"] - zone, axis=1) < sc.zone_radius)
    # the hand is 0.45 m from the zone centre from 2.3 s until 0.3 s before the pick window closes, and away at the start
    t = np.arange(T) * S.HS
    d = np.linalg.norm(I["hc"][0, :, HAND, :2] - zone[:2], axis=-1)
    assert np.allclose(d[(t >= 2.3) & (t <= wait + 0.7)], 0.45, atol=1e-6) and d[0] > 1


def test_runs_repeat_the_trajectories_per_measure():
    I = S.load(DATA / "manipulator_H10_locations.npz")
    I["wait"] = np.repeat([2.0, 7.22], 4)  # two opening times, to split every measure's block
    R = S.runs(I, ("lse", "sparsemax", "gm_pm01"), 0.2)
    assert R["measure"].tolist() == ["lse"] * 8 + ["sparsemax"] * 8 + ["gm_pm01"] * 8 and R["eps"] == 0.2
    for k in S.PER_TRAJECTORY:
        assert np.array_equal(R[k], np.concatenate([I[k]] * 3)), k
    assert R["H"] == 10 and R["n_h"] == I["n_h"] and R["hr"] is I["hr"]
    key = [(str(m), 0.2, float(w)) for m, w in zip(R["measure"], R["wait"])]
    groups, a = [], 0
    for b in range(1, len(key) + 1):  # brute-force oracle: consecutive runs with one (measure, eps, wait)
        if b == len(key) or key[b] != key[a]:
            groups.append((key[a][0], key[a][1], a, b, key[a][2]))
            a = b
    assert R["groups"] == groups and len(groups) == 6 and groups[1] == ("lse", 0.2, 4, 8, 7.22)


def test_initial_trajectories_violate_the_until_conjunct_alone():
    """On the float64 replay the until conjunct is within 3.3e-5 of -0.05 or -0.10 and is the exact robustness; the other
    three conjuncts are positive."""
    for name in ("H10", "H10_w2", "H10_locations"):
        I = S.load(DATA / ("manipulator_" + name + ".npz"))
        sc = S.scenario(I["H"], I["wait"][0])
        Z, M = workspace_warp.replay_predicates(I, sc, "cpu")
        assert Z.shape == M.shape == (len(I["V0"]), sc.samples, 3 + Wp.N_R * (1 + 2 * I["n_h"]))
        assert np.all(Z <= M + 1e-12) and np.all(M - Z <= 0.01 + 1e-9)  # each smoothed predicate is at most 0.01 below the exact one
        conj = np.stack([exact(p, M) for p in Wp.conj_programs(sc, I["n_h"])], -1)
        assert Wp.CONJUNCTS[2] == "order" and np.abs(conj[:, 2] - I["group"]).max() < 3.3e-5
        assert np.all(np.delete(conj, 2, 1) > 0.5)
        assert np.array_equal(exact(Wp.core_program(sc, I["n_h"]), M), conj.min(1))
