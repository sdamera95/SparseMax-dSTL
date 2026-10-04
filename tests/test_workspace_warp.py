"""Tests of the manipulator example on MuJoCo Warp (tasks.workspace_warp): the torque gradient of the until conjunct and
the first update of the optimization, against the paper's Table II and the recorded runs. Needs a CUDA device."""
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from sparsemax_dstl.tasks import workspace_scene as S
from sparsemax_dstl.tasks import workspace_warp
from sparsemax_dstl.warp import solver, solver_conjuncts

pytestmark = pytest.mark.skipif(wp.get_cuda_device_count() == 0, reason="needs a CUDA device")
wp.config.log_level = wp.LOG_WARNING
DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_until_torque_gradient_at_10_s():
    """Medians over the four trajectories against Table II: weight on the samples inside the zone and the norm of the
    torque gradient through them, for sparsemax and the sound log-sum-exp at eps = 0.2."""
    I = S.load(DATA / "manipulator_H10.npz")
    sc, prog, kw = S.until_program(10, 7.22, I["n_h"])
    Z, _ = workspace_warp.replay_predicates(I, sc, "cpu")
    inside = (Z[:, :, 2] < 0) & (np.arange(sc.samples) <= kw[-1])
    assert inside.sum(1).tolist() == [18, 18, 19, 19]
    g = workspace_warp.until_torque_gradient(I, sc, prog, kw, inside, [("sparsemax", 0.2), ("lse", 0.2)])
    assert not g["sign_flips_f32"].any() and g["copies_max_diff"] < 1e-4
    assert np.abs(g["w_viol32"][0] - 1).max() < 1e-5 and np.abs(g["grad_norm_viol"][0] / g["grad_norm"][0] - 1).max() < 1e-3
    assert abs(np.median(g["value32"][0]) + 0.2196) < 1e-4 and abs(np.median(g["value32"][1]) + 0.1773) < 2e-4
    assert abs(np.median(g["grad_norm_viol"][0]) - 16.818) < 2e-3
    assert abs(np.median(g["grad_norm_viol"][1]) - 3.623) < 2e-3 and np.all(g["grad_norm_viol"][1] < g["grad_norm"][1])


def test_first_update_under_sparsemax():
    """Eight violating trajectories, one constraint per conjunct: the exact robustness before and after the first update."""
    R = S.runs(S.load(DATA / "manipulator_H10_w2.npz"), ("sparsemax",), 0.2)
    chain, referee = workspace_warp.conjunct_chain(R)
    assert chain.K == 4 and len(R["groups"]) == 1
    state = solver_conjuncts.init_state(R["V0"], S.ALPHA0, chain.K)
    rho0, conj0 = referee(state["V"])
    assert np.abs(rho0 - R["group"]).max() < 3.3e-5 and np.all(np.argmin(conj0, 1) == 2)
    V, alpha, rec, conj = solver_conjuncts.iterate(chain, state, rho0, S.LAM, S.DELTA)
    assert np.all(rec[:, solver.TRACE_KEYS.index("accepted")] == 1) and np.abs(conj["r"][:, 2] - rec[:, 1]).max() == 0
    rho1, _ = referee(V)
    assert np.abs(rho1 - [0.0223, 0.0219, 0.0221, 0.0223, -0.0306, -0.0311, -0.0306, -0.0304]).max() < 1e-4
