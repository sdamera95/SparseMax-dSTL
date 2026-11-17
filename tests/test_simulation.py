"""The rigid-body simulation of both arms against a stored closed-loop record of the hold-up (tests/data): the model, one
rollout per call against physics steps taken one sample at a time, the replay of the record's targets, and the exact
robustness of every until and reach instance and the arms' clearance on the record."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("scipy")
from sparsemax_dstl.mpc import simulation as SIM  # noqa: E402
from sparsemax_dstl.tasks import dual_arm as D  # noqa: E402
from sparsemax_dstl.tasks import dual_arm_scenes as S  # noqa: E402

DATA = Path(__file__).parent / "data"


def test_model():
    m = SIM.model()
    assert (m.nq, m.nv, m.nu) == (D.NQ, D.NQ, 2 * D.NQ)
    assert m.opt.timestep == SIM.DT_PHYSICS
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for arm in SIM.JOINTS for n in arm]
    assert np.array_equal(m.jnt_actfrcrange[ids, 1], SIM.EFFORT)
    assert np.all(m.dof_armature[m.jnt_dofadr[ids]] == SIM.ARMATURE)


def test_rollout_against_steps_one_sample_at_a_time():
    f = np.load(DATA / "simulation_hold_up_replay.npz")
    Q = f["targets"][:10]
    sim = SIM.Simulation(f["x0"])
    X = sim.run(Q)
    ref = SIM.Simulation(f["x0"])
    mujoco.mj_setState(ref.m, ref.d, ref.s, SIM.FULL)
    out = []
    for q in Q:  # brute-force oracle: a Python loop over samples
        ref.d.ctrl[:D.NQ] = q
        mujoco.mj_step(ref.m, ref.d, nstep=SIM.SUBSTEPS)
        out.append(np.concatenate([ref.d.qpos[ref.qadr], ref.d.qvel[ref.vadr]]))
    np.testing.assert_allclose(X, np.stack(out), rtol=0, atol=1e-12)


def test_replay_of_recorded_targets():
    f = np.load(DATA / "simulation_hold_up_replay.npz")
    sim = SIM.Simulation(f["x0"])
    P, tg = int(f["samples_per_period"]), f["targets"]
    X = np.concatenate([sim.run(tg[i:i + P]) for i in range(0, len(tg), P)])  # control periods
    np.testing.assert_allclose(X[:, :D.NQ], f["states"][:, :D.NQ], rtol=0, atol=1e-9)
    np.testing.assert_allclose(X[:, D.NQ:], f["states"][:, D.NQ:], rtol=0, atol=1e-7)


def test_instances_and_clearance_on_a_record():
    f = np.load(DATA / "simulation_hold_up_record.npz")
    scene = S.hold_up(int(f["horizon"]), int(f["delay"]), states=f["record"])
    got = {(i["conjunct"], i["anchor"]): i for i in SIM.instances(scene, D.EPS_LENGTH)}
    for n, a, last, v in zip(f["names"], f["anchors"], f["last"], f["exact_smoothed"]):  # the recorded instances
        assert got[(str(n), int(a))]["last"] == int(last)
        np.testing.assert_allclose(got[(str(n), int(a))]["exact"], v, rtol=0, atol=1e-9)
    exact = {k: i["exact"] for k, i in {(i["conjunct"], i["anchor"]): i for i in SIM.instances(scene)}.items()}
    gap = np.array([exact[k] - got[k]["exact"] for k in exact])
    assert gap.min() >= 0 and gap.max() <= D.EPS_LENGTH / scene.cell_radius
    k0 = int(f["clearance_start"])
    cl = SIM.clearance(f["record"][k0:k0 + len(f["clearance"])])
    np.testing.assert_allclose(cl, f["clearance"], rtol=0, atol=1e-12)
