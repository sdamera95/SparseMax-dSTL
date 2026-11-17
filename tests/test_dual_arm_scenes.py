"""The dual-arm plant and the three problems of the sparse-differentiation examples against stored arrays (tests/data):
the URDF files and the constants read from them, the forward kinematics, and each problem's states, cells and conjuncts."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("scipy")
from sparsemax_dstl import stl  # noqa: E402
from sparsemax_dstl.tasks import dual_arm as D  # noqa: E402
from sparsemax_dstl.tasks import dual_arm_scenes as S  # noqa: E402

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("arm", ["right", "left"])
def test_urdf_files_and_joint_constants(arm):
    path = D.urdf_path(arm)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == D.URDF_SHA256[arm]
    joints = [j for j in ET.parse(path).getroot().iter("joint") if j.get("type") == "revolute"]
    a = {"right": 0, "left": 1}[arm]
    assert len(joints) == D.NJ
    for k, j in enumerate(joints):  # the arm's joints
        lim = j.find("limit")
        assert np.array_equal([float(v) for v in j.find("origin").get("xyz").split()], D.ORIGIN[a, k])
        assert np.array_equal([float(v) for v in j.find("axis").get("xyz").split()], D.AXIS[a, k])
        assert [float(lim.get("lower")), float(lim.get("upper"))] == D.LIMITS[a, k].tolist()


def test_forward_kinematics():
    f = np.load(DATA / "dual_arm_kinematics.npz")
    with jax.enable_x64(True):
        hand, C = jax.vmap(D.points)(jnp.asarray(f["q"]))
    np.testing.assert_allclose(np.asarray(hand), f["hand"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(C), f["spheres"], rtol=0, atol=1e-12)


def test_double_integrator():
    A, B = D.interval_matrices()
    x, u = np.linspace(-1, 1, D.NX), np.linspace(2, -2, D.NU)
    with jax.enable_x64(True):
        np.testing.assert_allclose(np.asarray(D.interval_map()(jnp.asarray(x), jnp.asarray(u))), A @ x + B @ u, rtol=1e-15)


@pytest.fixture(scope="module")
def scenes():
    return {"dual_arm_five_cell_zone.npz": S.five_cell_zone(100), "dual_arm_hold_up_delay14.npz": S.hold_up(100, 14),
            "dual_arm_circle.npz": S.circle(100), "dual_arm_circle_delay38.npz": S.circle(100, delay=38)}


@pytest.mark.parametrize("name", ["dual_arm_five_cell_zone.npz", "dual_arm_hold_up_delay14.npz", "dual_arm_circle.npz",
                                  "dual_arm_circle_delay38.npz"])
def test_scene(scenes, name):
    sc, f = scenes[name], np.load(DATA / name)
    assert np.array_equal(sc.states, f["states"])
    if "schedule" in f:
        assert np.array_equal(sc.schedule, f["schedule"])
    assert np.array_equal(sc.cells, f["cells"])
    want = json.loads(str(f["conjuncts"]))
    got = [{"name": c["name"], "kind": c["kind"], "anchors": np.asarray(c["anchors"]).tolist(), "formula": repr(c["formula"]),
            "nodes": [[n["kind"], np.asarray(n["preds"]).tolist()] for n in c["nodes"]], "gamma": c["gamma"]} for c in sc.conjuncts]
    assert got == want
    T_tr = max(int(np.max(c["anchors"])) + stl.horizon(stl.to_nnf(c["formula"])) for c in sc.conjuncts) + 1
    for c in sc.conjuncts:  # conjuncts (formula structure)
        assert c["window"] == stl.horizon(stl.to_nnf(c["formula"])) + 1
        prog = stl.compile_formula(c["formula"], T_tr, reads=[(c["formula"], c["anchors"])])
        assert prog.outputs[0][1].size == np.asarray(c["anchors"]).size


def test_predicates_on_the_schedule(scenes):
    """Every scheduled until instance starts with the visitor inside its cell: the release operand's exact value is
    negative at the anchor."""
    sc = scenes["dual_arm_five_cell_zone.npz"]
    with jax.enable_x64(True):
        P = np.asarray(jax.vmap(lambda x: sc.predicates(x, exact=True))(jnp.asarray(sc.schedule[:sc.n_episode])))
    assert P.shape == (sc.n_episode, sc.n_predicates)
    for c in sc.conjuncts:  # conjuncts (formula structure)
        if c["kind"] == "until":
            assert np.all(P[np.asarray(c["anchors"])][:, c["nodes"][1]["preds"]].min(-1) < 0)
