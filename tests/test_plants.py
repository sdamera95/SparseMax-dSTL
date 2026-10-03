import subprocess

import numpy as np
from mujoco import mjx

from sparsemax_dstl import plants


def efc_rows(m):
    return mjx.make_data(mjx.put_model(m, impl="jax"))._impl.efc_J.shape[0]


def test_menagerie_commit_is_pinned():
    head = subprocess.run(["git", "-C", str(plants.MENAGERIE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    assert head.stdout.strip() == plants.MENAGERIE_COMMIT


def test_panda_torque_motors_and_joint_limits():
    m = plants.panda(contacts=False)
    assert (m.nq, m.nv, m.nu) == (7, 7, 7)
    assert np.all(m.jnt_limited == 1)
    assert np.all(m.actuator_biastype == 0)
    np.testing.assert_array_equal(m.actuator_gainprm[:, 0], np.ones(7))
    np.testing.assert_array_equal(m.actuator_ctrlrange[:, 1], [87, 87, 87, 87, 12, 12, 12])
    assert np.all(m.geom_contype == 0) and np.all(m.geom_conaffinity == 0)
    assert efc_rows(m) == 7


def test_sliding_block_has_contact_rows():
    m = plants.sliding_block()
    assert m.nu == 1 and not np.any(m.jnt_limited)
    assert efc_rows(m) > 0
