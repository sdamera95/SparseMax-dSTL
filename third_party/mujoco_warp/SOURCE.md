# Source

The Python package of [etaoxing/mujoco_warp](https://github.com/etaoxing/mujoco_warp), branch `adjoint`, at commit `357a75d60a56d67d476942a1b6e54b3045ee8e87`: MuJoCo Warp 3.12.0 with reverse-mode derivatives. Its tests and test data are left out. The licence is Apache-2.0 (`LICENSE`).

Two files differ from that commit:

- `mujoco_warp/_src/collision_adjoint.py` adds the adjoint of box-box contact geometry, for the face-vertex and the edge-edge case, to the fork's adjoints of the other primitive pairs.
- `mujoco_warp/_src/constraint_adjoint.py` keeps a contact in the backward pass when any of its constraint rows is active; the fork tests the first row alone, which drops a contact with a pyramidal cone whose first row carries no force.
