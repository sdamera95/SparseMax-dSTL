"""The Franka Panda of the manipulator example in MuJoCo: the model with gravity compensation through the
actuators, its collision geometry, the time grid and control initializations."""

import math
import os
from fractions import Fraction
from functools import cache

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout

from .. import plants

DT = 0.002  # physics step, s
REPLAY_DT = 0.0005
NQ = 7  # the state is (q, qdot) in R^14, the control a torque command in R^7
SITE = "attachment_site"
BASE = "link0"
LINKS = ("link1", "link2", "link3", "link4", "link5", "link6", "link7")

R_GOAL = 0.05  # goal radius, m
R_OBS = 0.05  # obstacle radius, m
V_MAX = 0.1  # bound on the site's speed, m/s
# smoothing lengths of the norms: 1% of R_GOAL for lengths (m) and of V_MAX for the speed (m/s)
EPS_LENGTH = 5e-4
EPS_SPEED = 1e-3

TUNING_SEED = 2026092801
HELDOUT_SEED = 2026092802
N_TUNING = 10
N_HELDOUT = 30
N_INIT = 5
Q_SPREAD = np.array([0.8, 0.5, 0.5, 0.6, 0.6, 0.5, 0.6])
JOINT_MARGIN = 0.1
GOAL_MIN_Z = 0.15
MIN_SEPARATION = 0.25
OBS_FRACTION = (0.35, 0.65)
OBS_OFFSET = 0.06
CLEAR_MARGIN = 0.02
N_CANDIDATES = 4096
INIT_SIGMA = 0.01
INIT_CELL = Fraction(1, 25)
H_MAX = 16


# ------------------------------------------------------------------
# model

@cache
def model(timestep=DT):
    """The compiled model of model_spec(timestep). Cached; do not mutate the result."""
    return model_spec(timestep).compile()


def model_spec(timestep=DT):
    """The Panda without contacts (plants.panda_spec) with gravity compensation through the actuators and
    joint-level actuator force ranges equal to the actuator force ranges."""
    spec = plants.panda_spec(contacts=False)
    spec.option.timestep = timestep
    force = {a.target: np.array(a.forcerange) for a in spec.actuators}
    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1.0
    for j in spec.joints:
        j.actgravcomp = True
        j.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.actfrcrange = force[j.name]
    return spec


def torque_limit():
    """Upper end of every actuator's control range, (7,)."""
    return model().actuator_ctrlrange[:, 1].copy()


def replay(V, x0):
    """The float64 MuJoCo states (B, T, 14) at the interval boundaries under the normalized torques V (B, T - 1, 7) from
    x0 (14,), ten physics steps per interval."""
    mjm = model()
    B = len(V)
    datas = [mujoco.MjData(mjm) for _ in range(min(B, len(os.sched_getaffinity(0))))]
    U = np.repeat(np.asarray(V, np.float64) * np.asarray(torque_limit(), np.float64), 10, axis=1)
    state0 = np.broadcast_to(np.concatenate([[0.0], x0]), (B, 15)).copy()
    st, _ = mj_rollout.rollout(mjm, datas, state0, U)
    return np.concatenate([np.broadcast_to(x0, (B, 1, 14)), st[:, 9::10, 1:]], 1)


def _body(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def _chain(m, body):
    """Degrees of freedom of the joints between the world and body, in increasing order."""
    dofs = []
    while body > 0:
        dofs += range(m.body_dofadr[body], m.body_dofadr[body] + m.body_dofnum[body])
        body = m.body_parentid[body]
    return tuple(sorted(dofs))


# ------------------------------------------------------------------
# collision geometry

def collision_geoms(m, body):
    """Collision geoms of a body: those in geom group 3, the collision class of the Menagerie model."""
    return np.nonzero((m.geom_bodyid == body) & (m.geom_group == 3))[0]


def geom_vertices(m, g):
    """Vertices of mesh geom g in its body's frame."""
    assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    k = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(np.float64)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return m.geom_pos[g] + v @ R.reshape(3, 3).T


def reach_bounds(m, body, offset):
    """Upper bounds D_j >= |x - o_j| over all configurations, for the point x at offset (body frame) on
    body and the anchor o_j of every hinge joint on the body's chain, in chain order."""
    path = []
    b = body
    while b > 0:
        path.append(b)
        b = m.body_parentid[b]
    path = path[::-1]  # world -> body
    step = np.array([np.linalg.norm(m.body_pos[b]) for b in path])
    tail = np.concatenate([np.cumsum(step[::-1])[::-1][1:], [0.0]]) + np.linalg.norm(offset)
    D = []
    for i, b in enumerate(path):
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            D.append(np.linalg.norm(m.jnt_pos[j]) + tail[i])
    return np.array(D)


# ------------------------------------------------------------------
# time grid

def _exact(t):
    """A time as an exact fraction, reading a float by its decimal representation."""
    return t if isinstance(t, (Fraction, int)) else Fraction(str(t))


def _whole(ratio, what):
    if ratio.denominator != 1 or ratio < 1:
        raise ValueError(what + " must be a positive whole number, got " + str(ratio))
    return int(ratio)


def substeps(h, dt=DT):
    """Physics steps per sampling interval."""
    return _whole(_exact(h) / _exact(dt), "h / dt")


def intervals(H, h):
    """Number of sampling intervals N = H / h; a trajectory has N + 1 samples."""
    return _whole(_exact(H) / _exact(h), "H / h")


def window(a, b, h):
    """Sample indices [ceil(a/h), floor(b/h)] of a window [a, b] in seconds."""
    a, b, h = _exact(a), _exact(b), _exact(h)
    return math.ceil(a / h), math.floor(b / h)


# ------------------------------------------------------------------
# windows of the specifications, as fractions of the horizon

REACH = ((Fraction(1, 4), Fraction(1, 2)), (Fraction(3, 5), Fraction(4, 5)))
DWELL = Fraction(1, 10)


# ------------------------------------------------------------------
# initializations

def _config_box():
    m = model()
    lo, hi = m.jnt_range.T
    home = m.key_qpos[0]
    return np.maximum(home - Q_SPREAD, lo + JOINT_MARGIN), np.minimum(home + Q_SPREAD, hi - JOINT_MARGIN)


def initial_state(inst):
    """x0 = (q0, 0) for every instance: (..., 14)."""
    q0 = np.asarray(inst["q0"])
    return np.concatenate([q0, np.zeros_like(q0)], -1)


def initial_controls(seed, n, H, h, n_init=N_INIT):
    """Control initializations (n, n_init, H / h, 7): torques drawn from N(0, (INIT_SIGMA tau_max)^2) per
    joint, constant on cells of INIT_CELL seconds, clipped to the command limits, held at h, cut to H."""
    tau = torque_limit()
    rng = np.random.default_rng([seed, 1])
    W = rng.standard_normal((n, n_init, intervals(H_MAX, INIT_CELL), NQ)) * INIT_SIGMA * tau
    W = np.clip(W, -tau, tau)
    return np.repeat(W, intervals(INIT_CELL, h), axis=2)[:, :, :intervals(H, h)]
