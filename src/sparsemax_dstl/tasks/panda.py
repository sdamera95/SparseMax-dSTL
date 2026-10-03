"""Panda task environment for both robot studies (E007, protocol D006).

Status (D006 amendment of 2026-09-29): the study scenario is the human-robot shared workspace
of tasks.workspace (E019), which uses this module's plant, kinematics and dynamics. The random
virtual obstacle spheres, the instances and the specifications below are retired for study use
and remain as test fixtures.

State and control. The state is x = (q, qdot) in R^14 and the control u in R^7 is the
gravity-compensated joint torque command, held constant over each sampling interval h.
Physics runs at DT = 0.002 s with implicitfast, the Newton solver, 100 iterations and
tolerance 1e-8 (Menagerie's options, checked in the tests); h is a whole number of physics
steps, so changing h changes sampling and nothing about the physics.

Torque. The model carries MuJoCo's gravity compensation routed through the actuators: body
gravcomp = 1 on every body, actuatorgravcomp on every joint, and a joint-level actuator force
range equal to the actuator's force range. At every physics step MuJoCo (C, MJX and
MuJoCo-Warp alike) applies

    tau = clip(clip(u, -tau_max, tau_max) + tau_g(q), -tau_max, tau_max),
    tau_g(q) = -sum_b J_b(q)^T m_b g,

with tau_max = (87, 87, 87, 87, 12, 12, 12) N m, so the command and the total joint torque both
stay within Menagerie's force ranges. Joint limits stay in the dynamics; robot geoms carry no
contact rows.

Signals. Predicate scores (atoms) are functions of one state, positive when satisfied, stacked
on the last axis: scores(mx, inst, X) maps (..., T, 14) to (..., T, P), the layout of the STL
layer. The order is score_layout(): goals, speed, then clearances link-major. Each atom is a
distance-like margin divided by its own threshold, smoothed at zero (E014, user ruling
2026-09-28, option C):

    goal       (R_GOAL - n+(p - c)) / R_GOAL
    speed      (V_MAX - n+(pdot)) / V_MAX
    clearance  (n-(c_l - o) - R) / R,          R = r_l + R_OBS

    n+(x) = sqrt(|x|^2 + eps^2) >= |x|,   n-(x) = sqrt(|x|^2 + eps^2) - eps <= |x|,

with eps = EPS_LENGTH for positions and EPS_SPEED for the velocity. So each atom is at most
its exact normalized margin, a positive atom implies the exact distance or speed condition,
one unit is the atom's own threshold, and the loss is at most eps / threshold. Every atom is
infinitely differentiable; its gradient in the physical vector has norm at most
1 / threshold and its Hessian norm at most 1 / (eps threshold), reached at x = 0. n- is
computed as |x|^2 / (n+(x) + eps), the same function without cancellation near zero.

p is the attachment_site position and c_l the centre of the bounding sphere of one collision
geom of a link (link_spheres), both from MJX kinematics; pdot is the forward-mode derivative
of p along qdot. None of these paths touches the constraint solver. lipschitz() gives each
atom's declared bound in the infinity norm of the state, the form E010's guard consumes.

Referee. margins(mx, inst, X) gives the unsmoothed normalized margins in the same layout,
eps = 0 in the norms (root decision on E014, 2026-09-28):

    goal (R_GOAL - |p - c|) / R_GOAL,  speed (V_MAX - |pdot|) / V_MAX,  clearance (|c_l - o| - R) / R.

They are the referee's scores: every exact robustness reported as an outcome reads them,
while every optimized objective reads scores(). Each margin is at least its atom, and every
Panda formula uses its atoms without negation, so the referee's robustness is at least the
exact robustness on the atoms. margins() stops gradients at its input, so no derivative is
ever taken through it; the norm is not differentiable at zero length.

Dynamics. interval_map is f_h, rollout composes it, interval_jacobians gives D_x f_h and
D_u f_h by forward mode. All three take the physics step as an argument; the default is
E004's mjx_implicit.step, whose values equal mjx.step and whose derivatives differentiate
the converged constraint solve.

Specifications. core_spec (phi_H) and extension_specs build E005 formulas over the score
layout, with windows rounded by window().
"""

import math
from fractions import Fraction
from functools import cache

import mujoco
import numpy as np

from .. import plants

DT = 0.002
REPLAY_DT = 0.0005
NQ = 7
SITE = "attachment_site"
BASE = "link0"
LINKS = ("link1", "link2", "link3", "link4", "link5", "link6", "link7")

R_GOAL = 0.05
R_OBS = 0.05
V_MAX = 0.1
# eps is 1% of the smallest threshold of its physical quantity (E014): R_GOAL for lengths,
# V_MAX for the speed.
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
    """The task plant, compiled from model_spec(timestep). Cached; do not mutate the result."""
    return model_spec(timestep).compile()


def model_spec(timestep=DT):
    """plants.panda(contacts=False) plus gravity compensation through the actuators and
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
    return model().actuator_ctrlrange[:, 1].copy()


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
    """Collision geoms of a body; Menagerie's collision class is geom group 3."""
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
    """Upper bounds D_j >= |x - o_j| over all configurations, for the point x at offset (body
    frame) on body, and o_j the anchor of the j-th joint of the body's chain, in chain order.

    D_j is the joint anchor's offset in its body plus the lengths of the body offsets after
    that body up to body, plus |offset|: the triangle inequality along the chain, since
    rotations preserve lengths. Every joint on the chain must be a hinge."""
    path = []
    b = body
    while b > 0:
        path.append(b)
        b = m.body_parentid[b]
    path = path[::-1]  # world -> body
    step = np.array([np.linalg.norm(m.body_pos[b]) for b in path])
    tail = np.concatenate([np.cumsum(step[::-1])[::-1][1:], [0.0]]) + np.linalg.norm(offset)
    D = []
    for i, b in enumerate(path):  # over the bodies of the chain (model structure)
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
    """Sample indices [ceil(a/h), floor(b/h)] of a window [a, b] in seconds (D006 item 7)."""
    a, b, h = _exact(a), _exact(b), _exact(h)
    return math.ceil(a / h), math.floor(b / h)


# ------------------------------------------------------------------
# windows of the specifications

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
    """Paired control initializations (n, n_init, H / h, 7).

    Torques are piecewise constant on cells of INIT_CELL seconds, drawn i.i.d. from
    N(0, (INIT_SIGMA tau_max)^2) per joint and clipped to the command limits, over H_MAX
    seconds; they are then held at the sampling interval h and cut to the horizon H. The
    initial command, as a function of time, is therefore the same for every h that divides
    INIT_CELL, and the one for a shorter horizon is a prefix of the one for a longer one.
    """
    tau = torque_limit()
    rng = np.random.default_rng([seed, 1])
    W = rng.standard_normal((n, n_init, intervals(H_MAX, INIT_CELL), NQ)) * INIT_SIGMA * tau
    W = np.clip(W, -tau, tau)
    return np.repeat(W, intervals(INIT_CELL, h), axis=2)[:, :, :intervals(H, h)]
