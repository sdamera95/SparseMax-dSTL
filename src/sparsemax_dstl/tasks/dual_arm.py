"""The dual-arm manipulator of the sparse-differentiation examples: two seven-joint arms on one base, their forward
kinematics and collision spheres, the double integrator in joint space, and the predicates "outside a cell" and "at a
waypoint".

Frames are the base's (x forward, y to the robot's left, z up). Arm 0 is the right arm, arm 1 the left; joints 1 to 7 of
an arm are shoulder pitch, roll and yaw, elbow pitch, wrist yaw, pitch and roll. Link k's frame is
T_k = T_{k-1} Trans(ORIGIN_k) Rot(AXIS_k, q_k) from T_0 = Trans(PLACEMENT), and the hand is T_7 Trans(MOUNT_POS) Rot(MOUNT_ROT).
The state is x = (q_right, q_left, qd_right, qd_left) in R^28, the control u the joint accelerations in R^14, held over
each interval of H_STEP seconds.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

NJ = 7
NQ = 14
NX = 28
NU = 14
H_STEP = 0.02
JOINT_MARGIN = 0.15  # rad kept clear of each position limit
# joint drives of the simulation model
VELOCITY_LIMIT = 2.0  # rad/s
ACCELERATION_LIMIT = 50.0  # rad/s^2

# from the arm's URDF: joint origins in the parent link's frame (m), axes and position limits (rad), joints 1 to 7
ORIGIN = np.array([
    [[0.0, -0.096, 0.0], [0.0328, -0.075, 0.0], [-0.0328, 0.0, -0.12], [0.0095, -0.0325, -0.15],
     [-0.0095, 0.0325, -0.11], [0.0, 0.022, -0.06], [0.0225, -0.022, -0.064]],
    [[0.0, 0.096, 0.0], [0.0328, 0.075, 0.0], [-0.0328, 0.0, -0.12], [0.0095, -0.0325, -0.15],
     [-0.0095, 0.0325, -0.11], [0.0, -0.022, -0.06], [0.0225, 0.022, -0.064]]])
AXIS = np.array([
    [[0, -1, 0], [1, 0, 0], [0, 0, -1], [0, -1, 0], [0, 0, -1], [0, 1, 0], [1, 0, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, -1], [0, -1, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0]]], np.float64)
LIMITS = np.array([
    [[-1.05, 2.97], [-3.49, 0.35], [-2.79, 2.79], [-0.21, 2.355], [-2.79, 2.79], [-1.57, 1.57], [-1.57, 1.57]],
    [[-2.97, 1.05], [-0.35, 3.49], [-2.79, 2.79], [-0.21, 2.355], [-2.79, 2.79], [-1.57, 1.57], [-1.57, 1.57]]])
# the shoulder bases on the base (m), and the hand's mount on link 7: an offset and a half turn about link 7's x axis
PLACEMENT = np.array([[0.0, 0.0155, 0.89], [0.0, -0.0155, 0.89]])
MOUNT_POS = np.array([-0.045, 0.0, -0.0705])
MOUNT_ROT = np.diag([1.0, -1.0, -1.0])
# a start configuration of both arms, right joints 1 to 7 then left
START = np.array([0.10264, -0.01653, -0.06842, 1.08105, -1.49420, 0.00214, -0.43992,
                  -0.15378, 0.00932, 0.35089, 1.10520, 1.15322, -0.25960, 0.45310])

URDF = ("a7__right.nomesh.urdf", "a7__left.nomesh.urdf")  # the arms' URDFs without their meshes
URDF_SHA256 = {"right": "5e17c5e8cf0516ffc326cce06a93edd71cc37b60379f470681e26d0fe2507274",
               "left": "36e6162f5c662c633ed5803bb029bb9905c99e890bc5842f543fd7cd182fb8ed"}


def urdf_path(arm):
    """Path of an arm's URDF (arm 0 or "right", 1 or "left")."""
    i = {"right": 0, "left": 1}.get(arm, arm)
    return Path(__file__).parent / "data" / URDF[i]


def joint_bounds(margin=JOINT_MARGIN):
    """(lower, upper) of q (14,): the position limits less the margin."""
    L = LIMITS.reshape(NQ, 2)
    return L[:, 0] + margin, L[:, 1] - margin


def _rot(axis, angle):
    """Rotation matrices (..., 3, 3) about unit axes (..., 3) by angles (...)."""
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    o = jnp.zeros_like(x)
    K = jnp.stack([jnp.stack([o, -z, y], -1), jnp.stack([z, o, -x], -1), jnp.stack([-y, x, o], -1)], -2)
    s, c = jnp.sin(angle)[..., None, None], jnp.cos(angle)[..., None, None]
    return jnp.eye(3, dtype=angle.dtype) + s * K + (1 - c) * (K @ K)


def frames(q):
    """Rotations (2, 8, 3, 3) and origins (2, 8, 3) of links 1 to 7 and the hand of both arms at q (14,)."""
    dt = q.dtype
    qa = q.reshape(2, NJ)
    o, a = jnp.asarray(ORIGIN, dt), jnp.asarray(AXIS, dt)
    R = jnp.broadcast_to(jnp.eye(3, dtype=dt), (2, 3, 3))
    p = jnp.asarray(PLACEMENT, dt)
    Rs, ps = [], []
    for k in range(NJ):  # the joints of the kinematic chain
        p = p + jnp.einsum("aij,aj->ai", R, o[:, k])
        R = R @ _rot(a[:, k], qa[:, k])
        Rs.append(R)
        ps.append(p)
    ps.append(p + jnp.einsum("aij,j->ai", R, jnp.asarray(MOUNT_POS, dt)))
    Rs.append(R @ jnp.asarray(MOUNT_ROT, dt))
    return jnp.stack(Rs, 1), jnp.stack(ps, 1)


# collision spheres covering the links and hands of the simulation model: arm (0 right, 1 left), link (1 to 7; 8 the
# hand), centre in that link's frame (m), radius (m)
SPHERES = np.array([
    [0, 1, 0.0302, -0.115, 0.0229, 0.068],
    [0, 1, -0.0004, -0.066, 0.002, 0.068],
    [0, 1, -0.0309, -0.017, -0.0188, 0.068],
    [0, 2, -0.036, -0.0015, -0.1248, 0.0618],
    [0, 2, -0.0339, -0.0005, -0.0494, 0.0618],
    [0, 2, -0.0318, 0.0004, 0.026, 0.0618],
    [0, 3, -0.0021, 0.0044, 0.0043, 0.0653],
    [0, 3, 0.0052, 0.0032, -0.0877, 0.0653],
    [0, 3, 0.0124, 0.002, -0.1797, 0.0653],
    [0, 4, -0.0118, 0.0335, -0.1126, 0.0587],
    [0, 4, -0.0061, 0.0309, -0.0457, 0.0587],
    [0, 4, -0.0004, 0.0284, 0.0213, 0.0587],
    [0, 5, 0.0058, -0.0018, 0.0043, 0.0656],
    [0, 5, 0.0032, -0.0079, -0.09, 0.0656],
    [0, 6, 0.0019, 0.0088, 0.0214, 0.0533],
    [0, 6, -0.0057, -0.0115, -0.0359, 0.0533],
    [0, 6, -0.0132, -0.0319, -0.0932, 0.0533],
    [0, 7, -0.0474, -0.0026, -0.0798, 0.055400000000000005],
    [0, 7, -0.0289, -0.0017, -0.032, 0.055400000000000005],
    [0, 7, -0.0105, -0.0008, 0.0158, 0.055400000000000005],
    [0, 8, -0.0205, 0.0062, -0.0009, 0.0718],
    [0, 8, -0.0023, 0.0045, 0.0959, 0.0718],
    [0, 8, 0.0158, 0.0028, 0.1927, 0.0718],
    [0, 8, 0.0782, 0.1332, 0.0477, 0.035800000000000005],
    [0, 8, 0.0405, 0.0797, 0.0361, 0.035800000000000005],
    [0, 8, 0.0027, 0.0262, 0.0245, 0.035800000000000005],
    [1, 1, 0.0308, 0.1156, -0.0202, 0.068],
    [1, 1, -0.0001, 0.0662, -0.0012, 0.068],
    [1, 1, -0.0309, 0.0168, 0.0178, 0.068],
    [1, 2, -0.036, -0.0015, -0.1248, 0.0618],
    [1, 2, -0.0339, -0.0005, -0.0494, 0.0618],
    [1, 2, -0.0318, 0.0004, 0.026, 0.0618],
    [1, 3, 0.0119, 0.0005, -0.1804, 0.06470000000000001],
    [1, 3, 0.0047, 0.0035, -0.0881, 0.06470000000000001],
    [1, 3, -0.0026, 0.0065, 0.0042, 0.06470000000000001],
    [1, 4, 0.0006, 0.0285, 0.0209, 0.059300000000000005],
    [1, 4, -0.005, 0.0303, -0.0459, 0.059300000000000005],
    [1, 4, -0.0105, 0.0322, -0.1126, 0.059300000000000005],
    [1, 5, -0.0044, -0.0045, 0.005, 0.0663],
    [1, 5, 0.0004, 0.009, -0.092, 0.0663],
    [1, 6, 0.0121, -0.0052, 0.021, 0.057300000000000004],
    [1, 6, 0.0042, 0.0119, -0.0373, 0.057300000000000004],
    [1, 6, -0.0038, 0.029, -0.0955, 0.057300000000000004],
    [1, 7, -0.0078, -0.0033, 0.0136, 0.0536],
    [1, 7, -0.0302, -0.0035, -0.0328, 0.0536],
    [1, 7, -0.0526, -0.0037, -0.0792, 0.0536],
    [1, 8, 0.017, 0.0015, 0.1963, 0.0729],
    [1, 8, -0.0014, 0.0022, 0.0998, 0.0729],
    [1, 8, -0.0199, 0.003, 0.0033, 0.0729],
    [1, 8, 0.0762, -0.1283, 0.0757, 0.0369],
    [1, 8, 0.0398, -0.0769, 0.0514, 0.0369],
    [1, 8, 0.0033, -0.0255, 0.0271, 0.0369],
])
N_SPHERES = len(SPHERES)
ARM_OF = SPHERES[:, 0].astype(int)
LINK_OF = SPHERES[:, 1].astype(int)
RADIUS = SPHERES[:, 5]
DISTAL = np.flatnonzero(LINK_OF >= 4)  # the spheres on links 4 to 8: right 17, then left 17
ARM_PAIRS = np.array([(i, j) for i in np.flatnonzero((ARM_OF == 0) & (LINK_OF > 1)) for j in np.flatnonzero((ARM_OF == 1) & (LINK_OF > 1))])
HAND_POINT = SPHERES[[21, 47], 2:5]  # the middle sphere of each palm's cover, in the hand's frame


def points(q):
    """Hand points (2, 3) and sphere centres (52, 3) at q (14,)."""
    R, p = frames(q)
    arm, link = ARM_OF, LINK_OF - 1
    C = p[arm, link] + jnp.einsum("sij,sj->si", R[arm, link], jnp.asarray(SPHERES[:, 2:5], q.dtype))
    hand = p[:, NJ] + jnp.einsum("aij,aj->ai", R[:, NJ], jnp.asarray(HAND_POINT, q.dtype))
    return hand, C


def ik(targets, q_init, lo, hi, iterations=300, damping=1e-3):
    """Hand-point inverse kinematics of both arms by damped least squares over a batch, targets (B, 2, 3) and q_init (B, 14):
    q <- clip(q + J^T (J J^T + damping I)^-1 (target - p), lo, hi). Returns q (B, 14) and the distance left per arm (B, 2)."""
    hand = lambda q: points(q)[0].ravel()  # noqa: E731

    def one(target, q):
        def step(q, _):
            J = jax.jacfwd(hand)(q)
            dq = J.T @ jnp.linalg.solve(J @ J.T + damping * jnp.eye(6, dtype=q.dtype), target.ravel() - hand(q))
            return jnp.clip(q + dq, lo, hi), None

        q, _ = jax.lax.scan(step, q, None, length=iterations)
        return q, jnp.linalg.norm(points(q)[0] - target, axis=-1)

    return jax.vmap(one)(targets, q_init)


def interval_map(h=H_STEP):
    """f_h(x, u): the exact double integrator over h, q+ = q + h qd + h^2 u / 2, qd+ = qd + h u."""
    def f(x, u):
        q, qd = x[:NQ], x[NQ:]
        return jnp.concatenate([q + h * qd + 0.5 * h * h * u, qd + h * u])
    return f


def interval_matrices(h=H_STEP):
    """(A, B) with f_h(x, u) = A x + B u."""
    I = np.eye(NQ)
    A = np.block([[I, h * I], [0 * I, I]])
    B = np.concatenate([0.5 * h * h * I, h * I])
    return A, B


def rollout(x0, U, h=H_STEP):
    """States (N + 1, 28) from x0 (28,) under accelerations U (N, 14)."""
    f = interval_map(h)

    def step(x, u):
        y = f(x, u)
        return y, y

    _, X = jax.lax.scan(step, x0, U)
    return jnp.concatenate([x0[None], X])


# ------------------------------------------------------------------
# predicates

EPS_LENGTH = 5e-4  # smoothing length of the norms (m)
R_GOAL = 0.05  # a hand is at a waypoint within this distance (m)


def _norm_minus(v, eps):
    """sqrt(|v|^2 + eps^2) - eps, at most |v|; the exact norm for eps = 0."""
    s = jnp.sum(v * v, -1)
    return jnp.sqrt(s + eps * eps) - eps if eps > 0 else jnp.sqrt(s)


def _norm_plus(v, eps):
    """sqrt(|v|^2 + eps^2), at least |v|."""
    return jnp.sqrt(jnp.sum(v * v, -1) + eps * eps)


def outside(x, cells, radius, eps=EPS_LENGTH):
    """(n_cells * 34,) per cell and distal sphere i: (|c_i - z| - r_i - radius) / radius, positive when the sphere is
    outside the cell (the ball of the given radius about z)."""
    _, C = points(x[:NQ])
    r = jnp.asarray(RADIUS[DISTAL], x.dtype)
    d = _norm_minus(C[DISTAL][None] - cells[:, None], eps)
    return ((d - r[None] - radius) / radius).ravel()


def at_waypoint(x, waypoints, arms, eps=EPS_LENGTH):
    """(n,) per waypoint g of arm a: (R_GOAL - |p_a - g|) / R_GOAL, positive when the hand of arm a is within R_GOAL of g."""
    p, _ = points(x[:NQ])
    return (R_GOAL - _norm_plus(p[np.asarray(arms)] - jnp.asarray(waypoints, x.dtype), eps)) / R_GOAL


def predicate_joints(arms, n_cells):
    """(n_wp + 34 n_cells, 2) per predicate, the number of leading joints of each arm whose positions it reads: a waypoint
    predicate reads its arm's 7 joints, an outside predicate the joints 1 to min(link, 7) of its sphere's arm."""
    arms = np.asarray(arms)
    J = np.zeros((arms.size + n_cells * DISTAL.size, 2), np.int64)
    J[np.arange(arms.size), arms] = NJ
    s = np.tile(DISTAL, n_cells)
    J[arms.size + np.arange(s.size), ARM_OF[s]] = np.minimum(LINK_OF[s], NJ)
    return J
