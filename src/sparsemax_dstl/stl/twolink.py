"""Planar two-link arm predicates with closed-form derivatives, for tests.

The state is the joint-angle pair q = (q1, q2). The end effector is at

    p(q) = (l1 cos q1 + l2 cos(q1 + q2), l1 sin q1 + l2 sin(q1 + q2)).

Scores, positive when satisfied:

    reach      r_c^2 - |p - c|^2      (end effector inside the target disk)
    clearance  |p - o|^2 - r_o^2      (end effector outside the obstacle disk)
    elbow      q_max - q2             (linear joint bound, depends on q2 only)

The NumPy functions give their gradients and Hessians in closed form, as the
analytic reference for automatic differentiation.
"""
import jax.numpy as jnp
import numpy as np

from .oracle import analytic_derivatives
from .predicates import Predicate

L1, L2 = 1.0, 0.8
TARGET, TARGET_R = (1.2, 0.8), 0.5
OBSTACLE, OBSTACLE_R = (0.3, 1.2), 0.4
Q_MAX = 2.0


def position(q):
    c1, s1 = jnp.cos(q[0]), jnp.sin(q[0])
    c12, s12 = jnp.cos(q[0] + q[1]), jnp.sin(q[0] + q[1])
    return jnp.stack([L1 * c1 + L2 * c12, L1 * s1 + L2 * s12])


def reach(q):
    d = position(q) - jnp.asarray(TARGET, q.dtype)
    return TARGET_R ** 2 - jnp.sum(d * d)


def clearance(q):
    d = position(q) - jnp.asarray(OBSTACLE, q.dtype)
    return jnp.sum(d * d) - OBSTACLE_R ** 2


def elbow(q):
    return Q_MAX - q[1]


PREDICATES = (
    Predicate(reach, (0, 1), "reach"),
    Predicate(clearance, (0, 1), "clearance"),
    Predicate(elbow, (1,), "elbow"),
)


def _kinematics(Q):
    """p, Jacobian (N, 2, 2) and second derivatives (N, 2, 2, 2) of p, indexed [n, i, j, k]."""
    q1, q12 = Q[:, 0], Q[:, 0] + Q[:, 1]
    c1, s1, c12, s12 = np.cos(q1), np.sin(q1), np.cos(q12), np.sin(q12)
    p = np.stack([L1 * c1 + L2 * c12, L1 * s1 + L2 * s12], -1)
    J = np.stack([np.stack([-L1 * s1 - L2 * s12, -L2 * s12], -1),
                  np.stack([L1 * c1 + L2 * c12, L2 * c12], -1)], -2)
    Hx = np.stack([np.stack([-L1 * c1 - L2 * c12, -L2 * c12], -1),
                   np.stack([-L2 * c12, -L2 * c12], -1)], -2)
    Hy = np.stack([np.stack([-L1 * s1 - L2 * s12, -L2 * s12], -1),
                   np.stack([-L2 * s12, -L2 * s12], -1)], -2)
    return p, J, np.stack([Hx, Hy], 1)


def analytic(Q):
    """Scores (N, 3), gradients (N, 3, 2) and Hessians (N, 3, 2, 2) at states Q (N, 2)."""
    p, J, H = _kinematics(np.asarray(Q, dtype=np.float64))
    out = []
    for center, r, sign in ((TARGET, TARGET_R, -1.0), (OBSTACLE, OBSTACLE_R, 1.0)):
        d = p - np.asarray(center)
        g = sign * (np.sum(d * d, -1) - r ** 2)
        grad = sign * 2 * np.einsum("ni,nij->nj", d, J)
        hess = sign * 2 * (np.einsum("nij,nik->njk", J, J) + np.einsum("ni,nijk->njk", d, H))
        out.append((g, grad, hess))
    n = len(Q)
    out.append((Q_MAX - np.asarray(Q)[:, 1], np.broadcast_to([0.0, -1.0], (n, 2)), np.zeros((n, 2, 2))))
    return tuple(np.stack(parts, 1) for parts in zip(*out))


def analytic_chain(program, X, semantics="exact", beta=None):
    """Gradient (2T,) and Hessian (2T, 2T) of the root at t = 0 with respect to X (T, 2).

    Analytic reference: d2 rho/dX2 = Jz^T Hz Jz + sum_i (d rho/dz_i) d2 z_i/dX2,
    with Jz block diagonal in time, from the closed-form predicate derivatives
    and oracle.analytic_derivatives on the scores.
    """
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[0]
    g, dg, d2g = analytic(X)
    _, gz, hz = analytic_derivatives(program, g, semantics, beta)
    Jz = np.zeros((T, 3, T, 2))
    Jz[np.arange(T), :, np.arange(T), :] = dg
    Jz = Jz.reshape(3 * T, 2 * T)
    curv = np.zeros((T, 2, T, 2))
    curv[np.arange(T), :, np.arange(T), :] = np.einsum("ti,tijk->tjk", gz[0].reshape(T, 3), d2g)
    return gz[0] @ Jz, Jz.T @ hz[0] @ Jz + curv.reshape(2 * T, 2 * T)
