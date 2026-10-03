"""Planar two-link arm predicate scores as a Warp kernel.

Computes reach, clearance and elbow exactly as twolink.py does, for X of shape
(B, T, 2) into scores of shape (B, T, 3). The kernel is loop-free, so Warp's
generated adjoint is used and the launch is recorded on a tape as usual.
"""
from typing import Any

import warp as wp

from .twolink import L1, L2, OBSTACLE, OBSTACLE_R, Q_MAX, TARGET, TARGET_R

TX, TY = TARGET
OX, OY = OBSTACLE
# squared in Python double, as twolink.py does; a product inside the kernel would be float32
RT2 = TARGET_R ** 2
RO2 = OBSTACLE_R ** 2


@wp.kernel
def twolink_scores(X: wp.array3d(dtype=Any), scores: wp.array3d(dtype=Any)):
    b, t = wp.tid()
    q1 = X[b, t, 0]
    q2 = X[b, t, 1]
    l1 = type(q1)(L1)
    l2 = type(q1)(L2)
    px = l1 * wp.cos(q1) + l2 * wp.cos(q1 + q2)
    py = l1 * wp.sin(q1) + l2 * wp.sin(q1 + q2)
    dx = px - type(q1)(TX)
    dy = py - type(q1)(TY)
    scores[b, t, 0] = type(q1)(RT2) - (dx * dx + dy * dy)
    ex = px - type(q1)(OX)
    ey = py - type(q1)(OY)
    scores[b, t, 1] = ex * ex + ey * ey - type(q1)(RO2)
    scores[b, t, 2] = type(q1)(Q_MAX) - q2
