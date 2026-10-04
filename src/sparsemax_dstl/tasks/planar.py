"""Planar unicycle: the sampling period, the input bounds, the rollout, and smooth curves sampled into input sequences."""
import numpy as np

H = 0.1  # sampling period, s
V_MAX = 1.0  # m/s
W_MAX = np.pi / 2  # rad/s


# ------------------------------------------------------------------
# dynamics

def rollout(z0, u):
    """States (T + 1, 3) from z0 under inputs u (T, 2), unicycle with period H."""
    th = np.cumsum(np.concatenate([[z0[2]], H * u[:, 1]]))
    x = np.cumsum(np.concatenate([[z0[0]], H * u[:, 0] * np.cos(th[:-1])]))
    y = np.cumsum(np.concatenate([[z0[1]], H * u[:, 0] * np.sin(th[:-1])]))
    return np.stack([x, y, th], 1)


# ------------------------------------------------------------------
# smooth paths: pieces with matched position, heading and curvature at every junction

def hermite(p0, th0, k0, p1, th1, k1, scale=1.0, n=4000):
    """Quintic Hermite curve from (p0, heading th0, curvature k0) to (p1, th1, k1), dense points (n, 2). The end
    derivatives are L T and L^2 k N (T the unit tangent, N the left normal, L = scale times the chord)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = scale * np.hypot(*(p1 - p0))
    t0, t1 = np.array([np.cos(th0), np.sin(th0)]), np.array([np.cos(th1), np.sin(th1)])
    n0, n1 = np.array([-t0[1], t0[0]]), np.array([-t1[1], t1[0]])
    u = np.linspace(0, 1, n)[:, None]
    h = [1 - 10 * u**3 + 15 * u**4 - 6 * u**5, u - 6 * u**3 + 8 * u**4 - 3 * u**5, 0.5 * u**2 - 1.5 * u**3 + 1.5 * u**4 - 0.5 * u**5,
         0.5 * u**3 - u**4 + 0.5 * u**5, -4 * u**3 + 7 * u**4 - 3 * u**5, 10 * u**3 - 15 * u**4 + 6 * u**5]
    return h[0] * p0 + h[1] * L * t0 + h[2] * L**2 * k0 * n0 + h[3] * L**2 * k1 * n1 + h[4] * L * t1 + h[5] * p1


def arc(center, radius, phi0, phi1, n=4000):
    """Circular arc, counterclockwise for phi1 > phi0; dense points (n, 2)."""
    phi = np.linspace(phi0, phi1, n)
    return np.asarray(center, float) + radius * np.stack([np.cos(phi), np.sin(phi)], 1)


def leg(points, v_peak, up=0.3, down=0.3):
    """Positions at the samples along the dense polyline points, from rest to rest: a speed plateau at about v_peak
    (m/s) with smoothstep ramps over the fractions up and down of the leg's duration."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    s_dense = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(np.ceil(s_dense[-1] / (v_peak * H * (1 - (up + down) / 2))))
    tau = (np.arange(n) + 0.5) / n
    r = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))  # noqa: E731
    shape = r(tau / up) * r((1 - tau) / down)
    s = np.concatenate([[0.0], np.cumsum(shape)]) * s_dense[-1] / shape.sum()
    return np.stack([np.interp(s, s_dense, points[:, 0]), np.interp(s, s_dense, points[:, 1])], 1)


def inputs_from_positions(p, th0):
    """Inputs (len(p) - 1, 2) that move the unicycle through the sample positions p from heading th0:
    the heading of each moving step is the direction of its chord, held while the robot stands."""
    d = np.diff(p, axis=0)
    v = np.hypot(d[:, 0], d[:, 1]) / H
    moving = v > 1e-12
    ang = np.arctan2(d[:, 1], d[:, 0])
    idx = np.maximum.accumulate(np.where(moving, np.arange(len(d)), -1))
    th = np.where(idx >= 0, ang[np.maximum(idx, 0)], th0)
    th = np.concatenate([[th0], th])
    dth = (np.diff(th) + np.pi) % (2 * np.pi) - np.pi
    # the turn input of step t sets the heading of step t + 1 to that step's chord direction; step 0 moves along th0
    w = np.concatenate([[dth[0] + dth[1]], dth[2:], [0.0]]) / H
    return np.stack([v, w], 1)
