"""Planar unicycle with box regions: the constants of the system, the regions, the STL specification, and the two fixed
trajectories S1 and S2 built from smooth curves."""
import numpy as np

from ..stl import Always, And, Atom, Eventually, Not, Until

H = 0.1  # sampling period, s
V_MAX = 1.0  # m/s
W_MAX = np.pi / 2  # rad/s
Z0 = (1.0, 1.0, np.pi / 2)  # start: (1, 1), facing +y

# boxes (x1, x2, y1, y2), m
RED = (4.0, 6.0, 1.5, 8.5)
GREEN = (1.5, 3.5, 7.0, 9.0)
BLUE = (7.5, 9.5, 1.5, 3.5)
OBSTACLE = (6.4, 7.4, 3.9, 5.45)
BOUNDARY = (0.0, 10.0, 0.0, 10.0)
REGIONS = (RED, GREEN, BLUE, OBSTACLE, BOUNDARY)
NAMES = ("Red", "Green", "Blue", "Obstacle", "Boundary")



# ------------------------------------------------------------------
# specification

def box(i):
    """Region i of REGIONS as the conjunction of its predicates 4 i + j: x - x1, x2 - x, y - y1, y2 - y."""
    return And(*(Atom(4 * i + j) for j in range(4)))


def specification(a1, b1, a2, b2, T):
    """(specification, its until): (not Red U_[a1, b1] Green) and F_[a2, b2] G_[0, 2/H] Blue and G not Obstacle and
    G Boundary; intervals in samples, T samples."""
    hold = int(round(2 / H))
    until = Until((a1, b1), Not(box(0)), box(1))
    spec = And(until, Eventually((a2, b2), Always((0, hold), box(2))), Always((0, T - 1), Not(box(3))),
               Always((0, T - 1), box(4)))
    return spec, until


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


# ------------------------------------------------------------------
# the two trajectories

R_CLIP = 1.0  # radius of S1's arc at Red's edge, m
APEX_Y = 4.6
GREEN_STOP = (2.8, 8.0, np.pi / 3)  # position and heading of the stop in Green
BLUE_STOP = (8.5, 2.5, -np.pi / 2)  # position and heading of the stop in Blue
V_CRUISE = 0.8  # m/s, plateau speed of the long drives


def trajectories(wait, depth=0.06, level=0.04, down=0.12):
    """(u1, u2, timing): the inputs of S1 and S2 from Z0 and the specification's windows, for a wait of `wait` samples.
    S1 enters Red by depth (m) on an arc of radius R_CLIP and waits level (m) outside; S2 drives one curve to Green."""
    xa = 4.0 + depth
    c = (xa - R_CLIP, APEX_Y)
    phi_e = -np.arccos(1 - (depth + 0.15) / R_CLIP)  # the arc starts 0.15 m outside the edge
    phi_w = np.arccos(1 - (depth + level) / R_CLIP)
    entry = np.array(c) + R_CLIP * np.array([np.cos(phi_e), np.sin(phi_e)])
    wait_pt = np.array(c) + R_CLIP * np.array([np.cos(phi_w), np.sin(phi_w)])
    k = 1 / R_CLIP
    a = np.concatenate([hermite(Z0[:2], Z0[2], 0.0, entry, phi_e + np.pi / 2, k), arc(c, R_CLIP, phi_e, phi_w)[1:]])
    g = hermite(wait_pt, phi_w + np.pi / 2, k, GREEN_STOP[:2], GREEN_STOP[2], 0.0)
    pa = leg(a, V_CRUISE, up=0.2, down=down)
    pg = leg(g, V_CRUISE)
    p1 = np.concatenate([pa, np.repeat(pa[-1:], wait - 1, 0), pg])
    t_green = len(p1) - 1
    a1 = t_green + 5
    b1 = a1 + int(round(2 / H))
    leave = b1 + 5
    pblue = leg(sweep(), V_CRUISE)
    t_blue = leave + len(pblue) - 1
    a2, b2 = t_blue - 30, t_blue + 30
    T = b2 + int(round(2 / H)) + 1
    p1 = np.concatenate([p1, np.repeat(p1[-1:], leave - t_green, 0), pblue[1:]])
    p1 = np.concatenate([p1, np.repeat(p1[-1:], T - len(p1), 0)])
    p2 = leg(hermite(Z0[:2], Z0[2], 0.0, GREEN_STOP[:2], GREEN_STOP[2], 0.0), V_CRUISE)
    p2 = np.concatenate([p2, np.repeat(p2[-1:], leave - (len(p2) - 1), 0), pblue[1:]])
    p2 = np.concatenate([p2, np.repeat(p2[-1:], T - len(p2), 0)])
    timing = {"wait": wait, "depth": depth, "level": level, "t_green": t_green, "a1": a1, "b1": b1,
              "leave": leave, "t_blue": t_blue, "a2": a2, "b2": b2, "T": T}
    return inputs_from_positions(p1, Z0[2]), inputs_from_positions(p2, Z0[2]), timing


def sweep():
    """Dense points of the drive from the stop in Green to the stop in Blue."""
    return hermite(GREEN_STOP[:2], GREEN_STOP[2], 0.0, BLUE_STOP[:2], BLUE_STOP[2], 0.0, scale=1.2)


# ------------------------------------------------------------------
# names of the matched smoothings

METHODS = ("exact", "lse_plain", "gm01", "gm10", "sparsemax")
