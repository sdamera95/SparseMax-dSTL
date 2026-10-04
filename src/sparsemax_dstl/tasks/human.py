"""Scripted motion of a person as six capsules (torso, head, upper arms, forearms) in the robot base frame (z up,
metres), and the chains of spheres that cover the capsules."""
from dataclasses import dataclass

import numpy as np

NAMES = ("torso", "head", "upper_arm_r", "forearm_r", "upper_arm_l", "forearm_l")
RADII = np.array([0.15, 0.10, 0.05, 0.045, 0.05, 0.045])  # capsule radii in the order of NAMES, m
FLOOR = -0.9  # z of the floor in the robot base frame, m
HIP, SHOULDER, NECK, HEAD_TOP = 0.95, 1.45, 1.55, 1.78  # heights above the floor, m
SHOULDER_HALF = 0.2  # half the shoulder width, m
UPPER, FORE = 0.32, 0.45  # upper arm; forearm including the hand, m


@dataclass(frozen=True)
class Script:
    """Parameters of the motion: the person walks in to the standing point, and the right hand moves from rest to the
    reach target at t_reach and on to the hover point at t_release. Seconds, metres, radians."""
    zone: tuple = (0.5, 0.0)  # (x, y) of the zone centre
    phi: float = 0.0  # direction from the zone centre to the person
    stand: float = 0.45  # distance of the standing point from the zone centre
    walk: float = 1.0  # the person starts this much further out and arrives at t_arrive
    t_arrive: float = 1.0
    t_reach: float = 2.0
    t_move: float = 0.6  # duration of each move of the hand
    t_release: float = 3.4
    depth: float = 0.0  # the reach target lies this far beyond the zone centre, away from the person
    reach_z: float = 0.10
    hover: float = 0.05  # the hover point lies this far outside the zone's radius, on the person's side
    hover_z: float = 0.15
    zone_radius: float = 0.2
    anchor: object = None  # (B..., 3); when set, the hand returns to rest at t_release, then passes and waits near it
    t_pass: float = 6.0  # centre of the pass
    pass_width: float = 0.6  # duration of the cosine pass (pre_gap unset)
    pass_gap: float = 0.3  # distance of the pass point from the anchor
    standoff: float = 0.35  # distance of the standoff point, where the hand waits, from the anchor
    wait: float = 4.0  # duration of the hold at the standoff point
    pre_gap: object = None  # when set, the pass is a trapezoid that ramps in from this distance from the anchor
    pass_hold: float = 0.19  # duration of the trapezoid's hold at the pass point
    pass_ramp: float = 0.1  # duration of each ramp of the trapezoid
    standoff_from: object = None  # when set, the start of the hold at the standoff; the pass leaves and returns to it


def _step(t, t0, t1):
    """C1 cosine step from 0 at t0 to 1 at t1."""
    s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * s)


def _elbow(shoulder, hand, pole):
    """Two-link inverse kinematics: the elbow and the hand (..., 3), the hand pulled to 0.98 (UPPER + FORE) from the
    shoulder when it lies beyond that."""
    d = hand - shoulder
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    n = d / dist
    dist = np.minimum(dist, 0.98 * (UPPER + FORE))
    hand = shoulder + dist * n
    a = (UPPER ** 2 - FORE ** 2 + dist ** 2) / (2 * dist)
    h = np.sqrt(np.maximum(UPPER ** 2 - a ** 2, 0.0))
    perp = pole - np.sum(pole * n, -1, keepdims=True) * n
    perp /= np.linalg.norm(perp, axis=-1, keepdims=True)
    return shoulder + a * n + h * perp, hand


def capsules(script, times):
    """Endpoints (B..., T, 6, 2, 3) and radii (6,) of the human at the given times; B is the
    common shape of the script's fields (empty for scalars)."""
    f = lambda v: np.asarray(v, np.float64)[..., None, None]  # (B..., 1, 1): broadcasts over (T, 3)
    t = np.asarray(times, np.float64)[:, None]
    phi = f(script.phi)
    e = lambda k: np.eye(3)[k]
    u = np.cos(phi) * e(0) + np.sin(phi) * e(1)
    w = -np.sin(phi) * e(0) + np.cos(phi) * e(1)  # the human's left, facing -u
    zone = script.zone[0] * e(0) + script.zone[1] * e(1)
    stand = zone + f(script.stand) * u
    base = stand + (1 - _step(t, 0.0, f(script.t_arrive))) * f(script.walk) * u  # (B..., T, 3), z = 0
    z = lambda height: (FLOOR + height) * e(2)
    shoulder_r, shoulder_l = base - SHOULDER_HALF * w + z(SHOULDER), base + SHOULDER_HALF * w + z(SHOULDER)
    rest = shoulder_r - 0.7 * e(2) - 0.1 * u
    target = zone - f(script.depth) * u + f(script.reach_z) * e(2)
    hover = zone + f(script.zone_radius + np.asarray(script.hover)) * u + f(script.hover_z) * e(2)
    a = _step(t, f(script.t_reach), f(script.t_reach) + f(script.t_move))
    b = _step(t, f(script.t_release), f(script.t_release) + f(script.t_move))
    if script.anchor is None:
        hand = (1 - a) * rest + a * ((1 - b) * target + b * hover)
    else:
        hand = (1 - a) * rest + a * ((1 - b) * target + b * rest) + _pass_and_wait(script, t, f, rest)
    down = np.broadcast_to([0.0, 0.0, -1.0], hand.shape)
    elbow_r, hand = _elbow(shoulder_r, hand, down)
    elbow_l = shoulder_l - UPPER * e(2)
    hand_l = elbow_l - FORE * e(2)
    seg = [(base + z(HIP), base + z(SHOULDER)), (base + z(NECK), base + z(HEAD_TOP)), (shoulder_r, elbow_r),
           (elbow_r, hand), (shoulder_l, elbow_l), (elbow_l, hand_l)]
    shape = np.broadcast_shapes(*[np.shape(a) for pair in seg for a in pair])
    ends = np.stack([np.stack([np.broadcast_to(p, shape), np.broadcast_to(d, shape)], -2) for p, d in seg], -3)
    return ends, RADII.copy()  # ends (B..., T, 6, 2, 3)


def standing_shoulder(script):
    """The right shoulder of the standing human (after the walk), (B..., 3)."""
    phi = np.asarray(script.phi, np.float64)[..., None]
    u = np.concatenate([np.cos(phi), np.sin(phi), np.zeros_like(phi)], -1)
    w = np.concatenate([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], -1)
    zone = np.array([script.zone[0], script.zone[1], 0.0])
    return zone + np.asarray(script.stand, np.float64)[..., None] * u - SHOULDER_HALF * w + (FLOOR + SHOULDER) * np.eye(3)[2]


def pass_points(script):
    """The unit vector n (B..., 3) from the anchor to the standing right shoulder, and the pass
    and standoff points anchor + pass_gap n and anchor + standoff n."""
    anchor = np.asarray(script.anchor, np.float64)
    d = standing_shoulder(script) - anchor
    n = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return n, anchor + np.asarray(script.pass_gap, np.float64)[..., None] * n, anchor + np.asarray(script.standoff, np.float64)[..., None] * n


def _bump(t, t0, width):
    """C1 bump, 0 outside [t0 - width/2, t0 + width/2] and 1 at t0."""
    s = np.clip((t - t0) / width + 0.5, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(2 * np.pi * s)


def _pass_and_wait(script, t, f, rest):
    """Displacement of the right hand from rest by the pass and the wait, (B..., T, 3)."""
    if script.standoff_from is not None:
        return _pinned(script, t, f, rest)
    if script.pre_gap is not None:
        return _trapezoid(script, t, f, rest)
    _, p, q = pass_points(script)
    p, q = p[..., None, :], q[..., None, :]  # (B..., 1, 3)
    width = f(script.pass_width)
    t_wait = f(script.t_pass) + width / 2
    up = _step(t, t_wait, t_wait + f(script.t_move))
    down = _step(t, t_wait + f(script.t_move) + f(script.wait), t_wait + 2 * f(script.t_move) + f(script.wait))
    return _bump(t, f(script.t_pass), width) * (p - rest) + (up - down) * (q - rest)


def trapezoid_times(script):
    """(ramp-in start, hold start, hold end, ramp-out end, wait end) of the trapezoid pass, as arrays (B...)."""
    f = lambda v: np.asarray(v, np.float64)
    h0 = f(script.t_pass) - f(script.pass_hold) / 2
    h1 = f(script.t_pass) + f(script.pass_hold) / 2
    return h0 - f(script.pass_ramp), h0, h1, h1 + f(script.pass_ramp), h1 + f(script.pass_ramp) + f(script.wait)


def _trapezoid(script, t, f, rest):
    """Displacement of the right hand from rest by the trapezoid pass and the wait, (B..., T, 3)."""
    n, p, q = pass_points(script)
    p0 = np.asarray(script.anchor, np.float64) + np.asarray(script.pre_gap, np.float64)[..., None] * n
    p, q, p0 = p[..., None, :], q[..., None, :], p0[..., None, :]  # (B..., 1, 3)
    ta, h0, h1, tw, te = (f(v) for v in trapezoid_times(script))
    tm = f(script.t_move)
    return (_step(t, ta - tm, ta) * (p0 - rest) + _step(t, ta, h0) * (p - p0) + _step(t, h1, tw) * (q - p)
            - _step(t, te, te + tm) * (q - rest))


def pinned_times(script):
    """(standoff start, ramp-in start, hold start, hold end, ramp-out end, standoff end) with standoff_from set,
    as arrays (B...)."""
    f = lambda v: np.asarray(v, np.float64)
    h0 = f(script.t_pass) - f(script.pass_hold) / 2
    h1 = f(script.t_pass) + f(script.pass_hold) / 2
    s0 = f(script.standoff_from)
    return s0, h0 - f(script.pass_ramp), h0, h1, h1 + f(script.pass_ramp), s0 + f(script.wait)


def _pinned(script, t, f, rest):
    """Displacement of the right hand from rest by the hold at the standoff with the trapezoid pass inside it,
    (B..., T, 3)."""
    _, p, q = pass_points(script)
    p, q = p[..., None, :], q[..., None, :]  # (B..., 1, 3)
    s0, ta, h0, h1, tw, se = (f(v) for v in pinned_times(script))
    tm = f(script.t_move)
    return (_step(t, s0 - tm, s0) * (q - rest) + _step(t, ta, h0) * (p - q) + _step(t, h1, tw) * (q - p)
            - _step(t, se, se + tm) * (q - rest))


def spheres(endpoints, radii, spacing):
    """Sphere chains covering the capsules: centres (..., T, S, 3), radii (S,), and the capsule
    index of every sphere (S,), from endpoints (..., T, K, 2, 3)."""
    endpoints = np.asarray(endpoints, np.float64)
    # the lengths at the first sample set the number of spheres per capsule
    length = np.linalg.norm(endpoints.reshape((-1,) + endpoints.shape[-3:])[0, :, 1] - endpoints.reshape((-1,) + endpoints.shape[-3:])[0, :, 0], axis=-1)
    count = np.ceil(length / spacing).astype(int) + 1
    owner = np.repeat(np.arange(len(count)), count)
    frac = np.concatenate([np.linspace(0.0, 1.0, c) for c in count])
    gap = length[owner] / (count[owner] - 1)
    centres = endpoints[..., owner, 0, :] + frac[:, None] * (endpoints[..., owner, 1, :] - endpoints[..., owner, 0, :])
    return centres, np.sqrt(np.asarray(radii)[owner] ** 2 + (gap / 2) ** 2), owner
