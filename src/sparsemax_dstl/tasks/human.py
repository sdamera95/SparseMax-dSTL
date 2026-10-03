"""Scripted human capsules for the shared-workspace scenario (E019), in NumPy.

The human is K_h = 6 capsules, each a segment (proximal, distal) with a radius:

    torso, head, upper_arm_r, forearm_r, upper_arm_l, forearm_l

in the robot base frame (z up, metres). The robot base sits on the table top at z = 0; the
human stands on the floor at z = FLOOR across the table and faces the shared zone.

Motion (Script, every field an argument). The human's standing point is the zone centre plus
`stand` metres along the approach direction (cos phi, sin phi). It walks in from `walk`
metres further out, arriving at t_arrive. Its right hand then follows

    rest -> reach target -> hover point,

moving between t_reach and t_reach + t_move, holding at the target until t_release, and
moving to the hover point until t_release + t_move, where it stays. The reach target is the
zone centre moved `depth` metres toward the robot, at height `reach_z`; the hover point lies
`hover` metres outside the zone's edge toward the human, at height `hover_z`. So `depth`
sets the closest approach during the reach, t_release - t_reach how long the hand stays in
the zone, and `hover` how close the hand stays to the zone afterwards. Blends are
C1 cosine steps. The elbow follows two-link inverse kinematics with the pole pointing down;
a target beyond the arm's reach is clipped to 0.98 of it. The left arm hangs.

Pass and wait (E022; off unless `anchor` is set, so the defaults give the motion above).
`anchor` is a point of the arm's path (robot base frame), and n the unit vector from it to
the right shoulder of the standing human. With an anchor, the hand returns to rest at the
release instead of going to the hover point, and then:
- the pass: over [t_pass - pass_width/2, t_pass + pass_width/2] the hand moves on the straight
  line from rest to the pass point anchor + pass_gap n and back, with the C1 bump
  1/2 - 1/2 cos(2 pi (t - t_pass + pass_width/2) / pass_width), which is 1 at t_pass; so the
  hand is pass_gap from the anchor at t_pass, the closest approach of the pass when the
  angle at the pass point between the anchor and rest is at least 90 degrees
  (tests/test_regime.py checks the closest approach at one geometry, and the anchor pair's
  margins on regime instances);
- the wait: from t_wait = t_pass + pass_width/2 the hand moves to the standoff point
  anchor + standoff n (C1 step over t_move), holds there for `wait` seconds, and returns to
  rest over t_move. So the hand is `standoff` from the anchor over
  [t_wait + t_move, t_wait + t_move + wait].
The phases do not overlap when t_release + t_move <= t_pass - pass_width/2; the motion is
the sum of the phase displacements, which equals the sequential motion then.

Trapezoid pass (E024; off unless `pre_gap` is set, so the defaults give the bump above). With
pre_gap, the pass holds the hand at the pass point for pass_hold seconds and replaces the bump
and the separate move to the standoff:
- rest -> pre point anchor + pre_gap n, C1 step over [t_a - t_move, t_a], t_a = t_pass -
  pass_hold/2 - pass_ramp;
- pre point -> pass point over [t_a, t_a + pass_ramp];
- hold at the pass point over [t_pass - pass_hold/2, t_pass + pass_hold/2];
- pass point -> standoff point over [t_pass + pass_hold/2, t_w], t_w = t_pass + pass_hold/2 +
  pass_ramp;
- hold at the standoff for `wait` seconds from t_w, then back to rest over t_move.
All points lie on the line from the anchor to the right shoulder, so with pre_gap > standoff >
pass_gap the two ramps move the hand straight toward and away from the anchor.

Pinned plateau (E024, root's design revision 2 of 2026-09-30 08:55Z; off unless `standoff_from`
is set, which takes precedence over pre_gap): the hand holds at the standoff for `wait` seconds
from standoff_from (the handover dwell of the robot), and the trapezoid pass dips from the
standoff to the pass point and back inside that hold:
- rest -> standoff point, C1 step over [standoff_from - t_move, standoff_from];
- standoff -> pass point over [t_a, t_a + pass_ramp], hold at the pass point over
  [t_pass - pass_hold/2, t_pass + pass_hold/2], pass point -> standoff over the next pass_ramp;
- at the standoff until standoff_from + wait, then back to rest over t_move. With the
sampling interval h, a hold of pass_hold = (k - 1/2) h centred half a sample off the grid holds
exactly k samples, and a ramp of pass_ramp = 5 h passes through at most 5 samples. Points beyond
the arm's reach are clipped as above, so pass and standoff points must lie within
0.98 (UPPER + FORE) of the shoulder for the stated distances to hold.

capsules(script, times) returns endpoints (T, 6, 2, 3) and radii (6,), the layout of the
trajectory file (tasks.human_file). Script fields may be arrays of a common shape B; the
endpoints then have shape (B..., T, 6, 2, 3), for many scripts at once. spheres(endpoints, radii, spacing) covers each capsule
by a chain of spheres: n = ceil(length / spacing) + 1 centres evenly spaced on the segment,
radius sqrt(r^2 + (s/2)^2) with s the actual spacing, so every point within r of the
segment lies in some sphere. The number per capsule is fixed from the first sample's
lengths; segments are rigid in the script, so it holds for every sample.
"""
from dataclasses import dataclass

import numpy as np

NAMES = ("torso", "head", "upper_arm_r", "forearm_r", "upper_arm_l", "forearm_l")
RADII = np.array([0.15, 0.10, 0.05, 0.045, 0.05, 0.045])
FLOOR = -0.9
HIP, SHOULDER, NECK, HEAD_TOP = 0.95, 1.45, 1.55, 1.78  # heights above the floor, m
SHOULDER_HALF = 0.2
UPPER, FORE = 0.32, 0.45  # upper arm; forearm including the hand, m


@dataclass(frozen=True)
class Script:
    zone: tuple = (0.5, 0.0)
    phi: float = 0.0
    stand: float = 0.45
    walk: float = 1.0
    t_arrive: float = 1.0
    t_reach: float = 2.0
    t_move: float = 0.6
    t_release: float = 3.4
    depth: float = 0.0
    reach_z: float = 0.10
    hover: float = 0.05
    hover_z: float = 0.15
    zone_radius: float = 0.2
    anchor: object = None  # E022: a point of the arm's path, (3,) or (B..., 3); None keeps the motion above
    t_pass: float = 6.0
    pass_width: float = 0.6
    pass_gap: float = 0.3
    standoff: float = 0.35
    wait: float = 4.0
    pre_gap: object = None  # E024: the trapezoid pass (see the module docstring); None keeps the bump
    pass_hold: float = 0.19
    pass_ramp: float = 0.1
    standoff_from: object = None  # E024: the pinned plateau (see the module docstring); None keeps the pass above


def _step(t, t0, t1):
    """C1 cosine step from 0 at t0 to 1 at t1."""
    s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * s)


def _elbow(shoulder, hand, pole):
    """Two-link inverse kinematics: the elbow (T, 3) and the reachable hand (T, 3)."""
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
    """(t_a, hold start, hold end, t_w, wait end) of the trapezoid pass, as arrays (B...)."""
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
    """(standoff start, t_a, hold start, hold end, t_w, standoff end) of the pinned plateau, arrays (B...)."""
    f = lambda v: np.asarray(v, np.float64)
    h0 = f(script.t_pass) - f(script.pass_hold) / 2
    h1 = f(script.t_pass) + f(script.pass_hold) / 2
    s0 = f(script.standoff_from)
    return s0, h0 - f(script.pass_ramp), h0, h1, h1 + f(script.pass_ramp), s0 + f(script.wait)


def _pinned(script, t, f, rest):
    """Displacement of the right hand from rest by the pinned plateau with the pass inside it."""
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
    length = np.linalg.norm(endpoints.reshape((-1,) + endpoints.shape[-3:])[0, :, 1] - endpoints.reshape((-1,) + endpoints.shape[-3:])[0, :, 0], axis=-1)
    count = np.ceil(length / spacing).astype(int) + 1
    owner = np.repeat(np.arange(len(count)), count)
    frac = np.concatenate([np.linspace(0.0, 1.0, c) for c in count])  # over capsules (model structure)
    gap = length[owner] / (count[owner] - 1)
    centres = endpoints[..., owner, 0, :] + frac[:, None] * (endpoints[..., owner, 1, :] - endpoints[..., owner, 0, :])
    return centres, np.sqrt(np.asarray(radii)[owner] ** 2 + (gap / 2) ** 2), owner
