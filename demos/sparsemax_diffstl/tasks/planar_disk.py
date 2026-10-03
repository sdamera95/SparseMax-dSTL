"""E049 round 2: the planar unicycle example with disk regions (one predicate per region).

Predicates (P = 8), each positive when satisfied, from the position p = (x, y):

    0..3  inside Red, Green, Blue, Obstacle:  r_i - |p - c_i|   (outside is the negated atom, |p - c_i| - r_i)
    4..7  inside the workspace [0, 10]^2:     x, 10 - x, y, 10 - y

With one predicate per region every smoothing sees the exact signed distance to a region, so the
comparison between the smoothings is about the temporal operators (the until, eventually, always and
the top conjunction), which is what the theorems are about.

STL specification (intervals in samples):

    (not Red  U_[a1, b1]  Green)  and  F_[a2, b2] G_[0, 2/h] Blue  and  G_[0, T-1] not Obstacle  and  G_[0, T-1] Boundary.

The unicycle, its sampling period, the input box and the smoothings matched at one worst-case error per
node are those of planar.py (rollout, matched). matched below adds one pair to planar.matched, the sound
log-sum-exp ('lse_sound', round 3): at a node with m valid entries and beta = log(m) / eps, the minimum
-(1/beta) log sum exp(-beta z) (as the plain log-sum-exp) and the maximum (1/beta) log sum exp(beta z) - log(m) / beta
(semantics.lse_max: the plain maximum shifted down by log(m) / beta = eps). Both lie between the exact extremum
minus eps and the exact extremum, so the value is a lower bound of the exact robustness.
"""
import jax.numpy as jnp
import numpy as np

from ..stl import Always, And, Atom, Eventually, Not, Until, compile_formula, evaluate, lse_max, lse_min
from ..stl.program import read
from . import planar as P0

H = P0.H
U_MAX = np.array([P0.V_MAX, P0.W_MAX])  # input box: |v| <= 1 m/s, |omega| <= pi/2 rad/s
NAMES = ("Red", "Green", "Blue", "Obstacle")


def make_regions(red, green, blue, obstacle):
    """Regions as an (4, 3) array of (centre x, centre y, radius), in the order of NAMES."""
    return np.array([red, green, blue, obstacle], float)


def scores(xy, regions):
    """Leaf scores (..., T, 8) from positions (..., T, 2) and regions (4, 3)."""
    c = jnp.asarray(regions[:, :2])
    r = jnp.asarray(regions[:, 2])
    d = jnp.sqrt(jnp.sum((xy[..., None, :] - c) ** 2, -1))  # (..., T, 4)
    x, y = xy[..., 0:1], xy[..., 1:2]
    return jnp.concatenate([r - d, x, 10.0 - x, y, 10.0 - y], -1)


def specification(a1, b1, a2, b2, T):
    """(specification, its four conjuncts, the until); intervals in samples."""
    hold = int(round(2 / H))
    until = Until((a1, b1), Not(Atom(0)), Atom(1))
    conj = (until, Eventually((a2, b2), Always((0, hold), Atom(2))), Always((0, T - 1), Not(Atom(3))),
            Always((0, T - 1), And(Atom(4), Atom(5), Atom(6), Atom(7))))
    return And(*conj), conj, until


def matched(name, eps):
    """(max_reduce, min_reduce) at worst-case error eps per node: planar.matched for 'exact', 'lse_plain',
    'gm01', 'gm10', 'sparsemax', and the sound log-sum-exp for 'lse_sound' (beta = log(m) / eps per node,
    m the node's valid entries; one entry exact)."""
    if name != "lse_sound":
        return P0.matched(name, eps)

    def mx(z, param=None, mask=None):
        return P0._per_row(lse_max, z, jnp.log(P0._count(z, mask)) / eps, mask)

    def mn(z, param=None, mask=None):
        return P0._per_row(lse_min, z, jnp.log(P0._count(z, mask)) / eps, mask)
    return mx, mn


def values(S, timing, sem):
    """(specification at t = 0, its four conjuncts at t = 0) under the reduction pair sem; S (T, 8)."""
    spec, conj, _ = specification(timing["a1"], timing["b1"], timing["a2"], timing["b2"], timing["T"])
    prog = compile_formula(spec, timing["T"], reads=[(spec, [0])] + [(c, [0]) for c in conj])
    out = read(prog, evaluate(prog, S, sem))
    return out[..., 0], out[..., 1:]


# ------------------------------------------------------------------
# layout and the two trajectories

Z0 = (1.5, 1.0, np.pi / 2)  # start, facing +y
RED = (3.0, 4.5, 1.4)
GREEN = (5.6, 7.6, 1.0)
BLUE = (8.3, 2.6, 1.0)
OBSTACLE = (6.7, 4.3, 0.6)
REGIONS = make_regions(RED, GREEN, BLUE, OBSTACLE)
GREEN_STOP = (5.45, 7.4)  # 0.25 m from Green's centre: at a centre the predicate r - |p - c| has the apex of a cone
BLUE_STOP = (8.3, 2.85)  # 0.25 m from Blue's centre, for the same reason
GREEN_HEADING = -0.6  # heading at the stop in Green
BLUE_HEADING = -np.pi / 2
V_CRUISE = 0.6  # m/s, plateau speed of the drives


def _on_red(radius, phi):
    """Point at angle phi on the circle of the given radius about Red's centre, the clockwise heading there."""
    return np.array([RED[0] + radius * np.cos(phi), RED[1] + radius * np.sin(phi)]), phi - np.pi / 2


def blend(r0, r1, phi0, phi1, n=4000):
    """Dense points of the curve r(phi) = r0 + (r1 - r0) S(x) about Red's centre, x = (phi - phi0) / (phi1 - phi0),
    S the quintic smootherstep 6x^5 - 15x^4 + 10x^3 (zero first and second derivative at both ends, so the
    curve joins the circles of radius r0 and r1 with their heading and curvature)."""
    x = np.linspace(0, 1, n)
    r = r0 + (r1 - r0) * (6 * x**5 - 15 * x**4 + 10 * x**3)
    phi = phi0 + (phi1 - phi0) * x
    return np.stack([RED[0] + r * np.cos(phi), RED[1] + r * np.sin(phi)], 1)


def trajectories(wait, eps=0.1, depth=0.15, stand=0.23, clear=3.0, k_arc=12, down=0.08, phi_in=205.0, exit_turn=0.5):
    """S1, S2 and the specification's timing for one wait (samples standing at the wait spot); depth, stand
    and clear are in units of eps.

    S1 starts at Z0, drives a gentle S-curve onto the circle of radius R - depth*eps about Red's centre
    (inside Red by depth*eps), follows it clockwise at the plateau speed V_CRUISE for k_arc steps (constant
    turn rate, so those samples have equal depth), leaves Red on the blend r(phi) from R - depth*eps to
    R + stand*eps over exit_turn radians (smootherstep), slowing to rest at its end, stands for wait samples, then curves around Red's north side to
    its stop in Green (GREEN_STOP, heading GREEN_HEADING), stands until 0.5 s after Green's window closes, and
    drives a curve to its stop in Blue (BLUE_STOP, heading south). S2 drives one curve from Z0 past Red's
    west side, touching the circle of radius R + clear*eps at one point (heading north there, turning
    more gently than that circle, so the clearance has a single minimum), to the same stop in Green,
    stands there for the same time and takes the same curve to Blue. Green's window opens 0.5 s after S1
    reaches Green and lasts 2 s; Blue's start window is the 6 s centred on the arrival in Blue.
    Returns (u1, u2, timing); both start from Z0."""
    R = RED[2]
    r_in, r_out = R - depth * eps, R + stand * eps
    phi1 = np.radians(phi_in)
    dphi = k_arc * V_CRUISE * H / r_in
    phi2 = phi1 - dphi
    phi3 = phi2 - exit_turn
    p1, h1 = _on_red(r_in, phi1)
    p2, h2 = _on_red(r_in, phi2)
    p3, h3 = _on_red(r_out, phi3)
    a = np.concatenate([P0.hermite(Z0[:2], Z0[2], 0.0, p1, h1, -1 / r_in), P0.arc(RED[:2], r_in, phi1, phi2)[1:],
                        blend(r_in, r_out, phi2, phi3)[1:]])
    g1 = P0.hermite(p3, h3, -1 / r_out, GREEN_STOP, GREEN_HEADING, 0.0, scale=1.1)
    pa = P0.leg(a, V_CRUISE, up=0.2, down=down)
    pg = P0.leg(g1, V_CRUISE)
    q1 = np.concatenate([pa, np.repeat(pa[-1:], wait - 1, 0), pg])
    t_green = len(q1) - 1
    a1 = t_green + 5
    b1 = a1 + int(round(2 / H))
    leave = b1 + 5
    pblue = P0.leg(P0.hermite(GREEN_STOP, GREEN_HEADING, 0.0, BLUE_STOP, BLUE_HEADING, 0.0), V_CRUISE)
    t_blue = leave + len(pblue) - 1
    a2, b2 = t_blue - 30, t_blue + 30
    T = b2 + int(round(2 / H)) + 1
    q1 = np.concatenate([q1, np.repeat(q1[-1:], leave - t_green, 0), pblue[1:]])
    q1 = np.concatenate([q1, np.repeat(q1[-1:], T - len(q1), 0)])
    pc, hc = _on_red(R + clear * eps, np.pi)
    s2 = np.concatenate([P0.hermite(Z0[:2], Z0[2], 0.0, pc, hc, -0.25), P0.hermite(pc, hc, -0.25, GREEN_STOP, GREEN_HEADING, 0.0)[1:]])
    q2 = P0.leg(s2, V_CRUISE)
    if len(q2) - 1 > a1:
        raise ValueError("S2 reaches Green after the window opens")
    q2 = np.concatenate([q2, np.repeat(q2[-1:], leave - (len(q2) - 1), 0), pblue[1:]])
    q2 = np.concatenate([q2, np.repeat(q2[-1:], T - len(q2), 0)])
    timing = {"wait": wait, "eps": eps, "depth": depth * eps, "stand": stand * eps, "clear": clear * eps, "t_green": t_green,
              "a1": a1, "b1": b1, "leave": leave, "t_blue": t_blue, "a2": a2, "b2": b2, "T": T}
    return P0.inputs_from_positions(q1, Z0[2]), P0.inputs_from_positions(q2, Z0[2]), timing
