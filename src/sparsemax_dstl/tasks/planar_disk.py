"""Planar unicycle of Section V-A of the paper: four disk regions with one signed-distance predicate each, the
specification of Equation (18), and the fixed trajectories S1 and S2."""
import numpy as np

from ..stl import Always, And, Atom, Eventually, Not, Until
from . import planar as P0

H = P0.H
U_MAX = np.array([P0.V_MAX, P0.W_MAX])  # input box: |v| <= 1 m/s, |omega| <= pi/2 rad/s
NAMES = ("Red", "Green", "Blue", "Obstacle")


def make_regions(red, green, blue, obstacle):
    """Regions as an (4, 3) array of (centre x, centre y, radius), in the order of NAMES."""
    return np.array([red, green, blue, obstacle], float)


def specification(a1, b1, a2, b2, T):
    """(specification, its four conjuncts, the until) of Equation (18), intervals in samples. Predicates 0..3 are
    r_i - |p - c_i| for Red, Green, Blue, Obstacle and 4..7 are x, 10 - x, y, 10 - y."""
    hold = int(round(2 / H))
    until = Until((a1, b1), Not(Atom(0)), Atom(1))
    conj = (until, Eventually((a2, b2), Always((0, hold), Atom(2))), Always((0, T - 1), Not(Atom(3))),
            Always((0, T - 1), And(Atom(4), Atom(5), Atom(6), Atom(7))))
    return And(*conj), conj, until


# ------------------------------------------------------------------
# layout and the two trajectories

Z0 = (1.5, 1.0, np.pi / 2)  # start, facing +y
RED = (3.0, 4.5, 1.4)
GREEN = (5.6, 7.6, 1.0)
BLUE = (8.3, 2.6, 1.0)
OBSTACLE = (6.7, 4.3, 0.6)
REGIONS = make_regions(RED, GREEN, BLUE, OBSTACLE)
GREEN_STOP = (5.45, 7.4)  # 0.25 m from Green's centre, where the predicate r - |p - c| has a kink
BLUE_STOP = (8.3, 2.85)  # 0.25 m from Blue's centre
GREEN_HEADING = -0.6  # heading at the stop in Green
BLUE_HEADING = -np.pi / 2
V_CRUISE = 0.6  # m/s, plateau speed of the drives


def _on_red(radius, phi):
    """Point at angle phi on the circle of the given radius about Red's centre, and the clockwise heading there."""
    return np.array([RED[0] + radius * np.cos(phi), RED[1] + radius * np.sin(phi)]), phi - np.pi / 2


def blend(r0, r1, phi0, phi1, n=4000):
    """Dense points of the curve r(phi) = r0 + (r1 - r0) S(x) about Red's centre, x = (phi - phi0) / (phi1 - phi0),
    with S the quintic smootherstep 6x^5 - 15x^4 + 10x^3."""
    x = np.linspace(0, 1, n)
    r = r0 + (r1 - r0) * (6 * x**5 - 15 * x**4 + 10 * x**3)
    phi = phi0 + (phi1 - phi0) * x
    return np.stack([RED[0] + r * np.cos(phi), RED[1] + r * np.sin(phi)], 1)


def trajectories(wait, eps=0.1, depth=0.15, stand=0.23, clear=3.0, k_arc=12, down=0.08, phi_in=205.0, exit_turn=0.5):
    """(u1, u2, timing): the inputs of S1 and S2 from Z0 and the specification's windows, for a wait of `wait` samples.
    S1 runs depth eps inside Red on a concentric arc and waits stand eps outside; S2 passes clear eps west of Red."""
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
