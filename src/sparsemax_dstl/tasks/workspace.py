"""The manipulator beside a person (Section V-B of the paper): the scenario's parameters, the chains of spheres on the
robot, the layout of the predicates, the specification of Equation (19), and the person's scripted motions."""
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cache

import mujoco
import numpy as np

from ..stl import Always, And, Atom, Eventually, Or, Until
from . import human as Hm
from . import panda as P


@dataclass(frozen=True)
class Scenario:
    """Parameters of the scenario: thresholds, the zone, the windows as fractions of the horizon, sampling and sphere
    spacings."""
    d_min: float = 0.10  # separation threshold of sep, m
    d_slow: float = 0.30  # separation threshold of slow, m
    v_slow: float = 0.25  # speed threshold, m/s
    r_goal: float = 0.05  # radius about the pick and handover targets, m
    zone: tuple = (0.5, 0.0, 0.0)  # centre of the zone ball, m
    zone_radius: float = 0.2
    pick: tuple = (Fraction(1, 4), Fraction(1, 2))  # pick window, fractions of H
    handover: tuple = (Fraction(3, 5), Fraction(4, 5))  # window of the handover's start, fractions of H
    dwell: Fraction = Fraction(1, 10)  # time held at the handover target, fraction of H
    H: Fraction = Fraction(8)  # horizon, s
    h: Fraction = Fraction(1, 50)
    h_s: Fraction = Fraction(1, 50)  # STL sampling interval, s
    robot_spacing: float = 0.10  # largest spacing of the sphere centres along a capsule, m
    human_spacing: float = 0.12
    eps_length: float = 5e-4  # smoothing length of the norms of lengths, m
    until_hold: float = None  # distance of the pick target outside the zone ball in until_targets, m

    @property
    def eps_speed(self):
        return 0.01 * self.v_slow

    @property
    def samples(self):
        """Number of STL samples, H / h_s + 1."""
        n = Fraction(self.H) / Fraction(self.h_s)
        if n.denominator != 1:
            raise ValueError("H / h_s must be a whole number")
        return int(n) + 1


@dataclass(frozen=True)
class Plant:
    """The robot: a MuJoCo model, the site that carries the part, and the bodies whose collision geoms are covered by
    spheres."""
    model: object = field(default_factory=P.model)
    site: str = P.SITE
    bodies: tuple = P.LINKS

    def __hash__(self):
        return hash((id(self.model), self.site, self.bodies))


# ------------------------------------------------------------------
# robot sphere chains

def capsule(V):
    """Capsule (a, b, r) containing the points V (n, 3): principal axis, extent, largest distance."""
    mu = V.mean(0)
    axis = np.linalg.svd(V - mu)[2][0]
    t = (V - mu) @ axis
    lo, hi = t.min(), t.max()
    foot = mu + np.clip(t, lo, hi)[:, None] * axis
    return mu + lo * axis, mu + hi * axis, float(np.linalg.norm(V - foot, axis=-1).max())


@cache
def robot_spheres(plant, spacing):
    """Sphere chains covering every collision geom (group 3) of the plant's bodies, in the body
    frames: body (S,), centre (S, 3), radius (S,), geom (S,)."""
    m = plant.model
    body, centre, radius, geom = [], [], [], []
    for name in plant.bodies:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        for g in P.collision_geoms(m, b):
            a, e, r = capsule(P.geom_vertices(m, g))
            n = math.ceil(np.linalg.norm(e - a) / spacing) + 1
            gap = np.linalg.norm(e - a) / (n - 1)
            body += [b] * n
            geom += [g] * n
            centre.append(a + np.linspace(0, 1, n)[:, None] * (e - a))
            radius += [math.sqrt(r ** 2 + (gap / 2) ** 2)] * n
    return {"body": np.array(body), "centre": np.concatenate(centre), "radius": np.array(radius), "geom": np.array(geom)}


# ------------------------------------------------------------------
# atoms

def layout(plant, sc, n_human):
    """Names and state dependencies of the predicates, in the order of the last axis of the predicate values: pick,
    handover, zone, speed_i per robot sphere i, then sep_ij and slow_ij per robot sphere i and human sphere j."""
    m = plant.model
    s = robot_spheres(plant, sc.robot_spacing)
    site = P._chain(m, m.site_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, plant.site)])
    nq = m.nq
    names = ["pick", "handover", "zone"] + ["speed" + str(i) for i in range(len(s["body"]))]
    deps = [site] * 3 + [P._chain(m, b) + tuple(nq + k for k in P._chain(m, b)) for b in s["body"]]
    for kind in ("sep", "slow"):
        for i, b in enumerate(s["body"]):
            names += [kind + str(i) + "_" + str(j) for j in range(n_human)]
            deps += [P._chain(m, b)] * n_human
    return tuple(names), tuple(deps)


def lipschitz(plant, sc, human_radii, qdot_max):
    """Lipschitz bounds of the predicates in the infinity norm of the state, in layout() order, from
    panda.reach_bounds. The speed bounds hold on the box |qdot| <= qdot_max."""
    m = plant.model
    s = robot_spheres(plant, sc.robot_spacing)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, plant.site)
    Ds = P.reach_bounds(m, m.site_bodyid[site], m.site_pos[site]).sum()
    out = [Ds / sc.r_goal] * 2 + [Ds / sc.zone_radius]
    D = [P.reach_bounds(m, b, c) for b, c in zip(s["body"], s["centre"])]
    for d in D:
        k = np.arange(len(d))
        out.append((d.sum() + np.sum(np.asarray(qdot_max)[:len(d), None] * d[np.maximum.outer(k, k)])) / sc.v_slow)
    for thr in (sc.d_min, sc.d_slow):
        for d, r in zip(D, s["radius"]):
            out += list(d.sum() / (r + np.asarray(human_radii) + thr))
    return np.array(out)


# ------------------------------------------------------------------
# specifications

def specs(sc, n_robot, n_human):
    """(names, formulas, conjunction) of Equation (19) over layout(): order (zone U pick), handover (F G handover),
    separation (G and_ij sep_ij) and slowdown (G and_i ((and_j slow_ij) or speed_i)); windows in samples of h_s."""
    H, h = Fraction(sc.H), Fraction(sc.h_s)
    w = lambda a, b: P.window(a, b, h)
    speed0, sep0 = 3, 3 + n_robot
    slow0 = sep0 + n_robot * n_human
    sep = Always(w(0, H), And(*[Atom(sep0 + i * n_human + j) for i in range(n_robot) for j in range(n_human)]))
    slow = Always(w(0, H), And(*[Or(And(*[Atom(slow0 + i * n_human + j) for j in range(n_human)]), Atom(speed0 + i))
                                 for i in range(n_robot)]))
    order = Until(w(sc.pick[0] * H, sc.pick[1] * H), Atom(2), Atom(0))
    hand = Eventually(w(sc.handover[0] * H, sc.handover[1] * H), Always(w(0, sc.dwell * H), Atom(1)))
    names = ("order", "handover", "separation", "slowdown")
    rows = (order, hand, sep, slow)
    return names, rows, And(sep, slow, order, hand)


# ------------------------------------------------------------------
# the STL sampling interval

def sample_stride(sc, dt=P.DT):
    """Physics steps per STL sample, h_s / dt, a whole number."""
    k = Fraction(sc.h_s) / Fraction(str(dt))
    if k.denominator != 1:
        raise ValueError("h_s must be a whole number of physics steps")
    return int(k)


# ------------------------------------------------------------------
# instances

TUNING_SEED = 2026092901
HELDOUT_SEED = 2026092902
N_TUNING = 10
N_HELDOUT = 30
N_CANDIDATES = 2048
Q_SPREAD = 0.2
JOINT_MARGIN = 0.1  # rad inside every joint range, for q0 and the IK solutions
PICK_BOX = ((0.2, 0.6), (0.25, 0.55), (0.05, 0.30))  # x, |y| (either side), z of the pick target, m
HANDOVER_MIN_Z = 0.05
ZONE_MARGIN = 0.03
SEP_MARGIN = 0.02
IK_ITERATIONS = 200
IK_TOLERANCE = 1e-4
FILTER_STRIDE = 5  # the separation filter of the instances reads every fifth STL sample
SCRIPT_RANGES = {"phi": (-0.5, 0.5), "t_arrive": (0.5, 1.5), "t_reach": (1.8, 2.6), "hold": (0.4, 1.2),
                 "depth": (0.0, 0.08), "hover": (0.10, 0.20)}
PER_INSTANCE = ("q0", "q_pick", "q_handover", "pick", "handover", "start", "script", "human_ends", "human_centres",
                "candidate")


def script_of(sc, draw):
    """A human.Script from one draw of SCRIPT_RANGES (a dict of floats or of arrays)."""
    return Hm.Script(zone=tuple(sc.zone[:2]), zone_radius=sc.zone_radius, phi=draw["phi"], t_arrive=draw["t_arrive"],
                     t_reach=draw["t_reach"], t_release=np.asarray(draw["t_reach"]) + 0.6 + np.asarray(draw["hold"]),
                     depth=draw["depth"], hover=draw["hover"])


def human_inputs(script, sc):
    """Capsule endpoints (B..., T, 6, 2, 3), radii, sphere centres (B..., T, S_h, 3) and radii (S_h,)
    at the STL sampling times."""
    t = np.arange(sc.samples) * float(sc.h_s)
    ends, radii = Hm.capsules(script, t)
    centres, sr, owner = Hm.spheres(ends, radii, sc.human_spacing)
    return {"human_ends": ends, "human_capsule_radii": radii, "human_centres": centres, "human_radii": sr,
            "human_owner": owner}


# ------------------------------------------------------------------
# one close pass of the person's hand and a wait at a standoff

# Draw ranges of the pass and the wait, in addition to SCRIPT_RANGES (equal ends fix a value):
# - stand: standing distance of the human from the zone centre, m;
# - pass_margin, standoff_margin: pass_gap = (1 + pass_margin) R and standoff = (1 + standoff_margin) R in
#   human.Script, with R = r_anchor + r_hand + d_min and the anchor the robot sphere nearest to the standing
#   right shoulder at the handover configuration;
# - t_pass, pass_width, wait: seconds (human.Script).
REGIME_RANGES = {"stand": (0.6, 0.6), "pass_margin": (0.1, 0.1), "standoff_margin": (0.2, 0.2), "t_pass": (6.0, 6.0),
                 "pass_width": (0.6, 0.6), "wait": (4.0, 4.0)}
REGIME_CLEAR = 0.2  # in units of the predicates; used by workspace_mjx.regime_instances
# Further keys: pre_margin (pre_gap = (1 + pre_margin) R), pass_hold, pass_ramp and standoff_from (seconds, as in
# human.Script), and clear_above_pass (workspace_mjx.regime_instances then requires pass_margin + clear_above_pass
# in place of standoff_margin + REGIME_CLEAR).
REGIME_RANGES_PINNED = {"stand": (0.6, 0.6), "pass_margin": (0.1, 0.1), "standoff_margin": (0.2, 0.2), "t_pass": (6.01, 6.01),
                        "pass_width": (0.6, 0.6), "wait": (4.0, 4.0), "pass_hold": (0.19, 0.19), "pass_ramp": (0.1, 0.1),
                        "clear_above_pass": (0.3, 0.3), "standoff_from": (5.4, 5.4)}
REGIME_RANGES_TRAPEZOID = {"stand": (0.6, 0.6), "pass_margin": (0.1, 0.1), "standoff_margin": (0.2, 0.2), "t_pass": (6.01, 6.01),
                           "pass_width": (0.6, 0.6), "wait": (4.0, 4.0), "pre_margin": (0.5, 0.5), "pass_hold": (0.19, 0.19),
                           "pass_ramp": (0.1, 0.1), "clear_above_pass": (0.3, 0.3)}
# Two optional keys of a range dict: t_reach, the time the right hand starts its reach in place of the candidate's
# draw (t_release = t_reach + 0.6 + the drawn hold), and hand_from, the start in seconds of the span over which
# workspace_mjx.regime_instances places the robot at the handover configuration (by default handover[0] H).


def hand_sphere_radius(sc):
    """Radius of the right hand's sphere, the distal sphere of forearm_r in the human cover."""
    ends, radii = Hm.capsules(Hm.Script(), np.zeros(1))
    _, r, owner = Hm.spheres(ends, radii, sc.human_spacing)
    return float(r[owner == Hm.NAMES.index("forearm_r")][-1])


def regime_script_of(sc, draw, regime, C_hand, r_robot, site=None):
    """A human.Script with the pass and the wait, the anchor sphere's index (B...) and R (B...): draw as in script_of,
    regime a draw of REGIME_RANGES, C_hand (B..., S_r, 3) and r_robot (S_r,) the robot spheres at the handover pose."""
    base = script_of(sc, draw)
    if "t_reach" in regime:
        tr = np.asarray(regime["t_reach"], np.float64)
        base = replace(base, t_reach=tr, t_release=tr + 0.6 + np.asarray(draw["hold"]))
    shoulder = Hm.standing_shoulder(Hm.Script(zone=base.zone, phi=base.phi, stand=regime["stand"]))
    near = shoulder if "anchor_site" not in regime else np.asarray(site)  # the anchor is the robot sphere nearest to it
    a = np.argmin(np.linalg.norm(np.asarray(C_hand) - near[..., None, :], axis=-1), -1)
    anchor = np.take_along_axis(np.asarray(C_hand), a[..., None, None], -2)[..., 0, :]
    R = np.asarray(r_robot)[a] + hand_sphere_radius(sc) + sc.d_min
    trap = {} if "pre_margin" not in regime else {"pre_gap": (1 + np.asarray(regime["pre_margin"])) * R,
                                                  "pass_hold": regime["pass_hold"], "pass_ramp": regime["pass_ramp"]}
    if "standoff_from" in regime:
        trap = {"standoff_from": regime["standoff_from"], "pass_hold": regime["pass_hold"], "pass_ramp": regime["pass_ramp"]}
    s = Hm.Script(zone=base.zone, zone_radius=base.zone_radius, phi=base.phi, stand=regime["stand"], t_arrive=base.t_arrive,
                  t_reach=base.t_reach, t_release=base.t_release, depth=base.depth, hover=base.hover, anchor=anchor,
                  t_pass=regime["t_pass"], pass_width=regime["pass_width"], pass_gap=(1 + np.asarray(regime["pass_margin"])) * R,
                  standoff=(1 + np.asarray(regime["standoff_margin"])) * R, wait=regime["wait"], **trap)
    return s, a, R


def regime_episode(script, t):
    """Boolean (B..., T): the samples from the start of the hand's move toward the anchor to its return to rest."""
    f = lambda v: np.asarray(v, np.float64)[..., None]
    t = np.asarray(t, np.float64)
    if script.standoff_from is not None:  # from the move to the standoff to the return to rest
        s0, _, _, _, _, se = Hm.pinned_times(script)
        return (t >= f(s0) - f(script.t_move)) & (t <= f(se) + f(script.t_move))
    if script.pre_gap is not None:  # from the move to the pre point to the return to rest
        ta, _, _, _, te = Hm.trapezoid_times(script)
        return (t >= f(ta) - f(script.t_move)) & (t <= f(te) + f(script.t_move))
    return (t >= f(script.t_pass) - f(script.pass_width) / 2) & \
           (t <= f(script.t_pass) + f(script.pass_width) / 2 + 2 * f(script.t_move) + f(script.wait))


# ------------------------------------------------------------------
# the pick target just outside the zone (Scenario.until_hold set)
# until_instance places the pick target until_hold metres outside the zone ball and the handover target at
# handover_radius from its centre, on one ray from the centre. The person either holds the right hand at a reach
# target for the whole horizon ("zone") or stands 2 m from the zone centre with the hand at rest ("out").

UNTIL_PERSON = {"zone": {"stand": 0.6, "depth": -0.17, "reach_z": 0.03, "reach": True},
                "out": {"stand": 2.0, "depth": 0.0, "reach_z": 0.10, "reach": False}}


def until_targets(sc, direction, handover_radius):
    """(pick, handover) targets (3,) each: zone + (zone_radius + until_hold) n and zone +
    handover_radius n, n the unit vector of direction (3,) from the zone centre."""
    if sc.until_hold is None:
        raise ValueError("until_targets needs Scenario.until_hold")
    n = np.asarray(direction, np.float64) / np.linalg.norm(direction)
    z = np.asarray(sc.zone, np.float64)
    return z + (sc.zone_radius + sc.until_hold) * n, z + handover_radius * n


def until_script(sc, person):
    """The person of UNTIL_PERSON[person] as a human.Script, standing still and facing the zone along phi = 0: "zone"
    holds the right hand at the reach target from before t = 0 past the horizon, "out" keeps it at rest."""
    if sc.until_hold is None:
        raise ValueError("until_script needs Scenario.until_hold")
    p = UNTIL_PERSON[person]
    H = float(sc.H)
    t_reach = -1.0 if p["reach"] else 2 * H
    return Hm.Script(zone=tuple(sc.zone[:2]), zone_radius=sc.zone_radius, phi=0.0, stand=p["stand"], walk=0.0, t_arrive=0.5,
                     t_reach=t_reach, t_move=0.6, t_release=2 * H + 1.0, depth=p["depth"], reach_z=p["reach_z"])


def until_instance(sc, direction, handover_radius, person):
    """The dict of human_inputs for until_script, with the pick and handover targets of until_targets."""
    pick, hand = until_targets(sc, direction, handover_radius)
    out = human_inputs(until_script(sc, person), sc)
    out.update(pick=pick, handover=hand)
    return out


# ------------------------------------------------------------------
# the person working in the zone during the wait
# The person stands facing the zone with the right hand at zone - depth u + reach_z e_z (u the person's direction)
# from before t = 0, withdraws it over t_move seconds from t = w + release (w = pick[0] H, the opening of the pick
# window) to zone + (zone_radius + hover) u + hover_z e_z, and stays there to the end of the horizon.

UNTIL_WORK = {}


def until_work_script(sc, work):
    """The working person of the dict work (keys phi, stand, depth, reach_z, release, t_move, hover, hover_z) as a
    human.Script; the withdrawal starts release seconds after the pick window opens. Values may be arrays."""
    if sc.until_hold is None:
        raise ValueError("until_work_script needs Scenario.until_hold")
    w = float(sc.pick[0] * sc.H)
    tm = np.asarray(work["t_move"], np.float64)
    return Hm.Script(zone=tuple(sc.zone[:2]), zone_radius=sc.zone_radius, phi=work["phi"], stand=work["stand"], walk=0.0, t_arrive=0.5,
                     t_reach=-1.0 - tm, t_move=tm, t_release=w + np.asarray(work["release"], np.float64), depth=work["depth"],
                     reach_z=work["reach_z"], hover=work["hover"], hover_z=work["hover_z"])


def until_work_instance(sc, direction, handover_radius, work):
    """until_instance with the person of until_work_script."""
    pick, hand = until_targets(sc, direction, handover_radius)
    out = human_inputs(until_work_script(sc, work), sc)
    out.update(pick=pick, handover=hand)
    return out


# The visiting person stands `far` metres further back with the hand at rest, steps in over t_move seconds while
# reaching the hand to its target, stays for the plateau [v0, v1], and withdraws over t_move seconds, with
# v1 = w + 1 - lead - t_move and v0 = max(t_rest + t_move, v1 - length). The body is translated along the person's
# direction u.

def until_visit_inputs(sc, visit):
    """The dict of human_inputs for the visiting person, with the plateau (v0, v1) in seconds under the key visit, for
    the dict visit (keys phi, stand, depth, reach_z, t_move, hover, hover_z, far, t_rest, length, lead)."""
    if sc.until_hold is None:
        raise ValueError("until_visit_inputs needs Scenario.until_hold")
    w = float(sc.pick[0] * sc.H)
    tm = float(visit["t_move"])
    v1 = w + 1 - float(visit["lead"]) - tm
    v0 = max(float(visit["t_rest"]) + tm, v1 - float(visit["length"]))
    s = Hm.Script(zone=tuple(sc.zone[:2]), zone_radius=sc.zone_radius, phi=visit["phi"], stand=visit["stand"], walk=0.0, t_arrive=0.5,
                  t_reach=v0 - tm, t_move=tm, t_release=v1, depth=visit["depth"], reach_z=visit["reach_z"], hover=visit["hover"],
                  hover_z=visit["hover_z"])
    t = np.arange(sc.samples) * float(sc.h_s)
    ends, radii = Hm.capsules(s, t)
    off = float(visit["far"]) * (1 - Hm._step(t, v0 - tm, v0) + Hm._step(t, v1, v1 + tm))
    u = np.array([np.cos(visit["phi"]), np.sin(visit["phi"]), 0.0])
    ends = ends + off[:, None, None, None] * u
    centres, sr, owner = Hm.spheres(ends, radii, sc.human_spacing)
    return {"human_ends": ends, "human_capsule_radii": radii, "human_centres": centres, "human_radii": sr, "human_owner": owner, "visit": (v0, v1)}


def until_visit_instance(sc, direction, handover_radius, visit):
    """until_instance with the person of until_visit_inputs."""
    pick, hand = until_targets(sc, direction, handover_radius)
    out = until_visit_inputs(sc, visit)
    out.update(pick=pick, handover=hand)
    return out
