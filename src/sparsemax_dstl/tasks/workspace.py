"""Human-robot shared workspace with speed and separation monitoring (E019; M003 design point 2,
approved by the user; D006 amendment of 2026-09-29). A simplified form of the concept of
ISO/TS 15066 speed and separation monitoring, not the standard's protective separation
distance.

Scene. The robot (the Panda plant of tasks.panda by default; any MuJoCo arm through Plant)
stands on a table top at z = 0. A human (tasks.human) stands across the table and reaches
into a shared zone, the ball of radius zone_radius around the table point zone (x, y, z).
A ball, not a vertical cylinder, so that the arm may pass above the zone. The robot
picks a part at a pick pose outside the zone, then places it at a handover pose inside the
zone and holds it there.

Geometry. The robot's links are capsules fitted to their collision meshes (robot_spheres:
principal axis of the vertices, radius the largest vertex distance from the axis segment),
each covered by a chain of spheres of spacing robot_spacing, radius sqrt(r^2 + (s/2)^2). The
human's capsules are covered the same way with human_spacing (tasks.human.spheres). Both
covers contain their capsules, so a separation between sphere covers is a separation
between the capsules.

Atoms (E014 option C: a margin divided by its threshold, smoothed at zero length, at most
the exact normalized margin, with a Lipschitz bound; no atom is a negation of another), in
the order of layout():

    pick, handover     (r_goal - n+(p - g)) / r_goal                       goal form at the site p
    zone               (n-(p - z) - zone_radius) / zone_radius              outside the zone ball
    speed_i            (v_slow - n+(c_i')) / v_slow                         robot sphere centre speed
    sep_ij             (n-(c_i - h_j) - R_ij) / R_ij,   R_ij = r_i + r_j + d_min
    slow_ij            (n-(c_i - h_j) - S_ij) / S_ij,   S_ij = r_i + r_j + d_slow

with n+(x) = sqrt(|x|^2 + eps^2) and n-(x) = n+(x) - eps, eps_length for lengths and
eps_speed for speeds (1% of the smallest threshold of each quantity, as in E014). i runs over
robot spheres, j over human spheres; human sphere centres are time-indexed inputs from the
instance. The referee (margins) takes eps = 0 and stops gradients at the states.

Speed is taken at every robot sphere centre, paired with that sphere's separations: the
slow-down conjunct for sphere i is (and_j slow_ij) or speed_i, "every point of the robot is
far from the human or slow". A selection of the points facing the human would be a
non-smooth geometric choice; the disjunction makes it inside the formula, and at a far
sphere the slow disjunct holds whatever its speed. Under exact semantics this equals
and_j (slow_ij or speed_i) by distributivity, with 1 Or per robot sphere instead of one per
pair.

Specifications (specs), each free of negation, at the STL sampling interval h_s:

    separation   G_[0,H] and_{i,j} sep_ij
    slowdown     G_[0,H] and_i ((and_j slow_ij) or speed_i)
    order        zone U_[a H, b H] pick                  stay out of the zone until the pick
    handover     F_[c H, d H] G_[0, dwell H] handover    the handover pose lies inside the zone

The core study's specification is their conjunction; the extension's rows are the four
formulas, which E020 transcribes by their structure.

Sampling. The control is held over h; the physics steps at the plant's DT; the STL samples
every h_s. physics_rollout returns every physics state, sampled(states, h_s) the samples.
Every parameter that moves the regime (thresholds, zone, windows, sampling, sphere spacings,
the human's script) is an argument (Scenario, human.Script).
"""
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
    """Scenario parameters; every value that moves the regime is here (E019 brief)."""
    d_min: float = 0.10
    d_slow: float = 0.30
    v_slow: float = 0.25
    r_goal: float = 0.05
    zone: tuple = (0.5, 0.0, 0.0)
    zone_radius: float = 0.2
    pick: tuple = (Fraction(1, 4), Fraction(1, 2))
    handover: tuple = (Fraction(3, 5), Fraction(4, 5))
    dwell: Fraction = Fraction(1, 10)
    H: Fraction = Fraction(8)
    h: Fraction = Fraction(1, 50)
    h_s: Fraction = Fraction(1, 50)
    robot_spacing: float = 0.10
    human_spacing: float = 0.12
    eps_length: float = 5e-4
    until_hold: float = None  # E034: None keeps every function of this module unchanged; see until_instance

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
    """The robot behind the scenario: a MuJoCo model, the site that carries the part, and the
    bodies whose collision geoms are covered by spheres. Any arm of hinge joints fits."""
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
    for name in plant.bodies:  # over bodies (model structure)
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        for g in P.collision_geoms(m, b):  # over geoms (model structure)
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
    """Names and declared state dependencies of the atoms, in the order scores() returns."""
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
    """Declared bounds l with |g(x) - g(x')| <= l |x - x'|_inf per atom, in layout() order, as
    tasks.panda.lipschitz derives them: D_j bounds the distance of a point from joint j's
    anchor along the chain, n+ and n- and the height projection are 1-Lipschitz, and the speed
    bound holds on the box |qdot| <= qdot_max."""
    m = plant.model
    s = robot_spheres(plant, sc.robot_spacing)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, plant.site)
    Ds = P.reach_bounds(m, m.site_bodyid[site], m.site_pos[site]).sum()
    out = [Ds / sc.r_goal] * 2 + [Ds / sc.zone_radius]
    D = [P.reach_bounds(m, b, c) for b, c in zip(s["body"], s["centre"])]
    for d in D:  # over robot spheres (model structure)
        k = np.arange(len(d))
        out.append((d.sum() + np.sum(np.asarray(qdot_max)[:len(d), None] * d[np.maximum.outer(k, k)])) / sc.v_slow)
    for thr in (sc.d_min, sc.d_slow):  # the two separation kinds
        for d, r in zip(D, s["radius"]):  # over robot spheres (model structure)
            out += list(d.sum() / (r + np.asarray(human_radii) + thr))
    return np.array(out)


# ------------------------------------------------------------------
# specifications

def specs(sc, n_robot, n_human):
    """The four specifications (names, formulas) and the core conjunction, over layout()."""
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
FILTER_STRIDE = 5  # the separation filter reads every fifth STL sample; the witness reads all
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
# the designed regime (E022): one close pass of the hand and a wait at a standoff

# Draw ranges of the pass and the wait, in addition to SCRIPT_RANGES (equal ends fix a value):
# - stand: standing distance of the human from the zone centre, m;
# - pass_margin, standoff_margin: separation margins, in score units, of the pair (anchor
#   sphere, right-hand sphere) at the closest approach of the pass and during the wait, with
#   the robot at the handover configuration; the anchor is the robot sphere nearest to the
#   standing right shoulder there, so pass_gap = (1 + pass_margin) R and
#   standoff = (1 + standoff_margin) R with R = r_anchor + r_hand + d_min;
# - t_pass, pass_width, wait: seconds (human.Script).
REGIME_RANGES = {"stand": (0.6, 0.6), "pass_margin": (0.1, 0.1), "standoff_margin": (0.2, 0.2), "t_pass": (6.0, 6.0),
                 "pass_width": (0.6, 0.6), "wait": (4.0, 4.0)}
REGIME_CLEAR = 0.2  # score units; see regime_instances
# E024 (root's design revision of 2026-09-30 07:25Z): the trapezoid pass of tasks.human (pre_margin
# set) holding the hand at the pass margin for k = 10 samples at h = 0.02 s (pass_hold = 0.19 s
# centred at t_pass = 6.01 s, half a sample off the grid), ramps of pass_ramp = 0.1 s (at most 5
# samples) from the pre point at margin 0.5 and to the standoff; clear_above_pass: every
# separation margin outside the designed episode at least pass_margin + 0.3 = 0.4 score units at
# the reference configurations (the E022 rule, REGIME_CLEAR above the standoff, is unchanged; with
# 0.3 above the standoff it would ask for 0.5, and candidate 56's margin there is 0.4716).
# E024, root's design revision 2 (2026-09-30 08:55Z): the pinned plateau. The hand holds at the
# standoff (margin 0.2) from standoff_from = 5.4 s for the wait, which is also the handover dwell of
# the specification (the caller sets Scenario.dwell to the wait), with the trapezoid pass (margin
# 0.1, the 10 samples 5.92..6.10 s) dipping from the standoff inside it. The anchor stays E022's
# (the robot sphere nearest the standing shoulder at the handover configuration). The option
# anchor_site (the sphere nearest the part) rejects every candidate at 16 s, wait 4 s, on the
# separation filter (1165 to 1168 of the 1168 that pass the IK and zone checks, standing 0.6 to
# 0.9 m; gate E024/2026-09-30T0620Z, p_pinned2.out): other robot spheres lie between it and the human.
REGIME_RANGES_PINNED = {"stand": (0.6, 0.6), "pass_margin": (0.1, 0.1), "standoff_margin": (0.2, 0.2), "t_pass": (6.01, 6.01),
                        "pass_width": (0.6, 0.6), "wait": (4.0, 4.0), "pass_hold": (0.19, 0.19), "pass_ramp": (0.1, 0.1),
                        "clear_above_pass": (0.3, 0.3), "standoff_from": (5.4, 5.4)}
REGIME_RANGES_TRAPEZOID = {"stand": (0.6, 0.6), "pass_margin": (0.1, 0.1), "standoff_margin": (0.2, 0.2), "t_pass": (6.01, 6.01),
                           "pass_width": (0.6, 0.6), "wait": (4.0, 4.0), "pre_margin": (0.5, 0.5), "pass_hold": (0.19, 0.19),
                           "pass_ramp": (0.1, 0.1), "clear_above_pass": (0.3, 0.3)}
# E029 (the constrained demo on a 10 s horizon), two optional keys of a regime range dict, absent
# from every range above so that their scenarios are unchanged:
# - t_reach: the time the right hand starts its reach into the zone, in place of the candidate's
#   draw (t_release = t_reach + 0.6 + the drawn hold); a value beyond the horizon removes the reach;
# - hand_from: the start, in seconds, of the span over which the regime filter places the robot at
#   the handover configuration (by default the start of the handover window, handover[0] H).


def hand_sphere_radius(sc):
    """Radius of the right hand's sphere, the distal sphere of forearm_r in the human cover."""
    ends, radii = Hm.capsules(Hm.Script(), np.zeros(1))
    _, r, owner = Hm.spheres(ends, radii, sc.human_spacing)
    return float(r[owner == Hm.NAMES.index("forearm_r")][-1])


def regime_script_of(sc, draw, regime, C_hand, r_robot, site=None):
    """A human.Script with the pass and the wait: draw as in script_of, regime a draw of
    REGIME_RANGES (dicts of floats or of arrays of a common shape B), C_hand (B..., S_r, 3) the
    robot sphere centres at the handover configuration and r_robot (S_r,) their radii. Returns
    the script, the anchor sphere index (B...) and R (B...)."""
    base = script_of(sc, draw)
    if "t_reach" in regime:  # E029: the reach re-timed (beyond the horizon: no reach)
        tr = np.asarray(regime["t_reach"], np.float64)
        base = replace(base, t_reach=tr, t_release=tr + 0.6 + np.asarray(draw["hold"]))
    shoulder = Hm.standing_shoulder(Hm.Script(zone=base.zone, phi=base.phi, stand=regime["stand"]))
    near = shoulder if "anchor_site" not in regime else np.asarray(site)  # E024 pinned: the sphere nearest to the part
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
    """Boolean (B..., T): the samples of the designed episode, from the start of the pass to the
    hand's return to rest after the wait."""
    f = lambda v: np.asarray(v, np.float64)[..., None]
    t = np.asarray(t, np.float64)
    if script.standoff_from is not None:  # the pinned plateau: from the move to the standoff to the return to rest
        s0, _, _, _, _, se = Hm.pinned_times(script)
        return (t >= f(s0) - f(script.t_move)) & (t <= f(se) + f(script.t_move))
    if script.pre_gap is not None:  # the trapezoid: from the move to the pre point to the return to rest
        ta, _, _, _, te = Hm.trapezoid_times(script)
        return (t >= f(ta) - f(script.t_move)) & (t <= f(te) + f(script.t_move))
    return (t >= f(script.t_pass) - f(script.pass_width) / 2) & \
           (t <= f(script.t_pass) + f(script.pass_width) / 2 + 2 * f(script.t_move) + f(script.wait))


# ------------------------------------------------------------------
# E034: the demo on the Until (D006 amendment of 2026-10-01 03:29Z)
# The flag is Scenario.until_hold, a distance in metres; None (the default) is the pilot's scene, and
# nothing above reads the field. With the flag set, until_instance places the pick target on the
# zone ball's boundary, outside it by until_hold, so that a robot holding the part at the pick pose
# waits just outside the zone, and the handover target inside the zone on the same ray from the
# zone centre. The person (until_script) either keeps the right hand inside the zone for the whole
# horizon ("zone", the protocol's scene) or stands back from the table with the hand at rest
# ("out"). The windows are the Scenario's pick, handover and dwell, set by the caller.

UNTIL_PERSON = {"zone": {"stand": 0.6, "depth": -0.17, "reach_z": 0.03, "reach": True},
                "out": {"stand": 2.0, "depth": 0.0, "reach_z": 0.10, "reach": False}}


def until_targets(sc, direction, handover_radius):
    """(pick, handover) targets (3,) each: zone + (zone_radius + until_hold) n and zone +
    handover_radius n, n the unit vector of direction (3,) from the zone centre."""
    if sc.until_hold is None:
        raise ValueError("until_targets needs the E034 flag Scenario.until_hold")
    n = np.asarray(direction, np.float64) / np.linalg.norm(direction)
    z = np.asarray(sc.zone, np.float64)
    return z + (sc.zone_radius + sc.until_hold) * n, z + handover_radius * n


def until_script(sc, person):
    """The person of UNTIL_PERSON[person] as a human.Script facing the zone along phi = 0, standing
    still from t = 0 (no walk). "zone": the right hand moved to the reach target zone - depth u +
    reach_z e_z before t = 0 and held there past the horizon (depth < 0 puts it on the person's
    side of the zone centre); "out": no reach within the horizon, the hand at rest."""
    if sc.until_hold is None:
        raise ValueError("until_script needs the E034 flag Scenario.until_hold")
    p = UNTIL_PERSON[person]
    H = float(sc.H)
    t_reach = -1.0 if p["reach"] else 2 * H
    return Hm.Script(zone=tuple(sc.zone[:2]), zone_radius=sc.zone_radius, phi=0.0, stand=p["stand"], walk=0.0, t_arrive=0.5,
                     t_reach=t_reach, t_move=0.6, t_release=2 * H + 1.0, depth=p["depth"], reach_z=p["reach_z"])


def until_instance(sc, direction, handover_radius, person):
    """pick, handover (3,), the person's capsule ends, sphere centres (T, S_h, 3), radii and owners
    at the STL samples, for until_targets and until_script."""
    pick, hand = until_targets(sc, direction, handover_radius)
    out = human_inputs(until_script(sc, person), sc)
    out.update(pick=pick, handover=hand)
    return out


# ------------------------------------------------------------------
# E039: the person working in the zone during the wait (D006 amendment of 2026-10-01 06:52Z)
# Additive to E034's scene: until_script and UNTIL_PERSON above are unchanged. The person stands at the
# zone facing it along phi with the right hand inside the zone from before t = 0 (the hand point, the
# distal end of the forearm capsule, at zone - depth u + reach_z e_z, u the person's direction), keeps it
# there through the wait, withdraws over t_move seconds from t = w + release (w the opening of the pick
# window, sc.pick[0] H) to the standoff zone + (zone_radius + hover) u + hover_z e_z outside the zone, and
# stays there to the end of the horizon (through the handover and its dwell). Every value is a key of
# the dict passed to until_work_script; UNTIL_WORK names the settings used by examples/e039_*.

UNTIL_WORK = {}


def until_work_script(sc, work):
    """The working person of a dict work (keys phi, stand, depth, reach_z, release, t_move, hover, hover_z)
    as a human.Script, standing still from t = 0 (no walk); the withdrawal is timed from the pick window's
    opening of the Scenario sc. Values may be arrays of a common shape (one script per entry)."""
    if sc.until_hold is None:
        raise ValueError("until_work_script needs the E034 flag Scenario.until_hold")
    w = float(sc.pick[0] * sc.H)
    tm = np.asarray(work["t_move"], np.float64)
    return Hm.Script(zone=tuple(sc.zone[:2]), zone_radius=sc.zone_radius, phi=work["phi"], stand=work["stand"], walk=0.0, t_arrive=0.5,
                     t_reach=-1.0 - tm, t_move=tm, t_release=w + np.asarray(work["release"], np.float64), depth=work["depth"],
                     reach_z=work["reach_z"], hover=work["hover"], hover_z=work["hover_z"])


def until_work_instance(sc, direction, handover_radius, work):
    """until_instance with the working person of until_work_script in place of until_script's."""
    pick, hand = until_targets(sc, direction, handover_radius)
    out = human_inputs(until_work_script(sc, work), sc)
    out.update(pick=pick, handover=hand)
    return out


# E039, root's redirect of 2026-10-01 (after 07:26Z): the person comes near only while the arm is at rest at the hold pose. The person
# stands `far` metres further back (hand at rest) while the arm moves; once the arm is at rest (t_rest) the person steps in over
# t_move seconds to the stance of until_work_script's keys (phi, stand, depth, reach_z) while reaching the hand to its target, keeps
# it there for up to `length` seconds, and withdraws (hand to the point of hover and hover_z, body `far` metres back) over t_move
# seconds, done `lead` seconds before the arm leaves the pick pose at w + 1. So the near plateau is [v0, v1] with
# v1 = w + 1 - lead - t_move and v0 = max(t_rest + t_move, v1 - length). The body is translated rigidly along the person's
# direction u, so the capsules keep their lengths.

def until_visit_inputs(sc, visit):
    """Capsule endpoints, sphere centres (T, S_h, 3), radii and owners of the visiting person at the STL samples, and the plateau
    (v0, v1) in seconds, for the dict visit (keys of until_work_script plus far, t_rest, length, lead)."""
    if sc.until_hold is None:
        raise ValueError("until_visit_inputs needs the E034 flag Scenario.until_hold")
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
    """until_instance with the visiting person of until_visit_inputs."""
    pick, hand = until_targets(sc, direction, handover_radius)
    out = until_visit_inputs(sc, visit)
    out.update(pick=pick, handover=hand)
    return out
