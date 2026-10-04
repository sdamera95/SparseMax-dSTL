"""The scene of the manipulator example as the paper runs it (Section V-B, Appendix I-B): the scenario at a horizon and an
opening time of the pick window, the until conjunct's program, and the initial trajectories of examples/data."""
import json
from fractions import Fraction

import numpy as np

from . import workspace as W
from . import workspace_program as Wp

H_CTRL = Fraction(1, 50)  # sampling and control interval, s
HS = 0.02
HOLD = 0.005  # distance of the pick target outside the zone, m
ELEV = 0.6  # elevation of the ray from the zone centre that carries the pick and handover targets, rad
DIRECTION = (-float(np.cos(ELEV)), 0.0, float(np.sin(ELEV)))
HANDOVER_RADIUS = 0.10  # distance of the handover target from the zone centre, m
T_TR, F_LEN, DWELL = 0.8, 0.18, 0.8  # s: from the pick window to the handover window, that window's length, the dwell
LAM, DELTA, ALPHA0 = 0.01, 0.0, 0.002  # the solver's effort weight, the margin c of Eq. 4, the first step length
PER_TRAJECTORY = ("V0", "x0", "pick", "handover", "hc", "wait", "group", "location")


def frac(x):
    """Seconds as an exact fraction of two decimals."""
    return Fraction(str(round(float(x), 2)))


def scenario(H, w):
    """The Scenario at horizon H s with the pick window [w, w + 1] s and the handover window after it."""
    f0 = frac(w) + 1 + frac(T_TR)
    return W.Scenario(H=Fraction(H), h_s=H_CTRL, pick=(frac(w) / H, (frac(w) + 1) / H), handover=(f0 / H, (f0 + frac(F_LEN)) / H),
                      dwell=frac(DWELL) / H, until_hold=HOLD)


def longest_wait(H):
    """The latest opening time of the pick window that the horizon admits, H - 2.78 s."""
    return round(H - 1 - T_TR - F_LEN - DWELL, 2)


def until_program(H, w, n_h):
    """The scenario, the until conjunct's program (pruned to t = 0) and its witness samples kw (J,)."""
    sc = scenario(H, w)
    prog = Wp.conj_programs(sc, n_h)[Wp.CONJUNCTS.index("order")]
    inner = [s for s in prog.steps if s.label.endswith(".inner")]
    if len(inner) != 1 or prog.steps[-1].length != 1:
        raise ValueError("expected one Until read at t = 0")
    return sc, prog, inner[0].count.astype(int) - 2


def load(path):
    """The initial trajectories of a file of examples/data as a dict: normalized torques V0 (n, T - 1, 7), x0 (n, 14), pick and handover
    (n, 3), opening time wait (n,), initial exact robustness group (n,), horizon H, and the person's spheres hc (n, T, S, 3), hr (S,)."""
    z = np.load(path)
    I = {k: z[k] for k in ("V0", "pick", "handover", "wait", "group", "location")}
    n = len(I["V0"])
    H = int(round(I["V0"].shape[1] * HS))
    person = json.loads(str(z["person"]))
    waits, of = np.unique(I["wait"], return_inverse=True)
    visits = [W.until_visit_inputs(scenario(H, w), person) for w in waits]  # over the distinct opening times (a handful)
    I.update(x0=np.broadcast_to(z["x0"], (n, 14)), hc=np.stack([v["human_centres"] for v in visits]).astype(np.float32)[of],
             hr=visits[0]["human_radii"], n_h=len(visits[0]["human_radii"]), H=H, person=person)
    return I


def runs(I, measures, eps):
    """One run per measure and trajectory of I at node error eps: the arrays of I repeated per measure, the measure of each
    run, and groups, the blocks (measure, eps, first run, last run + 1, wait) of consecutive runs that share a program."""
    n = len(I["V0"])
    R = {k: v for k, v in I.items() if k not in PER_TRAJECTORY}
    R.update({k: np.tile(I[k], (len(measures),) + (1,) * (I[k].ndim - 1)) for k in PER_TRAJECTORY})
    R.update(measure=np.repeat(np.asarray(measures), n), eps=float(eps))
    cut = np.flatnonzero((R["measure"][1:] != R["measure"][:-1]) | (R["wait"][1:] != R["wait"][:-1])) + 1
    first, last = np.concatenate([[0], cut]), np.concatenate([cut, [len(R["wait"])]])
    R["groups"] = [(str(R["measure"][a]), float(eps), int(a), int(b), float(R["wait"][a])) for a, b in zip(first, last)]
    return R
