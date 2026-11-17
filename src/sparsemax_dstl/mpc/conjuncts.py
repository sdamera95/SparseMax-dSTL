"""Conjunct instances of a scene's specification over the receding horizon.

Each conjunct (an until or a reach of `tasks.dual_arm_scenes`) is compiled once over absolute time with one read per
instance, an instance being the formula at one anchor. Its per-sample nodes are the conjunction of a cell's outside
predicates for one arm ("and") or one waypoint predicate ("leaf"). The record's exact node values decide instances
(`stl.progression.decisions` at the margin). At control period k with horizon T, an instance is imposed when the record has
not decided it and, for an until, its whole window lies at or before sample k + T; for a reach, its anchor lies at or before
k. Rows are ordered reach conjuncts first, then untils, each in the scene's order, and within a conjunct by anchor.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ..stl import compile_formula, horizon, to_nnf
from ..stl import progression as pg
from ..tasks import dual_arm as D


def record_predicates(scene, states, length=0.0):
    """(L, P) predicate vectors along states (L, 28) for the record: the waypoint predicates with the exact norm and the
    outside predicates with the norm smoothed over `length` (m; 0 the exact norm)."""
    with jax.enable_x64(True):
        cells = jnp.asarray(scene.cells, jnp.float64)

        def one(x):
            return jnp.concatenate([D.at_waypoint(x, scene.waypoints, scene.waypoint_arms, 0.0), D.outside(x, cells, scene.cell_radius, length)])
        return np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(states, jnp.float64)))


def node_exact(nodes, P):
    """(L, n_nodes) the exact value of each per-sample node over predicate vectors P (L, n_pred)."""
    return np.stack([P[:, n["preds"]].min(-1) if n["kind"] == "and" else P[:, int(n["preds"][0])] for n in nodes], -1)


class Conjunct:
    """One conjunct: its program over the record's samples, the root rows of its instances (one per anchor), the record's
    decisions (per step (d, ok)) and its smoothing parameter gamma."""

    def __init__(self, c, program, record, delta):
        self.name, self.kind, self.formula, self.nodes = c["name"], c["kind"], c["formula"], c["nodes"]
        self.anchors, self.gamma = np.asarray(c["anchors"]), float(c["gamma"])
        self.window = horizon(to_nnf(self.formula)) + 1
        self.program = program
        self.root_step, self.root_rows = program.outputs[0]
        self.decisions = pg.decisions(program, node_exact(self.nodes, record), delta)

    def imposed(self, ks, T, n=None):
        """(K, n) root rows imposed at each period of ks (-1 for an empty slot) and their instance indices, in anchor
        order; n slots (default: the most imposed at one of these periods)."""
        ks = np.asarray(ks)
        d = self.decisions[self.root_step][0][self.root_rows]
        a = self.anchors
        inside = (a[None] + self.window - 1 <= ks[:, None] + T) if self.kind == "until" else (a[None] <= ks[:, None])
        on = (d[None] > ks[:, None]) & inside
        n = max(int(on.sum(1).max(initial=0)), 1) if n is None else n
        order = np.argsort(~on, axis=1, kind="stable")[:, :n]
        ok = np.take_along_axis(on, order, 1)
        return np.where(ok, self.root_rows[order], -1), np.where(ok, order, -1)


def conjuncts(scene, record, delta=None):
    """The scene's conjuncts in row order, decided on the record (L, P) of exact predicate vectors (record_predicates);
    conjuncts with the same formula and anchors share one program."""
    delta = scene.margin if delta is None else delta
    specs = [c for c in scene.conjuncts if c["kind"] == "reach"] + [c for c in scene.conjuncts if c["kind"] == "until"]
    L = max(int(np.max(c["anchors"])) + horizon(to_nnf(c["formula"])) for c in specs) + 1
    record = np.asarray(record)[:L]
    programs, out = {}, []
    for c in specs:
        key = (c["formula"], tuple(int(a) for a in c["anchors"]))
        if key not in programs:
            programs[key] = compile_formula(c["formula"], L, reads=[(c["formula"], [int(a) for a in c["anchors"]])])
        out.append(Conjunct(c, programs[key], record, delta))
    return out


def imposed_pairs(conjs, k, T):
    """[(conjunct name, anchor)] of the instances imposed at period k, in row order."""
    out = []
    for cj in conjs:
        _, inst = cj.imposed([k], T)
        out += [(cj.name, int(cj.anchors[i])) for i in inst[0] if i >= 0]
    return out
