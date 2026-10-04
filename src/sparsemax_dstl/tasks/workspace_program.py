"""The manipulator example's plant and compiled specification: the scene's Plant, the number of spheres on the robot, and the
programs of the specification and of its four conjuncts without the rows that the value at the first sample does not read."""
import numpy as np

from .. import stl
from ..stl import Program, Step
from . import workspace as W

plant = W.Plant()
N_R = len(W.robot_spheres(plant, W.Scenario().robot_spacing)["body"])
CONJUNCTS = ("separation", "slowdown", "order", "handover")


def reachable(program):
    """Boolean per step: the rows the root entry at t = 0 reads through valid window entries."""
    steps = program.steps
    reach = [np.zeros(s.length, bool) for s in steps]
    reach[-1][0] = True
    for s in reversed(range(len(steps))):  # over formula steps, consumers before sources
        step = steps[s]
        if step.kind == "atom" or not reach[s].any():
            continue
        sizes = [steps[j].length for j in step.sources]
        valid = np.arange(step.index.shape[1])[None, :] < step.count[:, None]
        hit = np.zeros(sum(sizes), bool)
        hit[step.index[reach[s][:, None] & valid]] = True
        offsets = np.cumsum([0] + sizes)
        for j, a, b in zip(step.sources, offsets[:-1], offsets[1:]):
            reach[j] |= hit[a:b]
    return reach


def prune(program):
    """The same program with every reduction step restricted to the rows the root at t = 0
    reads (atom steps kept whole); the root entry's value and its derivatives are unchanged."""
    reach = reachable(program)
    new, pos = [], []
    for s, step in enumerate(program.steps):  # over formula steps, sources before consumers
        if step.kind == "atom":
            new.append(step)
            pos.append(np.arange(step.length))
            continue
        keep = np.nonzero(reach[s])[0]
        offs = np.cumsum([0] + [new[j].length for j in step.sources])
        m = np.concatenate([np.where(pos[j] >= 0, pos[j] + o, -1) for j, o in zip(step.sources, offs[:-1])])
        cnt = step.count[keep]
        valid = np.arange(step.index.shape[1])[None, :] < cnt[:, None]
        idx = m[step.index[keep]]
        if len(keep) == 0 or np.any(idx[valid] < 0):
            raise ValueError("pruning lost a row that a kept row reads")
        new.append(Step(step.kind, len(keep), step.label, step.sources, np.where(valid, idx, 0).astype(np.int32),
                        cnt.astype(np.int32)))
        p = -np.ones(step.length, int)
        p[keep] = np.arange(len(keep))
        pos.append(p)
    return Program(program.formula, tuple(new), program.T, program.boundary, program.n_predicates)


def core_program(sc, n_h, pruned=True):
    """The compiled conjunction of workspace.specs for n_h human spheres, pruned by default."""
    prog = stl.compile_formula(W.specs(sc, N_R, n_h)[2], sc.samples)
    return prune(prog) if pruned else prog


def conj_programs(sc, n_h):
    """The four conjunct programs (pruned; the conjunct at t = 0 is the root) in CONJUNCTS order."""
    names, rows, _ = W.specs(sc, N_R, n_h)
    by = dict(zip(names, rows))
    return tuple(prune(stl.compile_formula(by[c], sc.samples)) for c in CONJUNCTS)
