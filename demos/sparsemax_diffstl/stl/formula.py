"""STL formulas over sampled traces, and their negation normal form.

A formula refers to predicates by integer index into a leaf-score array whose
last axis lists predicate scores, positive when satisfied. Intervals are
closed integer sample offsets [a, b] with 0 <= a <= b. Operators:

    Atom(i)            score of predicate i (Atom(i, True) is its negation)
    Not, And, Or, Implies
    Always(I, f), Eventually(I, f)
    Until(I, f, g)     closed prefix: f must also hold at the witness time
    Release(I, f, g)   exact dual of the closed-prefix Until

And and Or are n-ary. Their grouping is kept exactly as written, because the
smooth reductions are not associative.
"""
from dataclasses import dataclass


def _interval(interval):
    a, b = (int(v) for v in interval)
    if not 0 <= a <= b:
        raise ValueError("interval needs 0 <= a <= b, got " + str(interval))
    return (a, b)


@dataclass(frozen=True)
class Atom:
    index: int
    negated: bool = False


@dataclass(frozen=True)
class Not:
    child: object


@dataclass(frozen=True)
class And:
    children: tuple

    def __init__(self, *children):
        if len(children) == 0:
            raise ValueError("And needs at least one child")
        object.__setattr__(self, "children", tuple(children))


@dataclass(frozen=True)
class Or:
    children: tuple

    def __init__(self, *children):
        if len(children) == 0:
            raise ValueError("Or needs at least one child")
        object.__setattr__(self, "children", tuple(children))


@dataclass(frozen=True)
class Implies:
    left: object
    right: object


@dataclass(frozen=True)
class Always:
    interval: tuple
    child: object

    def __post_init__(self):
        object.__setattr__(self, "interval", _interval(self.interval))


@dataclass(frozen=True)
class Eventually:
    interval: tuple
    child: object

    def __post_init__(self):
        object.__setattr__(self, "interval", _interval(self.interval))


@dataclass(frozen=True)
class Until:
    interval: tuple
    left: object
    right: object

    def __post_init__(self):
        object.__setattr__(self, "interval", _interval(self.interval))


@dataclass(frozen=True)
class Release:
    interval: tuple
    left: object
    right: object

    def __post_init__(self):
        object.__setattr__(self, "interval", _interval(self.interval))


def to_nnf(f, negate=False):
    """Push negation down to the atoms, keeping the grouping of And and Or."""
    if isinstance(f, Atom):
        return Atom(f.index, f.negated != negate)
    if isinstance(f, Not):
        return to_nnf(f.child, not negate)
    if isinstance(f, Implies):
        return to_nnf(Or(Not(f.left), f.right), negate)
    if isinstance(f, (And, Or)):
        children = [to_nnf(c, negate) for c in f.children]
        flip = isinstance(f, And) == negate
        return Or(*children) if flip else And(*children)
    if isinstance(f, (Always, Eventually)):
        child = to_nnf(f.child, negate)
        flip = isinstance(f, Always) == negate
        return Eventually(f.interval, child) if flip else Always(f.interval, child)
    if isinstance(f, (Until, Release)):
        left, right = to_nnf(f.left, negate), to_nnf(f.right, negate)
        flip = isinstance(f, Until) == negate
        return Release(f.interval, left, right) if flip else Until(f.interval, left, right)
    raise TypeError("not an STL formula: " + repr(f))


def horizon(f):
    """Number of samples after t that the value at t reads."""
    if isinstance(f, Atom):
        return 0
    if isinstance(f, Not):
        return horizon(f.child)
    if isinstance(f, Implies):
        return max(horizon(f.left), horizon(f.right))
    if isinstance(f, (And, Or)):
        return max(horizon(c) for c in f.children)
    if isinstance(f, (Always, Eventually)):
        return f.interval[1] + horizon(f.child)
    if isinstance(f, (Until, Release)):
        return f.interval[1] + max(horizon(f.left), horizon(f.right))
    raise TypeError("not an STL formula: " + repr(f))


def atoms(f):
    """Set of predicate indices used by a formula."""
    if isinstance(f, Atom):
        return {f.index}
    if isinstance(f, Not):
        return atoms(f.child)
    if isinstance(f, (And, Or)):
        return set().union(*(atoms(c) for c in f.children))
    if isinstance(f, (Always, Eventually)):
        return atoms(f.child)
    return atoms(f.left) | atoms(f.right)
