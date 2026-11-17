"""Sparse derivatives of STL-constrained programs: the specification's patterns, the sparsemax supports, colorings and the
assembly of the constraint Jacobian and the Lagrangian Hessian."""
from .assembly import StructuralDerivatives
from .coloring import dense_derivatives
from .counts import count, report
from .patterns import Reference, prescribed

__all__ = ["StructuralDerivatives", "dense_derivatives", "count", "report", "Reference", "prescribed"]
