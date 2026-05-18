"""graphtm.core, Hierarchical Tsetlin Machine cores.

Exports:
  - HTMArchSpec, HierarchicalTM, HierarchicalTMMultiClass
      Canonical CPU/Numba HTM (parity oracle).
  - HGraphTMSpec, HierarchicalGraphTM, FiringClause, ClauseTree
      Graph-walking CUDA student (Option B). See
      `hierarchical_graph_tm.py` for the contract and
      `docs/ARCHITECTURE.md` §M3 for the project-wide interface.
"""
from .hierarchical_tm import (
    HierarchicalTM,
    HierarchicalTMMultiClass,
    HTMArchSpec,
)
from .hierarchical_graph_tm import (
    ClauseTree,
    FiringClause,
    HGraphTMSpec,
    HierarchicalGraphTM,
)

__all__ = [
    # Canonical CPU oracle
    "HTMArchSpec",
    "HierarchicalTM",
    "HierarchicalTMMultiClass",
    # Graph-walking CUDA student
    "HGraphTMSpec",
    "HierarchicalGraphTM",
    "FiringClause",
    "ClauseTree",
]
