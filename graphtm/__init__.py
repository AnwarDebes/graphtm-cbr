"""graphtm-cbr, Hierarchical Graph Tsetlin Machine with Counterfactual Boolean Recourse.

Subpackages:
  - core: canonical Hierarchical Tsetlin Machine (Granmo & Saha), CPU+Numba.
  - cuda: CUDA-C kernels (via PyCUDA) for forward + feedback at scale.
  - encoding: VSA hypervector ops + graph→Boolean encoder (BSC, k-hop).
  - recourse: clause-driven counterfactual graph-edit search.
  - distill: GNN teacher → HGTM student distillation.
  - data: dataset loaders (Ames / Hansen / Kazius mutagenicity).
"""
from .core.hierarchical_tm import (
    HierarchicalTM, HierarchicalTMMultiClass, HTMArchSpec,
)
from .encoding.hypervectors import (
    bind, bundle, one_hot, permute, random_hv, similarity,
    thermometer_encode, thermometer_encode_int,
)

__all__ = [
    "HTMArchSpec", "HierarchicalTM", "HierarchicalTMMultiClass",
    "bind", "bundle", "permute", "random_hv", "similarity",
    "thermometer_encode", "thermometer_encode_int", "one_hot",
]
