"""GIN teacher + HGTM student distillation (Module M4).

Public API:
  - `GINTeacher`           : 3-layer GINConv (32-d hidden, mean-pool).
  - `train_teacher`        : train the GIN teacher on `(GraphTensor, y)`.
  - `distill`              : hard-label distillation onto an HGTM student.
  - `DistillResult`        : dataclass returned by `distill(...)`.

Interface frozen by `docs/ARCHITECTURE.md` § M4.
"""
from .student import DistillResult, distill
from .teacher import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_N_ATOM_TYPES,
    DEFAULT_N_BOND_TYPES,
    GINTeacher,
    TeacherTrainHistory,
    graph_tensor_to_pyg,
    graphs_to_pyg_list,
    train_teacher,
)

__all__ = [
    "GINTeacher",
    "TeacherTrainHistory",
    "train_teacher",
    "distill",
    "DistillResult",
    "graph_tensor_to_pyg",
    "graphs_to_pyg_list",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_N_ATOM_TYPES",
    "DEFAULT_N_BOND_TYPES",
]
