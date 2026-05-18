"""Counterfactual recourse for HGTM (Module M5).

Public API:
  - GraphEdit, apply_edit, candidates_from_firing_clauses  (candidates)
  - greedy_minimal_edit, SearchTrace                       (search)
  - ValidityReport, validate, validate_smiles              (validity)
  - recourse_report                                        (output)

See `docs/ARCHITECTURE.md` §M5 and `research/06_graph_counterfactuals.md`
§4 for the full design rationale.
"""
from .candidates import (
    GraphEdit,
    apply_edit,
    candidates_from_firing_clauses,
)
from .output import recourse_report
from .search import SearchTrace, greedy_minimal_edit
from .validity import ValidityReport, validate, validate_smiles

__all__ = [
    "GraphEdit",
    "apply_edit",
    "candidates_from_firing_clauses",
    "SearchTrace",
    "greedy_minimal_edit",
    "ValidityReport",
    "validate",
    "validate_smiles",
    "recourse_report",
]
