"""JSON-friendly recourse report assembly.

`recourse_report` packages everything a downstream consumer (paper table,
benchmark log, regulatory case study) needs about a single counterfactual
search: the edits applied, pre/post predictions, validity, and runtime.

Matches the §4.6 contract in `research/06_graph_counterfactuals.md`:

    {
      "smiles_before": "...",
      "smiles_after":  "...",
      "edits":         [{"op": ..., "indices": (...), "new_value": ...}, ...],
      "prediction": {"before": {"class": int, "scores": [...]},
                     "after":  {"class": int, "scores": [...]}},
      "validity": {"sanitize_ok": bool, "lipinski_ok": bool,
                   "sa_score": float|None, "sa_ok": bool,
                   "overall_ok": bool, "notes": str,
                   "mw": float, "logp": float, "hba": int, "hbd": int},
      "latency_ms": {"total": float, "candidate_gen": float,
                     "search": float, "validate": float}
    }
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .candidates import GraphEdit
from .validity import ValidityReport

try:
    from rdkit import Chem
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    _HAVE_RDKIT = False


def _mol_to_smiles(mol: Any) -> Optional[str]:
    if mol is None or not _HAVE_RDKIT:
        return None
    if isinstance(mol, str):
        return mol
    try:
        return Chem.MolToSmiles(mol)
    except Exception:  # pragma: no cover
        return None


def _scores_to_list(s: Any) -> Optional[List[float]]:
    if s is None:
        return None
    a = np.asarray(s).ravel()
    return [float(x) for x in a.tolist()]


def _edit_to_dict(e: GraphEdit) -> Dict[str, Any]:
    return {
        "op": e.op,
        "indices": list(e.indices),
        "new_value": (int(e.new_value) if e.new_value is not None else None),
    }


def _validity_to_dict(v: Optional[ValidityReport]) -> Optional[Dict[str, Any]]:
    if v is None:
        return None
    if is_dataclass(v):
        d = asdict(v)
    else:  # pragma: no cover
        d = dict(getattr(v, "__dict__", {}))
    # Coerce numpy scalars / np.float to plain Python
    for k, val in list(d.items()):
        if isinstance(val, (np.floating, np.integer)):
            d[k] = val.item()
    return d


def recourse_report(
    *,
    orig_mol: Any = None,
    final_mol: Any = None,
    edits: Sequence[GraphEdit] = (),
    validity: Optional[ValidityReport] = None,
    pred_before: Optional[int] = None,
    pred_after: Optional[int] = None,
    scores_before: Any = None,
    scores_after: Any = None,
    latencies: Optional[Mapping[str, float]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a JSON-friendly dict for a single recourse instance.

    Keyword-only to avoid positional-call mistakes in downstream callers.
    All fields are optional; missing ones become ``None`` in the report.
    """
    report: Dict[str, Any] = {
        "smiles_before": _mol_to_smiles(orig_mol),
        "smiles_after": _mol_to_smiles(final_mol),
        "edits": [_edit_to_dict(e) for e in edits] if edits else [],
        "prediction": {
            "before": {
                "class": int(pred_before) if pred_before is not None else None,
                "scores": _scores_to_list(scores_before),
            },
            "after": {
                "class": int(pred_after) if pred_after is not None else None,
                "scores": _scores_to_list(scores_after),
            },
        },
        "validity": _validity_to_dict(validity),
        "latency_ms": dict(latencies) if latencies else {},
        "flipped": (
            None if pred_before is None or pred_after is None
            else bool(int(pred_before) != int(pred_after))
        ),
        "n_edits": len(edits) if edits else 0,
    }
    if extra:
        # Reserved for caller-specific fields (clauses_touched, etc.).
        report["extra"] = dict(extra)
    return report
