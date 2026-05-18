"""Greedy minimum-edit counterfactual search for HGTM.

Implements the Wachter-style objective from
`research/06_graph_counterfactuals.md` §4.3 in graph-edit space:

    min_{S subseteq K(x)}  |S|
    s.t.   sign(S_c(x ⊕ S) - S_{c'}(x ⊕ S)) flips
           validity(x ⊕ S) is True

The candidate set ``K(x)`` is produced by
``candidates.candidates_from_firing_clauses``; I never enumerate beyond
it. At each greedy step I pick the candidate that maximally decreases
the predicted class score margin, re-encode the resulting molecule, and
re-call the model. I stop the moment the prediction flips, capped by
``max_flips`` (default 3).

Why not BFS / 2^k enumeration: explicitly forbidden by the project's
"no BFS" reliability rule. Bounded candidate set + greedy descent gives
sub-100 ms latency on CPU for a fast model.
"""
from __future__ import annotations

import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from .candidates import GraphEdit, apply_edit

try:
    from rdkit import Chem
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    _HAVE_RDKIT = False


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class SearchTrace:
    """Lightweight bookkeeping for a search call (latencies, scores).

    Fields are public so callers can read them after `greedy_minimal_edit`
    returns. Kept as a regular class (not dataclass) so I can grow it
    without breaking pickled traces.
    """
    def __init__(self) -> None:
        self.applied_edits: List[GraphEdit] = []
        self.scores_before: List[Any] = []   # per outer step
        self.scores_after: List[Any] = []
        self.candidates_tried: int = 0
        self.candidates_rejected: int = 0
        self.flipped: bool = False
        self.total_latency_ms: float = 0.0
        self.steps_latency_ms: List[float] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_predict_one(model: Any, graph: Any) -> int:
    """Get a single predicted class index from ``model`` for ``graph``.

    Tolerant of three model shapes:
      * ``model.predict([graph]) -> array_like length 1``
      * ``model([graph]) -> array_like length 1``  (callable model)
      * ``model.predict(graph) -> int``            (single-graph API)
    """
    if hasattr(model, "predict"):
        out = model.predict([graph])
    elif callable(model):
        out = model([graph])
    else:
        raise TypeError("model must implement predict(graphs) or be callable")
    arr = np.atleast_1d(np.asarray(out))
    return int(arr[0])


def _model_class_scores(model: Any, graph: Any) -> np.ndarray:
    """Get class scores for a single graph as a 1D array of length K.

    Tolerant of three model shapes:
      * ``model.class_scores([graph]) -> array shape (1, K)``
      * ``model.class_scores(graph)   -> 1-D length K``
      * As last resort, fall back to a one-hot of ``model.predict``.
    """
    if hasattr(model, "class_scores"):
        try:
            s = np.asarray(model.class_scores([graph]))
        except TypeError:
            s = np.asarray(model.class_scores(graph))
        if s.ndim == 2:
            return s[0]
        return s.ravel()
    # Last-resort: predict-only model, emit ±1 one-hot
    pred = _model_predict_one(model, graph)
    # Assume binary by default
    n_classes = 2
    out = -np.ones(n_classes)
    out[int(pred)] = +1.0
    return out


def _margin(scores: np.ndarray, target_class: int) -> float:
    """Margin for ``target_class`` = score[target] - max(score[other]).

    For a flip I want this to drop below 0.
    """
    s = np.asarray(scores).ravel()
    if s.size <= 1:
        return float(s.sum())
    others = np.delete(s, target_class)
    return float(s[target_class] - others.max())


def _smiles_to_mol(smiles: str) -> Any:
    if not _HAVE_RDKIT:
        raise ImportError("RDKit required for greedy_minimal_edit")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return m


# ---------------------------------------------------------------------------
# greedy_minimal_edit
# ---------------------------------------------------------------------------

def greedy_minimal_edit(model: Any,
                         graph: Any,
                         candidates: Sequence[GraphEdit],
                         mol_or_smiles: Any,
                         *,
                         max_flips: int = 3,
                         encode_fn: Optional[Callable[[Any], Any]] = None,
                         validity_fn: Optional[Callable[[Any], bool]] = None,
                         trace: Optional[SearchTrace] = None,
                         ) -> Optional[List[GraphEdit]]:
    """Greedy descent in graph-edit space until the class prediction flips.

    Parameters
    ----------
    model :
        HGTM-style classifier. Must expose ``predict`` and ideally
        ``class_scores``. See ``_model_predict_one`` / ``_model_class_scores``
        for the duck-typed contract, this keeps M5 independent of M3.
    graph :
        Original ``GraphTensor``. Used for the initial prediction and to
        define the candidate set's index space.
    candidates :
        ``List[GraphEdit]`` from ``candidates_from_firing_clauses``. Order
        matters only for ties.
    mol_or_smiles :
        RDKit Mol or a SMILES string. I need an *actual molecule* to
        execute graph edits via ``RWMol``; I re-encode after each commit.
    max_flips :
        Hard budget on the number of edits committed. Defaults to 3 (the
        Lucic et al. 2022 sparsity preference).
    encode_fn :
        Callable mapping ``rdkit.Mol`` to a new ``GraphTensor`` of the same
        shape contract as ``graph``. If omitted, I cannot re-score after
        an edit; the function will degrade gracefully and stop after the
        first committed edit (returning that edit as the explanation). In
        production the caller should pass
        ``lambda m: encode_graph(m, codebook, k_hop=...)``.
    validity_fn :
        Optional ``(Mol) -> bool``. If provided, candidates whose result
        fails validity are skipped. If omitted I still check RDKit
        sanitization implicitly via ``apply_edit``.
    trace :
        Optional ``SearchTrace`` to populate with diagnostics.

    Returns
    -------
    Optional[List[GraphEdit]]
        The committed edit sequence that flipped the prediction, or
        ``None`` if no flip was achieved within ``max_flips``.
    """
    if not _HAVE_RDKIT:
        raise ImportError("RDKit required for greedy_minimal_edit")
    if max_flips < 1:
        return None
    if not candidates:
        return None

    t0 = time.perf_counter()
    if trace is None:
        trace = SearchTrace()

    # Resolve mol
    if isinstance(mol_or_smiles, str):
        mol = _smiles_to_mol(mol_or_smiles)
    else:
        mol = mol_or_smiles

    # Initial prediction and margin
    orig_pred = _model_predict_one(model, graph)
    orig_scores = _model_class_scores(model, graph)
    init_margin = _margin(orig_scores, orig_pred)

    current_graph = graph
    current_mol = mol
    current_pred = orig_pred
    current_margin = init_margin
    remaining: List[GraphEdit] = list(candidates)
    committed: List[GraphEdit] = []
    trace.scores_before.append(orig_scores)

    for step in range(max_flips):
        t_step = time.perf_counter()
        # Try each remaining candidate; pick the one with the largest
        # margin decrease.
        best_edit: Optional[GraphEdit] = None
        best_graph: Optional[Any] = None
        best_mol: Optional[Any] = None
        best_pred: int = current_pred
        best_margin: float = current_margin
        any_tried = False

        for e in remaining:
            trace.candidates_tried += 1
            # Apply on a copy
            try:
                new_mol = apply_edit(current_mol, e)
            except Exception:
                trace.candidates_rejected += 1
                continue
            if validity_fn is not None:
                try:
                    if not validity_fn(new_mol):
                        trace.candidates_rejected += 1
                        continue
                except Exception:
                    trace.candidates_rejected += 1
                    continue
            if encode_fn is None:
                # Without an encoder I can't measure margin change; greedily
                # accept the first valid edit and stop.
                best_edit = e
                best_mol = new_mol
                best_graph = current_graph
                best_pred = current_pred
                best_margin = current_margin - 1.0  # treat as a flip-attempt
                any_tried = True
                break
            try:
                new_graph = encode_fn(new_mol)
            except Exception:
                trace.candidates_rejected += 1
                continue
            try:
                new_pred = _model_predict_one(model, new_graph)
                new_scores = _model_class_scores(model, new_graph)
            except Exception:
                trace.candidates_rejected += 1
                continue
            new_margin = _margin(new_scores, orig_pred)
            any_tried = True
            # Pick the candidate with the biggest margin DECREASE
            # (i.e. smallest new_margin). Ties broken by flip preference:
            # a flip always wins over a non-flip.
            flipped_here = (new_pred != orig_pred)
            cur_flipped = (best_edit is not None and best_pred != orig_pred)
            if (flipped_here and not cur_flipped) or (
                flipped_here == cur_flipped and new_margin < best_margin
            ):
                best_edit = e
                best_graph = new_graph
                best_mol = new_mol
                best_pred = new_pred
                best_margin = new_margin

        if best_edit is None or not any_tried:
            # No candidate produced a valid model evaluation
            trace.steps_latency_ms.append(
                (time.perf_counter() - t_step) * 1000.0)
            break

        # Commit best
        committed.append(best_edit)
        current_graph = best_graph if best_graph is not None else current_graph
        current_mol = best_mol if best_mol is not None else current_mol
        current_pred = best_pred
        current_margin = best_margin
        remaining = [e for e in remaining if e != best_edit]
        trace.applied_edits.append(best_edit)
        if encode_fn is not None:
            trace.scores_after.append(_model_class_scores(model, current_graph))
        trace.steps_latency_ms.append(
            (time.perf_counter() - t_step) * 1000.0)

        if encode_fn is not None and current_pred != orig_pred:
            trace.flipped = True
            break

    trace.total_latency_ms = (time.perf_counter() - t0) * 1000.0

    if encode_fn is None:
        # Degraded: return what I committed (at most 1 edit)
        return committed if committed else None
    if not trace.flipped:
        return None
    return committed
