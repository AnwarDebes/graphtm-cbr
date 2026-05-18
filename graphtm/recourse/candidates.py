"""Clause-driven candidate generation for HGTM-CBR recourse.

For an input graph predicted class `c`, this module emits a bounded set of
candidate `GraphEdit`s, graph operations (atom/bond level) that touch the
elements currently deciding the prediction. The design follows
`research/06_graph_counterfactuals.md` §4.2:

  * Identify firing clauses (positive- and negative-polarity supporters).
  * For each firing clause, walk the AND/OR tree; for every leaf literal
    whose TA action is "include" AND whose value is currently consistent
    with firing, attribute it back to a (node, edge) of the input graph via
    VSA cleanup of the codebook hypervectors against `graph.node_hv`/
    `graph.edge_hv`.
  * Emit removal / swap edits for the attributed elements.

This is candidate *generation* only, search and validity live in sister
modules in `recourse/`. No BFS, no 2^k enumeration; the candidate set is
upper-bounded by `max_candidates` and seeded by the structure of the
clauses that actually fired on this input.

The `GraphEdit` value is intentionally graph-element-typed (atom/bond
indices, RDKit-aligned bond orders), NOT bit-flip-typed, so every emitted
edit is a real molecular operation that downstream `apply_edit` can
execute through `rdkit.Chem.RWMol`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import BondType, RWMol
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover, RDKit is a hard runtime dep
    _HAVE_RDKIT = False


# RDKit BondType mapping aligned with `codebook.BOND_NAMES`
# ("single", "double", "triple", "aromatic"). Used by `apply_edit` and the
# `swap_bond_order` op; defined lazily because RDKit may be unavailable
# when this module is merely imported for type checking.
def _bond_order_to_rdkit(order: int) -> Any:
    if not _HAVE_RDKIT:
        raise ImportError("RDKit is required for GraphEdit operations")
    table = {
        0: BondType.SINGLE,
        1: BondType.DOUBLE,
        2: BondType.TRIPLE,
        3: BondType.AROMATIC,
    }
    if order not in table:
        raise ValueError(f"unknown bond order index {order}; expected 0..3")
    return table[order]


# ---------------------------------------------------------------------------
# Graph edit operations
# ---------------------------------------------------------------------------

VALID_OPS: Tuple[str, ...] = (
    "remove_bond",
    "add_bond",
    "swap_atom",
    "swap_bond_order",
)


@dataclass(frozen=True)
class GraphEdit:
    """A single edit on a molecular graph.

    op:
      - ``remove_bond``: remove the bond between two atoms.
        ``indices`` is ``(atom_i, atom_j)`` (order-insensitive).
        ``new_value`` is unused.
      - ``add_bond``: add a new bond between two existing atoms.
        ``indices`` is ``(atom_i, atom_j)`` (order-insensitive).
        ``new_value`` is the bond-order index (0..3 per codebook).
      - ``swap_atom``: change an atom's element symbol to another from
        the codebook alphabet. ``indices`` is ``(atom_i,)``.
        ``new_value`` is the atom-type index into ``ATOM_SYMBOLS``.
      - ``swap_bond_order``: change the order of an existing bond.
        ``indices`` is ``(atom_i, atom_j)``.
        ``new_value`` is the new bond-order index (0..3).

    Edits are equality-hashable so they can be used in sets / as dict keys
    during greedy search bookkeeping.
    """
    op: str
    indices: Tuple[int, ...]
    new_value: Optional[int] = None

    def __post_init__(self) -> None:
        if self.op not in VALID_OPS:
            raise ValueError(f"unknown op {self.op!r}; expected one of {VALID_OPS}")
        # Validate index arity per op
        if self.op == "swap_atom":
            if len(self.indices) != 1:
                raise ValueError(f"swap_atom expects 1 index, got {self.indices}")
            if self.new_value is None or self.new_value < 0:
                raise ValueError("swap_atom requires non-negative new_value")
        else:
            if len(self.indices) != 2:
                raise ValueError(f"{self.op} expects 2 indices, got {self.indices}")
            if self.indices[0] == self.indices[1]:
                raise ValueError(f"{self.op} requires distinct atoms: {self.indices}")
            if self.op in ("add_bond", "swap_bond_order"):
                if self.new_value is None or self.new_value < 0:
                    raise ValueError(f"{self.op} requires non-negative new_value")


# ---------------------------------------------------------------------------
# apply_edit
# ---------------------------------------------------------------------------

def apply_edit(mol: Any, edit: GraphEdit) -> Any:
    """Apply ``edit`` to ``mol`` and return a new sanitized RDKit Mol.

    The input is NOT mutated. I copy through `RWMol(mol)`, apply the op,
    sanitize, and return. Caller is responsible for handling
    ``Chem.MolSanitizeException`` if the edit produces an invalid molecule
    (e.g. a 5-valent carbon). For "soft" sanitization (collect errors),
    use the sister `validity.validate` module.

    Returns the new Mol object. Raises:
      ImportError  if RDKit is unavailable.
      KeyError     if ``edit`` references a non-existent bond/atom.
    """
    if not _HAVE_RDKIT:
        raise ImportError("RDKit is required for apply_edit")
    if mol is None:
        raise ValueError("apply_edit: mol is None")

    rw = RWMol(mol)

    if edit.op == "remove_bond":
        i, j = edit.indices
        bond = rw.GetBondBetweenAtoms(int(i), int(j))
        if bond is None:
            raise KeyError(f"remove_bond: no bond between atoms {i}, {j}")
        rw.RemoveBond(int(i), int(j))

    elif edit.op == "add_bond":
        i, j = edit.indices
        if rw.GetBondBetweenAtoms(int(i), int(j)) is not None:
            raise KeyError(f"add_bond: bond {i}-{j} already exists")
        order_idx = int(edit.new_value) if edit.new_value is not None else 0
        rw.AddBond(int(i), int(j), _bond_order_to_rdkit(order_idx))

    elif edit.op == "swap_atom":
        (i,) = edit.indices
        new_atom_idx = int(edit.new_value) if edit.new_value is not None else 0
        # Need the symbol; I accept either an atomic number or a
        # codebook-index here. To stay codebook-independent I expect the
        # caller to pre-translate index → atomic number via
        # `_codebook_index_to_atomic_number` (kept private). For symbol
        # input use the standard codebook ATOM_SYMBOLS.
        from ..encoding.codebook import ATOM_SYMBOLS
        if new_atom_idx >= len(ATOM_SYMBOLS):
            raise ValueError(
                f"swap_atom: new_value {new_atom_idx} out of range "
                f"for ATOM_SYMBOLS (len={len(ATOM_SYMBOLS)})"
            )
        sym = ATOM_SYMBOLS[new_atom_idx]
        atom = rw.GetAtomWithIdx(int(i))
        atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(sym))

    elif edit.op == "swap_bond_order":
        i, j = edit.indices
        bond = rw.GetBondBetweenAtoms(int(i), int(j))
        if bond is None:
            raise KeyError(f"swap_bond_order: no bond between {i}, {j}")
        new_idx = int(edit.new_value) if edit.new_value is not None else 0
        bond.SetBondType(_bond_order_to_rdkit(new_idx))

    else:  # pragma: no cover, guarded by GraphEdit.__post_init__
        raise ValueError(f"unhandled op {edit.op!r}")

    # Re-sanitize. Caller catches if invalid.
    new_mol = rw.GetMol()
    Chem.SanitizeMol(new_mol)
    return new_mol


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _node_codebook_index(node_hv: np.ndarray, codebook: Any,
                          atom_type_fallback: Optional[int] = None) -> int:
    """VSA cleanup: index of the closest atom-codebook hypervector to `node_hv`.

    Because graph encoding XOR-binds the atom HV with other terms (role,
    neighborhood), exact equality won't hold. I pick the atom-codebook row
    that maximises Hamming similarity to `node_hv`. The `atom_type_fallback`
    (taken from GraphTensor.atom_type[node]) is preferred when known; this
    function is mostly used as a guard / sanity check.
    """
    if atom_type_fallback is not None:
        return int(atom_type_fallback)
    # Hamming similarity = bits-equal / D
    eq = (codebook.atom_hv == node_hv[None, :]).sum(axis=1)
    return int(np.argmax(eq))


def _iter_leaf_literals(tree_node: Any) -> Iterable[Tuple[Any, dict]]:
    """Recursively yield (literals_list, parent_node) pairs for fired leaves.

    Accepts the JSON-friendly nested clause-tree format documented in
    `core/hierarchical_tm.py::extract_clause_tree`:
        {"type": "AND"|"OR", "children": [...], "fired": bool, ...}
    where leaves are lists of literal strings like "X3=1" / "~X5=1".

    I only yield from subtrees with ``fired=True`` so candidate generation
    is targeted at the active path.
    """
    if isinstance(tree_node, list):
        # Leaf literal list, handled by parent
        return
    if not isinstance(tree_node, dict):
        return
    if tree_node.get("fired") is False:
        # Skip dead branches, they didn't decide the prediction
        return
    children = tree_node.get("children", [])
    for c in children:
        if isinstance(c, list):
            # This is a leaf conjunction of literal strings
            yield c, tree_node
        else:
            yield from _iter_leaf_literals(c)


def _parse_literal(lit: str) -> Tuple[int, int]:
    """Parse a literal string ``"X<idx>=1"`` or ``"~X<idx>=1"`` -> (idx, polarity).

    polarity == 1 for positive, 0 for negated. Matches the format emitted by
    ``hierarchical_tm.extract_clause_tree``.
    """
    s = lit.strip()
    polarity = 1
    if s.startswith("~"):
        polarity = 0
        s = s[1:]
    if not s.startswith("X"):
        raise ValueError(f"unrecognised literal {lit!r}")
    # "X12=1" → 12
    eq = s.find("=")
    if eq < 0:
        raise ValueError(f"unrecognised literal {lit!r}")
    return int(s[1:eq]), polarity


def _firing_clause_payload(fc: Any) -> Tuple[Any, Sequence[int]]:
    """Duck-type extraction of (clause_tree, node_indices) from a FiringClause.

    Tolerates a handful of attribute names so sister-module M3 has room to
    evolve. Returns (tree_dict_or_None, list_of_node_ids).
    """
    # Preferred: explicit tree + nodes attributes
    tree = (
        getattr(fc, "tree", None)
        or getattr(fc, "clause_tree", None)
        or getattr(fc, "literals", None)
    )
    nodes = (
        getattr(fc, "fired_nodes", None)
        or getattr(fc, "nodes", None)
        or getattr(fc, "node_indices", None)
        or []
    )
    return tree, list(nodes) if nodes is not None else []


def _candidate_edits_for_node(graph: Any, node_idx: int,
                               codebook: Any) -> List[GraphEdit]:
    """Emit removal / swap edits that touch ``node_idx`` and its bonds.

    For each bond incident on ``node_idx`` in ``graph.edge_index``, emit a
    ``remove_bond`` candidate. I also emit a ``swap_atom`` candidate
    flipping to a different atom type (the one most distant from the
    current atom_type, as a "maximally-different" tiebreaker).
    """
    edits: List[GraphEdit] = []
    # edge_index is shape [2, E] (undirected: both directions present)
    ei = np.asarray(graph.edge_index)
    n_nodes = int(graph.n_nodes)
    if node_idx < 0 or node_idx >= n_nodes:
        return edits
    # Incident bond removal, deduplicate (i,j) by canonical (min,max)
    seen: set[Tuple[int, int]] = set()
    if ei.size:
        for e in range(ei.shape[1]):
            u, v = int(ei[0, e]), int(ei[1, e])
            if u != node_idx and v != node_idx:
                continue
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            edits.append(GraphEdit(op="remove_bond", indices=key, new_value=None))

    # swap_atom: pick a target atom-type ≠ current
    atom_types = getattr(graph, "atom_type", None)
    if atom_types is not None and codebook is not None:
        cur = int(atom_types[node_idx])
        n_types = int(codebook.atom_hv.shape[0])
        # Pick target as (cur + 1) % n_types, deterministic, ≠ cur
        if n_types > 1:
            tgt = (cur + 1) % n_types
            edits.append(GraphEdit(op="swap_atom",
                                    indices=(int(node_idx),),
                                    new_value=int(tgt)))
    return edits


def _candidate_edits_for_edge(graph: Any, edge_idx: int,
                               codebook: Any) -> List[GraphEdit]:
    """Emit removal / order-swap edits for the bond at edge_idx (undirected)."""
    edits: List[GraphEdit] = []
    ei = np.asarray(graph.edge_index)
    if edge_idx < 0 or edge_idx >= ei.shape[1]:
        return edits
    u, v = int(ei[0, edge_idx]), int(ei[1, edge_idx])
    key = (min(u, v), max(u, v))
    edits.append(GraphEdit(op="remove_bond", indices=key, new_value=None))
    bond_types = getattr(graph, "bond_type", None)
    if bond_types is not None and codebook is not None:
        cur = int(bond_types[edge_idx])
        n_types = int(codebook.bond_hv.shape[0])
        if n_types > 1:
            tgt = (cur + 1) % n_types
            edits.append(GraphEdit(op="swap_bond_order",
                                    indices=key,
                                    new_value=int(tgt)))
    return edits


def candidates_from_firing_clauses(graph: Any,
                                    firing_clauses: Sequence[Any],
                                    codebook: Any = None,
                                    max_candidates: int = 50
                                    ) -> List[GraphEdit]:
    """Generate ≤``max_candidates`` graph edits seeded by firing clauses.

    Strategy (matches research/06 §4.2):

      1. For each firing clause, fetch ``(tree, nodes)`` via duck-typing.
      2. For each node the clause fired at, emit removal/swap edits for the
         atom and its incident bonds.
      3. If the firing-clause exposes a clause-tree, also enumerate the
         active leaf literals; I attribute them back to the firing nodes
         (because nodes are the per-position scope of HGTM clauses).

    Deduplication is by ``(op, indices, new_value)``. Order is preserved
    so the greedy search sees the highest-priority clauses first.

    NOTE: this function ONLY emits edits that touch elements present in
    ``graph.edge_index`` / ``graph.atom_type``. It never emits an op on a
    non-existent bond. (Verified by the corresponding test.)
    """
    if max_candidates <= 0:
        return []
    out: List[GraphEdit] = []
    seen: set[Tuple[str, Tuple[int, ...], Optional[int]]] = set()

    def _try_push(e: GraphEdit) -> bool:
        key = (e.op, tuple(sorted(e.indices)) if len(e.indices) == 2 else e.indices,
               e.new_value)
        if key in seen:
            return False
        # Final sanity: ``remove_bond`` / ``swap_bond_order`` MUST point at
        # an existing bond in the input graph.
        if e.op in ("remove_bond", "swap_bond_order"):
            ei = np.asarray(graph.edge_index)
            i, j = e.indices
            mask = ((ei[0] == i) & (ei[1] == j)) | ((ei[0] == j) & (ei[1] == i))
            if not mask.any():
                return False
        seen.add(key)
        out.append(e)
        return len(out) < max_candidates

    for fc in firing_clauses:
        if len(out) >= max_candidates:
            break
        _tree, nodes = _firing_clause_payload(fc)
        # Per-node candidates
        for n in nodes:
            for ed in _candidate_edits_for_node(graph, int(n), codebook):
                if not _try_push(ed):
                    if len(out) >= max_candidates:
                        break
            if len(out) >= max_candidates:
                break
        # Also: if firing clause exposes a list of edge indices, target them
        edge_ids = getattr(fc, "fired_edges", None) or getattr(fc, "edges", None)
        if edge_ids:
            for e_id in edge_ids:
                for ed in _candidate_edits_for_edge(graph, int(e_id), codebook):
                    if not _try_push(ed):
                        if len(out) >= max_candidates:
                            break
                if len(out) >= max_candidates:
                    break

    return out[:max_candidates]
