"""Tests for graphtm.recourse.candidates.

Covers:
  - GraphEdit dataclass validation (ops, arity, new_value).
  - apply_edit round-trip on benzene → methylbenzene-like ops.
  - candidates_from_firing_clauses bounding (≤ max_candidates) and the
    no-edit-on-nonexistent-bond invariant.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Sequence

import numpy as np
import pytest

from graphtm.recourse import GraphEdit, apply_edit, candidates_from_firing_clauses
from graphtm.recourse.candidates import (
    VALID_OPS,
    _parse_literal,
)

# RDKit fixtures (skip whole module if missing)
rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers, sister-module surrogates (M1 / M3 not built yet here)
# ---------------------------------------------------------------------------

@dataclass
class _MockCodebook:
    """Codebook mock with the same .atom_hv/.bond_hv/.role_hv attributes."""
    atom_hv: np.ndarray
    bond_hv: np.ndarray
    role_hv: np.ndarray


def _make_mock_codebook(n_atoms: int = 9, n_bonds: int = 4, D: int = 64,
                         seed: int = 0) -> _MockCodebook:
    rng = np.random.default_rng(seed)
    return _MockCodebook(
        atom_hv=rng.integers(0, 2, size=(n_atoms, D), dtype=np.uint8),
        bond_hv=rng.integers(0, 2, size=(n_bonds, D), dtype=np.uint8),
        role_hv=rng.integers(0, 2, size=(3, D), dtype=np.uint8),
    )


@dataclass
class _MockGraph:
    """Surrogate GraphTensor with the same attribute surface area."""
    n_nodes: int
    atom_type: np.ndarray
    edge_index: np.ndarray
    bond_type: np.ndarray
    node_hv: np.ndarray
    edge_hv: np.ndarray


def _mock_graph_from_mol(mol, codebook: _MockCodebook) -> _MockGraph:
    """Encode an RDKit mol as a minimal GraphTensor surrogate.

    atom_type idx 0=C, 1=N, ...; bond_type idx 0..3 by RDKit order;
    HVs are looked up from the codebook. This is NOT a faithful M1
    encoder, it's only for testing M5 in isolation.
    """
    n = mol.GetNumAtoms()
    # Use a permissive symbol map; fall back to 0 (carbon) for unknowns
    atom_map = {"C": 0, "N": 1, "O": 2, "F": 3, "P": 4, "S": 5, "Cl": 6,
                "Br": 7, "I": 8}
    atom_type = np.zeros(n, dtype=np.int32)
    for i in range(n):
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        atom_type[i] = atom_map.get(sym, 0)
    # Directed edges (both directions for undirected graph)
    bonds = []
    bond_types: List[int] = []
    bt_map = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = bt_map.get(b.GetBondType(), 0)
        bonds.append((i, j))
        bonds.append((j, i))
        bond_types.append(bt)
        bond_types.append(bt)
    if bonds:
        edge_index = np.array(bonds, dtype=np.int32).T  # [2, E]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)
    bond_type = np.array(bond_types, dtype=np.int32)
    node_hv = codebook.atom_hv[atom_type]
    if edge_index.size:
        edge_hv = codebook.bond_hv[bond_type]
    else:
        edge_hv = np.zeros((0, codebook.atom_hv.shape[1]), dtype=np.uint8)
    return _MockGraph(
        n_nodes=n,
        atom_type=atom_type,
        edge_index=edge_index,
        bond_type=bond_type,
        node_hv=node_hv,
        edge_hv=edge_hv,
    )


def _make_firing_clause(nodes=(), edges=(), tree=None):
    """Construct a duck-typed FiringClause surrogate."""
    return SimpleNamespace(fired_nodes=list(nodes),
                            fired_edges=list(edges),
                            tree=tree)


# ---------------------------------------------------------------------------
# GraphEdit validation
# ---------------------------------------------------------------------------

class TestGraphEdit:
    def test_remove_bond_two_indices(self):
        e = GraphEdit(op="remove_bond", indices=(0, 1))
        assert e.op == "remove_bond"
        assert e.indices == (0, 1)
        assert e.new_value is None

    def test_add_bond_requires_new_value(self):
        with pytest.raises(ValueError):
            GraphEdit(op="add_bond", indices=(0, 1))  # missing new_value

    def test_swap_atom_requires_one_index(self):
        with pytest.raises(ValueError):
            GraphEdit(op="swap_atom", indices=(0, 1), new_value=1)
        ok = GraphEdit(op="swap_atom", indices=(0,), new_value=1)
        assert ok.new_value == 1

    def test_distinct_atoms_required(self):
        with pytest.raises(ValueError):
            GraphEdit(op="remove_bond", indices=(0, 0))

    def test_rejects_unknown_op(self):
        with pytest.raises(ValueError):
            GraphEdit(op="explode_atom", indices=(0,), new_value=1)

    def test_valid_ops_constant(self):
        # Required by the architecture contract, frozen set of four ops
        assert set(VALID_OPS) == {
            "remove_bond", "add_bond", "swap_atom", "swap_bond_order",
        }

    def test_hashable(self):
        e1 = GraphEdit(op="remove_bond", indices=(0, 1))
        e2 = GraphEdit(op="remove_bond", indices=(0, 1))
        s = {e1, e2}
        assert len(s) == 1


# ---------------------------------------------------------------------------
# apply_edit
# ---------------------------------------------------------------------------

class TestApplyEdit:
    def test_remove_bond_breaks_chain(self):
        """Propane (CCC) -> ethane + methane after removing one C–C.

        I avoid aromatic rings here because removing one aromatic bond
        without re-kekulization will violate RDKit's aromaticity model
        (see `test_remove_bond_in_aromatic_ring_raises` below).
        """
        mol = Chem.MolFromSmiles("CCC")
        b = mol.GetBondBetweenAtoms(0, 1)
        assert b is not None
        edit = GraphEdit(op="remove_bond", indices=(0, 1))
        new = apply_edit(mol, edit)
        assert new is not None
        # The original bond should be gone
        assert new.GetBondBetweenAtoms(0, 1) is None
        # Same atom count
        assert new.GetNumAtoms() == mol.GetNumAtoms()
        # New mol re-sanitizes and round-trips to SMILES
        smi = Chem.MolToSmiles(new)
        assert smi  # non-empty

    def test_remove_bond_in_aromatic_ring_raises(self):
        """Removing a single aromatic-ring bond should be caught at sanitize.

        This is a documented sharp edge: callers must validate the edit
        with `validity.validate` rather than rely on `apply_edit` to fix
        the aromaticity model. The recourse search treats the resulting
        SanitizeException as a candidate rejection.
        """
        mol = Chem.MolFromSmiles("c1ccccc1")
        edit = GraphEdit(op="remove_bond", indices=(0, 1))
        # Should raise, invalid kekulization
        with pytest.raises(Exception):
            apply_edit(mol, edit)

    def test_add_bond_to_separate_atoms(self):
        """Methane + methane stub: build CH4 and add a fake C-C as a separate test.

        I construct two methanes (Cs only) and add a single bond.
        """
        rw = Chem.RWMol()
        a = rw.AddAtom(Chem.Atom("C"))
        b = rw.AddAtom(Chem.Atom("C"))
        mol = rw.GetMol()
        # Now add a bond via apply_edit
        edit = GraphEdit(op="add_bond", indices=(a, b), new_value=0)  # single
        new = apply_edit(mol, edit)
        bond = new.GetBondBetweenAtoms(a, b)
        assert bond is not None
        assert bond.GetBondType() == Chem.BondType.SINGLE

    def test_swap_atom_changes_element(self):
        """CH4 -> NH3 via swap_atom of atom 0."""
        # codebook ATOM_SYMBOLS: ("C","N","O","F","P","S","Cl","Br","I")
        # carbon idx 0 -> nitrogen idx 1
        mol = Chem.MolFromSmiles("C")
        edit = GraphEdit(op="swap_atom", indices=(0,), new_value=1)
        new = apply_edit(mol, edit)
        assert new.GetAtomWithIdx(0).GetSymbol() == "N"

    def test_swap_bond_order(self):
        """Ethane (CC) -> ethene (C=C) via swap_bond_order."""
        mol = Chem.MolFromSmiles("CC")
        edit = GraphEdit(op="swap_bond_order", indices=(0, 1), new_value=1)
        new = apply_edit(mol, edit)
        bond = new.GetBondBetweenAtoms(0, 1)
        assert bond is not None
        assert bond.GetBondType() == Chem.BondType.DOUBLE

    def test_remove_nonexistent_bond_raises(self):
        mol = Chem.MolFromSmiles("CC")
        # No bond between atoms 0 and 0 (caught by GraphEdit.__post_init__),
        # but a non-existent (0,1) on a single atom mol should KeyError.
        single = Chem.MolFromSmiles("C")  # only atom 0
        # Use a "valid" indices-pair the dataclass accepts but no bond exists
        # for. Atom 1 doesn't exist; expect a KeyError or ValueError from RDKit.
        with pytest.raises(Exception):
            apply_edit(single, GraphEdit(op="remove_bond", indices=(0, 5)))


# ---------------------------------------------------------------------------
# candidates_from_firing_clauses
# ---------------------------------------------------------------------------

class TestCandidates:
    def setup_method(self):
        self.cb = _make_mock_codebook()
        self.mol = Chem.MolFromSmiles("CCO")  # ethanol, 3 heavy atoms
        self.graph = _mock_graph_from_mol(self.mol, self.cb)

    def test_returns_at_most_max_candidates(self):
        # Firing every atom should produce way more candidates than 3 if
        # I don't cap. Verify the cap is enforced.
        fc = [_make_firing_clause(nodes=[0, 1, 2])]
        out = candidates_from_firing_clauses(self.graph, fc, self.cb,
                                              max_candidates=3)
        assert len(out) <= 3

    def test_empty_firing_returns_empty(self):
        out = candidates_from_firing_clauses(self.graph, [], self.cb,
                                              max_candidates=50)
        assert out == []

    def test_max_candidates_zero(self):
        fc = [_make_firing_clause(nodes=[0, 1, 2])]
        out = candidates_from_firing_clauses(self.graph, fc, self.cb,
                                              max_candidates=0)
        assert out == []

    def test_never_emits_edit_on_nonexistent_bond(self):
        # Construct a firing clause that "tries" to fire on a non-existent
        # node index. The candidate generator must skip those.
        fc = [_make_firing_clause(nodes=[5, 99])]  # nodes 5/99 don't exist
        out = candidates_from_firing_clauses(self.graph, fc, self.cb,
                                              max_candidates=50)
        # Edits should reference existing atoms only
        n = self.graph.n_nodes
        for e in out:
            for idx in e.indices:
                assert 0 <= idx < n, f"edit {e} touches non-existent atom {idx}"
            # remove_bond / swap_bond_order must point at an existing bond
            if e.op in ("remove_bond", "swap_bond_order"):
                i, j = e.indices
                bond_found = False
                ei = self.graph.edge_index
                for k in range(ei.shape[1]):
                    if (ei[0, k] == i and ei[1, k] == j) or (
                        ei[0, k] == j and ei[1, k] == i
                    ):
                        bond_found = True
                        break
                assert bond_found, f"{e} references non-existent bond"

    def test_remove_bond_edits_exist_in_graph(self):
        # Fire on atom 1 (the central C) which has bonds to 0 and 2.
        fc = [_make_firing_clause(nodes=[1])]
        out = candidates_from_firing_clauses(self.graph, fc, self.cb,
                                              max_candidates=50)
        remove_ops = [e for e in out if e.op == "remove_bond"]
        # At least the (0,1) and (1,2) bonds must be candidates
        keys = {tuple(sorted(e.indices)) for e in remove_ops}
        assert (0, 1) in keys
        assert (1, 2) in keys

    def test_swap_atom_candidate_uses_codebook_alphabet(self):
        fc = [_make_firing_clause(nodes=[0])]
        out = candidates_from_firing_clauses(self.graph, fc, self.cb,
                                              max_candidates=50)
        swap_ops = [e for e in out if e.op == "swap_atom"]
        assert len(swap_ops) >= 1
        for e in swap_ops:
            assert 0 <= e.new_value < self.cb.atom_hv.shape[0]
            # new_value must differ from the current atom type
            assert e.new_value != int(self.graph.atom_type[e.indices[0]])

    def test_dedup_across_firing_clauses(self):
        # Two clauses firing on the same node should not produce 2x edits
        fc = [_make_firing_clause(nodes=[0]), _make_firing_clause(nodes=[0])]
        out = candidates_from_firing_clauses(self.graph, fc, self.cb,
                                              max_candidates=50)
        keys = {(e.op, tuple(sorted(e.indices)), e.new_value) for e in out}
        assert len(keys) == len(out), "duplicate edits emitted"


class TestLiteralParser:
    def test_positive_literal(self):
        idx, pol = _parse_literal("X12=1")
        assert idx == 12
        assert pol == 1

    def test_negated_literal(self):
        idx, pol = _parse_literal("~X3=1")
        assert idx == 3
        assert pol == 0

    def test_malformed_literal_raises(self):
        with pytest.raises(ValueError):
            _parse_literal("Y3=1")
        with pytest.raises(ValueError):
            _parse_literal("garbage")
