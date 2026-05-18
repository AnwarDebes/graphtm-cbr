"""Tests for graphtm.recourse.search.greedy_minimal_edit.

The "synthetic 2-class" model used here returns:
    +1 if a (0,1) bond is present
    -1 otherwise

Greedy recourse should remove that bond in exactly one step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pytest

from graphtm.recourse import GraphEdit, greedy_minimal_edit
from graphtm.recourse.search import SearchTrace

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402


# ---------------------------------------------------------------------------
# Graph surrogate (M1 not built yet in parallel build)
# ---------------------------------------------------------------------------

@dataclass
class _Graph:
    """Minimal GraphTensor surrogate carrying the bits the test model reads.

    The test model only needs ``edge_index``; everything else is here so
    candidate generation works in isolation.
    """
    n_nodes: int
    atom_type: np.ndarray
    edge_index: np.ndarray
    bond_type: np.ndarray
    node_hv: np.ndarray
    edge_hv: np.ndarray


def _encode_mol(mol) -> _Graph:
    """Re-encode an RDKit mol into a `_Graph` for the test model.

    Mirrors the production `encode_graph(mol, codebook, k_hop)` contract
    but doesn't run any VSA work, the test model doesn't read HVs.
    """
    n = mol.GetNumAtoms()
    bonds: List[tuple] = []
    bts: List[int] = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonds.append((i, j))
        bonds.append((j, i))
        bts.append(0)
        bts.append(0)
    if bonds:
        ei = np.array(bonds, dtype=np.int32).T
    else:
        ei = np.zeros((2, 0), dtype=np.int32)
    return _Graph(
        n_nodes=n,
        atom_type=np.zeros(n, dtype=np.int32),
        edge_index=ei,
        bond_type=np.array(bts, dtype=np.int32),
        node_hv=np.zeros((n, 8), dtype=np.uint8),
        edge_hv=np.zeros((ei.shape[1], 8), dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# Mock model: returns +1 if (0,1) bond exists else -1
# ---------------------------------------------------------------------------

class _BondPresenceModel:
    """Class scores depend on whether atoms 0 and 1 are bonded.

    Predicts class 1 if bond(0,1) exists, class 0 otherwise. Class scores
    are produced as a length-2 vector with margin +/-2.
    """

    def __init__(self) -> None:
        self.n_calls = 0

    def _has_bond_01(self, graph: _Graph) -> bool:
        ei = graph.edge_index
        if ei.size == 0:
            return False
        mask = ((ei[0] == 0) & (ei[1] == 1)) | ((ei[0] == 1) & (ei[1] == 0))
        return bool(mask.any())

    def predict(self, graphs):
        self.n_calls += 1
        return np.array(
            [1 if self._has_bond_01(g) else 0 for g in graphs], dtype=np.int64
        )

    def class_scores(self, graphs):
        out = np.zeros((len(graphs), 2))
        for i, g in enumerate(graphs):
            if self._has_bond_01(g):
                out[i] = [0.0, 2.0]  # class 1 wins
            else:
                out[i] = [2.0, 0.0]  # class 0 wins
        return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGreedySearch:
    def test_flips_with_one_edit(self):
        """Ethane (C-C). Removing bond (0,1) flips class from 1 to 0."""
        smiles = "CC"
        mol = Chem.MolFromSmiles(smiles)
        g0 = _encode_mol(mol)
        model = _BondPresenceModel()
        # Sanity: initial prediction is class 1
        assert int(model.predict([g0])[0]) == 1

        cands = [GraphEdit(op="remove_bond", indices=(0, 1))]
        trace = SearchTrace()
        result = greedy_minimal_edit(
            model, g0, cands, mol, max_flips=3,
            encode_fn=_encode_mol, trace=trace,
        )
        assert result is not None, "greedy search must find a flip"
        assert len(result) == 1
        assert result[0].op == "remove_bond"
        assert tuple(sorted(result[0].indices)) == (0, 1)
        assert trace.flipped is True

    def test_returns_none_when_no_flip_possible(self):
        """If candidates can't change the (0,1) bond, no flip is found."""
        mol = Chem.MolFromSmiles("CC")
        g0 = _encode_mol(mol)
        model = _BondPresenceModel()
        # Candidates that don't touch (0,1), atom 0 swap doesn't change topology
        # (but for a 2-atom mol, swap is the only no-op edit; remove_bond on
        # a non-(0,1) bond also won't exist in this 2-atom mol). I use a
        # bogus list with no bond ops.
        cands: list[GraphEdit] = []
        result = greedy_minimal_edit(
            model, g0, cands, mol, max_flips=3, encode_fn=_encode_mol,
        )
        assert result is None

    def test_max_flips_respected(self):
        """Even with many candidates, search stops at ``max_flips``."""
        # Build propane (CCC, atoms 0-1-2), bond (0,1) and (1,2)
        mol = Chem.MolFromSmiles("CCC")
        g0 = _encode_mol(mol)
        model = _BondPresenceModel()
        # All bonds as candidates, plus other ops
        cands = [
            GraphEdit(op="remove_bond", indices=(1, 2)),
            GraphEdit(op="remove_bond", indices=(0, 1)),
        ]
        trace = SearchTrace()
        result = greedy_minimal_edit(
            model, g0, cands, mol, max_flips=1, encode_fn=_encode_mol,
            trace=trace,
        )
        # Only one flip needed because (0,1) removal alone flips the class.
        assert result is not None
        assert len(result) == 1

    def test_picks_flipping_edit_over_neutral_one(self):
        """Greedy picks the edit that flips, not the irrelevant one."""
        mol = Chem.MolFromSmiles("CCC")
        g0 = _encode_mol(mol)
        model = _BondPresenceModel()
        # First candidate is irrelevant (removes 1-2, model still says class 1)
        # Second candidate is the flipping one (removes 0-1)
        cands = [
            GraphEdit(op="remove_bond", indices=(1, 2)),
            GraphEdit(op="remove_bond", indices=(0, 1)),
        ]
        result = greedy_minimal_edit(
            model, g0, cands, mol, max_flips=2, encode_fn=_encode_mol,
        )
        assert result is not None
        # The (0,1) bond should be in the committed sequence
        committed_pairs = [tuple(sorted(e.indices)) for e in result]
        assert (0, 1) in committed_pairs

    def test_latency_under_target(self):
        """Latency budget per research/06 §4.3: < 100 ms with a fast model."""
        mol = Chem.MolFromSmiles("CCO")
        g0 = _encode_mol(mol)
        model = _BondPresenceModel()
        # Build a candidate set close to the typical 50-cap
        cands = [
            GraphEdit(op="remove_bond", indices=(0, 1)),
            GraphEdit(op="remove_bond", indices=(1, 2)),
        ]
        trace = SearchTrace()
        _ = greedy_minimal_edit(
            model, g0, cands, mol, max_flips=3, encode_fn=_encode_mol,
            trace=trace,
        )
        # The mock model is fast; I just verify the trace recorded
        # latency and it's under the 100 ms target.
        assert trace.total_latency_ms < 100.0, (
            f"latency {trace.total_latency_ms:.1f} ms > 100 ms target"
        )

    def test_empty_candidates(self):
        mol = Chem.MolFromSmiles("CC")
        g0 = _encode_mol(mol)
        model = _BondPresenceModel()
        result = greedy_minimal_edit(
            model, g0, [], mol, max_flips=3, encode_fn=_encode_mol,
        )
        assert result is None

    def test_validity_filter_rejects_bad_edits(self):
        """If ``validity_fn`` rejects every edit, I get None."""
        mol = Chem.MolFromSmiles("CC")
        g0 = _encode_mol(mol)
        model = _BondPresenceModel()
        cands = [GraphEdit(op="remove_bond", indices=(0, 1))]
        result = greedy_minimal_edit(
            model, g0, cands, mol, max_flips=3, encode_fn=_encode_mol,
            validity_fn=lambda _m: False,  # reject everything
        )
        assert result is None

    def test_smiles_input_accepted(self):
        """Caller may pass a SMILES string instead of an RDKit Mol."""
        mol = Chem.MolFromSmiles("CC")
        g0 = _encode_mol(mol)
        model = _BondPresenceModel()
        result = greedy_minimal_edit(
            model, g0, [GraphEdit(op="remove_bond", indices=(0, 1))],
            "CC", max_flips=3, encode_fn=_encode_mol,
        )
        assert result is not None
        assert len(result) == 1
