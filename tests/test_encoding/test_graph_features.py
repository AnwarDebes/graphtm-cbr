"""Unit tests for `encode_graph` and `GraphTensor`.

Uses both the duck-typed (PyG-style) input path so tests run without
RDKit, plus an opt-in RDKit branch that exercises the chemistry adaptor.
"""
from __future__ import annotations

import numpy as np
import pytest

from graphtm.encoding.codebook import make_codebook
from graphtm.encoding.graph_features import GraphTensor, encode_graph
from graphtm.encoding.hypervectors import xor_bind


# ---------------------------------------------------------------------------
# Duck-typed graph helpers
# ---------------------------------------------------------------------------

def _two_atom_graph() -> dict:
    # C-C single bond
    return {"atom_types": ["C", "C"], "edges": [(0, 1, "single")]}


def _ethanol_graph() -> dict:
    # C-C-O  (single, single)
    return {"atom_types": ["C", "C", "O"],
            "edges": [(0, 1, "single"), (1, 2, "single")]}


def _acetaldehyde_graph() -> dict:
    # C-C=O  (single, double), non-isomorphic w.r.t. bond types vs ethanol
    return {"atom_types": ["C", "C", "O"],
            "edges": [(0, 1, "single"), (1, 2, "double")]}


# ---------------------------------------------------------------------------
# Shape & dtype contract
# ---------------------------------------------------------------------------

def test_encode_graph_returns_graph_tensor():
    cb = make_codebook(seed=0)
    gt = encode_graph(_ethanol_graph(), cb, k_hop=2)
    assert isinstance(gt, GraphTensor)
    assert gt.n_nodes == 3
    assert gt.atom_type.shape == (3,)
    assert gt.atom_type.dtype == np.int32
    # 2 undirected edges -> 4 directed rows
    assert gt.edge_index.shape == (2, 4)
    assert gt.edge_index.dtype == np.int32
    assert gt.bond_type.shape == (4,)
    assert gt.bond_type.dtype == np.int32
    assert gt.node_hv.shape == (3, cb.D)
    assert gt.node_hv.dtype == np.uint8
    assert gt.edge_hv.shape == (4, cb.D)
    assert gt.edge_hv.dtype == np.uint8


def test_encode_graph_hvs_are_binary():
    cb = make_codebook(seed=0)
    gt = encode_graph(_ethanol_graph(), cb)
    assert set(np.unique(gt.node_hv).tolist()) <= {0, 1}
    assert set(np.unique(gt.edge_hv).tolist()) <= {0, 1}


# ---------------------------------------------------------------------------
# Algebraic / topology properties
# ---------------------------------------------------------------------------

def test_edge_hv_equals_atom_xor_bond_xor_atom():
    """Forward-direction edge HV should be exactly atom(u) ⊕ bond(b) ⊕ atom(v)."""
    cb = make_codebook(seed=0)
    g = _ethanol_graph()
    gt = encode_graph(g, cb, k_hop=2)

    # Row 0 = first edge in forward direction
    u = int(gt.edge_index[0, 0])
    v = int(gt.edge_index[1, 0])
    b = int(gt.bond_type[0])
    expected = xor_bind(
        xor_bind(cb.atom_hv[gt.atom_type[u]], cb.bond_hv[b]),
        cb.atom_hv[gt.atom_type[v]],
    )
    assert np.array_equal(gt.edge_hv[0], expected)


def test_edge_orientation_preserved_in_edge_index():
    """Both directions present; row 2k is (u,v), row 2k+1 is (v,u)."""
    cb = make_codebook(seed=0)
    gt = encode_graph(_ethanol_graph(), cb, k_hop=2)
    for k in range(gt.edge_index.shape[1] // 2):
        assert gt.edge_index[0, 2 * k] == gt.edge_index[1, 2 * k + 1]
        assert gt.edge_index[1, 2 * k] == gt.edge_index[0, 2 * k + 1]
        assert gt.bond_type[2 * k] == gt.bond_type[2 * k + 1]


def test_edge_hv_xor_self_inverse_recovers_endpoints():
    """edge_hv ⊕ atom(u) ⊕ bond(b) = atom(v), XOR is invertible (BSC)."""
    cb = make_codebook(seed=0)
    gt = encode_graph(_ethanol_graph(), cb)
    for k in range(gt.edge_index.shape[1]):
        u = int(gt.edge_index[0, k])
        v = int(gt.edge_index[1, k])
        b = int(gt.bond_type[k])
        recovered = xor_bind(
            xor_bind(gt.edge_hv[k], cb.atom_hv[gt.atom_type[u]]),
            cb.bond_hv[b],
        )
        assert np.array_equal(recovered, cb.atom_hv[gt.atom_type[v]])


def test_non_isomorphic_graphs_encode_to_distinct_tensors():
    """Ethanol vs acetaldehyde differ in one bond type; Hamming should be far above 0."""
    cb = make_codebook(seed=0)
    g1 = encode_graph(_ethanol_graph(), cb)
    g2 = encode_graph(_acetaldehyde_graph(), cb)
    # Compare per-node HVs of node 2 (the O atom, its incident bond differs)
    flip_count = np.sum(g1.node_hv[2] != g2.node_hv[2])
    # With 10% sparse atoms in D=8192, replacing one neighbour's bond changes
    # several hundred bits in the bundled neighbourhood HV, well above the
    # "different" floor of ~zero.
    assert flip_count > 50, f"node HVs too close: only {flip_count} bits differ"
    # Edge HVs must also disagree on the C=O / C-O bond
    # Find the edge with v=2 (the O atom)
    differ = False
    for k in range(g1.edge_hv.shape[0]):
        if g1.bond_type[k] != g2.bond_type[k]:
            differ = True
            assert not np.array_equal(g1.edge_hv[k], g2.edge_hv[k])
    assert differ, "test graphs do not actually have different bond types"


def test_two_isomorphic_graphs_encode_identically():
    """Same SMILES topology -> identical tensors with the same codebook."""
    cb = make_codebook(seed=99)
    g1 = encode_graph(_ethanol_graph(), cb)
    g2 = encode_graph(_ethanol_graph(), cb)
    assert np.array_equal(g1.node_hv, g2.node_hv)
    assert np.array_equal(g1.edge_hv, g2.edge_hv)
    assert np.array_equal(g1.atom_type, g2.atom_type)
    assert np.array_equal(g1.edge_index, g2.edge_index)
    assert np.array_equal(g1.bond_type, g2.bond_type)


def test_node_hv_k_hop_zero_is_just_self_atom_hv():
    """With k_hop=0 no neighbours are bundled, node HV == atom HV."""
    cb = make_codebook(seed=0)
    gt = encode_graph(_ethanol_graph(), cb, k_hop=0)
    for v in range(gt.n_nodes):
        assert np.array_equal(gt.node_hv[v], cb.atom_hv[gt.atom_type[v]])


def test_node_hv_changes_with_neighbourhood():
    """k_hop>0 must change the per-node HV from the bare atom HV."""
    cb = make_codebook(seed=0)
    gt = encode_graph(_ethanol_graph(), cb, k_hop=2)
    # At least one node should differ from its bare atom HV
    different = False
    for v in range(gt.n_nodes):
        if not np.array_equal(gt.node_hv[v], cb.atom_hv[gt.atom_type[v]]):
            different = True
            break
    assert different, "k_hop>0 produced identical node_hv to atom_hv"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_encode_graph_deterministic_per_codebook():
    cb1 = make_codebook(seed=11)
    cb2 = make_codebook(seed=11)
    g = _ethanol_graph()
    assert np.array_equal(encode_graph(g, cb1).node_hv,
                          encode_graph(g, cb2).node_hv)
    assert np.array_equal(encode_graph(g, cb1).edge_hv,
                          encode_graph(g, cb2).edge_hv)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_encode_graph_rejects_negative_k_hop():
    cb = make_codebook(seed=0)
    with pytest.raises(ValueError):
        encode_graph(_two_atom_graph(), cb, k_hop=-1)


def test_encode_graph_rejects_k_hop_above_codebook():
    cb = make_codebook(k_hop=2, seed=0)
    with pytest.raises(ValueError):
        encode_graph(_two_atom_graph(), cb, k_hop=3)


def test_encode_graph_rejects_unknown_object():
    cb = make_codebook(seed=0)
    with pytest.raises(TypeError):
        encode_graph(object(), cb)


def test_encode_graph_rejects_empty_atom_list():
    cb = make_codebook(seed=0)
    with pytest.raises(ValueError):
        encode_graph({"atom_types": [], "edges": []}, cb)


def test_encode_graph_rejects_out_of_range_edge_endpoint():
    cb = make_codebook(seed=0)
    bad = {"atom_types": ["C"], "edges": [(0, 5, "single")]}
    with pytest.raises(ValueError):
        encode_graph(bad, cb)


def test_encode_graph_rejects_unknown_bond_string():
    cb = make_codebook(seed=0)
    bad = {"atom_types": ["C", "C"], "edges": [(0, 1, "quintuple")]}
    with pytest.raises(ValueError):
        encode_graph(bad, cb)


# ---------------------------------------------------------------------------
# Optional RDKit branch, only runs if rdkit is installed
# ---------------------------------------------------------------------------

def test_encode_graph_rdkit_path_matches_duck_typed_path():
    """If RDKit is available, parsing ethanol via SMILES should encode the
    same atoms/bonds as the duck-typed dict (modulo atom order, which RDKit
    keeps as written in SMILES)."""
    Chem = pytest.importorskip("rdkit.Chem")
    cb = make_codebook(seed=0)
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    gt_rdkit = encode_graph(mol, cb, k_hop=2)
    gt_dict = encode_graph(_ethanol_graph(), cb, k_hop=2)
    # Atom-type vector should match (C, C, O)
    assert np.array_equal(gt_rdkit.atom_type, gt_dict.atom_type)
    # Bond-type vector should match (both single bonds)
    assert np.array_equal(np.sort(gt_rdkit.bond_type),
                          np.sort(gt_dict.bond_type))
    # Node HVs must match exactly (deterministic codebook + same topology)
    assert np.array_equal(gt_rdkit.node_hv, gt_dict.node_hv)
