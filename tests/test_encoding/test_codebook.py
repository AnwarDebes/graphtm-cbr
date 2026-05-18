"""Unit tests for the atom/bond/role codebook (`make_codebook`, `AtomBondCodebook`)."""
from __future__ import annotations

import numpy as np
import pytest

from graphtm.encoding.codebook import (
    ATOM_SYMBOLS,
    BOND_NAMES,
    AtomBondCodebook,
    make_codebook,
)


def test_make_codebook_default_shapes():
    cb = make_codebook()
    assert cb.atom_hv.shape == (9, 8192)
    assert cb.bond_hv.shape == (4, 8192)
    assert cb.role_hv.shape == (3, 8192)   # k_hop=2 -> 3 role HVs
    assert cb.D == 8192
    assert cb.sparsity == 0.10
    assert cb.atom_hv.dtype == np.uint8
    assert cb.bond_hv.dtype == np.uint8
    assert cb.role_hv.dtype == np.uint8


def test_make_codebook_deterministic_per_seed():
    a = make_codebook(seed=42)
    b = make_codebook(seed=42)
    assert np.array_equal(a.atom_hv, b.atom_hv)
    assert np.array_equal(a.bond_hv, b.bond_hv)
    assert np.array_equal(a.role_hv, b.role_hv)


def test_make_codebook_distinct_seeds_distinct():
    a = make_codebook(seed=1)
    b = make_codebook(seed=2)
    assert not np.array_equal(a.atom_hv, b.atom_hv)
    assert not np.array_equal(a.bond_hv, b.bond_hv)
    assert not np.array_equal(a.role_hv, b.role_hv)


def test_codebook_atoms_are_sparse_at_target_density():
    cb = make_codebook(seed=0)
    for row in cb.atom_hv:
        density = row.sum() / cb.D
        assert 0.095 <= density <= 0.105


def test_codebook_bonds_are_sparse_at_target_density():
    cb = make_codebook(seed=0)
    for row in cb.bond_hv:
        density = row.sum() / cb.D
        assert 0.095 <= density <= 0.105


def test_codebook_roles_are_sparse_at_target_density():
    cb = make_codebook(seed=0)
    for row in cb.role_hv:
        density = row.sum() / cb.D
        assert 0.095 <= density <= 0.105


def test_codebook_atoms_are_distinct_rows():
    cb = make_codebook(seed=0)
    for i in range(cb.atom_hv.shape[0]):
        for j in range(i + 1, cb.atom_hv.shape[0]):
            assert not np.array_equal(cb.atom_hv[i], cb.atom_hv[j])


def test_codebook_bonds_are_distinct_rows():
    cb = make_codebook(seed=0)
    for i in range(cb.bond_hv.shape[0]):
        for j in range(i + 1, cb.bond_hv.shape[0]):
            assert not np.array_equal(cb.bond_hv[i], cb.bond_hv[j])


def test_codebook_custom_sizes():
    cb = make_codebook(n_atom_types=5, n_bond_types=2, k_hop=3, D=1024,
                       sparsity=0.05, seed=7)
    assert cb.atom_hv.shape == (5, 1024)
    assert cb.bond_hv.shape == (2, 1024)
    assert cb.role_hv.shape == (4, 1024)
    assert cb.sparsity == 0.05
    for row in cb.atom_hv:
        # exact-density sampling -> within 1 bit of target
        assert abs(row.sum() / 1024 - 0.05) < 1.0 / 1024 + 1e-6


def test_canonical_alphabet_labels():
    """The library-wide symbol order must match the codebook row order."""
    assert ATOM_SYMBOLS == ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
    assert BOND_NAMES == ("single", "double", "triple", "aromatic")
    cb = make_codebook(seed=0)
    assert cb.atom_hv.shape[0] == len(ATOM_SYMBOLS)
    assert cb.bond_hv.shape[0] == len(BOND_NAMES)


def test_codebook_post_init_rejects_bad_shape():
    rng = np.random.default_rng(0)
    bad_atom = np.zeros((3, 64), dtype=np.uint8)  # wrong n_atom_types
    bond = np.zeros((4, 64), dtype=np.uint8)
    role = np.zeros((3, 64), dtype=np.uint8)
    with pytest.raises(ValueError):
        AtomBondCodebook(
            atom_hv=bad_atom, bond_hv=bond, role_hv=role,
            D=64, sparsity=0.1, n_atom_types=9, n_bond_types=4, k_hop=2,
        )


def test_codebook_post_init_rejects_wrong_dtype():
    atom = np.zeros((9, 64), dtype=np.uint32)
    bond = np.zeros((4, 64), dtype=np.uint8)
    role = np.zeros((3, 64), dtype=np.uint8)
    with pytest.raises(TypeError):
        AtomBondCodebook(
            atom_hv=atom, bond_hv=bond, role_hv=role,
            D=64, sparsity=0.1, n_atom_types=9, n_bond_types=4, k_hop=2,
        )


def test_make_codebook_invalid_args():
    with pytest.raises(ValueError):
        make_codebook(n_atom_types=0)
    with pytest.raises(ValueError):
        make_codebook(n_bond_types=0)
    with pytest.raises(ValueError):
        make_codebook(k_hop=-1)
