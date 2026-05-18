"""Unit tests for M1 hypervector primitives (`xor_bind`, `majority_bundle`, `sparse_bsc`).

Covers the frozen-interface contract from docs/ARCHITECTURE.md plus the
expected algebraic properties (XOR self-inverse, BSC sparsity, etc.).
"""
from __future__ import annotations

import numpy as np
import pytest

from graphtm.encoding.hypervectors import (
    majority_bundle,
    sparse_bsc,
    xor_bind,
)


# ---------------------------------------------------------------------------
# sparse_bsc
# ---------------------------------------------------------------------------

def test_sparse_bsc_density_within_tolerance():
    """10% sparsity must be within +/- 0.5% at D=8192."""
    D = 8192
    rng = np.random.default_rng(123)
    hv = sparse_bsc(D, 0.10, rng)
    density = hv.sum() / D
    assert 0.095 <= density <= 0.105, f"density {density} outside 10±0.5%"


def test_sparse_bsc_dtype_and_shape():
    rng = np.random.default_rng(0)
    hv = sparse_bsc(8192, 0.10, rng)
    assert hv.dtype == np.uint8
    assert hv.shape == (8192,)
    assert set(np.unique(hv).tolist()) <= {0, 1}


def test_sparse_bsc_deterministic_for_same_seed():
    a = sparse_bsc(8192, 0.10, np.random.default_rng(42))
    b = sparse_bsc(8192, 0.10, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_sparse_bsc_distinct_seeds_distinct_vectors():
    a = sparse_bsc(8192, 0.10, np.random.default_rng(1))
    b = sparse_bsc(8192, 0.10, np.random.default_rng(2))
    # Random sparse vectors should not be identical
    assert not np.array_equal(a, b)


def test_sparse_bsc_invalid_args():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sparse_bsc(0, 0.10, rng)
    with pytest.raises(ValueError):
        sparse_bsc(8192, 0.0, rng)
    with pytest.raises(ValueError):
        sparse_bsc(8192, 1.0, rng)


def test_sparse_bsc_multiple_sparsities_within_tolerance():
    """Density should track requested sparsity exactly to rounding."""
    D = 8192
    rng = np.random.default_rng(7)
    for s in (0.05, 0.10, 0.20):
        hv = sparse_bsc(D, s, rng)
        d = hv.sum() / D
        assert abs(d - s) < 1.0 / D + 1e-6, f"s={s} got density={d}"


# ---------------------------------------------------------------------------
# xor_bind
# ---------------------------------------------------------------------------

def test_xor_bind_is_self_inverse():
    rng = np.random.default_rng(0)
    a = sparse_bsc(8192, 0.10, rng)
    b = sparse_bsc(8192, 0.10, rng)
    ab = xor_bind(a, b)
    assert np.array_equal(xor_bind(ab, b), a)
    assert np.array_equal(xor_bind(ab, a), b)


def test_xor_bind_dtype_preserved():
    rng = np.random.default_rng(0)
    a = sparse_bsc(64, 0.10, rng)
    b = sparse_bsc(64, 0.10, rng)
    out = xor_bind(a, b)
    assert out.dtype == np.uint8
    assert out.shape == a.shape


def test_xor_bind_shape_mismatch_raises():
    a = np.zeros(8, dtype=np.uint8)
    b = np.zeros(16, dtype=np.uint8)
    with pytest.raises(ValueError):
        xor_bind(a, b)


def test_xor_bind_with_zero_is_identity():
    rng = np.random.default_rng(0)
    a = sparse_bsc(128, 0.10, rng)
    zero = np.zeros_like(a)
    assert np.array_equal(xor_bind(a, zero), a)


def test_xor_bind_commutative():
    rng = np.random.default_rng(2)
    a = sparse_bsc(256, 0.10, rng)
    b = sparse_bsc(256, 0.10, rng)
    assert np.array_equal(xor_bind(a, b), xor_bind(b, a))


# ---------------------------------------------------------------------------
# majority_bundle
# ---------------------------------------------------------------------------

def test_majority_bundle_simple_odd():
    stack = np.array(
        [[1, 1, 0, 0],
         [1, 0, 1, 0],
         [1, 0, 0, 1]],
        dtype=np.uint8,
    )
    # Per-column sums: 3, 1, 1, 1 -> > 1 is column 0 only
    out = majority_bundle(stack)
    assert out.dtype == np.uint8
    assert np.array_equal(out, np.array([1, 0, 0, 0], dtype=np.uint8))


def test_majority_bundle_tie_breaks_to_first_row():
    # Even number of rows -> ties resolved by row 0
    stack = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1]],
        dtype=np.uint8,
    )
    out = majority_bundle(stack)
    # All columns are ties (sum=1, n//2=1) -> take row 0
    assert np.array_equal(out, stack[0])


def test_majority_bundle_single_row_is_identity():
    rng = np.random.default_rng(1)
    a = sparse_bsc(128, 0.10, rng)
    stack = a[np.newaxis, :]
    out = majority_bundle(stack)
    assert np.array_equal(out, a)


def test_majority_bundle_unanimous_passes_through():
    stack = np.ones((5, 16), dtype=np.uint8)
    assert np.array_equal(majority_bundle(stack), np.ones(16, dtype=np.uint8))
    stack0 = np.zeros((5, 16), dtype=np.uint8)
    assert np.array_equal(majority_bundle(stack0), np.zeros(16, dtype=np.uint8))


def test_majority_bundle_rejects_empty_stack():
    with pytest.raises(ValueError):
        majority_bundle(np.zeros((0, 8), dtype=np.uint8))


def test_majority_bundle_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        majority_bundle(np.zeros(8, dtype=np.uint8))


def test_majority_bundle_dtype_uint8():
    stack = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=np.uint8)
    out = majority_bundle(stack)
    assert out.dtype == np.uint8
