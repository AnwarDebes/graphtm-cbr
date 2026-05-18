"""Binary hypervector helpers for HGTM graph encoding.

HGTM needs to encode graph context (node features + neighbor aggregates +
edge attributes) into a fixed-length Boolean vector that a flat
Tsetlin Machine can consume. Binary hypervectors give us a principled
way to do this with role-filler binding and permutation-based composition.

Operations:
  - bind(a, b) / xor_bind(a, b) : XOR (binding, invertible)
  - bundle(a, b, ...) / majority_bundle(stack) : majority over inputs
  - permute(a, k)               : circular shift by k (encodes role/hop)
  - sparse_bsc(d, sparsity, rng): 10%-sparse BSC hypervector (uint8 0/1)

These are the canonical VSA (Vector Symbolic Architectures) operations.
"""
from __future__ import annotations

import numpy as np


def random_hv(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random Boolean hypervector, uniform 0/1 in dim bits."""
    return rng.integers(0, 2, size=dim, dtype=np.uint32)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR-bind two hypervectors (their bound is dissimilar to either)."""
    return np.bitwise_xor(a, b)


def bundle(vectors: list[np.ndarray]) -> np.ndarray:
    """Majority-bundle hypervectors into a superposition.

    Each output bit is 1 iff the majority of inputs have it as 1.
    Ties broken by toggling, keeps the result balanced.
    """
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    arr = np.stack(vectors).astype(np.int32)
    sums = arr.sum(axis=0)
    n = arr.shape[0]
    out = (sums > n // 2).astype(np.uint32)
    # Handle ties: if n is even and sum == n//2, randomly break (here: parity bit)
    if n % 2 == 0:
        ties = sums == n // 2
        if ties.any():
            # Deterministic tie-break: take bit from first vector
            out[ties] = arr[0][ties]
    return out


# ---------------------------------------------------------------------------
# M1 frozen-interface ops (uint8 BSC variant, used by codebook + graph_features)
# ---------------------------------------------------------------------------

def xor_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR-bind two uint8 0/1 hypervectors element-wise (BSC binding)."""
    if a.shape != b.shape:
        raise ValueError(f"xor_bind shape mismatch: {a.shape} vs {b.shape}")
    # bitwise_xor on uint8 0/1 returns uint8 0/1 (preserves dtype)
    return np.bitwise_xor(a.astype(np.uint8, copy=False),
                          b.astype(np.uint8, copy=False))


def majority_bundle(stack: np.ndarray) -> np.ndarray:
    """Bit-wise majority of an [N, D] uint8 stack, returns uint8 0/1.

    Bit is 1 iff strictly more than half the inputs have it set. For even
    N, ties (== N/2) take the bit from the first row, which keeps the
    result balanced and deterministic across calls with the same input.
    """
    if stack.ndim != 2:
        raise ValueError(f"majority_bundle expects [N, D]; got shape {stack.shape}")
    if stack.shape[0] == 0:
        raise ValueError("majority_bundle requires at least one row")
    n = stack.shape[0]
    sums = stack.astype(np.int32).sum(axis=0)
    out = (sums > n // 2).astype(np.uint8)
    if n % 2 == 0:
        ties = sums == (n // 2)
        if ties.any():
            out[ties] = stack[0].astype(np.uint8)[ties]
    return out


def sparse_bsc(d: int, sparsity: float, rng: np.random.Generator) -> np.ndarray:
    """Sparse BSC atom: uint8 0/1 of length d with ~`sparsity` fraction of ones.

    Implementation samples exactly `round(sparsity * d)` distinct one-positions
    from rng so density is exact (not just in expectation), keeps tests stable.
    """
    if d <= 0:
        raise ValueError(f"sparse_bsc: d must be positive, got {d}")
    if not (0.0 < sparsity < 1.0):
        raise ValueError(f"sparse_bsc: sparsity must be in (0,1), got {sparsity}")
    k = int(round(sparsity * d))
    if k < 1:
        k = 1
    if k >= d:
        k = d - 1
    out = np.zeros(d, dtype=np.uint8)
    idx = rng.choice(d, size=k, replace=False)
    out[idx] = 1
    return out


def permute(v: np.ndarray, k: int = 1) -> np.ndarray:
    """Cyclic-shift permutation by k positions. Encodes role/position."""
    return np.roll(v, k)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Hamming similarity in [0, 1], 1 = identical, 0.5 = random."""
    if a.shape != b.shape:
        raise ValueError("hypervector shape mismatch")
    return 1.0 - float(np.bitwise_xor(a, b).sum()) / a.size


def thermometer_encode(value: float, n_bits: int, lo: float = 0.0,
                       hi: float = 1.0) -> np.ndarray:
    """Thermometer-encode a continuous value into n_bits.

    The Granmo-standard way to feed continuous features into a TM:
    bit i = 1 iff value > lo + i*(hi-lo)/n_bits.
    """
    if hi <= lo:
        raise ValueError(f"thermometer_encode: hi ({hi}) must exceed lo ({lo})")
    thresholds = np.linspace(lo, hi, n_bits + 1)[1:-1]
    bits = (value > thresholds).astype(np.uint32)
    # Append the upper-bound bit so length matches n_bits
    upper = np.array([1 if value > thresholds[-1] else 0], dtype=np.uint32)
    return np.concatenate([bits, upper]) if len(bits) < n_bits else bits[:n_bits]


def thermometer_encode_int(value: int, n_bits: int) -> np.ndarray:
    """Thermometer-encode a non-negative integer (capped at n_bits).

    bit i = 1 iff value > i.
    """
    out = np.zeros(n_bits, dtype=np.uint32)
    cap = min(value, n_bits)
    if cap > 0:
        out[:cap] = 1
    return out


def one_hot(idx: int, n: int) -> np.ndarray:
    """One-hot encoding, bit `idx` set, rest 0."""
    v = np.zeros(n, dtype=np.uint32)
    if 0 <= idx < n:
        v[idx] = 1
    return v
