"""graphtm/cuda/memory.py, GPU memory management for HGTM CUDA backend (M2).

CudaTAState owns the bit-plane uint32 storage for Tsetlin Automaton
state on device, and provides host↔device transfer helpers used by the
training loop in M3.

Layout (matches research/03 §5 and the M2 contract in docs/ARCHITECTURE.md):

    ta_state : uint32[CLAUSES, LA_CHUNKS, STATE_BITS]

where

    LITERALS_PER_CLAUSE = R * IA * IF * LA * 2 * LF
    LA_CHUNKS           = ceil(LITERALS_PER_CLAUSE / 32)
    STATE_BITS          = 8                 (= range [0, 255])

The "include action" bit for literal `lit` of clause `c` is the top
bit-plane bit:

    action = (ta_state[c, lit//32, STATE_BITS-1] >> (lit % 32)) & 1

The full state value of TA `lit` of clause `c` is reconstructed by OR-
ing the 8 bit-planes for that bit position (see `to_host_states`).

This file deliberately does NOT call into PyCUDA at import time, the
intent (per M2 spec) is that the module imports cleanly on a host
without a working CUDA toolchain, and only raises on first kernel
launch. I make a single best-effort lazy import in `_lazy_cuda`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# lazy PyCUDA import
_CUDA_LOAD_ERR: Optional[BaseException] = None
_cuda_driver = None


def _lazy_cuda():
    """Import pycuda.driver lazily so module import never fails."""
    global _cuda_driver, _CUDA_LOAD_ERR
    if _cuda_driver is not None:
        return _cuda_driver
    if _CUDA_LOAD_ERR is not None:
        raise RuntimeError(
            "PyCUDA is not importable on this host. Module M2 (graphtm.cuda) "
            "requires a working CUDA toolchain and `pycuda` package. "
            "Original import error: " + repr(_CUDA_LOAD_ERR)
        )
    try:
        import pycuda.driver as cuda  # type: ignore
        import pycuda.autoinit  # type: ignore  # noqa: F401
    except Exception as e:  # ImportError, RuntimeError (no GPU), etc.
        _CUDA_LOAD_ERR = e
        raise RuntimeError(
            "PyCUDA is not importable on this host. Module M2 (graphtm.cuda) "
            "requires a working CUDA toolchain and `pycuda` package. "
            f"Original import error: {e!r}"
        )
    _cuda_driver = cuda
    return cuda


# public types
@dataclass(frozen=True)
class TAStateShape:
    """Compile-time TA-state geometry. M3 wraps an HGraphTMSpec into this.

    Mirrors the canonical HTM tree dims from research/02 §1.
    """
    C: int           # CLAUSES
    R: int           # ROOT_FACTORS
    IA: int          # INTERIOR_ALTERNATIVES
    IF: int          # INTERIOR_FACTORS
    LA: int          # LEAF_ALTERNATIVES
    LF: int          # LEAF_FACTORS
    state_bits: int = 8

    @property
    def features(self) -> int:
        return self.R * self.IF * self.LF

    @property
    def literals_per_clause(self) -> int:
        return self.R * self.IA * self.IF * self.LA * 2 * self.LF

    @property
    def la_chunks(self) -> int:
        return (self.literals_per_clause + 31) // 32

    @property
    def total_uint32(self) -> int:
        return self.C * self.la_chunks * self.state_bits


class CudaTAState:
    """Owns device-side bit-plane TA state and host transfer helpers.

    Usage (typical):
        sh = TAStateShape(C=16, R=2, IA=2, IF=2, LA=10, LF=2)
        ta = CudaTAState(sh)
        ta.allocate()                  # GPU buffer
        ta.init_centre()               # all lower planes 1, top plane 0
        ta.copy_from_host(host_arr)    # optionally override (e.g. M3 init)
        ...
        host_arr = ta.copy_to_host()
        actions  = ta.action_bits_host()   # [C, LITERALS] uint8 0/1
    """

    def __init__(self, shape: TAStateShape):
        self.shape = shape
        self._gpu_buf = None         # pycuda DeviceAllocation
        self._allocated_bytes = 0

    # --- properties for downstream code ---

    @property
    def nbytes(self) -> int:
        return self.shape.total_uint32 * 4

    @property
    def gpu_ptr(self):
        if self._gpu_buf is None:
            raise RuntimeError(
                "CudaTAState.allocate() must be called before gpu_ptr."
            )
        return self._gpu_buf

    # --- alloc / free ---

    def allocate(self) -> None:
        cuda = _lazy_cuda()
        if self._gpu_buf is not None:
            return
        self._gpu_buf = cuda.mem_alloc(self.nbytes)
        self._allocated_bytes = self.nbytes

    def free(self) -> None:
        if self._gpu_buf is not None:
            self._gpu_buf.free()
            self._gpu_buf = None
            self._allocated_bytes = 0

    # --- init helpers ---

    def init_centre_host(self) -> np.ndarray:
        """Return a host-side bit-plane array initialised to centre state
        (`= 2^(STATE_BITS-1) - 1`, action=0 for every TA). Lower planes
        all-ones, top plane zero. Matches cair's `prepare` kernel.

        Note on parity: the C reference's `tm_initialize`
        (`TsetlinMachine.c:59-65`) uses a per-pair coin flip, one of
        every (pos, neg) literal pair starts on the include side, the
        other on exclude. That is the canonical HTM init for the CPU
        oracle in graphtm/core. M3 may set the host array to a matching
        pattern and then call `copy_from_host`. This helper is the cair-
        style fast default.
        """
        sh = self.shape
        a = np.zeros((sh.C, sh.la_chunks, sh.state_bits), dtype=np.uint32)
        a[:, :, : sh.state_bits - 1] = np.uint32(0xFFFFFFFF)
        # top plane is already 0
        return a

    def init_centre(self) -> None:
        """Allocate (if needed) and upload the centre-state init array."""
        if self._gpu_buf is None:
            self.allocate()
        host = self.init_centre_host()
        self.copy_from_host(host)

    def init_canonical_host(self, seed: int = 42) -> np.ndarray:
        """Canonical Granmo init: per (pos, neg) literal pair coin-flip, one
        side starts on include (action=1), the other on exclude (action=0).
        See `vendors/HeirarchicalTM_experiments/TsetlinMachine.c:59-65`.

        Without this, action bits all start at 0 → every clause is an empty
        AND → fires unconditionally → class_sum saturates → feedback
        probability collapses → no learning.
        """
        sh = self.shape
        rng = np.random.default_rng(seed)
        a = self.init_centre_host()
        L = sh.literals_per_clause            # = R*IA*IF*LA*2*LF
        LF = sh.LF
        n_leaves = L // (2 * LF)              # pos-block + neg-block per leaf
        # For each clause × leaf, for each (pos_n, neg_n) pair:
        #   coin = rng.random() < 0.5 → set pos action=1, else neg action=1.
        # The state value below the action bit is left at centre - 1 (already
        # set by init_centre_host), so a single Type Ia / Ib step can flip it.
        top_plane = sh.state_bits - 1
        coins = rng.random(size=(sh.C, n_leaves, LF)) < 0.5
        for c in range(sh.C):
            for leaf in range(n_leaves):
                base = leaf * 2 * LF
                for n in range(LF):
                    pos_lit = base + n
                    neg_lit = base + LF + n
                    if coins[c, leaf, n]:
                        target = pos_lit
                    else:
                        target = neg_lit
                    chunk = target // 32
                    bit = target % 32
                    a[c, chunk, top_plane] |= np.uint32(1) << bit
        return a

    def init_canonical(self, seed: int = 42) -> None:
        if self._gpu_buf is None:
            self.allocate()
        self.copy_from_host(self.init_canonical_host(seed))

    # --- transfer ---

    def copy_from_host(self, host_arr: np.ndarray) -> None:
        cuda = _lazy_cuda()
        if self._gpu_buf is None:
            self.allocate()
        if host_arr.dtype != np.uint32:
            raise TypeError(
                f"copy_from_host expects uint32, got {host_arr.dtype}"
            )
        expected = (self.shape.C, self.shape.la_chunks, self.shape.state_bits)
        if host_arr.shape != expected:
            raise ValueError(
                f"copy_from_host: expected shape {expected}, "
                f"got {host_arr.shape}"
            )
        cuda.memcpy_htod(self._gpu_buf, np.ascontiguousarray(host_arr))

    def copy_to_host(self) -> np.ndarray:
        cuda = _lazy_cuda()
        if self._gpu_buf is None:
            raise RuntimeError("copy_to_host before allocate()")
        sh = self.shape
        host = np.empty((sh.C, sh.la_chunks, sh.state_bits), dtype=np.uint32)
        cuda.memcpy_dtoh(host, self._gpu_buf)
        return host

    # M3 contract aliases.
    def to_host(self) -> np.ndarray:
        return self.copy_to_host()

    def to_host_clause(self, clause_id: int) -> np.ndarray:
        full = self.copy_to_host()
        return full[clause_id]

    def reseed(self, seed: int) -> None:
        # Bit-plane storage carries no RNG; feedback kernel reseeds per-launch.
        self._reseed_value = int(seed)

    # --- host-side decode helpers (no device calls) ---

    def action_bits_host(self, host_arr: Optional[np.ndarray] = None
                         ) -> np.ndarray:
        """Decode the top bit-plane into per-literal action bits.

        Returns uint8[C, LITERALS_PER_CLAUSE] in {0, 1}.
        """
        if host_arr is None:
            host_arr = self.copy_to_host()
        sh = self.shape
        L = sh.literals_per_clause
        top = host_arr[:, :, sh.state_bits - 1]          # [C, LA_CHUNKS]
        out = np.zeros((sh.C, L), dtype=np.uint8)
        for lit in range(L):
            chunk = lit // 32
            pos   = lit % 32
            out[:, lit] = ((top[:, chunk] >> pos) & 1).astype(np.uint8)
        return out

    def state_values_host(self, host_arr: Optional[np.ndarray] = None
                          ) -> np.ndarray:
        """Decode bit-planes into integer state values.

        Returns int32[C, LITERALS_PER_CLAUSE] in [0, 2^STATE_BITS - 1].
        """
        if host_arr is None:
            host_arr = self.copy_to_host()
        sh = self.shape
        L = sh.literals_per_clause
        out = np.zeros((sh.C, L), dtype=np.int32)
        for lit in range(L):
            chunk = lit // 32
            pos   = lit % 32
            v = np.zeros(sh.C, dtype=np.int32)
            for b in range(sh.state_bits):
                bit = ((host_arr[:, chunk, b] >> pos) & 1).astype(np.int32)
                v |= bit << b
            out[:, lit] = v
        return out

    # --- canonical-coordinate index helper ---

    def lit_index(self, j: int, k: int, l: int, m: int, n: int) -> int:
        """Linearise canonical clause coordinate (j,k,l,m,n) → literal id.

        Matches the same formula compiled into kernels.cu's
        `lit_index` device-inline. `n` ∈ [0, 2*LF), first LF positive
        literals, next LF negated.
        """
        sh = self.shape
        return (((j * sh.IA + k) * sh.IF + l) * sh.LA * 2 * sh.LF
                + m * 2 * sh.LF + n)
