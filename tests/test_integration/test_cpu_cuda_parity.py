"""CPU-reference vs CUDA-kernel forward-path numerical parity.

The CPU reference is `graphtm.core.hierarchical_tm.HierarchicalTM` (the
Granmo & Saha canonical port). The CUDA path is `graphtm.cuda._kernels`
(M2). Both must produce IDENTICAL clause outputs on the same seed +
same Boolean input, this is invariant 2 in `docs/ARCHITECTURE.md`.

Skip cleanly when CUDA is unavailable; do NOT silently degrade to a
"CPU-only parity" since that would be tautological.

The "same Boolean input" subtlety: HierarchicalTM consumes a flat
`[n_features]` Boolean vector with `n_features == R * IF * LF`. The CUDA
kernel `clause_forward_pernode` consumes packed per-node hypervectors, so
I have to pick a parity surface that both kernels can produce. I
compare on the FLAT clause-output array `[C]`, since both are required
to expose `clause_output` (per the M3 contract and the existing
`HierarchicalTM.calculate_clause_output`).

If either side does not yet expose a flat-input parity entry point this
test skips with an informative reason (M1/M2/M3 may still be landing).
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CUDA availability, single check, no silent fallback
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    """Per invariant 4: one and only one device check."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------

def test_cpu_cuda_forward_parity():
    """Same seed, same packed input, identical clause outputs.

    Skipped if either:
      * CUDA is not visible to the host (no GPU / no driver),
      * M2 `graphtm.cuda._kernels.CudaKernels` is not yet present,
      * M3 has not exposed a parity surface (forward-only entry).

    Hard-asserts ELEMENTWISE EQUALITY on the clause_output vector once
    both surfaces are present.
    """
    if not _cuda_available():
        pytest.skip("CUDA not available, parity is not testable on this host.")
    pytest.importorskip(
        "pycuda",
        reason="pycuda not installed; CUDA kernels unreachable.",
    )

    cuda_kern_mod = pytest.importorskip(
        "graphtm.cuda._kernels",
        reason="M2 CUDA kernels module not yet present.",
    )
    if not hasattr(cuda_kern_mod, "CudaKernels"):
        pytest.skip("graphtm.cuda._kernels.CudaKernels not yet present.")

    from graphtm.core.hierarchical_tm import HierarchicalTM, HTMArchSpec

    # Tiny architecture, keep parity check fast and tractable.
    R, IA, IF, LA, LF = 2, 2, 2, 4, 2
    n_features = R * IF * LF   # = 8
    n_clauses = 4
    n_states = 50
    threshold = 20

    spec = HTMArchSpec(
        n_features=n_features,
        n_clauses=n_clauses,
        root_factors=R,
        interior_alternatives=IA,
        interior_factors=IF,
        leaf_alternatives=LA,
        leaf_factors=LF,
        n_states=n_states,
        threshold=threshold,
        s=3.9,
        seed=42,
    )
    rng = np.random.default_rng(42)
    X = rng.integers(0, 2, size=n_features).astype(np.int32)

    # CPU reference
    cpu_tm = HierarchicalTM(spec)
    cpu_tm.calculate_clause_output(X)
    cpu_clause_out = np.asarray(cpu_tm.clause_output, dtype=np.int32).copy()

    # CUDA path
    # I use the CPU TM's initial TA state and feed it into the CUDA kernel.
    # The M2 contract for CudaKernels.forward returns the same [C] clause
    # output given identical bit-packed TA state and identical input.
    if not hasattr(cuda_kern_mod, "forward_from_flat_input"):
        pytest.skip(
            "graphtm.cuda._kernels has no flat-input parity entry "
            "(forward_from_flat_input), parity surface not yet exposed."
        )
    cuda_clause_out = cuda_kern_mod.forward_from_flat_input(
        ta_state=cpu_tm.ta_state, X=X, spec=spec,
    )
    cuda_clause_out = np.asarray(cuda_clause_out, dtype=np.int32).ravel()

    assert cuda_clause_out.shape == cpu_clause_out.shape, (
        f"shape mismatch: CPU {cpu_clause_out.shape} vs CUDA {cuda_clause_out.shape}"
    )
    diff = np.abs(cuda_clause_out - cpu_clause_out)
    assert int(diff.max()) == 0, (
        f"CUDA != CPU clause outputs (max abs diff = {int(diff.max())}). "
        f"CPU = {cpu_clause_out.tolist()}, CUDA = {cuda_clause_out.tolist()}"
    )


def test_cuda_kernels_module_imports_when_cuda_present():
    """Smoke: if CUDA is available the kernels module must NOT silently fail.

    Per invariant 4 ("no silent CPU fallback"), `graphtm.cuda._kernels`
    must either succeed in compiling or raise an informative error, it
    must not silently load a CPU stub. I tolerate "not implemented yet"
    (M2 may still be landing) by skipping.
    """
    if not _cuda_available():
        pytest.skip("CUDA not available on this host.")
    pytest.importorskip("pycuda")
    try:
        import graphtm.cuda._kernels  # noqa: F401
    except ImportError:
        pytest.skip("M2 graphtm.cuda._kernels not yet present.")
    except Exception as e:   # noqa: BLE001
        # Any other error means the module exists but failed to compile.
        # That is a real defect, surface it via a normal failure.
        raise AssertionError(
            f"graphtm.cuda._kernels failed to load on a CUDA host: {e!r}"
        ) from e
