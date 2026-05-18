"""CPU-reference vs CUDA-kernel head-to-head speedup table.

Builds the SAME logical TM in two backends:
  - CPU reference: `graphtm.core.hierarchical_tm.HierarchicalTM`
                   (Granmo & Saha canonical port, NumPy + optional Numba)
  - CUDA path:     `graphtm.cuda._kernels.CudaKernels` (forward-only)

Times the forward pass on identical inputs and reports the per-call
speedup. The CUDA backend uses an explicit `forward_from_flat_input`
adapter (M2 contract) that takes the same Boolean feature vector the
CPU reference consumes, so the comparison is apples-to-apples.

If the CUDA backend is unavailable, the script prints "CUDA unavailable"
and exits 0 (you can still see the CPU-only side from this file's
output). Per invariant 4 I do NOT secretly run a CPU-only "GPU" pass.

Run:
    python benchmarks/cpu_vs_cuda.py [--n-iter 200]
                                     [--clauses 200,1000,5000]
                                     [--out results/cpu_vs_cuda.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

# Ensure the repo root is on sys.path so `import graphtm` works when this
# script is run directly (`python benchmarks/cpu_vs_cuda.py`) without
# installing the package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


_LOG = logging.getLogger("benchmarks.cpu_vs_cuda")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU reference vs CUDA kernel forward-pass speedup."
    )
    p.add_argument("--clauses", type=str, default="200,1000,5000",
                   help="Comma-separated clause counts to sweep.")
    p.add_argument("--n-iter", type=int, default=200,
                   help="Timed forward calls per backend per clause count.")
    p.add_argument("--n-warmup", type=int, default=10,
                   help="Warmup calls per backend.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--out", type=str,
                   default="results/cpu_vs_cuda.json",
                   help="Output JSON path.")
    return p.parse_args(argv)


def _try_import_cuda():
    """Return (CudaKernels, forward_fn) or (None, None) if unavailable.

    `forward_fn` is the parity-surface entrypoint
    `graphtm.cuda._kernels.forward_from_flat_input` (per the M2 contract).
    If the kernels module exists but the entry isn't exposed, returns
    (CudaKernels, None), caller decides how to surface that.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        import pycuda  # noqa: F401
        from graphtm.cuda._kernels import CudaKernels
    except Exception:
        return None, None
    forward_fn = getattr(__import__(
        "graphtm.cuda._kernels", fromlist=["forward_from_flat_input"]
    ), "forward_from_flat_input", None)
    return CudaKernels, forward_fn


def _bench_one(C: int, *, n_iter: int, n_warmup: int, seed: int) -> dict:
    from graphtm.core.hierarchical_tm import HierarchicalTM, HTMArchSpec

    R, IA, IF, LA, LF = 2, 2, 2, 8, 2
    n_features = R * IF * LF   # = 8
    spec = HTMArchSpec(
        n_features=n_features,
        n_clauses=C,
        root_factors=R,
        interior_alternatives=IA,
        interior_factors=IF,
        leaf_alternatives=LA,
        leaf_factors=LF,
        n_states=100,
        threshold=int(min(4 * C, 4000)),
        s=3.9,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=n_features).astype(np.int32)

    # CPU reference
    cpu_tm = HierarchicalTM(spec)
    # Warmup
    for _ in range(n_warmup):
        cpu_tm.calculate_clause_output(X)
    cpu_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        cpu_tm.calculate_clause_output(X)
        cpu_times.append(time.perf_counter() - t0)
    cpu_median = statistics.median(cpu_times)
    cpu_row = {
        "backend": "cpu",
        "median_sec": float(cpu_median),
        "stdev_sec": float(statistics.pstdev(cpu_times)) if len(cpu_times) > 1 else 0.0,
    }

    # CUDA path (skip if unavailable)
    CudaKernels, forward_fn = _try_import_cuda()
    if CudaKernels is None:
        return {"C": C, "cpu": cpu_row, "cuda": None,
                "reason_skipped": "CUDA / pycuda / graphtm.cuda._kernels unavailable"}
    if forward_fn is None:
        return {"C": C, "cpu": cpu_row, "cuda": None,
                "reason_skipped": (
                    "graphtm.cuda._kernels.forward_from_flat_input "
                    "entry not exposed (M2 contract not finalised)"
                )}

    import torch
    # Warmup
    for _ in range(n_warmup):
        _ = forward_fn(ta_state=cpu_tm.ta_state, X=X, spec=spec)
    torch.cuda.synchronize()
    cuda_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = forward_fn(ta_state=cpu_tm.ta_state, X=X, spec=spec)
        torch.cuda.synchronize()
        cuda_times.append(time.perf_counter() - t0)
    cuda_median = statistics.median(cuda_times)
    cuda_row = {
        "backend": "cuda",
        "median_sec": float(cuda_median),
        "stdev_sec": float(statistics.pstdev(cuda_times)) if len(cuda_times) > 1 else 0.0,
    }
    speedup = float(cpu_median / cuda_median) if cuda_median > 0 else float("inf")
    return {
        "C": C,
        "cpu": cpu_row,
        "cuda": cuda_row,
        "speedup_cuda_over_cpu": speedup,
    }


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = _parse_args(argv)
    try:
        clause_counts = [int(c.strip()) for c in args.clauses.split(",") if c.strip()]
    except ValueError as e:
        raise SystemExit(f"invalid --clauses: {e}") from e

    rows = []
    for C in clause_counts:
        _LOG.info("benchmarking C=%d ...", C)
        rows.append(_bench_one(
            C, n_iter=args.n_iter, n_warmup=args.n_warmup, seed=args.seed,
        ))
        _LOG.info("    -> %s", rows[-1])

    payload = {
        "tool": "benchmarks/cpu_vs_cuda.py",
        "seed": args.seed,
        "n_iter": args.n_iter,
        "rows": rows,
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOG.info("results written to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
