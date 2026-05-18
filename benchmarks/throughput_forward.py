"""Forward-kernel throughput micro-benchmark.

Measures the per-sample wall-clock cost of the CUDA forward pass across a
small grid of clause counts. The model is held identical except for `C`
(`n_clauses`), so the result curve is a clean (samples/sec) vs (clauses)
plot.

Methodology:
  - 64 synthetic 8-node graphs, 16 features per node, 256-bit hypervectors.
  - For each C in {200, 1000, 5000}:
      * build a HierarchicalGraphTM with that C,
      * warm up 10 forward calls (CUDA JIT, kernel paging),
      * time 100 forward calls in a tight loop,
      * report median + p95 latency and samples/sec.
  - All times via `torch.cuda.synchronize()` + `time.perf_counter()`.

Failure mode: if the CUDA stack is not available, exit 1 with a clear
message, per invariant 4 ("no silent CPU fallback"), this benchmark
refuses to run on CPU and emit fake "GPU" numbers.

Run:
    python benchmarks/throughput_forward.py [--clause-counts 200,1000]
                                            [--batch-size 64]
                                            [--n-iter 100]
                                            [--out results/throughput_forward.json]
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
from typing import List, Sequence

# Ensure the repo root is on sys.path so `import graphtm` works when this
# script is run directly (`python benchmarks/throughput_forward.py`).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


_LOG = logging.getLogger("benchmarks.throughput_forward")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forward-kernel throughput vs n_clauses."
    )
    p.add_argument("--clause-counts", type=str, default="200,1000,5000",
                   help="Comma-separated list of clause counts to sweep.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Synthetic mini-batch size (number of graphs).")
    p.add_argument("--n-iter", type=int, default=100,
                   help="Timed forward iterations per clause count.")
    p.add_argument("--n-warmup", type=int, default=10,
                   help="Warmup iterations before timing.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--out", type=str,
                   default="results/throughput_forward.json",
                   help="Output JSON path.")
    return p.parse_args(argv)


def _require_cuda() -> None:
    """Per invariant 4: hard-fail if CUDA is not available."""
    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "torch is required for this benchmark. Install it via "
            "`pip install torch>=2.6`."
        ) from e
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device visible. This benchmark refuses to run on "
            "CPU, see docs/ARCHITECTURE.md invariant 4 (no silent CPU "
            "fallback)."
        )


def _make_synthetic_graphs(n_graphs: int, seed: int):
    """64 small graphs with 4-8 atoms each, 256-bit codebook."""
    from graphtm.encoding.codebook import make_codebook
    from graphtm.encoding.graph_features import encode_graph

    codebook = make_codebook(D=256, sparsity=0.10, seed=seed)
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n_graphs):
        n_nodes = int(rng.integers(4, 9))
        atom_types = rng.integers(0, codebook.n_atom_types, size=n_nodes).tolist()
        edges = [(u, u + 1, int(rng.integers(0, codebook.n_bond_types)))
                 for u in range(n_nodes - 1)]
        graphs.append(encode_graph(
            {"atom_types": atom_types, "edges": edges},
            codebook,
            k_hop=codebook.k_hop,
        ))
    return graphs, codebook


def _bench_one_clause_count(C: int, graphs, codebook, *,
                            n_iter: int, n_warmup: int,
                            seed: int) -> dict:
    """Time `n_iter` forward calls and return latency + throughput stats."""
    import torch
    from graphtm.core.hierarchical_graph_tm import (
        HGraphTMSpec, HierarchicalGraphTM,
    )

    spec = HGraphTMSpec(
        n_classes=2,
        n_clauses=C,
        threshold=int(min(2000, 4 * C)),
        s=3.9,
        n_states=100,
        R=2, IA=2, IF=2, LA=8, LF=2,
        D_bits=codebook.D,
        max_nodes=16,
        seed=seed,
    )
    model = HierarchicalGraphTM(spec, device="cuda")

    # Warm up, first call paginates kernels + allocates workspace.
    for _ in range(n_warmup):
        _ = model.class_scores(graphs)
    torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = model.class_scores(graphs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    p95 = times_sorted[int(0.95 * (len(times_sorted) - 1))]
    n_graphs = len(graphs)
    return {
        "C": C,
        "n_graphs_per_batch": n_graphs,
        "n_iter": n_iter,
        "median_sec_per_batch": float(median),
        "p95_sec_per_batch": float(p95),
        "samples_per_sec": float(n_graphs / median),
        "stdev_sec": float(statistics.pstdev(times)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = _parse_args(argv)
    _require_cuda()

    try:
        clause_counts = [int(c.strip()) for c in args.clause_counts.split(",") if c.strip()]
    except ValueError as e:
        raise SystemExit(f"invalid --clause-counts: {e}") from e
    if not clause_counts:
        raise SystemExit("--clause-counts must contain at least one value.")

    graphs, codebook = _make_synthetic_graphs(args.batch_size, args.seed)
    _LOG.info("synthesised %d graphs, D=%d", len(graphs), codebook.D)

    rows = []
    for C in clause_counts:
        _LOG.info("benchmarking C=%d ...", C)
        try:
            row = _bench_one_clause_count(
                C, graphs, codebook,
                n_iter=args.n_iter, n_warmup=args.n_warmup, seed=args.seed,
            )
        except Exception as e:   # noqa: BLE001
            _LOG.error("C=%d failed: %r", C, e)
            row = {"C": C, "error": repr(e)}
        rows.append(row)
        _LOG.info("    -> %s", row)

    payload = {
        "tool": "benchmarks/throughput_forward.py",
        "seed": args.seed,
        "n_iter": args.n_iter,
        "batch_size": args.batch_size,
        "rows": rows,
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOG.info("results written to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
