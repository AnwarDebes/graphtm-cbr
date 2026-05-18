"""Full-train wall-clock benchmark.

Trains a HierarchicalGraphTM student on a 1000-graph synthetic 2-class
problem and reports:
  - wall-clock per epoch (median, p95)
  - total wall-clock to a chosen epoch budget
  - per-epoch training accuracy (optional sanity)

This is the budget benchmark I cite in the paper: it should clear
30 min on one A100/V100/RTX 4090 with the production spec.

Per invariant 4 (no silent CPU fallback), this benchmark refuses to run
without CUDA, there is no point timing CPU-only paths and calling them
"GPU".

Run:
    python benchmarks/throughput_train.py [--n-graphs 1000]
                                          [--epochs 10]
                                          [--clauses 1000]
                                          [--out results/throughput_train.json]
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
# script is run directly (`python benchmarks/throughput_train.py`).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


_LOG = logging.getLogger("benchmarks.throughput_train")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-train wall-clock for HierarchicalGraphTM."
    )
    p.add_argument("--n-graphs", type=int, default=1000,
                   help="Synthetic dataset size.")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs to run end-to-end.")
    p.add_argument("--clauses", type=int, default=1000,
                   help="Number of clauses in the student.")
    p.add_argument("--threshold", type=int, default=2000,
                   help="Class-sum saturation T.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--out", type=str,
                   default="results/throughput_train.json",
                   help="Output JSON path.")
    return p.parse_args(argv)


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as e:
        raise SystemExit("torch required (pip install torch>=2.6).") from e
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device visible. This benchmark refuses CPU mode, see "
            "docs/ARCHITECTURE.md invariant 4 (no silent CPU fallback)."
        )


def _make_synthetic_graphs(n_graphs: int, seed: int):
    from graphtm.encoding.codebook import make_codebook
    from graphtm.encoding.graph_features import encode_graph

    codebook = make_codebook(D=256, sparsity=0.10, seed=seed)
    rng = np.random.default_rng(seed)
    graphs = []
    labels = np.zeros(n_graphs, dtype=np.int64)
    for i in range(n_graphs):
        n_nodes = int(rng.integers(5, 12))
        atom_types = rng.integers(0, codebook.n_atom_types, size=n_nodes).tolist()
        edges = []
        for u in range(n_nodes - 1):
            edges.append((u, u + 1, int(rng.integers(0, codebook.n_bond_types))))
        # Random extra edges for cyclic graphs.
        for _ in range(rng.integers(0, 3)):
            u = int(rng.integers(0, n_nodes))
            v = int(rng.integers(0, n_nodes))
            if u != v:
                edges.append((u, v, int(rng.integers(0, codebook.n_bond_types))))
        graphs.append(encode_graph(
            {"atom_types": atom_types, "edges": edges},
            codebook,
            k_hop=codebook.k_hop,
        ))
        has_c = any(a == 0 for a in atom_types)
        has_db = any(b == 1 for (_, _, b) in edges)
        labels[i] = 1 if (has_c and has_db) else 0
    return graphs, labels, codebook


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = _parse_args(argv)
    _require_cuda()

    import torch
    from graphtm.core.hierarchical_graph_tm import (
        HGraphTMSpec, HierarchicalGraphTM,
    )

    graphs, y, codebook = _make_synthetic_graphs(args.n_graphs, args.seed)
    _LOG.info("synthesised %d graphs, %d positive", len(graphs), int(y.sum()))

    spec = HGraphTMSpec(
        n_classes=2,
        n_clauses=args.clauses,
        threshold=args.threshold,
        s=3.9,
        n_states=100,
        R=2, IA=2, IF=2, LA=8, LF=2,
        D_bits=codebook.D,
        max_nodes=16,
        seed=args.seed,
    )
    model = HierarchicalGraphTM(spec, device="cuda")
    _LOG.info("model built: %d clauses, T=%d", args.clauses, args.threshold)

    # timed training
    epoch_times: list[float] = []
    t_total_start = time.perf_counter()
    for ep in range(args.epochs):
        t0 = time.perf_counter()
        model.fit(graphs, y, epochs=1)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        _LOG.info("epoch %2d/%d  wallclock=%.3fs", ep + 1, args.epochs, dt)
    total_wallclock = time.perf_counter() - t_total_start

    # post-training: predict accuracy sanity
    preds = model.predict(graphs)
    train_acc = float((preds.ravel() == y.ravel()).mean())

    epoch_times_sorted = sorted(epoch_times)
    median_ep = epoch_times_sorted[len(epoch_times_sorted) // 2]
    p95_ep = epoch_times_sorted[int(0.95 * (len(epoch_times_sorted) - 1))]

    payload = {
        "tool": "benchmarks/throughput_train.py",
        "seed": args.seed,
        "n_graphs": args.n_graphs,
        "clauses": args.clauses,
        "threshold": args.threshold,
        "epochs": args.epochs,
        "median_sec_per_epoch": float(median_ep),
        "p95_sec_per_epoch": float(p95_ep),
        "stdev_sec": float(statistics.pstdev(epoch_times)) if len(epoch_times) > 1 else 0.0,
        "total_wallclock_sec": float(total_wallclock),
        "train_accuracy": train_acc,
        "epoch_times_sec": epoch_times,
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOG.info("total wallclock: %.2fs (median epoch=%.2fs)", total_wallclock, median_ep)
    _LOG.info("train accuracy: %.3f", train_acc)
    _LOG.info("results written to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
