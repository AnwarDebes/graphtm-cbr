"""Smoke tests for `graphtm.distill` (Module M4).

These are MICRO-DATA smoke tests, not full integration. The real
integration test (200-graph micro-AMES) lives in M7.

Three contracts:
  1. `GINTeacher.forward` returns logits of shape [B, 2] on a 5-graph
     mock batch.
  2. `train_teacher` on a 50-graph synthetic dataset beats random by
     ≥ 5 pp val-AUROC (i.e. ≥ 0.55).
  3. `distill` runs end-to-end on a 50-graph synthetic with a stub
     `HierarchicalGraphTM` that records every `.fit()` call's args.

Sister modules (M1/M3) are deliberately mocked so M4 is testable in
isolation while the parallel agents are still implementing them.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytest


# torch + torch-geometric are required by M4 hard-rule "raise on import
# with a clear message". If the host environment cannot import them
# (e.g. local numpy/pandas binary mismatch on a workstation), skip the
# whole file at collection time, this is honest about what I can test
# and what I cannot.
torch = pytest.importorskip("torch", reason="M4 distillation needs PyTorch")
try:
    import torch_geometric  # noqa: F401
    from torch_geometric.loader import DataLoader  # noqa: F401
except Exception as e:
    pytest.skip(
        f"M4 distillation needs torch-geometric (got: {type(e).__name__}: {e})",
        allow_module_level=True,
    )


# Sister-module M1 may not have landed yet; I synthesise a compatible
# GraphTensor stub if the real one is missing. This keeps M4 testable in
# isolation per the parallel-build contract.
try:
    from graphtm.encoding.graph_features import GraphTensor  # type: ignore
except Exception:  # pragma: no cover
    @dataclass
    class GraphTensor:  # type: ignore[no-redef]
        n_nodes: int
        atom_type: np.ndarray
        edge_index: np.ndarray
        bond_type: np.ndarray
        node_hv: np.ndarray = field(
            default_factory=lambda: np.zeros((0, 8), dtype=np.uint8)
        )
        edge_hv: np.ndarray = field(
            default_factory=lambda: np.zeros((0, 8), dtype=np.uint8)
        )

# Now import the module-under-test. I force a re-import in case a sister
# test loaded a partial state.
if "graphtm.distill" in sys.modules:
    distill_pkg = importlib.reload(sys.modules["graphtm.distill"])
else:
    distill_pkg = importlib.import_module("graphtm.distill")

from graphtm.distill import (  # noqa: E402
    DistillResult,
    GINTeacher,
    distill,
    graph_tensor_to_pyg,
    graphs_to_pyg_list,
    train_teacher,
)


# Synthetic-graph factory
def _make_synthetic_graph(
    rng: np.random.Generator,
    *,
    label: int,
    n_nodes_min: int = 6,
    n_nodes_max: int = 14,
    n_atom_types: int = 9,
    n_bond_types: int = 4,
) -> "GraphTensor":
    """Build a small labelled graph where class-1 has a different atom-type
    distribution from class-0. A 3-layer GIN with mean-pool reliably picks
    up class-conditional node-type frequencies → enough signal for the
    sanity test ("better than random") but trivial enough to converge in
    a few epochs.
    """
    n = int(rng.integers(n_nodes_min, n_nodes_max + 1))
    if label == 1:
        # Class-1: atoms biased toward types 0-3 (representing CNOP).
        probs = np.array(
            [0.30, 0.25, 0.20, 0.10, 0.05, 0.03, 0.03, 0.02, 0.02]
        )[:n_atom_types]
    else:
        # Class-0: biased toward types 4-8 (heavier / inert atoms).
        probs = np.array(
            [0.02, 0.03, 0.03, 0.05, 0.20, 0.25, 0.20, 0.12, 0.10]
        )[:n_atom_types]
    probs = probs / probs.sum()
    atom_type = rng.choice(n_atom_types, size=n, p=probs).astype(np.int32)

    # Random tree-ish structure: spanning tree + a few extra edges.
    edges = []
    for v in range(1, n):
        u = int(rng.integers(0, v))
        edges.append((u, v))
    # Add 1-3 extra random edges to make it a graph (not just a tree).
    n_extra = int(rng.integers(1, 4))
    for _ in range(n_extra):
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u != v:
            edges.append((u, v))
    # Undirected: include both directions.
    bidir = []
    for u, v in edges:
        bidir.append((u, v))
        bidir.append((v, u))
    edge_index = np.array(bidir, dtype=np.int32).T  # [2, n_edges]
    n_edges = edge_index.shape[1]
    bond_type = rng.integers(0, n_bond_types, size=n_edges).astype(np.int32)

    node_hv = np.zeros((n, 8), dtype=np.uint8)
    edge_hv = np.zeros((n_edges, 8), dtype=np.uint8)
    return GraphTensor(
        n_nodes=n,
        atom_type=atom_type,
        edge_index=edge_index,
        bond_type=bond_type,
        node_hv=node_hv,
        edge_hv=edge_hv,
    )


def _make_synthetic_dataset(
    n_graphs: int, *, seed: int = 7
) -> tuple[List["GraphTensor"], np.ndarray]:
    rng = np.random.default_rng(int(seed))
    graphs: List[GraphTensor] = []
    labels = np.zeros(n_graphs, dtype=np.int64)
    for i in range(n_graphs):
        y = int(i % 2)  # exact 50/50 split for clean signal
        labels[i] = y
        graphs.append(_make_synthetic_graph(rng, label=y))
    # Shuffle so train/val splits are not trivially ordered.
    perm = rng.permutation(n_graphs)
    return [graphs[i] for i in perm], labels[perm]


# Stub HierarchicalGraphTM
class StubHierarchicalGraphTM:
    """Records `.fit()` invocations; `.predict()` echoes the most recent
    training labels for the matching graph IDs (id-based, so train and
    val graphs get matched by `id(g)`). A held-out graph it has never
    seen gets the dataset-mode label.

    This is a *behavioural* stub, it is enough to exercise distill's
    bookkeeping, scoring, and metric logic without standing up the real
    M3 HGTM. Fidelity to teacher is therefore expected to be 1.0 on the
    train fold and ≈ majority-class baseline on the val fold.
    """

    def __init__(self):
        self.fit_calls: List[dict] = []
        self._memo: dict[int, int] = {}
        self._mode: int = 0

    def fit(self, graphs, y, epochs):
        y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
        self.fit_calls.append(
            {
                "n_graphs": len(graphs),
                "n_labels": int(y_arr.shape[0]),
                "epochs": int(epochs),
                "y_first_5": y_arr[:5].tolist(),
            }
        )
        for g, yi in zip(graphs, y_arr):
            self._memo[id(g)] = int(yi)
        if y_arr.size > 0:
            self._mode = int(np.bincount(y_arr).argmax())

    def predict(self, graphs):
        return np.array(
            [self._memo.get(id(g), self._mode) for g in graphs],
            dtype=np.int64,
        )

    def class_scores(self, graphs):
        # Pretend margins of ±1, enough for AUROC to be defined.
        preds = self.predict(graphs)
        scores = np.zeros((len(graphs), 2), dtype=np.float64)
        scores[np.arange(len(graphs)), preds] = 1.0
        return scores


# Test 1: forward shape
def test_gin_teacher_forward_shape_on_batch_of_5():
    """GINTeacher must return [B, 2] logits on a 5-graph mock batch."""
    rng = np.random.default_rng(0)
    graphs = [
        _make_synthetic_graph(rng, label=i % 2) for i in range(5)
    ]
    pyg = graphs_to_pyg_list(graphs, np.array([i % 2 for i in range(5)]))
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(pyg)

    model = GINTeacher()
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (5, 2), (
        f"expected logits shape (5, 2), got {tuple(logits.shape)}"
    )
    # Sanity: probabilities sum to 1 across classes.
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(
        probs.sum(dim=-1), torch.ones(5), atol=1e-5
    )


def test_graph_tensor_to_pyg_roundtrip():
    """The conversion must preserve node count, edge count, and atom types."""
    rng = np.random.default_rng(1)
    g = _make_synthetic_graph(rng, label=1)
    pyg = graph_tensor_to_pyg(g, y=1)
    assert pyg.x.shape == (g.n_nodes, 9)
    assert pyg.edge_index.shape == (2, g.edge_index.shape[1])
    assert pyg.edge_attr.shape == (g.edge_index.shape[1], 4)
    # Atom-type one-hot must point at the right index for at least one node.
    decoded = pyg.x.argmax(dim=-1).numpy()
    assert (decoded == g.atom_type).sum() >= 1, "atom-type encoding lost"


def test_gin_teacher_parameter_count_is_modest():
    """GIN parameter count for the M4 spec should fit on a footnote, i.e.
    well under 100k. This guards against accidental scale-up."""
    model = GINTeacher()
    n_params = model.n_parameters()
    assert n_params < 100_000, (
        f"GIN teacher has {n_params} parameters, M4 contract calls for a "
        "small 32-d hidden net."
    )
    assert n_params > 1_000, (
        f"GIN teacher has only {n_params} parameters, likely mis-wired."
    )


# Test 2: train_teacher beats random
def test_train_teacher_beats_random_on_50_graph_synthetic():
    """50-graph class-conditional dataset; teacher must beat random by ≥ 5 pp.

    Random AUROC = 0.5; I require ≥ 0.55 on a stratified held-out fold.
    Deterministic via seed=42.
    """
    graphs, y = _make_synthetic_dataset(50, seed=7)
    # Use fewer epochs in the smoke test (saves CI wall-time).
    model, soft, hard, auroc = train_teacher(
        graphs,
        y,
        epochs=30,
        lr=1e-2,
        batch_size=8,
        val_frac=0.30,
        class_balanced=True,
        seed=42,
        device="cpu",
    )
    assert soft.shape == (50,)
    assert hard.shape == (50,)
    assert set(np.unique(hard).tolist()).issubset({0, 1})
    assert auroc >= 0.55, (
        f"Teacher AUROC {auroc:.3f} did not exceed random+0.05, "
        "either the synthetic signal is weaker than expected or the GIN "
        "training loop is broken."
    )


def test_train_teacher_is_deterministic_under_seed():
    """Two runs with the same seed produce the same soft predictions."""
    graphs, y = _make_synthetic_dataset(40, seed=11)
    _, soft_a, _, _ = train_teacher(
        graphs, y, epochs=5, batch_size=8, seed=123, device="cpu"
    )
    _, soft_b, _, _ = train_teacher(
        graphs, y, epochs=5, batch_size=8, seed=123, device="cpu"
    )
    np.testing.assert_allclose(
        soft_a, soft_b, atol=1e-5,
        err_msg="train_teacher should be deterministic under fixed seed",
    )


# Test 3: distill end-to-end with stub student
def test_distill_records_fit_calls_and_returns_metric_curve():
    """distill must call student.fit() exactly `epochs` times in
    per_epoch_eval mode, return a metric dict with the expected keys,
    and produce numerically sane values."""
    graphs, y_true = _make_synthetic_dataset(50, seed=13)
    # Use the teacher's hard predictions as the distillation target.
    rng = np.random.default_rng(0)
    # I don't need to train a real teacher here, emulate one by adding
    # a small amount of noise to y_true so the teacher ≠ truth but is
    # mostly correct.
    flip = rng.random(50) < 0.1
    y_teacher_hard = np.where(flip, 1 - y_true, y_true).astype(np.int64)

    student = StubHierarchicalGraphTM()
    metrics = distill(
        student,
        graphs,
        y_teacher_hard,
        y_true,
        epochs=3,
        val_frac=0.2,
        seed=42,
    )

    # Right number of fit calls (one per epoch in per_epoch_eval mode).
    assert len(student.fit_calls) == 3, (
        f"distill should call student.fit 3 times (per epoch); "
        f"got {len(student.fit_calls)}"
    )
    # Each call should pass epochs=1 in per-epoch mode.
    for call in student.fit_calls:
        assert call["epochs"] == 1
        assert call["n_graphs"] == call["n_labels"]
        assert call["n_graphs"] > 0

    # Metric-dict contract.
    for key in (
        "epoch",
        "val_acc_truth",
        "val_acc_teacher",
        "val_auroc_truth",
        "train_acc_teacher",
        "final_val_auroc_truth",
        "final_val_acc_truth",
        "final_fidelity_to_teacher",
    ):
        assert key in metrics, f"distill metrics missing '{key}'"

    # Each per-epoch curve must have one entry per epoch.
    assert len(metrics["epoch"]) == 3
    assert len(metrics["val_acc_truth"]) == 3
    assert len(metrics["val_acc_teacher"]) == 3
    assert len(metrics["val_auroc_truth"]) == 3
    assert len(metrics["train_acc_teacher"]) == 3

    # Train fidelity should be exactly 1.0 once the stub has memoised
    # all train graphs.
    assert metrics["train_acc_teacher"][-1] == pytest.approx(1.0, abs=1e-6)


def test_distill_no_per_epoch_eval_path():
    """When per_epoch_eval=False, distill issues ONE fit call with
    epochs=N and the curves are empty."""
    graphs, y_true = _make_synthetic_dataset(40, seed=19)
    y_teacher_hard = y_true.copy()

    student = StubHierarchicalGraphTM()
    metrics = distill(
        student,
        graphs,
        y_teacher_hard,
        y_true,
        epochs=5,
        val_frac=0.2,
        seed=0,
        per_epoch_eval=False,
    )
    assert len(student.fit_calls) == 1
    assert student.fit_calls[0]["epochs"] == 5
    # Empty curves but final scalars exist.
    assert metrics["epoch"] == []
    assert "final_val_auroc_truth" in metrics
    assert "final_val_acc_truth" in metrics
    assert "final_fidelity_to_teacher" in metrics


def test_distill_returns_dataclass_when_requested():
    """`return_object=True` yields a DistillResult object."""
    graphs, y_true = _make_synthetic_dataset(40, seed=23)
    y_teacher_hard = y_true.copy()

    student = StubHierarchicalGraphTM()
    result = distill(
        student,
        graphs,
        y_teacher_hard,
        y_true,
        epochs=2,
        val_frac=0.2,
        seed=0,
        return_object=True,
    )
    assert isinstance(result, DistillResult)
    assert len(result.epoch) == 2


def test_distill_rejects_mismatched_lengths():
    graphs, y_true = _make_synthetic_dataset(20, seed=29)
    y_teacher_hard = y_true[:-1]  # off-by-one on purpose
    student = StubHierarchicalGraphTM()
    with pytest.raises(ValueError, match="mismatched lengths"):
        distill(student, graphs, y_teacher_hard, y_true, epochs=1)


def test_distill_rejects_zero_epochs():
    graphs, y_true = _make_synthetic_dataset(20, seed=31)
    y_teacher_hard = y_true.copy()
    student = StubHierarchicalGraphTM()
    with pytest.raises(ValueError, match="epochs must be"):
        distill(student, graphs, y_teacher_hard, y_true, epochs=0)
