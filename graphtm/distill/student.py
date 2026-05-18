"""HGTM-student distillation driver (Module M4).

The HGTM student is trained to *imitate the teacher's hard predictions*
on the training graphs. Hard-label distillation is the deliberate choice:
soft-label distillation has been empirically destructive for Tsetlin
Machines (the TA include/exclude voting is fragile to fractional
targets), losing ~11.7 pp on prior runs, see research/02 and the
M4 contract in `docs/ARCHITECTURE.md`.

Pipeline:
  1. Build `(graphs, y_teacher_hard)` as training data.
  2. Hold out a stratified slice (`val_frac`) to monitor accuracy /
     fidelity to teacher and accuracy against ground-truth labels.
  3. Train the HGTM `student` for `epochs` rounds (`student.fit`).
  4. Return a metrics dict with the final student AUROC, fidelity to
     teacher, fidelity to ground truth, and the per-epoch curve.

The student is treated as a black box that obeys the M3 contract
(`HierarchicalGraphTM.fit(graphs, y, epochs)` and `.predict(graphs) ->
np.ndarray`). This file imports nothing from M3 at top-level so the
two agents stay decoupled; the type hint is a forward string.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from graphtm.core.hierarchical_graph_tm import HierarchicalGraphTM
    from graphtm.encoding.graph_features import GraphTensor


# Helper metrics
def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney-U binary AUROC; degenerate folds → 0.5.

    Kept local so this module has no scikit-learn import, sister modules
    may or may not pull it in.
    """
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError("AUROC shape mismatch")
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(counts.size, dtype=np.float64)
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        ranks = avg[inv]
    rank_sum_pos = ranks[: pos.size].sum()
    u = rank_sum_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("accuracy shape mismatch")
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def _stratified_holdout(
    n: int, y: np.ndarray, val_frac: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Same logic as teacher's holdout; deliberate duplicate to keep the
    two files decoupled (M4 lives in a single sub-package; both helpers
    are private)."""
    y = np.asarray(y).reshape(-1)
    if n <= 1:
        return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64)
    val_n = max(1, int(round(val_frac * n)))
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_pos_val = max(0, int(round(val_frac * idx_pos.size)))
    n_neg_val = max(0, val_n - n_pos_val)
    n_neg_val = min(n_neg_val, idx_neg.size)
    val_idx = np.concatenate([idx_pos[:n_pos_val], idx_neg[:n_neg_val]])
    train_idx = np.concatenate([idx_pos[n_pos_val:], idx_neg[n_neg_val:]])
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


# Soft-output extraction
def _student_scores(model: "HierarchicalGraphTM",
                     graphs: Sequence["GraphTensor"]) -> np.ndarray:
    """Get a soft per-graph score from the student.

    Prefers `class_scores()` (returns per-class margin sums) if present;
    falls back to `predict()` (hard labels in {0,1}). I accept both
    interfaces because M3 may delay `class_scores` until later.

    Returns array of shape [N] giving the positive-class margin.
    """
    if hasattr(model, "class_scores"):
        try:
            cs = np.asarray(model.class_scores(list(graphs)), dtype=np.float64)
            if cs.ndim == 2 and cs.shape[1] >= 2:
                # Use class-1 margin minus class-0 margin as the score.
                return cs[:, 1] - cs[:, 0]
            if cs.ndim == 1:
                return cs
        except NotImplementedError:
            pass
    preds = np.asarray(model.predict(list(graphs)), dtype=np.int64)
    return preds.astype(np.float64)


# Public API
@dataclass
class DistillResult:
    """Container for `distill()` return values; the call site can also
    treat it as a `dict` via `result.to_dict()`.

    Fields:
      - epoch: per-epoch index (0..epochs-1)
      - val_acc_truth: student accuracy on held-out vs ground truth
      - val_acc_teacher: student accuracy on held-out vs teacher hard label
      - val_auroc_truth: student AUROC on held-out vs ground truth
      - train_acc_teacher: student accuracy on train vs teacher hard label
    """
    epoch: List[int] = field(default_factory=list)
    val_acc_truth: List[float] = field(default_factory=list)
    val_acc_teacher: List[float] = field(default_factory=list)
    val_auroc_truth: List[float] = field(default_factory=list)
    train_acc_teacher: List[float] = field(default_factory=list)
    final_val_auroc_truth: float = 0.5
    final_val_acc_truth: float = 0.0
    final_fidelity_to_teacher: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "epoch": list(self.epoch),
            "val_acc_truth": list(self.val_acc_truth),
            "val_acc_teacher": list(self.val_acc_teacher),
            "val_auroc_truth": list(self.val_auroc_truth),
            "train_acc_teacher": list(self.train_acc_teacher),
            "final_val_auroc_truth": self.final_val_auroc_truth,
            "final_val_acc_truth": self.final_val_acc_truth,
            "final_fidelity_to_teacher": self.final_fidelity_to_teacher,
        }


def distill(
    student: "HierarchicalGraphTM",
    graphs: List["GraphTensor"],
    y_teacher_hard: np.ndarray,
    y_true: np.ndarray,
    *,
    epochs: int,
    val_frac: float = 0.15,
    seed: int = 42,
    per_epoch_eval: bool = True,
    return_object: bool = False,
) -> Dict[str, object]:
    """Hard-label distillation: train `student` on `(graphs, y_teacher_hard)`,
    monitor on a held-out fold against `y_true`.

    Hard labels only, see module docstring for the rationale.

    Args:
        student: an `HierarchicalGraphTM` (M3), anything with
                 `.fit(graphs, y, epochs)` and `.predict(graphs)` works.
        graphs:  list of `GraphTensor` (M1).
        y_teacher_hard: ndarray [N] of {0,1}, the teacher's hard
                        predictions (the *distillation target*).
        y_true:  ndarray [N] of {0,1}, ground-truth labels, used ONLY
                 for the monitor. Never seen by the student.
        epochs:  total HGTM training epochs.
        val_frac: held-out fraction for monitoring.
        seed: numpy + torch seeding.
        per_epoch_eval: if True, train the student in 1-epoch increments
                        and record metrics after each pass. If False,
                        train in one `student.fit(..., epochs=epochs)`
                        call and only measure the final state.

    Returns:
        dict with metric curves (epoch, val_acc_truth, val_acc_teacher,
        val_auroc_truth, train_acc_teacher) plus `final_*` scalars.
    """
    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))

    y_t = np.asarray(y_teacher_hard).astype(np.int64).reshape(-1)
    y_g = np.asarray(y_true).astype(np.int64).reshape(-1)
    n = len(graphs)
    if y_t.shape[0] != n or y_g.shape[0] != n:
        raise ValueError(
            f"distill: mismatched lengths, graphs={n}, "
            f"y_teacher_hard={y_t.shape[0]}, y_true={y_g.shape[0]}"
        )
    if epochs < 1:
        raise ValueError(f"distill: epochs must be ≥ 1, got {epochs}")

    train_idx, val_idx = _stratified_holdout(n, y_t, val_frac, rng)
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]

    # Teacher labels are the training target. Ground truth is held out
    # for evaluation only.
    y_train_target = y_t[train_idx]
    y_train_truth = y_g[train_idx]
    y_val_target = y_t[val_idx]
    y_val_truth = y_g[val_idx]

    out = DistillResult()

    if per_epoch_eval:
        # Train one epoch at a time so I can record per-epoch metrics.
        for ep in range(int(epochs)):
            student.fit(train_graphs, y_train_target, epochs=1)
            # Train fidelity: predictions vs teacher's hard labels.
            train_pred = np.asarray(
                student.predict(train_graphs), dtype=np.int64
            ).reshape(-1)
            train_fid = _accuracy(y_train_target, train_pred)
            if len(val_graphs) > 0:
                val_pred = np.asarray(
                    student.predict(val_graphs), dtype=np.int64
                ).reshape(-1)
                val_acc_t = _accuracy(y_val_truth, val_pred)
                val_acc_tch = _accuracy(y_val_target, val_pred)
                val_scores = _student_scores(student, val_graphs)
                val_auroc = _binary_auroc(y_val_truth, val_scores)
            else:
                val_acc_t = float("nan")
                val_acc_tch = float("nan")
                val_auroc = float("nan")
            out.epoch.append(ep)
            out.train_acc_teacher.append(train_fid)
            out.val_acc_truth.append(val_acc_t)
            out.val_acc_teacher.append(val_acc_tch)
            out.val_auroc_truth.append(val_auroc)
    else:
        student.fit(train_graphs, y_train_target, epochs=int(epochs))

    # Final-state metrics on the held-out fold (also covers no-per-epoch case).
    if len(val_graphs) > 0:
        val_pred = np.asarray(
            student.predict(val_graphs), dtype=np.int64
        ).reshape(-1)
        val_scores = _student_scores(student, val_graphs)
        out.final_val_acc_truth = _accuracy(y_val_truth, val_pred)
        out.final_fidelity_to_teacher = _accuracy(y_val_target, val_pred)
        out.final_val_auroc_truth = _binary_auroc(y_val_truth, val_scores)
    else:
        out.final_val_acc_truth = float("nan")
        out.final_fidelity_to_teacher = float("nan")
        out.final_val_auroc_truth = float("nan")

    if return_object:
        return out  # type: ignore[return-value]
    return out.to_dict()


__all__ = [
    "DistillResult",
    "distill",
]
