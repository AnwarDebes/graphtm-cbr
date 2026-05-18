"""GIN teacher for the HGTM distillation pipeline (Module M4).

The teacher is a small, conventional Graph Isomorphism Network (Xu et al.,
ICLR 2019, *How Powerful Are Graph Neural Networks?*). Its job is *not*
to be the headline model, it is a per-graph soft label provider whose
hard-thresholded predictions seed the HGTM student via distillation.

Architecture (frozen by `docs/ARCHITECTURE.md` § M4):
  - 3 GINConv layers, 32-d hidden, ReLU + BatchNorm between
  - mean-pool over nodes → graph embedding
  - 2-class linear head (logits)
  - Input is a PyTorch-Geometric `Data` batch with `x`, `edge_index`,
    `edge_attr` (atom-type one-hot or learned embedding, bond-type
    one-hot or learned embedding).

The teacher must use *raw graph message passing*, bag-of-atoms is
explicitly disallowed (M4 contract, invariant #1).

Why GIN and not GCN/AttentiveFP? GIN-AttrMasking is the strongest GNN
baseline on TDC AMES (0.842 AUROC, research/05). I keep the
architecture vanilla so the *distillation gap* attributable to HGTM
recourse is unambiguous.

Public API:
  - `GINTeacher(torch.nn.Module)`
  - `train_teacher(graphs, y, epochs=80, lr=1e-3, batch_size=32, seed=42)`
      → (model, soft_predictions, hard_predictions, val_auroc)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

# Mandatory dependency, fail loudly per M4 hard-rules.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "graphtm.distill.teacher requires PyTorch. Install with "
        "`pip install torch>=2.6`."
    ) from e

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINConv, global_mean_pool
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "graphtm.distill.teacher requires torch-geometric. Install with "
        "`pip install torch-geometric>=2.7`."
    ) from e

if TYPE_CHECKING:
    # Avoid an import cycle in module-load order; only used for type hints.
    from graphtm.encoding.graph_features import GraphTensor


# Atom/bond type cardinalities default to HGraphTMSpec defaults
# (n_atom_types=9, n_bond_types=4 in docs/ARCHITECTURE.md § M3).
DEFAULT_N_ATOM_TYPES = 9
DEFAULT_N_BOND_TYPES = 4
DEFAULT_HIDDEN_DIM = 32


# Conversion helpers
def graph_tensor_to_pyg(
    g: "GraphTensor",
    *,
    n_atom_types: int = DEFAULT_N_ATOM_TYPES,
    n_bond_types: int = DEFAULT_N_BOND_TYPES,
    y: Optional[int] = None,
) -> Data:
    """Convert one `GraphTensor` to a PyG `Data` object.

    `x` is a one-hot encoding of `atom_type` (shape [n_nodes, n_atom_types]),
    `edge_attr` is a one-hot encoding of `bond_type`
    (shape [n_edges, n_bond_types]). `edge_index` is passed through.

    No bag-of-atoms, the per-node tensor is preserved; the GIN does
    the aggregation, not us.
    """
    n = int(g.n_nodes)
    atom_type = np.asarray(g.atom_type, dtype=np.int64)
    if atom_type.shape != (n,):
        raise ValueError(
            f"graph_tensor_to_pyg: atom_type shape {atom_type.shape} != ({n},)"
        )
    x = torch.zeros((n, n_atom_types), dtype=torch.float32)
    valid = (atom_type >= 0) & (atom_type < n_atom_types)
    if valid.any():
        idx = np.where(valid)[0]
        x[idx, atom_type[valid]] = 1.0

    edge_index = torch.as_tensor(np.asarray(g.edge_index, dtype=np.int64))
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"graph_tensor_to_pyg: edge_index shape {tuple(edge_index.shape)} "
            "must be [2, n_edges]"
        )

    e = edge_index.size(1)
    bond_type = np.asarray(g.bond_type, dtype=np.int64).reshape(-1)
    if bond_type.shape[0] != e:
        raise ValueError(
            f"graph_tensor_to_pyg: bond_type length {bond_type.shape[0]} != "
            f"n_edges {e}"
        )
    edge_attr = torch.zeros((e, n_bond_types), dtype=torch.float32)
    if e > 0:
        valid_b = (bond_type >= 0) & (bond_type < n_bond_types)
        if valid_b.any():
            idx_b = np.where(valid_b)[0]
            edge_attr[idx_b, bond_type[valid_b]] = 1.0

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([int(y)], dtype=torch.long)
    return data


def graphs_to_pyg_list(
    graphs: List["GraphTensor"],
    y: Optional[np.ndarray] = None,
    *,
    n_atom_types: int = DEFAULT_N_ATOM_TYPES,
    n_bond_types: int = DEFAULT_N_BOND_TYPES,
) -> List[Data]:
    """Vectorised wrapper around `graph_tensor_to_pyg` for a list of graphs."""
    if y is not None and len(y) != len(graphs):
        raise ValueError(
            f"graphs_to_pyg_list: len(y) {len(y)} != len(graphs) {len(graphs)}"
        )
    out: List[Data] = []
    for i, g in enumerate(graphs):
        out.append(
            graph_tensor_to_pyg(
                g,
                n_atom_types=n_atom_types,
                n_bond_types=n_bond_types,
                y=None if y is None else int(y[i]),
            )
        )
    return out


# Model
def _gin_mlp(in_dim: int, hidden: int) -> nn.Sequential:
    """Standard GINConv MLP, two linear layers with ReLU + BN, per Xu 2019."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


class GINTeacher(nn.Module):
    """Three-layer GIN with mean-pool and a 2-class linear head.

    Forward expects a PyG batch with `.x`, `.edge_index`, `.batch` attributes.
    `.edge_attr` is accepted for interface symmetry but the vanilla GIN does
    not use edge features inside the convolution; I keep them on the `Data`
    object so downstream tasks (recourse) can still read them.

    Output: raw logits of shape [B, 2]. Soft probability for "positive class"
    (label = 1) is `softmax(logits, dim=-1)[:, 1]`.
    """

    def __init__(
        self,
        n_atom_types: int = DEFAULT_N_ATOM_TYPES,
        n_bond_types: int = DEFAULT_N_BOND_TYPES,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        n_classes: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_atom_types = int(n_atom_types)
        self.n_bond_types = int(n_bond_types)
        self.hidden_dim = int(hidden_dim)
        self.n_classes = int(n_classes)
        self.dropout_p = float(dropout)

        # Three GINConv layers, first lifts from one-hot atom features to
        # hidden, subsequent two operate on hidden→hidden.
        self.conv1 = GINConv(_gin_mlp(self.n_atom_types, self.hidden_dim))
        self.conv2 = GINConv(_gin_mlp(self.hidden_dim, self.hidden_dim))
        self.conv3 = GINConv(_gin_mlp(self.hidden_dim, self.hidden_dim))

        self.dropout = nn.Dropout(self.dropout_p)
        self.head = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, data: Data) -> torch.Tensor:
        """Compute logits for each graph in the batch.

        Args:
            data: a PyG batch (a `Batch`, or a single `Data` with no
                `.batch` attribute, in that case I treat it as one graph).
        Returns:
            logits tensor of shape [n_graphs, n_classes].
        """
        x = data.x
        edge_index = data.edge_index
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        g = global_mean_pool(h, batch)
        g = self.dropout(g)
        return self.head(g)

    @torch.no_grad()
    def predict_proba(self, loader: "DataLoader") -> np.ndarray:
        """Run inference; return softmax probabilities of shape [N, n_classes]."""
        self.eval()
        device = next(self.parameters()).device
        out: List[np.ndarray] = []
        for batch in loader:
            batch = batch.to(device)
            logits = self.forward(batch)
            p = F.softmax(logits, dim=-1).cpu().numpy()
            out.append(p)
        if not out:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        return np.concatenate(out, axis=0)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Training
def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary AUROC by Mann-Whitney U / rank statistic.

    Self-contained so the test does not require scikit-learn. Handles
    degenerate cases (single-class folds) by returning 0.5.
    """
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError("AUROC shape mismatch")
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Rank-based U; ties get average rank.
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    # Average rank for ties.
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        # average rank per unique value
        sums = np.zeros(counts.size, dtype=np.float64)
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        ranks = avg[inv]
    rank_sum_pos = ranks[: pos.size].sum()
    u = rank_sum_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _set_all_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _stratified_holdout(
    n: int, y: np.ndarray, val_frac: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified single-split into (train, val) index arrays.

    Used purely for the in-loop val-AUROC monitor; the *real* scaffold
    split for AMES comes from `graphtm.data.ames`.
    """
    y = np.asarray(y).reshape(-1)
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


def _balanced_sampler_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency per-sample weights for `WeightedRandomSampler`.

    Returns weights with length len(y); each weight = 1 / count(class_of_i).
    """
    y = np.asarray(y).astype(np.int64).reshape(-1)
    unique, counts = np.unique(y, return_counts=True)
    inv = {int(c): 1.0 / float(n) for c, n in zip(unique, counts)}
    return np.array([inv[int(c)] for c in y], dtype=np.float64)


@dataclass
class TeacherTrainHistory:
    """Per-epoch loss + val-AUROC log; useful for diagnostics."""
    train_loss: List[float]
    val_auroc: List[float]


def train_teacher(
    graphs: List["GraphTensor"],
    y: np.ndarray,
    *,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    val_frac: float = 0.15,
    class_balanced: bool = True,
    device: Optional[str] = None,
    seed: int = 42,
    n_atom_types: int = DEFAULT_N_ATOM_TYPES,
    n_bond_types: int = DEFAULT_N_BOND_TYPES,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    return_history: bool = False,
) -> Tuple[GINTeacher, np.ndarray, np.ndarray, float]:
    """Train the GIN teacher on a list of `GraphTensor` + labels.

    Optimiser: AdamW. Loss: cross-entropy on 2-class logits (equivalent to
    BCEWithLogits on the positive-class margin, but keeps the same head
    shape as multi-task generalisation later).

    Optional class-balanced sampling (WeightedRandomSampler), toggled on
    by default since AMES is mildly imbalanced (~55/45) and the labels of
    interest for distillation are the rare-toxicophore positives.

    Returns:
        (model, soft_predictions, hard_predictions, val_auroc)

        - `soft_predictions`: array of shape [N], the probability the
          teacher assigns to class 1 for *every* training graph.
        - `hard_predictions`: argmax of teacher output per graph; this is
          what M4's distillation step consumes (hard-label distillation
          only, soft-label was empirically destructive for TMs, see
          research/02 and `student.py`).
        - `val_auroc`: AUROC on the held-out stratified split.
    """
    _set_all_seeds(seed)
    rng = np.random.default_rng(int(seed))

    y_arr = np.asarray(y).astype(np.int64).reshape(-1)
    if y_arr.shape[0] != len(graphs):
        raise ValueError(
            f"train_teacher: len(graphs)={len(graphs)} != len(y)={y_arr.shape[0]}"
        )
    n = y_arr.shape[0]
    if n < 4:
        raise ValueError(
            f"train_teacher: need at least 4 graphs, got {n}"
        )

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(dev)

    pyg_list = graphs_to_pyg_list(
        graphs, y_arr, n_atom_types=n_atom_types, n_bond_types=n_bond_types
    )

    train_idx, val_idx = _stratified_holdout(n, y_arr, val_frac, rng)
    train_data = [pyg_list[i] for i in train_idx]
    val_data = [pyg_list[i] for i in val_idx]
    all_data = pyg_list  # for final inference

    # Sampler / loader.
    if class_balanced and len(train_data) > 0:
        w = _balanced_sampler_weights(y_arr[train_idx])
        g_torch = torch.Generator()
        g_torch.manual_seed(int(seed))
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(w, dtype=torch.double),
            num_samples=len(train_data),
            replacement=True,
            generator=g_torch,
        )
        train_loader = DataLoader(
            train_data, batch_size=batch_size, sampler=sampler
        )
    else:
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    all_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False)

    model = GINTeacher(
        n_atom_types=n_atom_types,
        n_bond_types=n_bond_types,
        hidden_dim=hidden_dim,
    ).to(device_t)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = TeacherTrainHistory(train_loss=[], val_auroc=[])

    for epoch in range(int(epochs)):
        model.train()
        ep_loss = 0.0
        ep_n = 0
        for batch in train_loader:
            batch = batch.to(device_t)
            optim.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y.view(-1))
            loss.backward()
            optim.step()
            ep_loss += float(loss.item()) * batch.num_graphs
            ep_n += int(batch.num_graphs)
        avg_loss = ep_loss / max(1, ep_n)
        history.train_loss.append(avg_loss)

        # Val-AUROC each epoch (cheap; ~hundreds of graphs).
        if len(val_data) > 0:
            probs = model.predict_proba(val_loader)
            auroc = _binary_auroc(y_arr[val_idx], probs[:, 1])
        else:
            auroc = 0.5
        history.val_auroc.append(auroc)

    # Final inference over the whole set.
    probs_all = model.predict_proba(all_loader)
    soft = probs_all[:, 1].astype(np.float32)
    hard = (soft >= 0.5).astype(np.int64)

    final_auroc = history.val_auroc[-1] if history.val_auroc else 0.5

    if return_history:
        return model, soft, hard, final_auroc, history  # type: ignore[return-value]
    return model, soft, hard, final_auroc


__all__ = [
    "GINTeacher",
    "TeacherTrainHistory",
    "graph_tensor_to_pyg",
    "graphs_to_pyg_list",
    "train_teacher",
    "DEFAULT_N_ATOM_TYPES",
    "DEFAULT_N_BOND_TYPES",
    "DEFAULT_HIDDEN_DIM",
]
