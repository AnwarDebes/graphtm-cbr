"""Molecule/graph -> per-node + per-edge BSC hypervector tensors.

Implements the frozen `encode_graph` interface from docs/ARCHITECTURE.md:

  - `GraphTensor`: dataclass holding per-node atom types, edge index, bond
    types, per-node hypervectors, and per-edge hypervectors. All HV arrays
    are uint8 0/1 of length `codebook.D`.

  - `encode_graph(mol, codebook, k_hop=2) -> GraphTensor`:
      * accepts an RDKit Mol or a duck-typed graph object (see
        `_extract_graph` for the contract);
      * per-edge HV = atom(u) XOR bond(b) XOR atom(v)  (BSC binding);
      * per-node HV = majority-bundle of:
           - atom_hv[type(v)] (self at hop 0),
           - permute(atom_hv[type(u)], r) for every neighbour u at hop r,
             r in 1..k_hop. Permutation by `r` tags the hop role
             (HDGL/Granmo convention, see research/04 §2).

  - The graph stores edges in both directions (undirected). Edge HV is
    *orientation-preserving* (atom(u) XOR bond(b) XOR atom(v) differs from
    atom(v) XOR bond(b) XOR atom(u) only if `atom(u) != atom(v)`, which is
    a fact of XOR + non-commuting role-of-position, here implicit; I keep
    both orientations so each direction has its own row).

No bag-of-atoms summarization is produced, only per-node and per-edge
tensors leave this module.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .codebook import AtomBondCodebook, ATOM_SYMBOLS, BOND_NAMES
from .hypervectors import majority_bundle, permute, xor_bind


# ---------------------------------------------------------------------------
# Canonical lookups
# ---------------------------------------------------------------------------

_ATOM_INDEX: dict[str, int] = {sym: i for i, sym in enumerate(ATOM_SYMBOLS)}

# RDKit BondType integer codes (Chem.BondType): SINGLE=1, DOUBLE=2,
# TRIPLE=3, AROMATIC=12. I accept either the integer code or the string
# name, mapping both to [0..3].
_BOND_INDEX_BY_NAME: dict[str, int] = {name: i for i, name in enumerate(BOND_NAMES)}
_BOND_INDEX_BY_RDKIT_INT: dict[int, int] = {1: 0, 2: 1, 3: 2, 12: 3}


# ---------------------------------------------------------------------------
# GraphTensor, frozen interface (see docs/ARCHITECTURE.md M1)
# ---------------------------------------------------------------------------

@dataclass
class GraphTensor:
    """Per-node + per-edge hypervector tensors for one graph."""

    n_nodes: int
    atom_type: np.ndarray   # [n_nodes] int32
    edge_index: np.ndarray  # [2, n_edges] int32  (undirected: both directions)
    bond_type: np.ndarray   # [n_edges] int32
    node_hv: np.ndarray     # [n_nodes, D] uint8
    edge_hv: np.ndarray     # [n_edges, D] uint8


# ---------------------------------------------------------------------------
# Input adaptors, RDKit Mol OR duck-typed dict-like graph
# ---------------------------------------------------------------------------

def _atom_index(symbol: str) -> int:
    """Map atom symbol to codebook index. Unknown -> last index (Iodine slot).

    Returning a valid slot for unknown elements keeps `encode_graph` total;
    the caller should pre-filter SMILES if they want a hard error instead.
    """
    return _ATOM_INDEX.get(symbol, len(ATOM_SYMBOLS) - 1)


def _bond_index(value: Any) -> int:
    """Map bond type (int code, str name, or RDKit BondType) to codebook idx."""
    # Strings, try direct name lookup
    if isinstance(value, str):
        key = value.lower()
        if key in _BOND_INDEX_BY_NAME:
            return _BOND_INDEX_BY_NAME[key]
        # Allow upper-case RDKit-style names ("SINGLE", "DOUBLE", ...)
        key2 = value.upper()
        for i, name in enumerate(BOND_NAMES):
            if name.upper() == key2:
                return i
        raise ValueError(f"unknown bond type: {value!r}")
    # RDKit BondType enum, convert via int()
    try:
        as_int = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"unknown bond type: {value!r}") from e
    if as_int in _BOND_INDEX_BY_RDKIT_INT:
        return _BOND_INDEX_BY_RDKIT_INT[as_int]
    if 0 <= as_int < len(BOND_NAMES):
        # Plain 0..3 index (PyG-style)
        return as_int
    raise ValueError(f"unknown bond type: {value!r}")


def _extract_graph(mol: Any) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Return (atom_indices, edges) where edges = list of (u, v, bond_idx).

    Accepts either:
      * an RDKit `Mol`, has `GetAtoms()` returning items with `GetSymbol()`,
        and `GetBonds()` returning items with `GetBeginAtomIdx`/`GetEndAtomIdx`/
        `GetBondType` (int-convertible).
      * a duck-typed object (e.g. dict) with `atom_types` (list[int|str]) and
        `edges` (iterable of (u, v, bond_type)). `atom_types` may be int (used
        as-is, validated against codebook) or str (looked up in `ATOM_SYMBOLS`).

    Edges are returned in input order, each as a single undirected pair ,
    `encode_graph` doubles them.
    """
    # RDKit branch
    if hasattr(mol, "GetAtoms") and hasattr(mol, "GetBonds"):
        atom_indices = [_atom_index(a.GetSymbol()) for a in mol.GetAtoms()]
        edges: list[tuple[int, int, int]] = []
        for b in mol.GetBonds():
            u = int(b.GetBeginAtomIdx())
            v = int(b.GetEndAtomIdx())
            bidx = _bond_index(b.GetBondType())
            edges.append((u, v, bidx))
        return atom_indices, edges

    # Duck-typed dict-like branch
    if isinstance(mol, dict):
        atoms_raw = mol.get("atom_types")
        edges_raw = mol.get("edges")
    else:
        atoms_raw = getattr(mol, "atom_types", None)
        edges_raw = getattr(mol, "edges", None)
    if atoms_raw is None or edges_raw is None:
        raise TypeError(
            "encode_graph: expected an RDKit Mol or an object with "
            "`atom_types` and `edges`; got " + type(mol).__name__
        )

    atom_indices = []
    for a in atoms_raw:
        if isinstance(a, str):
            atom_indices.append(_atom_index(a))
        else:
            atom_indices.append(int(a))

    edges = []
    for e in edges_raw:
        if len(e) != 3:
            raise ValueError(f"edge must be (u, v, bond_type); got {e!r}")
        u, v, btype = e
        edges.append((int(u), int(v), _bond_index(btype)))
    return atom_indices, edges


# ---------------------------------------------------------------------------
# k-hop neighbourhood walk
# ---------------------------------------------------------------------------

def _build_adjacency(n_nodes: int,
                     edges: Iterable[tuple[int, int, int]]
                     ) -> list[list[tuple[int, int]]]:
    """Adjacency list with bond indices, `adj[u]` = list of (v, bond_idx)."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for u, v, b in edges:
        if u == v:
            continue   # ignore self-loops; codebook supplies the self HV
        adj[u].append((v, b))
        adj[v].append((u, b))
    return adj


def _hop_neighbours(adj_with_bond: list[list[tuple[int, int]]],
                    start: int, k_hop: int,
                    ) -> list[tuple[int, int, list[int]]]:
    """BFS-yield (node, hop, bond_chain) for nodes 1..k_hop hops from `start`.

    `bond_chain` is the sequence of bond indices along the discovered path.
    Each node is yielded once (BFS-shortest path). The start node itself is
    not yielded, the caller adds the self-HV at hop 0 separately.
    """
    if k_hop <= 0:
        return []
    visited = {start}
    out: list[tuple[int, int, list[int]]] = []
    queue: deque[tuple[int, int, list[int]]] = deque()
    queue.append((start, 0, []))
    while queue:
        node, hop, chain = queue.popleft()
        if hop >= k_hop:
            continue
        for nbr, bond_idx in adj_with_bond[node]:
            if nbr in visited:
                continue
            visited.add(nbr)
            new_chain = chain + [bond_idx]
            out.append((nbr, hop + 1, new_chain))
            queue.append((nbr, hop + 1, new_chain))
    return out


# ---------------------------------------------------------------------------
# encode_graph, the public entrypoint
# ---------------------------------------------------------------------------

def encode_graph(mol: Any,
                 codebook: AtomBondCodebook,
                 k_hop: int = 2,
                 ) -> GraphTensor:
    """Encode a molecule/graph into per-node + per-edge BSC hypervectors."""
    if k_hop < 0:
        raise ValueError(f"k_hop must be >= 0, got {k_hop}")
    if k_hop > codebook.k_hop:
        raise ValueError(
            f"k_hop={k_hop} exceeds codebook.k_hop={codebook.k_hop}; "
            "rebuild the codebook with a larger k_hop."
        )

    atom_indices, edges_raw = _extract_graph(mol)
    n_nodes = len(atom_indices)
    if n_nodes == 0:
        raise ValueError("encode_graph: molecule has zero atoms")
    for a in atom_indices:
        if not (0 <= a < codebook.n_atom_types):
            raise ValueError(
                f"atom index {a} out of range [0, {codebook.n_atom_types})"
            )
    for u, v, b in edges_raw:
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            raise ValueError(f"edge endpoint out of range: ({u},{v})")
        if not (0 <= b < codebook.n_bond_types):
            raise ValueError(
                f"bond index {b} out of range [0, {codebook.n_bond_types})"
            )

    D = codebook.D
    atom_hv = codebook.atom_hv
    bond_hv = codebook.bond_hv

    # ---- Atom-type tensor (int32) ----
    atom_type = np.asarray(atom_indices, dtype=np.int32)

    # ---- Edge tensors (undirected: both directions) ----
    n_und_edges = len(edges_raw)
    n_dir_edges = 2 * n_und_edges
    edge_index = np.zeros((2, n_dir_edges), dtype=np.int32)
    bond_type = np.zeros(n_dir_edges, dtype=np.int32)
    edge_hv = np.zeros((n_dir_edges, D), dtype=np.uint8)
    for i, (u, v, b) in enumerate(edges_raw):
        triple_uv = xor_bind(xor_bind(atom_hv[atom_indices[u]], bond_hv[b]),
                             atom_hv[atom_indices[v]])
        triple_vu = xor_bind(xor_bind(atom_hv[atom_indices[v]], bond_hv[b]),
                             atom_hv[atom_indices[u]])
        # Forward direction (u -> v)
        edge_index[0, 2 * i]     = u
        edge_index[1, 2 * i]     = v
        bond_type[2 * i]         = b
        edge_hv[2 * i]           = triple_uv
        # Reverse direction (v -> u), orientation-preserving row
        edge_index[0, 2 * i + 1] = v
        edge_index[1, 2 * i + 1] = u
        bond_type[2 * i + 1]     = b
        edge_hv[2 * i + 1]       = triple_vu

    # ---- Per-node hypervector tensor with k-hop role-bundled neighbourhood ----
    # Per research/04 §3: for each hop-r neighbour reached via bond chain
    # b_1,...,b_r, accumulate atom(u) XOR bond(b_1) XOR ... XOR bond(b_r),
    # then permute by `r` to tag the hop role and bundle into node_hv[v].
    adj = _build_adjacency(n_nodes, edges_raw)
    node_hv = np.zeros((n_nodes, D), dtype=np.uint8)
    for v in range(n_nodes):
        self_hv = atom_hv[atom_indices[v]]
        if k_hop == 0:
            node_hv[v] = self_hv
            continue
        rows: list[np.ndarray] = [self_hv]
        for nbr, hop, bond_chain in _hop_neighbours(adj, v, k_hop):
            msg = atom_hv[atom_indices[nbr]].copy()
            for b_idx in bond_chain:
                msg = xor_bind(msg, bond_hv[b_idx])
            # Role-tag by cyclic shift = `hop` positions (HDGL/Granmo lab).
            rows.append(permute(msg, hop).astype(np.uint8, copy=False))
        stack = np.stack(rows, axis=0).astype(np.uint8, copy=False)
        node_hv[v] = majority_bundle(stack)

    return GraphTensor(
        n_nodes=n_nodes,
        atom_type=atom_type,
        edge_index=edge_index,
        bond_type=bond_type,
        node_hv=node_hv,
        edge_hv=edge_hv,
    )
