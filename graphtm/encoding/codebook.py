"""Atom/bond/role codebook of sparse BSC hypervectors.

Per research/04 §3 the codebook is the atomic alphabet for graph encoding:
fixed random sparse-BSC hypervectors for atom types, bond types, and one
hop-role marker per hop in 0..k_hop. Drawn once with an explicit seed so
the encoding is fully reproducible.

Defaults match the ToxBenchmark vocabulary used elsewhere in graphtm:
  - 9 atom types  : C, N, O, F, P, S, Cl, Br, I
  - 4 bond types  : single, double, triple, aromatic
  - k_hop = 2     : ECFP4-equivalent radius
  - D = 8192      : 128 * 64-bit words, ~150-item bundling capacity
  - sparsity = 0.10 : SBDR regime (Rachkovskij 2001)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hypervectors import sparse_bsc


@dataclass
class AtomBondCodebook:
    """Frozen alphabet of sparse-BSC hypervectors used by `encode_graph`."""

    atom_hv: np.ndarray            # [n_atom_types, D] uint8 0/1
    bond_hv: np.ndarray            # [n_bond_types, D] uint8 0/1
    role_hv: np.ndarray            # [k_hop+1,     D] uint8 0/1
    D: int = 8192
    sparsity: float = 0.10
    n_atom_types: int = 9
    n_bond_types: int = 4
    k_hop: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate shapes/dtypes, the rest of M1 trusts these invariants."""
        if self.atom_hv.shape != (self.n_atom_types, self.D):
            raise ValueError(
                f"atom_hv shape {self.atom_hv.shape} != "
                f"({self.n_atom_types}, {self.D})"
            )
        if self.bond_hv.shape != (self.n_bond_types, self.D):
            raise ValueError(
                f"bond_hv shape {self.bond_hv.shape} != "
                f"({self.n_bond_types}, {self.D})"
            )
        if self.role_hv.shape != (self.k_hop + 1, self.D):
            raise ValueError(
                f"role_hv shape {self.role_hv.shape} != "
                f"({self.k_hop + 1}, {self.D})"
            )
        for name, arr in (("atom_hv", self.atom_hv),
                          ("bond_hv", self.bond_hv),
                          ("role_hv", self.role_hv)):
            if arr.dtype != np.uint8:
                raise TypeError(f"{name} dtype must be uint8, got {arr.dtype}")


def make_codebook(
    n_atom_types: int = 9,
    n_bond_types: int = 4,
    k_hop: int = 2,
    D: int = 8192,
    sparsity: float = 0.10,
    seed: int = 0,
) -> AtomBondCodebook:
    """Build a deterministic AtomBondCodebook from `seed` (sparse BSC atoms)."""
    if n_atom_types <= 0:
        raise ValueError(f"n_atom_types must be > 0, got {n_atom_types}")
    if n_bond_types <= 0:
        raise ValueError(f"n_bond_types must be > 0, got {n_bond_types}")
    if k_hop < 0:
        raise ValueError(f"k_hop must be >= 0, got {k_hop}")

    rng = np.random.default_rng(seed)
    atom_hv = np.stack(
        [sparse_bsc(D, sparsity, rng) for _ in range(n_atom_types)], axis=0
    )
    bond_hv = np.stack(
        [sparse_bsc(D, sparsity, rng) for _ in range(n_bond_types)], axis=0
    )
    role_hv = np.stack(
        [sparse_bsc(D, sparsity, rng) for _ in range(k_hop + 1)], axis=0
    )
    return AtomBondCodebook(
        atom_hv=atom_hv,
        bond_hv=bond_hv,
        role_hv=role_hv,
        D=D,
        sparsity=sparsity,
        n_atom_types=n_atom_types,
        n_bond_types=n_bond_types,
        k_hop=k_hop,
        seed=seed,
    )


# Canonical alphabet labels, kept here so downstream modules can map back
# from indices to human-readable symbols (recourse needs this for RDKit ops).
ATOM_SYMBOLS: tuple[str, ...] = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
BOND_NAMES: tuple[str, ...] = ("single", "double", "triple", "aromatic")
