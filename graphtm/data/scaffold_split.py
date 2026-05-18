"""Bemis-Murcko scaffold splitting for molecular datasets.

Why this exists
---------------
Random splits leak scaffold identity from train into test: two close analogues
of the same chemotype end up on opposite sides and the model effectively
memorises the chemotype. Scaffold splits (Bemis & Murcko, J Med Chem 39:2887,
1996) collapse every molecule to its ring-system core and force *whole
scaffolds* to lie on a single side of the split. This is the same protocol
TDC ADMET-Group, MoleculeNet, and OGB use, so leaderboard numbers are directly
comparable.

Convention (matches MoleculeNet / OGB / TDC): the largest scaffold buckets go
into the training set, the smallest into validation/test. The smallest
scaffolds are by definition the most "novel" relative to the training data,
which is exactly what I want a held-out set to measure.

This module has *no* dependency on M1/M6 internals: it operates on plain
SMILES strings and returns NumPy index arrays. That keeps it cheap to call
from `ames.py`, `tox21.py`, or any future loader.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

# `MissingDependency` is defined in `graphtm.data.__init__`; importing via the
# package keeps the symbol shared across `ames.py`, `kazius.py`, `tox21.py`,
# and `scaffold_split.py` without an extra module.
from graphtm.data import MissingDependency


def _murcko_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES for one molecule.

    A molecule that fails to parse, or whose scaffold cannot be computed,
    falls back to an empty-string scaffold which keeps it in its own bucket
    of "no-scaffold" molecules. This matches the MoleculeNet implementation.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:   # pragma: no cover - env hint
        raise MissingDependency(
            "rdkit is required for scaffold splitting. "
            "Install via `pip install rdkit` or `conda install -c conda-forge rdkit`."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(
    smiles_list: Sequence[str],
    frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split SMILES indices into train/valid/test by Bemis-Murcko scaffold.

    Parameters
    ----------
    smiles_list
        Sequence of SMILES strings, one per molecule.
    frac
        (train_frac, valid_frac, test_frac) summing to 1.0 (within 1e-6).
    seed
        Permutation seed for shuffling equally-sized scaffold groups so that
        the split is deterministic per seed but tie-breaks are not always the
        same lexical order.

    Returns
    -------
    (train_idx, valid_idx, test_idx)
        Three disjoint `np.ndarray[int]` index arrays covering exactly
        `range(len(smiles_list))`.

    Algorithm
    ---------
    1. Compute Murcko scaffold of every SMILES.
    2. Group molecule indices by scaffold string.
    3. Sort groups by (size desc, scaffold_string) to keep deterministic order.
    4. Walk the sorted groups: append the entire group to train until train is
       full; then to valid until valid is full; remainder goes to test.

    The "largest scaffolds → train" rule (MoleculeNet, OGB, TDC) makes the
    test split contain the rarest chemotypes, which is the regime that
    matters for the headline scaffold-AUROC number.
    """
    n = len(smiles_list)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    if abs(sum(frac) - 1.0) > 1e-6:
        raise ValueError(f"frac must sum to 1.0, got {frac} (sum={sum(frac)})")
    if any(f < 0 for f in frac):
        raise ValueError(f"frac entries must be non-negative, got {frac}")

    # 1. Scaffold every molecule.
    scaffolds: Dict[str, List[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffolds[_murcko_scaffold(smi)].append(i)

    # 2. Sort groups by size desc, then seed-permuted tie-break inside each
    #    size bucket. Two ingredients:
    #      (a) shuffle indices *inside* each group  ↦ defeats input-order bias.
    #      (b) attach a seed-randomised rank to each group ↦ makes ties across
    #          equal-size groups deterministic-per-seed but not the same for
    #          different seeds, satisfying the "deterministic per seed" rule
    #          while still letting `seed` produce meaningfully different splits.
    rng = np.random.default_rng(seed)
    groups: List[List[int]] = list(scaffolds.values())
    for g in groups:
        rng.shuffle(g)
    tiebreaks = rng.random(len(groups))
    groups_with_keys = list(zip(groups, tiebreaks))
    groups_with_keys.sort(key=lambda pair: (-len(pair[0]), pair[1]))
    groups = [g for g, _ in groups_with_keys]

    # 3. Greedy fill train, valid, then test.
    n_train = int(np.floor(frac[0] * n))
    n_valid = int(np.floor(frac[1] * n))
    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []

    for grp in groups:
        if len(train_idx) + len(grp) <= n_train:
            train_idx.extend(grp)
        elif len(valid_idx) + len(grp) <= n_valid:
            valid_idx.extend(grp)
        else:
            test_idx.extend(grp)

    # Final sort restores ascending order inside each split (downstream-friendly).
    out_train = np.array(sorted(train_idx), dtype=np.int64)
    out_valid = np.array(sorted(valid_idx), dtype=np.int64)
    out_test = np.array(sorted(test_idx), dtype=np.int64)

    # Sanity invariants, disjoint and exhaustive.
    assert len(out_train) + len(out_valid) + len(out_test) == n, (
        f"scaffold_split lost molecules: "
        f"{len(out_train)}+{len(out_valid)}+{len(out_test)} != {n}"
    )
    assert len(np.intersect1d(out_train, out_valid)) == 0
    assert len(np.intersect1d(out_train, out_test)) == 0
    assert len(np.intersect1d(out_valid, out_test)) == 0

    return out_train, out_valid, out_test


__all__ = ["scaffold_split"]
