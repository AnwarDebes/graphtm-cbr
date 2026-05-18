"""TDC AMES (Hansen 2009) loader → `GraphTensor` list + labels + metadata.

Why this exists
---------------
The primary HGTM-CBR benchmark is TDC AMES (Hansen 2009 + Kazius 2005 union;
canonical TDC source size n=7,255, post-sanitisation runtime size n=7,278,
binary mutagenicity, scaffold split). Per `research/05_benchmark_choice.md`
this is the only common-leaderboard dataset that *both* (a) has a real
topology gap (Morgan-FP MLP 0.794 vs ZairaChem 0.871) and (b) plugs directly
into ICH M7(R2) §6.1. This loader is the on-ramp.

Pipeline
--------
1. Fetch the TDC AMES CSV (TDC's `Tox(name='AMES')`); the file is cached in
   `~/.graphtm_cache/tdc_ames.csv`. If the upstream is unavailable, fall back
   to the cached CSV that the user has dropped into `~/.graphtm_cache/`.
2. SMILES → RDKit Mol → `Chem.SanitizeMol`; molecules that fail are dropped
   and counted in the returned metadata.
3. Mol → `GraphTensor` via the M1 `encode_graph` interface (I **do not**
   reach into M1 internals). If M1 isn't on the import path yet, I raise a
   clear `NotImplementedError` so the parallel build doesn't get silently
   blocked.
4. Compute split indices over the kept molecules per `split` policy:
     - ``"scaffold"`` → Bemis-Murcko split via `scaffold_split` (this module).
     - ``"random"``   → RNG-shuffled 80/10/10 split.
     - ``"none"``     → all molecules in `train`; empty valid/test.

API
---
Per the M6 contract (`docs/ARCHITECTURE.md`):

    graphs, labels, metadata = load_tdc_ames(split="scaffold", seed=42)

* `graphs`, full list of `GraphTensor` (length = n_kept), input order
   preserved (sanitised subset of the source CSV order).
* `labels`, np.int64 array (length = n_kept), aligned with `graphs`.
* `metadata`, dict (see schema below).

Metadata dict carries:
    - ``n_total_smiles``  : rows in source CSV
    - ``n_kept``          : rows that survived sanitisation
    - ``n_dropped``       : rows that failed to sanitise
    - ``n_atoms_max``     : maximum heavy-atom count in kept set
    - ``class_balance``   : dict {0: n_neg, 1: n_pos}
    - ``split_indices``   : {"train": idx, "valid": idx, "test": idx}
                            (np.int64 arrays into `graphs` / `labels`)
    - ``split_sizes``     : (n_train, n_valid, n_test)
    - ``smiles``          : list of kept SMILES strings (aligned with graphs)
    - ``source``          : "tdc" | "csv_cache" | "url_mirror"
    - ``cache_path``      : absolute path to the cached CSV
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from graphtm.data import MissingDependency
from graphtm.data.scaffold_split import scaffold_split


# Default TDC ADMET-Group AMES mirror (Hansen 2009 + Kazius 2005 union;
# canonical TDC source size 7,255 SMILES; post-sanitisation runtime size 7,278
# (5822 train / 727 valid / 729 test on scaffold split); binary label `Y` in
# {0,1}). When TDC isn't available I still try a small set of public mirror
# URLs before erroring.
_TDC_AMES_FALLBACK_URLS: Tuple[str, ...] = (
    "https://tdcommons.ai/static/data/adme/Ames.csv",
    "https://huggingface.co/datasets/tdcommons/ames/resolve/main/Ames.csv",
)


def _resolve_cache_dir(cache_dir: str) -> Path:
    """Expand ``~`` and create the directory if absent."""
    out = Path(os.path.expanduser(cache_dir)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fetch_tdc_ames(cache_dir: Path) -> Tuple[Path, str]:
    """Return (csv_path, source_tag). Source is 'tdc' or 'csv_cache' or 'url_mirror'.

    Order of attempts:
      1. TDC `Tox(name='AMES')`, primary path.
      2. CSV already present in `cache_dir/tdc_ames.csv`.
      3. HTTPS fallback download to `cache_dir/tdc_ames.csv`.
    """
    cached = cache_dir / "tdc_ames.csv"

    # Attempt 1: TDC python API.
    try:
        from tdc.single_pred import Tox  # type: ignore
        data = Tox(name="AMES", path=str(cache_dir))
        df = data.get_data()
        if not cached.exists():
            df.to_csv(cached, index=False)
        return cached, "tdc"
    except ImportError:
        # Fall through to cache / URL fallback.
        pass
    except Exception:
        # TDC present but its download stack failed (e.g. network down,
        # upstream URL change). Fall through.
        pass

    # Attempt 2: pre-existing CSV cache.
    if cached.exists() and cached.stat().st_size > 0:
        return cached, "csv_cache"

    # Attempt 3: public HTTPS mirrors.
    last_err: Optional[Exception] = None
    for url in _TDC_AMES_FALLBACK_URLS:
        try:
            urllib.request.urlretrieve(url, cached)
            if cached.exists() and cached.stat().st_size > 0:
                return cached, "url_mirror"
        except Exception as exc:
            last_err = exc
            continue

    raise MissingDependency(
        "Could not obtain TDC AMES dataset. "
        "Install TDC via `pip install PyTDC`, OR drop the CSV at "
        f"{cached}. Last error: {last_err!r}."
    )


def _parse_ames_csv(csv_path: Path) -> Tuple[List[str], np.ndarray]:
    """Return (smiles_list, labels_np) from a TDC-style CSV.

    TDC AMES schema: columns `Drug_ID, Drug, Y` (Drug = SMILES, Y in {0,1}).
    I tolerate header variants (``SMILES``/``smiles``/``Drug``;
    ``Y``/``label``/``Activity``) in case the cache was populated by a non-TDC
    mirror.
    """
    import csv as _csv

    smiles_col_candidates = ("Drug", "SMILES", "smiles", "Canonical_SMILES")
    label_col_candidates = ("Y", "y", "label", "Label", "Activity", "activity")

    with csv_path.open("r", newline="") as f:
        reader = _csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        smi_col = next(
            (c for c in smiles_col_candidates if c in reader.fieldnames), None
        )
        lbl_col = next(
            (c for c in label_col_candidates if c in reader.fieldnames), None
        )
        if smi_col is None or lbl_col is None:
            raise ValueError(
                f"CSV {csv_path} columns {reader.fieldnames!r} do not contain a "
                f"recognised SMILES + label pair. Looked for {smiles_col_candidates} "
                f"and {label_col_candidates}."
            )

        smiles: List[str] = []
        labels: List[int] = []
        for row in reader:
            s = row[smi_col]
            y = row[lbl_col]
            if not s or y in (None, ""):
                continue
            try:
                yi = int(float(y))
            except ValueError:
                continue
            smiles.append(s)
            labels.append(yi)

    return smiles, np.asarray(labels, dtype=np.int64)


def _sanitize_smiles(smiles: List[str], labels: np.ndarray):
    """Return (kept_mols, kept_smiles, kept_labels, n_dropped).

    Each SMILES is parsed via `Chem.MolFromSmiles` then explicitly run through
    `Chem.SanitizeMol`; molecules that raise during sanitisation are dropped
    and counted (a returned diagnostic, not a silent loss).
    """
    try:
        from rdkit import Chem
    except ImportError as exc:   # pragma: no cover - env hint
        raise MissingDependency(
            "rdkit is required to parse SMILES. "
            "Install via `pip install rdkit` or `conda install -c conda-forge rdkit`."
        ) from exc

    kept_mols = []
    kept_smiles: List[str] = []
    kept_labels: List[int] = []
    n_dropped = 0

    for smi, y in zip(smiles, labels.tolist()):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            n_dropped += 1
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            n_dropped += 1
            continue
        kept_mols.append(mol)
        kept_smiles.append(smi)
        kept_labels.append(int(y))

    return (
        kept_mols,
        kept_smiles,
        np.asarray(kept_labels, dtype=np.int64),
        n_dropped,
    )


def _encode_with_m1(mols) -> list:
    """Convert RDKit mols to `GraphTensor` via the M1 `encode_graph` interface.

    M1 is being implemented by a sister agent; this loader does not depend
    on its internals. If `encode_graph` is not yet importable I raise a
    descriptive `NotImplementedError` so the parallel build never silently
    runs on un-encoded molecules.
    """
    try:
        from graphtm.encoding.codebook import make_codebook
        from graphtm.encoding.graph_features import encode_graph  # type: ignore
    except ImportError as exc:
        raise NotImplementedError(
            "graphtm.encoding.graph_features.encode_graph (M1) is not "
            "available yet. The data loader has produced sanitised RDKit "
            "molecules but cannot turn them into GraphTensors until M1 is "
            "merged. Original import error: "
            f"{exc!r}"
        ) from exc

    cb = make_codebook(seed=0)
    return [encode_graph(m, cb, k_hop=cb.k_hop) for m in mols]


def _random_split(
    n: int, frac: Tuple[float, float, float], seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy-RNG 3-way split (used only when split='random')."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(np.floor(frac[0] * n))
    n_valid = int(np.floor(frac[1] * n))
    train = np.sort(perm[:n_train])
    valid = np.sort(perm[n_train : n_train + n_valid])
    test = np.sort(perm[n_train + n_valid :])
    return train, valid, test


def load_tdc_ames(
    split: str = "scaffold",
    seed: int = 42,
    cache_dir: str = "~/.graphtm_cache",
    frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    encode: bool = True,
):
    """Load TDC AMES → (graphs, labels, metadata).

    Matches the M6 contract in `docs/ARCHITECTURE.md`. The split indices live
    in `metadata['split_indices']` so the function signature stays a flat
    3-tuple while still exposing train/valid/test partitioning.

    Parameters
    ----------
    split : {"scaffold", "random", "none"}
        - ``"scaffold"``, Bemis-Murcko scaffold split (default; comparable to
          TDC leaderboard).
        - ``"random"``, RNG-shuffled split with `seed`.
        - ``"none"``, all molecules in `train`; empty valid/test indices.
    seed : int
        RNG seed for `random` split and tie-breaks in scaffold-split shuffling.
        Splits are deterministic given seed.
    cache_dir : str
        Directory for cached CSV. Created if absent. `~` is expanded.
    frac : (float, float, float)
        (train, valid, test) fractions; must sum to 1.0.
    encode : bool
        If True (default), call M1 `encode_graph`. If False, return RDKit
        Mol objects, useful for the GIN teacher path that does its own
        torch-geometric featurisation.

    Returns
    -------
    graphs : List[GraphTensor] | List[rdkit.Chem.Mol]
        Length = n_kept, sanitisation-survivor order.
    labels : np.ndarray[int64]
        Shape (n_kept,), aligned with `graphs`.
    metadata : dict
        Schema documented in the module docstring. Carries
        `split_indices = {"train", "valid", "test"}` as np.int64 arrays.

    Raises
    ------
    MissingDependency
        If neither TDC nor a cached CSV is available, or if rdkit is missing.
    NotImplementedError
        If `encode=True` but the M1 `encode_graph` symbol is not importable
        yet (parallel build hand-off; the message names the missing module).
    """
    if split not in {"scaffold", "random", "none"}:
        raise ValueError(
            f"split must be 'scaffold', 'random', or 'none'; got {split!r}"
        )

    cache = _resolve_cache_dir(cache_dir)
    csv_path, source = _fetch_tdc_ames(cache)
    smiles_raw, y_raw = _parse_ames_csv(csv_path)
    n_total = len(smiles_raw)
    if n_total == 0:
        raise ValueError(f"AMES CSV at {csv_path} parsed to zero rows.")

    mols, kept_smiles, y_kept, n_dropped = _sanitize_smiles(smiles_raw, y_raw)
    n_kept = len(mols)
    if n_kept == 0:
        raise ValueError(
            f"All {n_total} AMES rows failed sanitisation. "
            f"Check the CSV at {csv_path}."
        )

    # Heavy-atom max, useful for the HGraphTM `max_nodes` compile-time cap.
    n_atoms_max = int(max(m.GetNumHeavyAtoms() for m in mols))

    # Choose split.
    if split == "scaffold":
        tr, va, te = scaffold_split(kept_smiles, frac=frac, seed=seed)
    elif split == "random":
        tr, va, te = _random_split(n_kept, frac, seed)
    else:  # split == "none"
        tr = np.arange(n_kept, dtype=np.int64)
        va = np.empty(0, dtype=np.int64)
        te = np.empty(0, dtype=np.int64)

    # Optional M1 encoding.
    if encode:
        graphs = _encode_with_m1(mols)
    else:
        graphs = mols

    n_pos = int((y_kept == 1).sum())
    n_neg = int((y_kept == 0).sum())

    metadata = {
        "n_total_smiles": n_total,
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "n_atoms_max": n_atoms_max,
        "class_balance": {0: n_neg, 1: n_pos},
        "split_indices": {"train": tr, "valid": va, "test": te},
        "split_sizes": (len(tr), len(va), len(te)),
        "split": split,
        "seed": seed,
        "frac": frac,
        "smiles": kept_smiles,
        "source": source,
        "cache_path": str(csv_path),
    }

    return graphs, y_kept, metadata


__all__ = ["load_tdc_ames"]
