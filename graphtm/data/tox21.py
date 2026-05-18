"""TDC Tox21 confirmatory loader → (graphs, labels, metadata).

Why this exists
---------------
Tox21 is the confirmatory benchmark in `research/05_benchmark_choice.md`:
12 nuclear-receptor / stress-response binary tasks, scaffold split, ~8k
molecules. NR-AhR is the most topology-dependent single task (GCN 0.886 vs
RF 0.81 per `arXiv 1703.00564`). I expose every Tox21 sub-task through the
same loader; default is NR-AhR per the project plan.

The interface mirrors `load_tdc_ames` so downstream code (teacher training,
distillation, evaluation) can be reused without per-dataset branches.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from graphtm.data import MissingDependency
from graphtm.data.scaffold_split import scaffold_split


# Canonical 12-task list (Mayr 2016 / MoleculeNet).
TOX21_TASKS: Tuple[str, ...] = (
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
)

# TDC mirrors the MoleculeNet "tox21.csv.gz" file when called as
# `Tox(name='Tox21')`. I also keep a public URL fallback for offline CI.
_TOX21_FALLBACK_URLS: Tuple[str, ...] = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
)


def _resolve_cache_dir(cache_dir: str) -> Path:
    """Expand ``~`` and create the directory if absent."""
    out = Path(os.path.expanduser(cache_dir)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fetch_tox21(cache_dir: Path, task: str) -> Tuple[Path, str]:
    """Return (csv_path, source_tag).

    Strategy:
      1. TDC `Tox(name='Tox21_<task>')`, primary path.
      2. Pre-existing CSV at `cache_dir/tox21_<task>.csv`.
      3. Public MoleculeNet mirror downloaded once and filtered per `task`.
    """
    task_clean = task.replace("/", "_")
    cached = cache_dir / f"tox21_{task_clean}.csv"

    # Attempt 1: TDC python API.
    try:
        from tdc.single_pred import Tox  # type: ignore
        data = Tox(name=f"Tox21_{task}", path=str(cache_dir))
        df = data.get_data()
        if not cached.exists():
            df.to_csv(cached, index=False)
        return cached, "tdc"
    except ImportError:
        pass
    except Exception:
        # Try generic Tox(name='Tox21') fallback (older TDC versions
        # didn't expose per-task pickles).
        try:
            from tdc.single_pred import Tox  # type: ignore
            data = Tox(name="Tox21", label_name=task, path=str(cache_dir))
            df = data.get_data()
            if not cached.exists():
                df.to_csv(cached, index=False)
            return cached, "tdc"
        except Exception:
            pass

    # Attempt 2: cached CSV.
    if cached.exists() and cached.stat().st_size > 0:
        return cached, "csv_cache"

    # Attempt 3: download MoleculeNet mirror, filter to selected task.
    last_err: Optional[Exception] = None
    moleculenet_raw = cache_dir / "tox21_moleculenet_raw.csv.gz"
    if not moleculenet_raw.exists():
        for url in _TOX21_FALLBACK_URLS:
            try:
                urllib.request.urlretrieve(url, moleculenet_raw)
                if moleculenet_raw.exists() and moleculenet_raw.stat().st_size > 0:
                    break
            except Exception as exc:
                last_err = exc
                continue
        else:
            raise MissingDependency(
                "Could not obtain Tox21 dataset. Install TDC via "
                "`pip install PyTDC`, OR drop a CSV at "
                f"{cached}. Last error: {last_err!r}."
            )

    # Filter MoleculeNet csv to the requested task: keep rows where task ∈ {0,1}
    # (MoleculeNet uses NaN for "not assayed"; I drop those).
    import csv as _csv
    import gzip

    with gzip.open(moleculenet_raw, "rt", newline="") as f:
        reader = _csv.DictReader(f)
        if reader.fieldnames is None or task not in reader.fieldnames:
            raise ValueError(
                f"Tox21 MoleculeNet CSV at {moleculenet_raw} does not contain "
                f"column {task!r}. Columns: {reader.fieldnames}"
            )
        smi_col = "smiles" if "smiles" in reader.fieldnames else "SMILES"
        rows = []
        for row in reader:
            v = row[task]
            if v in (None, "", "nan", "NaN"):
                continue
            try:
                yi = int(float(v))
            except ValueError:
                continue
            rows.append({"Drug": row[smi_col], "Y": yi})

    with cached.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["Drug", "Y"])
        writer.writeheader()
        writer.writerows(rows)

    return cached, "url_mirror"


def _parse_csv(csv_path: Path) -> Tuple[List[str], np.ndarray]:
    """Identical schema-tolerance to `ames._parse_ames_csv`."""
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
                f"recognised SMILES + label pair."
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
    """SMILES → sanitised RDKit Mol; drop unparseable; return diagnostics."""
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
    """See `ames._encode_with_m1`, same M1 hand-off contract."""
    try:
        from graphtm.encoding.codebook import make_codebook
        from graphtm.encoding.graph_features import encode_graph  # type: ignore
    except ImportError as exc:
        raise NotImplementedError(
            "graphtm.encoding.graph_features.encode_graph (M1) is not "
            "available yet. The Tox21 loader has produced sanitised RDKit "
            "molecules but cannot turn them into GraphTensors until M1 is "
            f"merged. Original import error: {exc!r}"
        ) from exc

    cb = make_codebook(seed=0)
    return [encode_graph(m, cb, k_hop=cb.k_hop) for m in mols]


def _random_split(
    n: int, frac: Tuple[float, float, float], seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(np.floor(frac[0] * n))
    n_valid = int(np.floor(frac[1] * n))
    train = np.sort(perm[:n_train])
    valid = np.sort(perm[n_train : n_train + n_valid])
    test = np.sort(perm[n_train + n_valid :])
    return train, valid, test


def load_tox21(
    task: str = "NR-AhR",
    split: str = "scaffold",
    seed: int = 42,
    cache_dir: str = "~/.graphtm_cache",
    frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    encode: bool = True,
):
    """Load one Tox21 sub-task → (graphs, labels, metadata).

    Parameters
    ----------
    task : str
        One of `TOX21_TASKS`. Default ``"NR-AhR"`` per project plan.
    split : {"scaffold", "random", "none"}
        Same semantics as `load_tdc_ames`.
    seed : int
        RNG seed (deterministic per seed).
    cache_dir : str
        Directory for cached CSV. `~` is expanded; created if absent.
    frac : (float, float, float)
        (train, valid, test) fractions.
    encode : bool
        If True (default), invoke M1 `encode_graph`. If False, return Mols.

    Returns
    -------
    graphs : List[GraphTensor] | List[rdkit.Chem.Mol]
    labels : np.ndarray[int64]
    metadata : dict
        Schema identical to `load_tdc_ames`, plus ``task`` field with the
        Tox21 task name.

    Raises
    ------
    MissingDependency
        If neither TDC nor a cached/mirror CSV is available, or rdkit missing.
    NotImplementedError
        If `encode=True` but M1 `encode_graph` not yet importable.
    ValueError
        If `task` is not in `TOX21_TASKS`.
    """
    if task not in TOX21_TASKS:
        raise ValueError(
            f"task must be one of {TOX21_TASKS}; got {task!r}"
        )
    if split not in {"scaffold", "random", "none"}:
        raise ValueError(
            f"split must be 'scaffold', 'random', or 'none'; got {split!r}"
        )

    cache = _resolve_cache_dir(cache_dir)
    csv_path, source = _fetch_tox21(cache, task)
    smiles_raw, y_raw = _parse_csv(csv_path)
    n_total = len(smiles_raw)
    if n_total == 0:
        raise ValueError(f"Tox21 CSV at {csv_path} parsed to zero rows.")

    mols, kept_smiles, y_kept, n_dropped = _sanitize_smiles(smiles_raw, y_raw)
    n_kept = len(mols)
    if n_kept == 0:
        raise ValueError(
            f"All {n_total} Tox21 rows failed sanitisation. Check {csv_path}."
        )

    n_atoms_max = int(max(m.GetNumHeavyAtoms() for m in mols))

    if split == "scaffold":
        tr, va, te = scaffold_split(kept_smiles, frac=frac, seed=seed)
    elif split == "random":
        tr, va, te = _random_split(n_kept, frac, seed)
    else:
        tr = np.arange(n_kept, dtype=np.int64)
        va = np.empty(0, dtype=np.int64)
        te = np.empty(0, dtype=np.int64)

    if encode:
        graphs = _encode_with_m1(mols)
    else:
        graphs = mols

    n_pos = int((y_kept == 1).sum())
    n_neg = int((y_kept == 0).sum())

    metadata = {
        "task": task,
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


__all__ = ["load_tox21", "TOX21_TASKS"]
