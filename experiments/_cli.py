"""Shared argparse + logging helpers for the experiments/ scripts (M7).

This module intentionally has no dependencies on the rest of graphtm; it is
loaded by every entrypoint script before any heavy import so that ``--help``
remains fast.

Public surface:
    add_common_args(parser)   ← --seed, --dataset, --cache_dir, --device
    resolve_project_root()    ← always returns <repo>/  (parent of experiments/)
    results_dir()             ← <project_root>/results, ensured to exist
    setup_logging(script)     ← stdout + results/<script>_<ts>.log
    timestamp()               ← compact UTC tag used in artefact names
    safe_write_json(path, ...)  ← writes JSON, refuses paths outside project root
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_project_root() -> Path:
    """Return the absolute path of the project root (parent of experiments/).

    I resolve relative to this file rather than to ``os.getcwd`` so that
    artefacts always land under the same root regardless of where the script
    is invoked from.
    """
    return Path(__file__).resolve().parent.parent


def results_dir() -> Path:
    """Return ``<project_root>/results``, creating it if absent."""
    root = resolve_project_root() / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def timestamp() -> str:
    """Compact UTC tag suitable for filenames: ``YYYYMMDD_HHMMSS``."""
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _assert_within_root(path: Path) -> Path:
    """Refuse to write anywhere outside the project root.

    This is a hard rule for the M7 module: artefacts must live under
    ``<project_root>/`` (typically ``results/``). Symlink shenanigans are
    blocked by resolving both sides.
    """
    root = resolve_project_root().resolve()
    candidate = path.expanduser().resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside project root: {candidate} "
            f"(root = {root})"
        ) from exc
    return candidate


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--seed``, ``--dataset``, ``--cache_dir``, ``--device``.

    These flags are shared across all M7 scripts. Scripts may add additional
    flags after calling this helper.
    """
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for numpy / torch / CUDA. Default: 42.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ames",
        choices=["ames", "tox21"],
        help="Dataset to load via graphtm.data. Default: ames.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for dataset downloads / encoded graphs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device. Default: cuda. CPU paths are not benchmarked.",
    )
    return parser


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(script_name: str, *, tag: Optional[str] = None) -> Path:
    """Set up a logger that writes to stdout AND ``results/<script>_<ts>.log``.

    Returns the log-file path.
    """
    ts = tag or timestamp()
    log_path = results_dir() / f"{script_name}_{ts}.log"

    root_logger = logging.getLogger()
    # Wipe any prior handlers, scripts can be re-invoked in the same process
    # (e.g. by full_pipeline subprocess that imports them).
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(fmt)
    stream.setLevel(logging.INFO)
    root_logger.addHandler(stream)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    root_logger.addHandler(fh)

    logging.getLogger(script_name).info("log file: %s", log_path)
    return log_path


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    """Seed numpy + python + torch (+ cuda if available). No global state."""
    import random as _random
    _random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

class _NumpyJSONEncoder(json.JSONEncoder):
    """Numpy-aware JSON encoder. Falls back to ``str`` for anything exotic."""

    def default(self, o: Any) -> Any:  # noqa: D401
        try:
            import numpy as np  # local import, keep --help fast
        except ImportError:
            np = None  # type: ignore[assignment]

        if np is not None:
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def safe_write_json(path: str | os.PathLike, payload: Any) -> Path:
    """Write ``payload`` as JSON to ``path``. Refuses paths outside the project root."""
    target = _assert_within_root(Path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False, cls=_NumpyJSONEncoder)
    return target


def safe_open_text(path: str | os.PathLike, mode: str = "w"):
    """Open a text file for writing under the project root."""
    if any(ch in mode for ch in ("w", "x", "a")):
        target = _assert_within_root(Path(path))
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target = Path(path)
    return target.open(mode, encoding="utf-8")
