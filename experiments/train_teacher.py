"""Train the GIN teacher and save soft + hard predictions.

CLI:
    python -m experiments.train_teacher \
        --dataset ames \
        --epochs 80 \
        --seed 42 \
        --output  results/teacher_pred_<ts>.json \
        --model_path results/teacher_<ts>.pt

The script:
  1. Loads the chosen dataset via ``graphtm.data`` (M6).
  2. Calls ``graphtm.distill.teacher.train_teacher`` (M4) to fit a 3-layer GIN.
  3. Computes soft (sigmoid) and hard (argmax/threshold) predictions over the
     full training split and any held-out split provided by the loader.
  4. Persists the model checkpoint with ``torch.save`` and a JSON sidecar
     containing predictions, labels, split indices, and metadata.

This script is import-only against the M4/M6 contracts; if those modules are
not yet present the script raises ``NotImplementedError`` naming the module.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

from experiments._cli import (
    add_common_args,
    resolve_project_root,
    results_dir,
    safe_write_json,
    seed_all,
    setup_logging,
    timestamp,
)

log = logging.getLogger("train_teacher")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_teacher",
        description="Train GIN teacher and save predictions for distillation.",
    )
    add_common_args(p)
    p.add_argument("--epochs", type=int, default=80, help="GIN training epochs.")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to JSON sidecar (predictions, labels, splits, metadata). "
            "Default: results/teacher_pred_<ts>.json"
        ),
    )
    p.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to save .pt checkpoint. Default: results/teacher_<ts>.pt",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional run tag substituted into default output paths.",
    )
    return p


# ---------------------------------------------------------------------------
# Dataset dispatch
# ---------------------------------------------------------------------------

def _load_dataset(dataset: str, cache_dir: str | None):
    """Dispatch to the right M6 loader.

    Returns ``(graphs, y, meta)``; ``meta`` is the dict returned by the loader
    and is expected to carry split indices.
    """
    log.info("loading dataset: %s", dataset)
    try:
        if dataset == "ames":
            from graphtm.data.ames import load_tdc_ames

            graphs, y, meta = load_tdc_ames(split="scaffold")
        elif dataset == "tox21":
            from graphtm.data.tox21 import load_tox21

            graphs, y, meta = load_tox21(task="NR-AhR", split="scaffold")
        else:
            raise ValueError(f"unknown dataset: {dataset}")
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            f"M6 loader for '{dataset}' is not available, "
            f"graphtm.data is owned by Module M6. Original error: {exc}"
        ) from exc

    log.info(
        "loaded %d graphs (cache_dir=%s); y shape=%r",
        len(graphs), cache_dir, getattr(y, "shape", None),
    )
    return graphs, y, meta


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _train_teacher(graphs, y, *, epochs: int, lr: float):
    """Delegate to ``graphtm.distill.teacher.train_teacher`` (M4)."""
    try:
        from graphtm.distill.teacher import train_teacher as _tt
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            "graphtm.distill.teacher.train_teacher not available, "
            "Module M4 (distill/) owns this symbol. Original error: " + str(exc)
        ) from exc

    log.info("training GIN teacher: epochs=%d lr=%.2e", epochs, lr)
    t0 = time.perf_counter()
    model, soft = _tt(graphs, y, epochs=epochs, lr=lr)
    dt = time.perf_counter() - t0
    log.info("teacher trained in %.1fs", dt)
    return model, soft, dt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    ts = args.tag or timestamp()
    setup_logging("train_teacher", tag=ts)
    log.info("args: %s", vars(args))

    seed_all(args.seed)

    out_json = Path(args.output) if args.output else results_dir() / f"teacher_pred_{ts}.json"
    out_model = Path(args.model_path) if args.model_path else results_dir() / f"teacher_{ts}.pt"

    # M6, load data
    graphs, y, meta = _load_dataset(args.dataset, args.cache_dir)

    # M4, train teacher
    model, soft, train_seconds = _train_teacher(
        graphs, y, epochs=args.epochs, lr=args.lr,
    )

    # ---- predictions
    import numpy as np  # heavy import deferred
    soft = np.asarray(soft)
    if soft.ndim == 1:
        # binary task, interpret as P(class=1)
        hard = (soft >= 0.5).astype(np.int32)
    else:
        hard = soft.argmax(axis=-1).astype(np.int32)

    # ---- persist model
    out_model.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        torch.save(
            {
                "state_dict": model.state_dict() if hasattr(model, "state_dict") else None,
                "args": vars(args),
                "dataset": args.dataset,
                "seed": args.seed,
            },
            out_model,
        )
        log.info("saved teacher checkpoint: %s", out_model)
    except ImportError as exc:
        log.warning("torch not available, skipping .pt save: %s", exc)

    # ---- persist JSON
    payload: Dict[str, Any] = {
        "schema": "graphtm-cbr/teacher_pred/v1",
        "timestamp_utc": ts,
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "n_graphs": int(len(graphs)),
        "train_seconds": float(train_seconds),
        "soft_pred": soft.tolist(),
        "hard_pred": hard.tolist(),
        "y_true": np.asarray(y).tolist() if y is not None else None,
        "split": _serialize_splits(meta),
        "model_path": str(out_model),
        "meta": _safe_meta(meta),
    }
    written = safe_write_json(out_json, payload)
    log.info("saved teacher predictions: %s", written)

    # surface to caller (full_pipeline)
    print(f"TEACHER_PRED_JSON={written}")
    print(f"TEACHER_MODEL_PATH={out_model}")
    return 0


def _serialize_splits(meta: Any) -> Dict[str, Any] | None:
    """Best-effort serialization of split indices from the loader meta dict."""
    if not isinstance(meta, dict):
        return None
    out: Dict[str, Any] = {}
    for key in ("train_idx", "valid_idx", "test_idx", "train", "valid", "test"):
        v = meta.get(key)
        if v is None:
            continue
        try:
            out[key] = [int(i) for i in v]
        except (TypeError, ValueError):
            out[key] = v
    return out or None


def _safe_meta(meta: Any) -> Any:
    """Strip non-JSON-serializable entries from meta."""
    if not isinstance(meta, dict):
        return None
    clean = {}
    for k, v in meta.items():
        try:
            import json as _j

            _j.dumps(v)
            clean[k] = v
        except (TypeError, ValueError):
            clean[k] = repr(v)[:200]
    return clean


if __name__ == "__main__":
    # Ensure the project root is importable when invoked as a script.
    root = resolve_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.exit(main())
