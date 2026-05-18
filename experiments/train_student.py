"""Train the HierarchicalGraphTM student by distillation from the GIN teacher.

CLI:
    python -m experiments.train_student \
        --dataset ames \
        --teacher_pred results/teacher_pred_<ts>.json \
        --clauses 2000 \
        --T 50 \
        --s 5.0 \
        --epochs 100 \
        --seed 42 \
        --output  results/student_metrics_<ts>.json \
        --model_path results/student_<ts>.pkl

The script:
  1. Loads encoded graphs via the M6 loaders (must match the dataset that was
     used for the teacher run).
  2. Loads the teacher soft predictions from the JSON sidecar emitted by
     ``train_teacher.py``.
  3. Constructs ``HierarchicalGraphTM(HGraphTMSpec(...))`` (M3).
  4. Calls ``graphtm.distill.student.distill`` (M4) for ``--epochs`` epochs.
  5. Persists the trained student via ``pickle`` and a JSON metrics file.

If a required symbol from M3/M4/M6 is missing, the script raises
``NotImplementedError`` naming the responsible module.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
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

log = logging.getLogger("train_student")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_student",
        description="Distill the HierarchicalGraphTM student from teacher soft preds.",
    )
    add_common_args(p)
    p.add_argument(
        "--teacher_pred",
        type=str,
        required=True,
        help="Path to the teacher JSON sidecar (output of train_teacher.py).",
    )
    p.add_argument("--clauses", type=int, default=2000, help="Number of clauses (C).")
    p.add_argument("--T", type=int, default=50, help="Vote-sum clip threshold T.")
    p.add_argument("--s", type=float, default=5.0, help="Specificity s.")
    p.add_argument("--epochs", type=int, default=100, help="Student training epochs.")
    p.add_argument(
        "--n_states", type=int, default=200,
        help="TA states (n_states in HGraphTMSpec).",
    )
    p.add_argument(
        "--max_nodes", type=int, default=80,
        help="Compile-time per-graph node cap.",
    )
    p.add_argument(
        "--D_bits", type=int, default=8192,
        help="Hypervector dimension (D_bits in HGraphTMSpec).",
    )
    p.add_argument(
        "--k_hop", type=int, default=2,
        help="k-hop role binding depth.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to JSON metrics sidecar. Default: results/student_metrics_<ts>.json",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to save student .pkl. Default: results/student_<ts>.pkl",
    )
    p.add_argument(
        "--tag", type=str, default=None,
        help="Optional run tag substituted into default output paths.",
    )
    return p


# ---------------------------------------------------------------------------
# Dataset (must match the teacher run)
# ---------------------------------------------------------------------------

def _load_dataset(dataset: str):
    try:
        if dataset == "ames":
            from graphtm.data.ames import load_tdc_ames

            return load_tdc_ames(split="scaffold")
        elif dataset == "tox21":
            from graphtm.data.tox21 import load_tox21

            return load_tox21(task="NR-AhR", split="scaffold")
        raise ValueError(f"unknown dataset: {dataset}")
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            f"M6 loader for '{dataset}' is not available, "
            f"graphtm.data is owned by Module M6. Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Build the student
# ---------------------------------------------------------------------------

def _build_student(args: argparse.Namespace, *, n_classes: int):
    """Instantiate ``HierarchicalGraphTM(HGraphTMSpec(...))`` (M3)."""
    try:
        from graphtm.core.hierarchical_graph_tm import (
            HGraphTMSpec,
            HierarchicalGraphTM,
        )
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            "graphtm.core.hierarchical_graph_tm.{HGraphTMSpec,HierarchicalGraphTM} "
            "not available, Module M3 (core/) owns these symbols. "
            f"Original error: {exc}"
        ) from exc

    spec = HGraphTMSpec(
        n_classes=n_classes,
        n_clauses=args.clauses,
        threshold=args.T,
        s=args.s,
        n_states=args.n_states,
        D_bits=args.D_bits,
        k_hop=args.k_hop,
        max_nodes=args.max_nodes,
        seed=args.seed,
    )
    log.info("HGraphTMSpec: %s", spec)
    student = HierarchicalGraphTM(spec, device=args.device)
    return student, spec


# ---------------------------------------------------------------------------
# Distill
# ---------------------------------------------------------------------------

def _distill(student, graphs, y_teacher, y_true, *, epochs: int) -> Dict[str, Any]:
    try:
        from graphtm.distill.student import distill
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            "graphtm.distill.student.distill not available, "
            "Module M4 (distill/) owns this symbol. Original error: " + str(exc)
        ) from exc

    log.info("running distillation: epochs=%d n_graphs=%d", epochs, len(graphs))
    t0 = time.perf_counter()
    metrics = distill(student, graphs, y_teacher=y_teacher, y_true=y_true, epochs=epochs)
    dt = time.perf_counter() - t0
    log.info("distillation finished in %.1fs", dt)
    if not isinstance(metrics, dict):
        log.warning("distill() did not return a dict, wrapping for serialization")
        metrics = {"raw": repr(metrics)}
    metrics.setdefault("distill_seconds", dt)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    ts = args.tag or timestamp()
    setup_logging("train_student", tag=ts)
    log.info("args: %s", vars(args))

    seed_all(args.seed)

    out_json = Path(args.output) if args.output else results_dir() / f"student_metrics_{ts}.json"
    out_model = Path(args.model_path) if args.model_path else results_dir() / f"student_{ts}.pkl"

    # ---- load teacher preds
    teacher_path = Path(args.teacher_pred).expanduser().resolve()
    if not teacher_path.exists():
        log.error("teacher prediction file not found: %s", teacher_path)
        return 2
    with teacher_path.open("r", encoding="utf-8") as f:
        teacher_blob = json.load(f)
    log.info(
        "loaded teacher preds: schema=%s dataset=%s n=%d",
        teacher_blob.get("schema"), teacher_blob.get("dataset"),
        len(teacher_blob.get("soft_pred", [])),
    )

    if teacher_blob.get("dataset") and teacher_blob["dataset"] != args.dataset:
        log.warning(
            "teacher dataset (%s) != requested dataset (%s), proceeding anyway",
            teacher_blob["dataset"], args.dataset,
        )

    import numpy as np

    y_teacher = np.asarray(teacher_blob["soft_pred"])
    y_true = np.asarray(teacher_blob["y_true"]) if teacher_blob.get("y_true") is not None else None

    # ---- load data
    graphs, y, _meta = _load_dataset(args.dataset)
    if y_true is None:
        y_true = np.asarray(y)
    if len(graphs) != y_teacher.shape[0]:
        log.warning(
            "graph count (%d) != teacher pred count (%d); distill driver must align",
            len(graphs), y_teacher.shape[0],
        )

    # ---- decide n_classes
    if y_teacher.ndim == 1:
        n_classes = 2
    else:
        n_classes = int(y_teacher.shape[-1])

    # ---- build student
    student, spec = _build_student(args, n_classes=n_classes)

    # ---- distill
    metrics = _distill(
        student, graphs,
        y_teacher=y_teacher, y_true=y_true,
        epochs=args.epochs,
    )

    # ---- save model (pickle, the contract says .pkl)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_model.open("wb") as f:
            pickle.dump(student, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("saved student model: %s", out_model)
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        log.warning("could not pickle student (likely contains CUDA handles): %s", exc)

    # ---- save metrics JSON
    payload: Dict[str, Any] = {
        "schema": "graphtm-cbr/student_metrics/v1",
        "timestamp_utc": ts,
        "dataset": args.dataset,
        "seed": args.seed,
        "spec": _spec_to_dict(spec),
        "epochs": args.epochs,
        "n_graphs": int(len(graphs)),
        "teacher_pred_path": str(teacher_path),
        "model_path": str(out_model),
        "metrics": metrics,
    }
    written = safe_write_json(out_json, payload)
    log.info("saved student metrics: %s", written)

    print(f"STUDENT_METRICS_JSON={written}")
    print(f"STUDENT_MODEL_PATH={out_model}")
    return 0


def _spec_to_dict(spec: Any) -> Dict[str, Any]:
    """Best-effort dataclass→dict for HGraphTMSpec."""
    try:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(spec):
            return asdict(spec)
    except ImportError:
        pass
    return {k: getattr(spec, k) for k in dir(spec) if not k.startswith("_") and not callable(getattr(spec, k))}


if __name__ == "__main__":
    root = resolve_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.exit(main())
