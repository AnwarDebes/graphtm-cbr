"""Per-molecule recourse evaluation: clause-walk + greedy edit + validity stats.

CLI:
    python -m experiments.eval_recourse \
        --model_path results/student_<ts>.pkl \
        --dataset    ames \
        --n_test     200 \
        --max_flips  3 \
        --output_md  results/recourse_<ts>.md \
        --output_json results/recourse_<ts>.json

For each test molecule predicted positive by the student:
  1. ``model.firing_clauses(graph)``                          (M3)
  2. ``candidates_from_firing_clauses(graph, firing, ...)``    (M5)
  3. ``greedy_minimal_edit(model, graph, candidates, ...)``      (M5)
  4. RDKit validity via ``recourse.validity.validate``         (M5)
  5. Record (edits, validity flags, latency).

Aggregates:
  - % recourse success (count flipped to negative)
  - mean / median flip count
  - latency p50 / p90 / p99
  - validity-rate (% of edited molecules that pass RDKit sanitize)

Outputs:
  - JSON sidecar with per-molecule records and aggregate metrics
  - Markdown report with the same content human-formatted
"""
from __future__ import annotations

import argparse
import logging
import math
import pickle
import statistics
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments._cli import (
    add_common_args,
    resolve_project_root,
    results_dir,
    safe_open_text,
    safe_write_json,
    seed_all,
    setup_logging,
    timestamp,
)

log = logging.getLogger("eval_recourse")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_recourse",
        description="Evaluate clause-driven counterfactual recourse over a test set.",
    )
    add_common_args(p)
    p.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the pickled student model (.pkl) from train_student.py.",
    )
    p.add_argument(
        "--n_test", type=int, default=200,
        help="Number of test molecules to evaluate (predicted-positive subset).",
    )
    p.add_argument(
        "--max_flips", type=int, default=3,
        help="Greedy edit budget per molecule.",
    )
    p.add_argument(
        "--max_candidates", type=int, default=50,
        help="Cap on edit candidates generated per firing-clause set.",
    )
    p.add_argument(
        "--output_md", type=str, default=None,
        help="Markdown report path. Default: results/recourse_<ts>.md",
    )
    p.add_argument(
        "--output_json", type=str, default=None,
        help="JSON sidecar path. Default: results/recourse_<ts>.json",
    )
    p.add_argument(
        "--tag", type=str, default=None,
        help="Optional run tag.",
    )
    return p


# ---------------------------------------------------------------------------
# Data + model loading
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
            f"M6 loader for '{dataset}' is not available, graphtm.data is "
            f"owned by Module M6. Original error: {exc}"
        ) from exc


def _resolve_test_indices(meta: Any, *, n_total: int, n_test: int) -> List[int]:
    """Return up to ``n_test`` indices from the test split, falling back to tail."""
    if isinstance(meta, dict):
        for key in ("test_idx", "test"):
            idx = meta.get(key)
            if idx is None:
                continue
            try:
                return [int(i) for i in idx][:n_test]
            except (TypeError, ValueError):
                pass
    # fallback: last n_test indices
    start = max(0, n_total - n_test)
    return list(range(start, n_total))


def _load_student(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Recourse helpers (import M3/M5 lazily)
# ---------------------------------------------------------------------------

def _firing_clauses(model, graph):
    if not hasattr(model, "firing_clauses"):
        raise NotImplementedError(
            "model.firing_clauses(graph) is not available, Module M3 (core/) "
            "owns this method."
        )
    return model.firing_clauses(graph)


def _candidates_from_firing(graph, firing, *, max_candidates: int):
    try:
        from graphtm.recourse.candidates import candidates_from_firing_clauses
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            "graphtm.recourse.candidates.candidates_from_firing_clauses not "
            "available, Module M5 (recourse/) owns this symbol. "
            f"Original error: {exc}"
        ) from exc
    return candidates_from_firing_clauses(graph, firing, max_candidates=max_candidates)


def _greedy_edit(model, graph, candidates, *, max_flips: int):
    try:
        from graphtm.recourse.search import greedy_minimal_edit
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            "graphtm.recourse.search.greedy_minimal_edit not available, "
            "Module M5 (recourse/) owns this symbol. Original error: " + str(exc)
        ) from exc
    return greedy_minimal_edit(model, graph, candidates, max_flips=max_flips)


def _validate(mol_after_edit):
    try:
        from graphtm.recourse.validity import validate
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            "graphtm.recourse.validity.validate not available, "
            "Module M5 (recourse/) owns this symbol. Original error: " + str(exc)
        ) from exc
    return validate(mol_after_edit)


def _apply_edits(graph, edits):
    """Apply a list of GraphEdit ops to produce a post-edit graph/mol.

    The actual implementation lives in M5 (graph mutation requires RDKit
    awareness). I probe the recourse package for an ``apply_edits`` helper;
    fall back to a flag in the record if it's not present.
    """
    try:
        from graphtm.recourse.candidates import apply_edits as _ae

        return _ae(graph, edits)
    except (ImportError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    sv = sorted(values)
    k = (len(sv) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sv[int(k)])
    return float(sv[f] + (sv[c] - sv[f]) * (k - f))


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = len(records)
    succeeded = [r for r in records if r["recourse_found"]]
    flips = [r["n_flips"] for r in succeeded if r["n_flips"] is not None]
    latencies = [r["latency_ms"] for r in records if r["latency_ms"] is not None]
    valid = [r for r in succeeded if r.get("validity", {}).get("valid")]

    return {
        "n_total": n_total,
        "n_recourse_success": len(succeeded),
        "recourse_success_rate": (len(succeeded) / n_total) if n_total else 0.0,
        "mean_flip_count": float(statistics.mean(flips)) if flips else float("nan"),
        "median_flip_count": float(statistics.median(flips)) if flips else float("nan"),
        "max_flip_count": max(flips) if flips else None,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p90": _percentile(latencies, 0.90),
        "latency_ms_p99": _percentile(latencies, 0.99),
        "latency_ms_mean": float(statistics.mean(latencies)) if latencies else float("nan"),
        "validity_rate": (len(valid) / len(succeeded)) if succeeded else 0.0,
    }


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

def _write_markdown(path: Path, args, aggregate: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    with safe_open_text(path, "w") as f:
        f.write("# Recourse evaluation report\n\n")
        f.write(f"- Generated: `{timestamp()}` UTC\n")
        f.write(f"- Dataset: `{args.dataset}`\n")
        f.write(f"- Model: `{args.model_path}`\n")
        f.write(f"- Seed: `{args.seed}`\n")
        f.write(f"- max_flips: `{args.max_flips}`  max_candidates: `{args.max_candidates}`\n\n")

        f.write("## Aggregate metrics\n\n")
        f.write("| metric | value |\n|---|---|\n")
        for k, v in aggregate.items():
            f.write(f"| `{k}` | {_fmt(v)} |\n")
        f.write("\n")

        f.write("## Per-molecule records (first 25)\n\n")
        f.write("| idx | pred | recourse | n_flips | latency_ms | valid | edits |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in records[:25]:
            edits = r.get("edits") or []
            edits_str = "; ".join(_fmt_edit(e) for e in edits) if edits else "-"
            f.write(
                f"| {r['idx']} | {r['pred']} | {r['recourse_found']} | "
                f"{r['n_flips']} | {_fmt(r['latency_ms'])} | "
                f"{r.get('validity', {}).get('valid')} | {edits_str} |\n"
            )


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def _fmt_edit(e: Any) -> str:
    if isinstance(e, dict):
        return f"{e.get('op')}({e.get('indices')}->{e.get('new_value')})"
    if is_dataclass(e):
        d = asdict(e)
        return f"{d.get('op')}({d.get('indices')}->{d.get('new_value')})"
    return str(e)


def _edit_to_dict(e: Any) -> Any:
    if is_dataclass(e):
        return asdict(e)
    if isinstance(e, dict):
        return e
    return str(e)


def _validity_to_dict(v: Any) -> Dict[str, Any]:
    if v is None:
        return {"valid": None}
    if is_dataclass(v):
        return asdict(v)
    if isinstance(v, dict):
        return v
    if hasattr(v, "__dict__"):
        return {k: getattr(v, k) for k in vars(v)}
    return {"raw": str(v)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    ts = args.tag or timestamp()
    setup_logging("eval_recourse", tag=ts)
    log.info("args: %s", vars(args))

    seed_all(args.seed)

    out_md = Path(args.output_md) if args.output_md else results_dir() / f"recourse_{ts}.md"
    out_json = Path(args.output_json) if args.output_json else results_dir() / f"recourse_{ts}.json"

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        log.error("model file not found: %s", model_path)
        return 2

    log.info("loading student model: %s", model_path)
    student = _load_student(model_path)

    log.info("loading dataset: %s", args.dataset)
    graphs, y, meta = _load_dataset(args.dataset)
    test_idx = _resolve_test_indices(meta, n_total=len(graphs), n_test=args.n_test)
    log.info("test set: %d graphs (of %d total)", len(test_idx), len(graphs))

    # Predict to find positives
    test_graphs = [graphs[i] for i in test_idx]
    log.info("scoring student on test split")
    try:
        preds = student.predict(test_graphs)
    except AttributeError as exc:
        raise NotImplementedError(
            "student.predict() not available, Module M3 (core/) owns this method."
        ) from exc

    import numpy as np

    preds = np.asarray(preds)
    pos_mask = (preds.argmax(axis=-1) == 1) if preds.ndim > 1 else (preds >= 0.5)
    pos_idx_local = [i for i, m in enumerate(pos_mask) if bool(m)]
    log.info("predicted-positive molecules: %d / %d", len(pos_idx_local), len(test_idx))

    records: List[Dict[str, Any]] = []
    for j, local_i in enumerate(pos_idx_local):
        i = test_idx[local_i]
        graph = graphs[i]

        t0 = time.perf_counter()
        try:
            firing = _firing_clauses(student, graph)
            candidates = _candidates_from_firing(
                graph, firing, max_candidates=args.max_candidates,
            )
            edits = _greedy_edit(
                student, graph, candidates, max_flips=args.max_flips,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
        except NotImplementedError:
            # propagate, this is a hard contract failure
            raise
        except Exception as exc:  # noqa: BLE001, record-and-continue policy
            dt_ms = (time.perf_counter() - t0) * 1000.0
            log.warning("molecule %d: recourse failed with %s", i, exc)
            records.append({
                "idx": int(i),
                "pred": int(preds[local_i].argmax()) if preds.ndim > 1 else float(preds[local_i]),
                "n_firing_clauses": None,
                "n_candidates": None,
                "recourse_found": False,
                "n_flips": None,
                "edits": None,
                "validity": {"valid": None, "error": str(exc)},
                "latency_ms": dt_ms,
            })
            continue

        # Validate post-edit molecule
        validity: Dict[str, Any] = {"valid": None}
        mol_after = _apply_edits(graph, edits) if edits else None
        if edits and mol_after is not None:
            try:
                vr = _validate(mol_after)
                validity = _validity_to_dict(vr)
            except NotImplementedError:
                raise
            except Exception as exc:  # noqa: BLE001
                validity = {"valid": False, "error": str(exc)}

        records.append({
            "idx": int(i),
            "pred": int(preds[local_i].argmax()) if preds.ndim > 1 else float(preds[local_i]),
            "n_firing_clauses": _safe_len(firing),
            "n_candidates": _safe_len(candidates),
            "recourse_found": bool(edits),
            "n_flips": _safe_len(edits) if edits else None,
            "edits": [_edit_to_dict(e) for e in edits] if edits else None,
            "validity": validity,
            "latency_ms": dt_ms,
        })

        if (j + 1) % 25 == 0 or j == len(pos_idx_local) - 1:
            log.info("recourse: %d / %d done", j + 1, len(pos_idx_local))

    aggregate = _aggregate(records)
    log.info("aggregate metrics: %s", aggregate)

    payload = {
        "schema": "graphtm-cbr/recourse_eval/v1",
        "timestamp_utc": ts,
        "dataset": args.dataset,
        "model_path": str(model_path),
        "seed": args.seed,
        "n_test": int(args.n_test),
        "max_flips": int(args.max_flips),
        "max_candidates": int(args.max_candidates),
        "aggregate": aggregate,
        "records": records,
    }
    written_json = safe_write_json(out_json, payload)
    log.info("saved JSON sidecar: %s", written_json)

    _write_markdown(out_md, args, aggregate, records)
    log.info("saved Markdown report: %s", out_md)

    print(f"RECOURSE_JSON={written_json}")
    print(f"RECOURSE_MD={out_md}")
    return 0


def _safe_len(x: Any) -> Optional[int]:
    try:
        return int(len(x))
    except TypeError:
        return None


if __name__ == "__main__":
    root = resolve_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.exit(main())
