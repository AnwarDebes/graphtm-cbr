"""End-to-end orchestration: teacher → student → recourse → unified report.

CLI:
    python -m experiments.full_pipeline \
        --seed 42 \
        --budget_minutes 30 \
        [--dataset ames] \
        [--teacher_epochs 80] \
        [--student_epochs 100] \
        [--clauses 2000] [--T 50] [--s 5.0] \
        [--n_test 200] [--max_flips 3]

This script shells out to ``train_teacher``, ``train_student``, and
``eval_recourse`` via ``subprocess`` so that each phase has its own log file
and process-level isolation. The unified ``results/full_run_<ts>.json``
captures:
  - phase invocation arguments
  - artefact paths emitted by each phase
  - wall-clock elapsed time
  - exit codes
  - budget status (under / over)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments._cli import (
    add_common_args,
    resolve_project_root,
    results_dir,
    safe_write_json,
    seed_all,
    setup_logging,
    timestamp,
)

log = logging.getLogger("full_pipeline")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="full_pipeline",
        description="Orchestrate teacher → student → recourse end-to-end.",
    )
    add_common_args(p)
    p.add_argument(
        "--budget_minutes", type=float, default=30.0,
        help="Wall-clock budget for the full run (used for reporting; "
             "subprocesses are NOT hard-killed mid-phase).",
    )
    # Teacher
    p.add_argument("--teacher_epochs", type=int, default=80)
    p.add_argument("--teacher_lr", type=float, default=1e-3)
    # Student
    p.add_argument("--student_epochs", type=int, default=100)
    p.add_argument("--clauses", type=int, default=2000)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--s", type=float, default=5.0)
    p.add_argument("--n_states", type=int, default=200)
    p.add_argument("--D_bits", type=int, default=8192)
    p.add_argument("--k_hop", type=int, default=2)
    p.add_argument("--max_nodes", type=int, default=80)
    # Recourse
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--max_flips", type=int, default=3)
    p.add_argument("--max_candidates", type=int, default=50)
    p.add_argument(
        "--tag", type=str, default=None,
        help="Optional run tag substituted into default artefact paths.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path to unified results JSON. Default: results/full_run_<ts>.json",
    )
    p.add_argument(
        "--skip_recourse", action="store_true",
        help="Skip phase 3 (recourse). Useful for fast iteration.",
    )
    return p


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run_phase(name: str, argv: List[str]) -> Dict[str, Any]:
    """Run a phase script via ``subprocess.run``. Stream stdout, return record.

    I always invoke ``python -m experiments.<name>`` so the package import
    path is consistent regardless of cwd.
    """
    root = resolve_project_root()
    cmd = [sys.executable, "-m", f"experiments.{name}", *argv]
    log.info("==> phase %s: %s", name, " ".join(cmd))

    env = os.environ.copy()
    # Make sure subprocesses can import `experiments` and `graphtm`.
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{pp}" if pp else str(root)

    t0 = time.perf_counter()
    proc = subprocess.run(  # noqa: PLW1510, I explicitly check returncode
        cmd,
        capture_output=True,
        text=True,
        cwd=str(root),
        env=env,
        check=False,
    )
    dt = time.perf_counter() - t0

    # Stream captured stdout/stderr to my own log (and stdout).
    if proc.stdout:
        for line in proc.stdout.rstrip("\n").splitlines():
            log.info("[%s] %s", name, line)
    if proc.stderr:
        for line in proc.stderr.rstrip("\n").splitlines():
            log.warning("[%s][stderr] %s", name, line)

    artefacts = _extract_artefacts(proc.stdout or "")
    log.info("<== phase %s done in %.1fs (rc=%d)", name, dt, proc.returncode)

    return {
        "name": name,
        "argv": argv,
        "returncode": int(proc.returncode),
        "seconds": float(dt),
        "stdout_artefacts": artefacts,
    }


def _extract_artefacts(stdout: str) -> Dict[str, str]:
    """Pick up ``KEY=value`` markers printed by the phase scripts.

    Phase scripts print machine-readable markers on stdout like::
        TEACHER_PRED_JSON=/abs/path/results/teacher_pred_xxx.json
    """
    out: Dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        if k.isupper() and v and not v.startswith(" "):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Phase argv builders
# ---------------------------------------------------------------------------

def _teacher_argv(args, *, tag: str, pred_path: Path, model_path: Path) -> List[str]:
    return [
        "--dataset", args.dataset,
        "--seed", str(args.seed),
        "--device", args.device,
        "--epochs", str(args.teacher_epochs),
        "--lr", str(args.teacher_lr),
        "--output", str(pred_path),
        "--model_path", str(model_path),
        "--tag", tag,
        *((["--cache_dir", args.cache_dir]) if args.cache_dir else []),
    ]


def _student_argv(args, *, tag: str, teacher_pred: Path,
                  metrics_path: Path, model_path: Path) -> List[str]:
    return [
        "--dataset", args.dataset,
        "--seed", str(args.seed),
        "--device", args.device,
        "--teacher_pred", str(teacher_pred),
        "--clauses", str(args.clauses),
        "--T", str(args.T),
        "--s", str(args.s),
        "--epochs", str(args.student_epochs),
        "--n_states", str(args.n_states),
        "--D_bits", str(args.D_bits),
        "--k_hop", str(args.k_hop),
        "--max_nodes", str(args.max_nodes),
        "--output", str(metrics_path),
        "--model_path", str(model_path),
        "--tag", tag,
        *((["--cache_dir", args.cache_dir]) if args.cache_dir else []),
    ]


def _recourse_argv(args, *, tag: str, model_path: Path,
                   out_json: Path, out_md: Path) -> List[str]:
    return [
        "--dataset", args.dataset,
        "--seed", str(args.seed),
        "--device", args.device,
        "--model_path", str(model_path),
        "--n_test", str(args.n_test),
        "--max_flips", str(args.max_flips),
        "--max_candidates", str(args.max_candidates),
        "--output_md", str(out_md),
        "--output_json", str(out_json),
        "--tag", tag,
        *((["--cache_dir", args.cache_dir]) if args.cache_dir else []),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    ts = args.tag or timestamp()
    setup_logging("full_pipeline", tag=ts)
    log.info("args: %s", vars(args))
    log.info("project root: %s", resolve_project_root())

    seed_all(args.seed)

    results = results_dir()
    out_json = Path(args.output) if args.output else results / f"full_run_{ts}.json"

    teacher_pred = results / f"teacher_pred_{ts}.json"
    teacher_model = results / f"teacher_{ts}.pt"
    student_metrics = results / f"student_metrics_{ts}.json"
    student_model = results / f"student_{ts}.pkl"
    recourse_md = results / f"recourse_{ts}.md"
    recourse_json = results / f"recourse_{ts}.json"

    phases: List[Dict[str, Any]] = []
    pipeline_t0 = time.perf_counter()
    budget_sec = float(args.budget_minutes) * 60.0

    # ---- Phase 1: teacher
    p1 = _run_phase("train_teacher", _teacher_argv(
        args, tag=ts, pred_path=teacher_pred, model_path=teacher_model,
    ))
    phases.append(p1)
    if p1["returncode"] != 0:
        return _finalize(args, ts, out_json, phases, pipeline_t0, budget_sec, status="failed_teacher")

    # ---- Phase 2: student
    p2 = _run_phase("train_student", _student_argv(
        args, tag=ts, teacher_pred=teacher_pred,
        metrics_path=student_metrics, model_path=student_model,
    ))
    phases.append(p2)
    if p2["returncode"] != 0:
        return _finalize(args, ts, out_json, phases, pipeline_t0, budget_sec, status="failed_student")

    # ---- Phase 3: recourse
    if not args.skip_recourse:
        p3 = _run_phase("eval_recourse", _recourse_argv(
            args, tag=ts, model_path=student_model,
            out_json=recourse_json, out_md=recourse_md,
        ))
        phases.append(p3)
        if p3["returncode"] != 0:
            return _finalize(args, ts, out_json, phases, pipeline_t0, budget_sec, status="failed_recourse")

    return _finalize(args, ts, out_json, phases, pipeline_t0, budget_sec, status="ok")


def _finalize(args, ts: str, out_json: Path, phases: List[Dict[str, Any]],
              pipeline_t0: float, budget_sec: float, *, status: str) -> int:
    elapsed = time.perf_counter() - pipeline_t0
    over_budget = elapsed > budget_sec
    aggregated = _aggregate_artefacts(phases)

    payload = {
        "schema": "graphtm-cbr/full_run/v1",
        "timestamp_utc": ts,
        "iso_timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "status": status,
        "args": vars(args),
        "seed": args.seed,
        "elapsed_seconds": float(elapsed),
        "budget_seconds": float(budget_sec),
        "over_budget": bool(over_budget),
        "phases": phases,
        "artefacts": aggregated,
    }
    written = safe_write_json(out_json, payload)
    log.info(
        "pipeline done: status=%s elapsed=%.1fs (budget=%.1fs)",
        status, elapsed, budget_sec,
    )
    log.info("unified results: %s", written)
    if over_budget:
        log.warning(
            "WALL-CLOCK over budget by %.1fs (target <= 30 min for production runs)",
            elapsed - budget_sec,
        )
    # Final marker so callers can scrape stdout
    print(f"FULL_RUN_JSON={written}")
    return 0 if status == "ok" else 1


def _aggregate_artefacts(phases: List[Dict[str, Any]]) -> Dict[str, str]:
    """Flatten per-phase stdout artefacts into one dict for the unified JSON."""
    out: Dict[str, str] = {}
    for ph in phases:
        for k, v in ph.get("stdout_artefacts", {}).items():
            out[k] = v
    return out


if __name__ == "__main__":
    root = resolve_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.exit(main())
