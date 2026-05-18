"""experiments/, Module M7 entrypoint scripts for graphtm-cbr.

This package contains the CLI scripts that drive the four phases of the
graphtm-cbr build:

  - train_teacher.py  : phase 1, GIN teacher training + soft+hard predictions
  - train_student.py  : phase 2, HierarchicalGraphTM student distillation
  - eval_recourse.py  : phase 3, per-molecule recourse + validity + latency
  - full_pipeline.py  : phase 4, orchestrate phases 1-3 via subprocess

Shared CLI helpers live in `_cli.py`.

All scripts:
  - write artefacts under ``<project_root>/results/``
  - log to stdout AND to ``results/<script>_<timestamp>.log``
  - take an explicit ``--seed`` for reproducibility
  - import-only against the frozen interface in ``docs/ARCHITECTURE.md``

These scripts are runnable both via ``python -m experiments.<name>`` and
``python experiments/<name>.py``.
"""
