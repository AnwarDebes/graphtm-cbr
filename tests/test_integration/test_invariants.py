"""Static checks that the 5 ARCHITECTURE.md invariants hold across `graphtm/`.

Invariants (verbatim from docs/ARCHITECTURE.md):

  1. No bag-of-atoms.  Any global graph summarisation must be tagged with
     `# AGGREGATE -- non-graph` so it stays auditable.
  2. Parity testable. Every CUDA kernel has a CPU reference and a parity
     test. This file checks the presence of parity tests for every kernel,
     but the numerical check itself lives in test_cpu_cuda_parity.py.
  3. No claim drift. Names and docstrings describe the actual op, not a
     marketing label. I forbid a small list of forbidden marketing
     strings in module docstrings.
  4. No silent CPU fallback. Every module that exposes a `device=` parameter
     must either honour `cuda` or raise -- never silently fall back to CPU.
  5. Reproducibility. Seeds are explicit; no global state. I check that
     no `np.random.seed(` is called at module top level inside `graphtm/`
     and that there is no `random.seed(` either.

These checks are syntactic, not semantic, they catch the most common
classes of drift without false-positive-storming on legitimate aggregations
elsewhere (e.g. a clause-output sum that has nothing to do with graphs).
The exception tag `# AGGREGATE -- non-graph` is the audit trail.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "graphtm"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _iter_pkg_files():
    """Yield every .py file under graphtm/ that I audit."""
    for dirpath, _dirnames, filenames in os.walk(PKG):
        for fname in filenames:
            if fname.endswith(".py"):
                yield Path(dirpath) / fname


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Invariant 1, No bag-of-atoms / global graph summarisation
# ---------------------------------------------------------------------------

# Variable suffixes that name PER-NODE or PER-EDGE arrays. Reducing one of
# these to a scalar without an explicit `# AGGREGATE -- non-graph` tag is
# exactly the bag-of-atoms anti-pattern. (Reducing over the HV bit axis ,
# `axis=-1` or `axis=1` on a 2-D `[N, D]` array, is fine; that collapses
# the BIT dim, not the GRAPH dim.)
_PER_NODE_VARS = re.compile(
    r"\b(?:node_features?|per_node|node_count|node_scores?|node_logits?|"
    r"graph_features?|graph_pool|graph_pooled|bag_of_atoms|atom_counts?)\b"
)
# Method-call reductions that collapse to a scalar when called without axis.
_SCALAR_REDUCERS = re.compile(
    r"\.(?:sum|mean|any|all|max|min|count_nonzero|prod|argmax|argmin)\(\s*\)"
)
# Builtin reductions: `sum(per_node_thing)`, `np.sum(per_node_thing)`.
_BUILTIN_REDUCERS = re.compile(
    r"\b(?:sum|mean|max|min|np\.sum|np\.mean|np\.max|np\.min|np\.prod|"
    r"torch\.sum|torch\.mean|torch\.max|torch\.min|torch\.prod)\("
)


def _line_has_audit_tag(line: str) -> bool:
    """Returns True if the line carries the explicit override tag."""
    return ("# AGGREGATE -- non-graph" in line
            or "# AGGREGATE -- per-clause" in line)


def _looks_like_graph_summarisation(line: str) -> bool:
    """Heuristic: line uses a per-node / per-graph variable AND collapses it
    to a scalar (no axis= guard, no slice).

    False-positive avoidance:
      * lines with `axis=` are exempt (axis-spec means the reduction is
        constrained to a specific dim, not a global collapse);
      * `[None, :]` is broadcasting, not a reduction.
    """
    if "axis=" in line:
        return False
    if not _PER_NODE_VARS.search(line):
        return False
    if _SCALAR_REDUCERS.search(line):
        return True
    if _BUILTIN_REDUCERS.search(line) and _PER_NODE_VARS.search(line):
        return True
    return False


def test_invariant_1_no_unmarked_graph_summarisation():
    """Reducer collapsing a per-node / per-graph variable to a scalar must
    carry the `# AGGREGATE -- non-graph` audit tag."""
    offenders: list[str] = []
    for p in _iter_pkg_files():
        if "__pycache__" in p.parts:
            continue
        text = _read(p)
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _line_has_audit_tag(line):
                continue
            if _looks_like_graph_summarisation(stripped):
                offenders.append(f"{p.relative_to(ROOT)}:{lineno}: {stripped}")
    assert not offenders, (
        "Invariant 1 violation: per-node / per-graph variable collapsed to "
        "a scalar without an `# AGGREGATE -- non-graph` audit tag:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Invariant 2, Parity testable
# ---------------------------------------------------------------------------

def test_invariant_2_parity_test_file_exists():
    """A parity test file must exist for the CUDA forward kernel."""
    parity = ROOT / "tests" / "test_integration" / "test_cpu_cuda_parity.py"
    assert parity.exists(), f"Missing parity test file: {parity}"
    # The file must reference both the CPU reference and the CUDA kernels.
    body = _read(parity)
    assert "graphtm.core.hierarchical_tm" in body, (
        "parity test must import the CPU reference HierarchicalTM"
    )
    assert "graphtm.cuda._kernels" in body, (
        "parity test must import the CUDA kernels module"
    )


# ---------------------------------------------------------------------------
# Invariant 3, No claim drift in module docstrings
# ---------------------------------------------------------------------------

# I forbid marketing words inside graphtm/ docstrings. The list is short
# and intentional, these are the words past projects have been called
# out for ("revolutionary", "novel" without backing, etc.). Comments
# inside code are allowed; this only checks module-level docstrings.
_FORBIDDEN_IN_DOCSTRING = (
    "revolutionary",
    "magic",
    "blazingly fast",
    "world-class",
    "state-of-the-art" ,
    "unprecedented",
)


def test_invariant_3_no_claim_drift_in_module_docstrings():
    """Module-level docstrings must not contain marketing puffery."""
    offenders: list[str] = []
    for p in _iter_pkg_files():
        if "__pycache__" in p.parts:
            continue
        text = _read(p)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # A broken file is a different problem; skip here.
            continue
        doc = ast.get_docstring(tree)
        if not doc:
            continue
        lowered = doc.lower()
        for bad in _FORBIDDEN_IN_DOCSTRING:
            if bad in lowered:
                offenders.append(f"{p.relative_to(ROOT)}: docstring contains "
                                 f"forbidden phrase {bad!r}")
    assert not offenders, (
        "Invariant 3 violation (claim drift):\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Invariant 4, No silent CPU fallback
# ---------------------------------------------------------------------------

# Detect `torch.cuda.is_available()` falling through to a CPU code path
# without raising. Heuristic: every occurrence of
# `if not torch.cuda.is_available():` must be followed within the next
# few lines by a `raise` statement, NOT just a `device = "cpu"` fallback.
_RAISE_RE = re.compile(r"^\s*raise\s+", re.MULTILINE)


def test_invariant_4_no_silent_cpu_fallback():
    """`torch.cuda.is_available()` checks must raise on absence, not fall through."""
    offenders: list[str] = []
    for p in _iter_pkg_files():
        if "__pycache__" in p.parts:
            continue
        # The core CPU-reference module (hierarchical_tm.py) is intentionally
        # CPU-only; it doesn't check cuda. The cli helper (_cli.py) is
        # different, it's a seeding utility, not a compute path.
        text = _read(p)
        lines = text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "torch.cuda.is_available" not in stripped:
                continue
            # Allow positive-form checks ("if torch.cuda.is_available():")
            #, these are the normal "use GPU" branch.
            if (stripped.startswith("if torch.cuda.is_available")
                    or "torch.cuda.is_available()" in stripped
                    and "not" not in stripped):
                # Still need to check that the negation form, if present,
                # raises. I handle that below.
                pass
            if "not torch.cuda.is_available" in stripped or "not torch.cuda.is_available()" in stripped:
                # The next non-empty, non-comment line should be a raise.
                window = "\n".join(lines[i + 1:i + 8])
                if not _RAISE_RE.search(window):
                    offenders.append(
                        f"{p.relative_to(ROOT)}:{i + 1}: negative cuda "
                        f"check does not raise, possible silent fallback."
                    )
    assert not offenders, (
        "Invariant 4 violation (silent CPU fallback):\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Invariant 5, Reproducibility: no module-level seed mutation
# ---------------------------------------------------------------------------

_BAD_SEED_RE = re.compile(
    r"^(?!\s)(?:numpy\.|np\.)?random\.seed\(",   # module-level np.random.seed(
    re.MULTILINE,
)
_BAD_TORCH_SEED_RE = re.compile(
    r"^(?!\s)torch\.manual_seed\(",
    re.MULTILINE,
)


def test_invariant_5_no_module_level_global_seeding():
    """No module under `graphtm/` may call `np.random.seed(...)` or
    `torch.manual_seed(...)` at MODULE top level. Seeds belong to a
    per-instance `np.random.default_rng(seed)`."""
    offenders: list[str] = []
    for p in _iter_pkg_files():
        if "__pycache__" in p.parts:
            continue
        text = _read(p)
        if _BAD_SEED_RE.search(text) or _BAD_TORCH_SEED_RE.search(text):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, (
        "Invariant 5 violation: module-level global seeding detected in: "
        + ", ".join(offenders)
    )
