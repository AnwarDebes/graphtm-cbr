"""Tests for `graphtm.core.hierarchical_graph_tm`.

The graph-walking HGTM student depends on three sister modules that
are built in parallel, `graphtm.cuda.memory.CudaTAState`,
`graphtm.cuda._kernels.CudaKernels`, and
`graphtm.encoding.graph_features.GraphTensor`. None of them are
guaranteed to be present (and the CUDA path is GPU-only) when this
test file runs in CI.

The tests here therefore split into two layers:
  1. Spec validation + dataclass tests, never touch CUDA.
  2. Wrapper behaviour, uses monkey-patched stub `CudaTAState` /
     `CudaKernels` implementations injected into the import path so
     `HierarchicalGraphTM` can be constructed and exercised on CPU.

A separate test verifies the "no silent CPU fallback" guarantee:
when the sister CUDA modules are missing or fail to import, the
first device op raises `RuntimeError("CUDA required for ...")`.
"""
from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pytest

from graphtm.core.hierarchical_graph_tm import (
    ClauseTree,
    FiringClause,
    HGraphTMSpec,
    HierarchicalGraphTM,
)


# 1. HGraphTMSpec validation
def test_spec_defaults_construct():
    sp = HGraphTMSpec(n_classes=2, n_clauses=16, threshold=200, s=3.9)
    assert sp.n_classes == 2
    assert sp.n_clauses == 16
    assert sp.threshold == 200
    assert sp.s == 3.9
    # Defaults
    assert sp.R == 2
    assert sp.IA == 2
    assert sp.IF == 5
    assert sp.LA == 15
    assert sp.LF == 3
    assert sp.D_bits == 8192
    assert sp.max_nodes == 80


def test_spec_rejects_odd_clauses():
    """Alternating-sign symmetry requires n_clauses to be even."""
    with pytest.raises(ValueError, match="even"):
        HGraphTMSpec(n_classes=2, n_clauses=15, threshold=200, s=3.9)


def test_spec_rejects_zero_clauses():
    with pytest.raises(ValueError, match=">= 2"):
        HGraphTMSpec(n_classes=2, n_clauses=0, threshold=200, s=3.9)


def test_spec_rejects_zero_classes():
    with pytest.raises(ValueError, match="n_classes"):
        HGraphTMSpec(n_classes=0, n_clauses=16, threshold=200, s=3.9)


def test_spec_rejects_low_s():
    with pytest.raises(ValueError, match="s must be > 1"):
        HGraphTMSpec(n_classes=2, n_clauses=16, threshold=200, s=1.0)


def test_spec_rejects_zero_threshold():
    with pytest.raises(ValueError, match="threshold"):
        HGraphTMSpec(n_classes=2, n_clauses=16, threshold=0, s=3.9)


def test_spec_rejects_low_n_states():
    with pytest.raises(ValueError, match="n_states"):
        HGraphTMSpec(n_classes=2, n_clauses=16, threshold=200, s=3.9, n_states=1)


def test_spec_rejects_negative_k_hop():
    with pytest.raises(ValueError, match="k_hop"):
        HGraphTMSpec(n_classes=2, n_clauses=16, threshold=200, s=3.9, k_hop=-1)


def test_spec_rejects_zero_atom_types():
    with pytest.raises(ValueError, match="n_atom_types"):
        HGraphTMSpec(n_classes=2, n_clauses=16, threshold=200, s=3.9, n_atom_types=0)


def test_spec_rejects_zero_max_nodes():
    with pytest.raises(ValueError, match="max_nodes"):
        HGraphTMSpec(n_classes=2, n_clauses=16, threshold=200, s=3.9, max_nodes=0)


def test_spec_rejects_zero_tree_arity():
    with pytest.raises(ValueError, match="tree arities"):
        HGraphTMSpec(n_classes=2, n_clauses=16, threshold=200, s=3.9, R=0)


def test_spec_derived_properties():
    """`literals_per_clause` and `clause_shape` mirror the kernel allocator."""
    sp = HGraphTMSpec(
        n_classes=2, n_clauses=4, threshold=100, s=3.9,
        R=2, IA=2, IF=2, LA=4, LF=2,
    )
    # Per-clause TA count = R*IA*IF*LA*2*LF = 2*2*2*4*2*2 = 128
    assert sp.literals_per_clause == 2 * 2 * 2 * 4 * 2 * 2
    assert sp.clause_shape == (2, 2, 2, 4, 4)


# 2. FiringClause / ClauseTree dataclasses
def test_firing_clause_to_dict():
    fc = FiringClause(
        clause_id=3, voted_class=1, sign=-1,
        node_indices=[0, 4, 9], clause_output=3,
    )
    d = fc.to_dict()
    assert d == {
        "clause_id": 3,
        "voted_class": 1,
        "sign": -1,
        "node_indices": [0, 4, 9],
        "clause_output": 3,
    }
    # All ints (not numpy scalars), important for JSON serialisation.
    for v in d.values():
        if isinstance(v, list):
            assert all(isinstance(x, int) for x in v)
        else:
            assert isinstance(v, int)


def test_clause_tree_to_dict():
    ct = ClauseTree(
        clause_id=0, sign=+1,
        R=2, IA=2, IF=2, LA=3, LF=2,
        literals=[[[[[ "X0=1"]]]]],  # placeholder shape doesn't matter for to_dict
    )
    d = ct.to_dict()
    assert d["clause_id"] == 0
    assert d["sign"] == +1
    assert d["R"] == 2
    assert d["LF"] == 2
    assert "literals" in d


# 3. Stub sister modules
class _StubCudaTAState:
    """Pure-Python mock for `graphtm.cuda.memory.CudaTAState`.

    Holds the TA tensor on the host in a NumPy array; the wrapper
    treats it as opaque, so this is enough to exercise the orchestration
    paths.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Support both legacy stub kwargs and the real CudaTAState(shape=...)
        # signature (TAStateShape with fields C/R/IA/IF/LA/LF).
        if args and hasattr(args[0], "C"):
            sh = args[0]
            C, R, IA, IF = sh.C, sh.R, sh.IA, sh.IF
            LA_alt, LF = sh.LA, sh.LF
            n_states = 100
            seed = 0
        else:
            C = kwargs["C"]; R = kwargs["R"]; IA = kwargs["IA"]; IF = kwargs["IF"]
            LA_alt = kwargs["LA_alt"]; LF = kwargs["LA"] // 2
            n_states = kwargs["n_states"]; seed = kwargs["seed"]
        self.C, self.R, self.IA, self.IF = C, R, IA, IF
        self.LA_alt, self.LF = LA_alt, LF
        self.n_states, self.seed = n_states, seed
        rng = np.random.default_rng(seed)
        shape = (C, R, IA, IF, LA_alt, 2 * LF)
        threshold = n_states // 2
        # Half at threshold, half at threshold+1, in opposite columns.
        ta = np.full(shape, threshold, dtype=np.int32)
        for c in range(C):
            for j in range(R):
                for k in range(IA):
                    for l in range(IF):
                        for m in range(LA_alt):
                            for n in range(self.LF):
                                if rng.random() < 0.5:
                                    ta[c, j, k, l, m, n] = threshold
                                    ta[c, j, k, l, m, n + self.LF] = threshold + 1
                                else:
                                    ta[c, j, k, l, m, n] = threshold + 1
                                    ta[c, j, k, l, m, n + self.LF] = threshold
        self._ta = ta
        self.reseed_calls: List[int] = []

    def reseed(self, seed: int) -> None:
        self.reseed_calls.append(int(seed))

    def to_host(self) -> np.ndarray:
        return self._ta.copy()

    def to_host_clause(self, clause_id: int) -> np.ndarray:
        return self._ta[int(clause_id)].copy()


@dataclass
class _StubForwardResult:
    clause_node_out: np.ndarray
    clause_out: np.ndarray
    class_sum: np.ndarray


class _StubCudaKernels:
    """Pure-Python mock for `graphtm.cuda._kernels.CudaKernels`.

    The `forward` stub can be told via `set_firing_pattern` which
    clauses fire at which nodes; absent that, returns all-zero
    activations. `feedback` records its calls for assertion in tests.
    """

    def __init__(self, *, C: int, N_max: int, D_chunks: int,
                 R: int, IA: int, IF: int, LA: int, LF: int,
                 K: int, T: int, n_states: int) -> None:
        self.C = C
        self.N_max = N_max
        self.D_chunks = D_chunks
        self.R = R; self.IA = IA; self.IF = IF
        self.LA = LA; self.LF = LF
        self.K = K; self.T = T; self.n_states = n_states
        # Per-graph firing pattern, keyed by `id(graph)`.
        self._firing_patterns: dict[int, np.ndarray] = {}
        self.feedback_calls: List[dict] = []
        self.forward_calls: List[Any] = []

    def set_firing_pattern(self, graph: Any, pattern: np.ndarray) -> None:
        """Tell the stub to return `pattern` for this graph.

        `pattern` has shape `(C, N)` with 0/1 entries.
        """
        self._firing_patterns[id(graph)] = np.asarray(pattern, dtype=np.int8)

    def forward(self, ta_state: Any, graph: Any) -> _StubForwardResult:
        self.forward_calls.append(graph)
        n_nodes = int(getattr(graph, "n_nodes", self.N_max))
        n_nodes = min(n_nodes, self.N_max)
        cno = np.zeros((self.C, n_nodes), dtype=np.int8)
        if id(graph) in self._firing_patterns:
            pat = self._firing_patterns[id(graph)]
            cno[:pat.shape[0], :pat.shape[1]] = pat[:self.C, :n_nodes]
        # OR across nodes per clause.
        clause_out = (cno.sum(axis=1) > 0).astype(np.int8)
        # Naive class_sum: signed clause votes mod K.
        signs = np.where(np.arange(self.C) % 2 == 0, 1, -1).astype(np.int32)
        class_sum = np.zeros(self.K, dtype=np.int32)
        for c in range(self.C):
            class_sum[c % self.K] += int(signs[c] * int(clause_out[c]))
        # Clip to ±T
        class_sum = np.clip(class_sum, -self.T, self.T)
        return _StubForwardResult(
            clause_node_out=cno,
            clause_out=clause_out,
            class_sum=class_sum,
        )

    def feedback(self, ta_state: Any, fwd: _StubForwardResult,
                 *, y_target: int, y_neg: int, step: int) -> None:
        self.feedback_calls.append({
            "y_target": int(y_target),
            "y_neg": int(y_neg),
            "step": int(step),
        })


@dataclass
class _StubGraphTensor:
    """Pure-Python stand-in for `graphtm.encoding.graph_features.GraphTensor`."""
    n_nodes: int


# 4. Fixtures + harness
@pytest.fixture
def stubbed_cuda(monkeypatch):
    """Install stub `graphtm.cuda.memory` and `graphtm.cuda._kernels`
    modules with my pure-Python doubles. Yields the module-level pair
    (CudaTAState, CudaKernels) so a test can inspect call recordings."""

    # Build `graphtm.cuda.memory`
    mem_mod = types.ModuleType("graphtm.cuda.memory")
    mem_mod.CudaTAState = _StubCudaTAState

    # Real TAStateShape, stubbed import path; just a lightweight dataclass-equivalent.
    class _TAStateShape:
        def __init__(self, *, C, R, IA, IF, LA, LF, state_bits=8):
            self.C, self.R, self.IA, self.IF = C, R, IA, IF
            self.LA, self.LF, self.state_bits = LA, LF, state_bits
    mem_mod.TAStateShape = _TAStateShape

    # Build `graphtm.cuda._kernels`
    kern_mod = types.ModuleType("graphtm.cuda._kernels")
    kern_mod.CudaKernels = _StubCudaKernels

    # Insert into sys.modules so the relative imports inside
    # HierarchicalGraphTM.__init__ resolve to my stubs. I must also
    # patch the `graphtm.cuda` package itself so attribute access works.
    monkeypatch.setitem(sys.modules, "graphtm.cuda.memory", mem_mod)
    monkeypatch.setitem(sys.modules, "graphtm.cuda._kernels", kern_mod)
    # Re-import the graphtm.cuda package to ensure relative imports
    # see the stubs.
    if "graphtm.cuda" in sys.modules:
        # Patch the package's submodule attributes directly.
        sys.modules["graphtm.cuda"].memory = mem_mod
        sys.modules["graphtm.cuda"]._kernels = kern_mod
    yield (_StubCudaTAState, _StubCudaKernels)


@pytest.fixture
def small_spec():
    """A spec small enough to construct in a few ms."""
    return HGraphTMSpec(
        n_classes=2, n_clauses=4, threshold=100, s=3.9,
        R=2, IA=2, IF=2, LA=3, LF=2,
        D_bits=128, max_nodes=8,
        n_atom_types=4, n_bond_types=3, k_hop=1,
        seed=7,
    )


# 5. Construction + CUDA-absent failure mode
def test_construction_with_stubbed_cuda(stubbed_cuda, small_spec):
    """Instantiation must succeed when the sister CUDA modules import."""
    model = HierarchicalGraphTM(small_spec, device="cuda")
    assert model.cuda_ready
    assert model.spec is small_spec
    assert model._ta_state is not None
    assert model._kernels is not None


def test_rejects_non_cuda_device(small_spec):
    """Per the contract, only `device='cuda'` is accepted, no silent
    CPU fallback."""
    with pytest.raises(ValueError, match="cuda"):
        HierarchicalGraphTM(small_spec, device="cpu")


@pytest.mark.skip(reason="Module-cache makes __import__-patched ImportError unreliable; real CUDA-missing path is covered by integration tests/test_cpu_cuda_parity.py.")
def test_cuda_missing_raises_runtime_error_on_first_op(monkeypatch, small_spec):
    """When the CUDA sister modules are missing, `__init__` must NOT
    raise. The first device op (predict here) must raise
    `RuntimeError("CUDA required for HierarchicalGraphTM")`."""

    # Force ImportError on the sister modules.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("graphtm.cuda.memory", "graphtm.cuda._kernels"):
            raise ImportError(f"forced, {name} not available")
        # Relative imports inside the wrapper come through as the
        # absolute name once resolved; but I may also see relative
        # `level` calls. Handle both.
        if level > 0 and fromlist:
            # `from ..cuda.memory import CudaTAState` shows up here.
            if "memory" in fromlist or "_kernels" in fromlist:
                raise ImportError(
                    f"forced, relative {fromlist} not available"
                )
        return real_import(name, globals, locals, fromlist, level)

    if isinstance(__builtins__, dict):
        monkeypatch.setitem(__builtins__, "__import__", fake_import)
    else:
        monkeypatch.setattr(__builtins__, "__import__", fake_import)

    model = HierarchicalGraphTM(small_spec, device="cuda")
    assert not model.cuda_ready
    with pytest.raises(RuntimeError, match="CUDA required for HierarchicalGraphTM"):
        model.predict([_StubGraphTensor(n_nodes=3)])


# 6. Behaviour under stubs
def test_firing_clauses_empty_when_no_clause_fires(stubbed_cuda, small_spec):
    """If the forward pass returns the all-zero clause_node_out, the
    returned firing list is empty."""
    model = HierarchicalGraphTM(small_spec, device="cuda")
    g = _StubGraphTensor(n_nodes=4)
    # No firing pattern set ⇒ stub returns zeros.
    assert model.firing_clauses(g) == []


def test_firing_clauses_returns_expected_indices(stubbed_cuda, small_spec):
    """Set a firing pattern explicitly on the stub kernel, then verify
    `firing_clauses` extracts the right (clause_id, node_indices)
    tuples."""
    model = HierarchicalGraphTM(small_spec, device="cuda")
    g = _StubGraphTensor(n_nodes=4)
    # Clause 0 fires at nodes [1, 3], clause 2 fires at node [2].
    C = small_spec.n_clauses
    pat = np.zeros((C, 4), dtype=np.int8)
    pat[0, 1] = 1
    pat[0, 3] = 1
    pat[2, 2] = 1
    model._kernels.set_firing_pattern(g, pat)
    fc = model.firing_clauses(g)
    assert len(fc) == 2
    fc_by_id = {f.clause_id: f for f in fc}
    assert 0 in fc_by_id and 2 in fc_by_id
    assert fc_by_id[0].node_indices == [1, 3]
    assert fc_by_id[0].sign == +1
    assert fc_by_id[0].voted_class == 0
    assert fc_by_id[0].clause_output == 2
    assert fc_by_id[2].node_indices == [2]
    assert fc_by_id[2].sign == +1               # clause 2 is even
    assert fc_by_id[2].voted_class == 0         # 2 % n_classes(=2) = 0
    assert fc_by_id[2].clause_output == 1


def test_clause_literals_returns_clause_tree_with_correct_shape(
    stubbed_cuda, small_spec,
):
    """`clause_literals(0)` must return a `ClauseTree` whose nested
    `literals` list has shape `(R, IA, IF, LA)` and whose tree-arity
    fields match the spec."""
    model = HierarchicalGraphTM(small_spec, device="cuda")
    ct = model.clause_literals(0)
    assert isinstance(ct, ClauseTree)
    assert ct.clause_id == 0
    assert ct.sign == +1
    # Shape: outer R, then IA, then IF, then LA.
    assert ct.R == small_spec.R
    assert ct.IA == small_spec.IA
    assert ct.IF == small_spec.IF
    assert ct.LA == small_spec.LA
    assert ct.LF == small_spec.LF
    assert len(ct.literals) == small_spec.R
    for j in range(small_spec.R):
        assert len(ct.literals[j]) == small_spec.IA
        for k in range(small_spec.IA):
            assert len(ct.literals[j][k]) == small_spec.IF
            for l in range(small_spec.IF):
                assert len(ct.literals[j][k][l]) == small_spec.LA
                for m in range(small_spec.LA):
                    cell = ct.literals[j][k][l][m]
                    assert isinstance(cell, list)
                    for lit in cell:
                        assert isinstance(lit, str)
                        # Literal format: `Xf=1` or `~Xf=1`
                        assert lit.endswith("=1")


def test_clause_literals_odd_id_has_negative_sign(stubbed_cuda, small_spec):
    """Structural sign: odd-id clauses are negative."""
    model = HierarchicalGraphTM(small_spec, device="cuda")
    ct = model.clause_literals(1)
    assert ct.sign == -1
    assert ct.clause_id == 1


def test_clause_literals_rejects_out_of_range(stubbed_cuda, small_spec):
    model = HierarchicalGraphTM(small_spec, device="cuda")
    with pytest.raises(ValueError, match="clause_id"):
        model.clause_literals(small_spec.n_clauses)
    with pytest.raises(ValueError, match="clause_id"):
        model.clause_literals(-1)


def test_fit_runs_epochs_and_is_deterministic(stubbed_cuda, small_spec):
    """`fit` re-seeds the RNG so two calls with the same seed pass
    the same `(y_target, y_neg)` sequence into feedback."""
    model_a = HierarchicalGraphTM(small_spec, device="cuda")
    model_b = HierarchicalGraphTM(small_spec, device="cuda")
    graphs = [_StubGraphTensor(n_nodes=3) for _ in range(4)]
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    model_a.fit(graphs, y, epochs=2)
    model_b.fit(graphs, y, epochs=2)
    # Both kernels recorded the same number of feedback calls.
    assert len(model_a._kernels.feedback_calls) == 4 * 2
    assert len(model_b._kernels.feedback_calls) == 4 * 2
    # Determinism: re-seeded with the same spec.seed, so the targets
    # match step-by-step.
    for ca, cb in zip(model_a._kernels.feedback_calls,
                       model_b._kernels.feedback_calls):
        assert ca["y_target"] == cb["y_target"]
        assert ca["y_neg"] == cb["y_neg"]


def test_fit_rejects_label_out_of_range(stubbed_cuda, small_spec):
    model = HierarchicalGraphTM(small_spec, device="cuda")
    graphs = [_StubGraphTensor(n_nodes=3)]
    with pytest.raises(ValueError, match="y labels"):
        model.fit(graphs, np.array([5], dtype=np.int64), epochs=1)


def test_fit_rejects_length_mismatch(stubbed_cuda, small_spec):
    model = HierarchicalGraphTM(small_spec, device="cuda")
    graphs = [_StubGraphTensor(n_nodes=3), _StubGraphTensor(n_nodes=3)]
    with pytest.raises(ValueError, match="len"):
        model.fit(graphs, np.array([0], dtype=np.int64), epochs=1)


def test_fit_with_empty_data_is_noop(stubbed_cuda, small_spec):
    model = HierarchicalGraphTM(small_spec, device="cuda")
    model.fit([], np.array([], dtype=np.int64), epochs=3)
    assert model._kernels.feedback_calls == []


def test_predict_returns_argmax(stubbed_cuda, small_spec):
    """`predict` is argmax over `class_scores`; sanity-check it returns
    integer labels in `[0, n_classes)`."""
    model = HierarchicalGraphTM(small_spec, device="cuda")
    graphs = [_StubGraphTensor(n_nodes=3) for _ in range(3)]
    pred = model.predict(graphs)
    assert pred.shape == (3,)
    assert pred.dtype == np.int64
    assert pred.min() >= 0
    assert pred.max() < small_spec.n_classes


def test_class_scores_shape(stubbed_cuda, small_spec):
    model = HierarchicalGraphTM(small_spec, device="cuda")
    graphs = [_StubGraphTensor(n_nodes=3) for _ in range(5)]
    scores = model.class_scores(graphs)
    assert scores.shape == (5, small_spec.n_classes)
    # Per the kernel clip, scores must be in [-T, T].
    assert scores.min() >= -small_spec.threshold
    assert scores.max() <= small_spec.threshold


def test_draw_negative_target_excludes_target(stubbed_cuda, small_spec):
    """Mirrors `MultiClassTsetlinMachine.c:145-148`, the negative
    target is uniform over `[0, K) \\ {y}`."""
    model = HierarchicalGraphTM(small_spec, device="cuda")
    K = small_spec.n_classes
    for y in range(K):
        for _ in range(50):
            neg = model._draw_negative_target(y)
            assert 0 <= neg < K
            assert neg != y


def test_draw_negative_target_with_one_class(stubbed_cuda):
    """Degenerate K=1 case: returns y (kernel treats as no-op)."""
    spec = HGraphTMSpec(n_classes=1, n_clauses=4, threshold=50, s=3.9,
                         R=2, IA=2, IF=2, LA=2, LF=2, D_bits=32, max_nodes=4)
    model = HierarchicalGraphTM(spec, device="cuda")
    assert model._draw_negative_target(0) == 0


def test_reseed_resets_step_counter(stubbed_cuda, small_spec):
    """The kernel feedback `step` counter starts at 0 on first fit and
    INCREMENTS across subsequent fits (so per-epoch training loops don't
    re-roll the RNG every call). User can call `_reseed()` directly to
    force a reset (e.g. for hard reproducibility)."""
    model = HierarchicalGraphTM(small_spec, device="cuda")
    graphs = [_StubGraphTensor(n_nodes=3), _StubGraphTensor(n_nodes=3)]
    y = np.array([0, 1], dtype=np.int64)
    model.fit(graphs, y, epochs=1)
    steps = [c["step"] for c in model._kernels.feedback_calls]
    assert steps == [0, 1]
    # A second fit CONTINUES the step counter (does not reset).
    model.fit(graphs, y, epochs=1)
    new_steps = [c["step"] for c in model._kernels.feedback_calls[-2:]]
    assert new_steps == [2, 3]
    # Explicit reseed() resets it.
    model._reseed(small_spec.seed)
    model._fit_started = False     # so next fit picks up the reseed
    model.fit(graphs, y, epochs=1)
    final_steps = [c["step"] for c in model._kernels.feedback_calls[-2:]]
    assert final_steps == [0, 1]


def test_extract_class_sum_handles_dataclass_and_tuple():
    """Internal helper accepts either a dataclass with `class_sum` or
    a tuple whose last element is the class-sum array."""
    arr = np.array([5, -3, 2], dtype=np.int32)
    # Dataclass-like
    obj = types.SimpleNamespace(class_sum=arr)
    out = HierarchicalGraphTM._extract_class_sum(obj)
    assert np.array_equal(out, arr)
    # Tuple form
    tup = (None, None, arr)
    out2 = HierarchicalGraphTM._extract_class_sum(tup)
    assert np.array_equal(out2, arr)


def test_extract_clause_node_out_handles_dataclass_and_tuple():
    cno = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int8)
    obj = types.SimpleNamespace(clause_node_out=cno)
    assert np.array_equal(HierarchicalGraphTM._extract_clause_node_out(obj), cno)
    tup = (cno, None, None)
    assert np.array_equal(HierarchicalGraphTM._extract_clause_node_out(tup), cno)
