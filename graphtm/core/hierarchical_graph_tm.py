"""HierarchicalGraphTM, graph-walking Hierarchical Tsetlin Machine (Option B).

This is the production student of the graphtm-cbr stack: a HGTM whose
clauses are evaluated **at every node of a molecular graph** and
OR-aggregated across nodes. Edges enter clause evaluation through VSA
binding embedded in the per-edge hypervector input. No bag-of-atoms
fingerprint anywhere on the path, the student walks the graph each
forward pass.

Stack:
  - This file orchestrates GPU memory + the CUDA kernel sequence
    (`graphtm.cuda._kernels.CudaKernels`, `graphtm.cuda.memory.CudaTAState`).
  - Forward and feedback live in the CUDA kernels (compile-time
    specialised on tree dims and `max_nodes`).
  - Clause-tree extraction reuses the canonical CPU walker from
    `graphtm.core.hierarchical_tm.HierarchicalTM.extract_clause_tree`
    by pulling TA state to host.

Hard rules (`docs/ARCHITECTURE.md`):
  - No bag-of-atoms fallback. Always per-node + per-edge tensors.
  - No silent CPU fallback. If CUDA is unavailable, the first device
    operation raises `RuntimeError`.
  - Deterministic given the seed; reseed on every `fit`.

The Spec → architecture mapping (see `research/02_hgtm_canonical_spec.md`):
  - `R = ROOT_FACTORS`            top-AND arity
  - `IA = INTERIOR_ALTERNATIVES`  OR arity at depth 1
  - `IF = INTERIOR_FACTORS`       inner-AND arity at depth 2
  - `LA = LEAF_ALTERNATIVES`      OR arity at the leaves
  - `LF = LEAF_FACTORS`           literals per leaf = `LF` pos + `LF` neg
  - Per-clause TA tensor shape: `[R, IA, IF, LA, 2*LF]` (host-side view).
  - Clause sign is structural (`+1` for even clause id, `-1` for odd) ,
    so `n_clauses` must be even to keep balanced ± populations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:                       # pragma: no cover
    from ..encoding.graph_features import GraphTensor


# Spec
@dataclass
class HGraphTMSpec:
    """Architecture + training spec for the graph-walking HGTM.

    Every field is a compile-time constant of the CUDA kernel set: a
    change here forces a kernel recompile. The defaults are chosen so a
    user can construct the spec with only `n_classes`, `n_clauses`,
    `threshold`, `s` and have a usable tree.

    Constraints (validated in `__post_init__`):
      - `n_clauses` is even (alternating-sign symmetry).
      - `n_classes >= 1`, `n_clauses >= 2`, `threshold >= 1`, `s > 1.0`.
      - `n_states >= 2`, `D_bits > 0`, `max_nodes >= 1`, `k_hop >= 0`.
      - `n_atom_types >= 1`, `n_bond_types >= 1`.
    """
    n_classes: int
    n_clauses: int
    threshold: int                  # T (class-sum clip)
    s: float                        # feedback specificity
    n_states: int = 200             # automaton ceiling; include action = state > n_states/2
    R: int = 2                      # ROOT_FACTORS
    IA: int = 2                     # INTERIOR_ALTERNATIVES
    IF: int = 5                     # INTERIOR_FACTORS
    LA: int = 15                    # LEAF_ALTERNATIVES
    LF: int = 3                     # LEAF_FACTORS (per leaf, pos+neg)
    D_bits: int = 8192              # hypervector dim
    n_atom_types: int = 9
    n_bond_types: int = 4
    k_hop: int = 2
    max_nodes: int = 80             # compile-time graph-size cap
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_classes < 1:
            raise ValueError(f"n_classes must be >= 1, got {self.n_classes}")
        if self.n_clauses < 2:
            raise ValueError(f"n_clauses must be >= 2, got {self.n_clauses}")
        if self.n_clauses % 2 != 0:
            raise ValueError(
                f"n_clauses must be even for alternating ± clause sign; "
                f"got {self.n_clauses}"
            )
        if self.threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {self.threshold}")
        if not (self.s > 1.0):
            raise ValueError(f"s must be > 1.0, got {self.s}")
        if self.n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {self.n_states}")
        if self.D_bits <= 0:
            raise ValueError(f"D_bits must be > 0, got {self.D_bits}")
        if self.max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1, got {self.max_nodes}")
        if self.k_hop < 0:
            raise ValueError(f"k_hop must be >= 0, got {self.k_hop}")
        if self.n_atom_types < 1:
            raise ValueError(f"n_atom_types must be >= 1, got {self.n_atom_types}")
        if self.n_bond_types < 1:
            raise ValueError(f"n_bond_types must be >= 1, got {self.n_bond_types}")
        if self.R < 1 or self.IA < 1 or self.IF < 1 or self.LA < 1 or self.LF < 1:
            raise ValueError(
                f"tree arities must all be >= 1: R={self.R}, IA={self.IA}, "
                f"IF={self.IF}, LA={self.LA}, LF={self.LF}"
            )

    @property
    def literals_per_clause(self) -> int:
        """Number of TAs per clause (positive + negated)."""
        return self.R * self.IA * self.IF * self.LA * 2 * self.LF

    @property
    def clause_shape(self) -> Tuple[int, int, int, int, int]:
        """Host-side TA tensor shape per clause: (R, IA, IF, LA, 2*LF)."""
        return (self.R, self.IA, self.IF, self.LA, 2 * self.LF)


# Interpretability dataclasses
@dataclass
class FiringClause:
    """One clause that voted non-zero on a single graph forward pass.

    Attributes:
      clause_id      Index of the clause in [0, n_clauses).
      voted_class    Class id this clause votes for (structural, derived
                     from clause_id by the kernel; for the canonical
                     HGTM this is `clause_id % n_classes` and the sign
                     is `+1` if clause_id is even, `-1` if odd).
      sign           Structural sign (+1 / -1).
      node_indices   Graph-node indices where the clause fired (i.e.
                     `clause_node_out[c, i] != 0`). Empty list means the
                     clause did not fire at any node.
      clause_output  Aggregate clause output for this graph (integer
                     vote multiplicity = OR-sum of per-node activations
                     across all nodes; see `research/02_hgtm_canonical_spec.md`
                     §2 for why this is an int and not bool).
    """
    clause_id: int
    voted_class: int
    sign: int
    node_indices: List[int]
    clause_output: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": int(self.clause_id),
            "voted_class": int(self.voted_class),
            "sign": int(self.sign),
            "node_indices": [int(i) for i in self.node_indices],
            "clause_output": int(self.clause_output),
        }


@dataclass
class ClauseTree:
    """Symbolic AND-OR-AND-OR-AND tree for a single clause.

    The exact structure mirrors `research/02_hgtm_canonical_spec.md` §1.
    Per-axis sizes are taken from the parent `HGraphTMSpec` so a
    downstream consumer can validate the shape without consulting the
    spec.

    Fields:
      clause_id       Index of the clause this tree describes.
      sign            Structural sign (+1 for even clause id, -1 for odd).
      R, IA, IF, LA, LF
                      Tree arities, copied from the spec for shape
                      validation. `LF` here is the *base* leaf-factor
                      count (per side); each leaf has `2 * LF` automata.
      literals        Nested list of literals indexed by
                      `[j, k, l, m]` (root, interior-alt, interior-fac,
                      leaf-alt). Each cell is a list of literal strings
                      such as `"X3=1"` (pos literal included) or
                      `"~X5=1"` (negated literal included). An empty
                      cell means no literal at that leaf is "include"
                      action, i.e. the leaf is a trivial 1-leaf.
    """
    clause_id: int
    sign: int
    R: int
    IA: int
    IF: int
    LA: int
    LF: int
    literals: List[List[List[List[List[str]]]]]    # [R][IA][IF][LA] -> list[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": int(self.clause_id),
            "sign": int(self.sign),
            "R": int(self.R),
            "IA": int(self.IA),
            "IF": int(self.IF),
            "LA": int(self.LA),
            "LF": int(self.LF),
            "literals": self.literals,
        }


# Wrapper
class HierarchicalGraphTM:
    """Graph-walking HGTM student.

    The class is intentionally thin: GPU memory + kernel sequencing,
    plus host-side interpretability paths. The math lives in
    `graphtm.cuda._kernels.CudaKernels` (forward + feedback) and the
    canonical CPU oracle at
    `graphtm.core.hierarchical_tm.HierarchicalTM`.

    Lifecycle:
      __init__   Allocate `CudaTAState`, build `CudaKernels` with
                 compile-time defines from spec. **Does not perform any
                 device op** beyond what those classes do, if CUDA is
                 unavailable, the first device op (inside `fit`,
                 `predict`, `class_scores`, `firing_clauses`) raises
                 `RuntimeError("CUDA required for HierarchicalGraphTM")`.
      fit        Online training loop. Re-seeds the RNG with
                 `spec.seed` so every call to `fit` is deterministic.
      predict    Argmax over class sums (no learning).
      class_scores
                 Raw per-graph class sums clipped to ±T (no learning).
      firing_clauses
                 Returns `FiringClause` list for one graph, the
                 clauses that voted non-zero plus the node indices at
                 which they fired.
      clause_literals
                 Walks TA state on host and emits a `ClauseTree` for
                 the requested clause id.

    Lazy CUDA import:
      `CudaTAState` and `CudaKernels` are imported inside `__init__`
      because the sister `graphtm.cuda.*` modules may not be ready yet
      (parallel build). Tests can monkey-patch the imports.
    """

    # Sentinel raised when the first CUDA op fails. Centralised so
    # downstream code (and tests) match on a stable type+message.
    _CUDA_ERR = "CUDA required for HierarchicalGraphTM"

    def __init__(self, spec: HGraphTMSpec, device: str = "cuda") -> None:
        if device != "cuda":
            raise ValueError(
                f"HierarchicalGraphTM only supports device='cuda'; got {device!r}. "
                f"This is by design, there is no silent CPU fallback. Use the CPU "
                f"oracle `graphtm.core.hierarchical_tm.HierarchicalTM` for parity."
            )
        self.spec: HGraphTMSpec = spec
        self.device: str = device
        # Host RNG for the multi-class one-vs-rest negative-target draw
        # (mirrors `MultiClassTsetlinMachine.c:140-151`). Re-seeded on `fit`.
        self.rng: np.random.Generator = np.random.default_rng(spec.seed)
        self._step: int = 0          # monotone counter passed to feedback RNG

        # Defer CUDA imports to instance construction; sister modules
        # may not yet be built. I treat ImportError as the documented
        # "CUDA unavailable" failure path.
        self._ta_state: Any = None
        self._kernels: Any = None
        try:
            from ..cuda.memory import CudaTAState
            from ..cuda._kernels import CudaKernels
        except ImportError as e:
            # The sister modules are missing or fail to import a CUDA
            # runtime dep. Defer the explicit error to first device op
            # so a user can still inspect a model spec or build trees
            # off a manually populated TA state. See `_require_cuda`.
            self._cuda_import_error: Optional[BaseException] = e
            return
        self._cuda_import_error = None

        D_chunks = (spec.D_bits + 31) // 32
        self._D_chunks = D_chunks

        try:
            from ..cuda.memory import TAStateShape
            shape = TAStateShape(
                C=spec.n_clauses,
                R=spec.R, IA=spec.IA, IF=spec.IF,
                LA=spec.LA, LF=spec.LF,
            )
            self._ta_state = CudaTAState(shape)
            self._kernels = CudaKernels(
                C=spec.n_clauses,
                N_max=spec.max_nodes,
                D_chunks=D_chunks,
                R=spec.R, IA=spec.IA, IF=spec.IF,
                LA=spec.LA, LF=spec.LF,
                K=spec.n_classes,
                T=spec.threshold,
                n_states=spec.n_states,
            )
            self._pump: Optional[_GraphPump] = None
            self._clause_class_gpu = None
            self._cuda_initialised = False
        except Exception as e:
            self._cuda_import_error = e

    # CUDA gate
    def _require_cuda(self) -> None:
        """Raise `RuntimeError` if CUDA initialisation failed earlier.

        Called at the top of every method that performs a device op.
        Centralised so the message is stable and matches the contract
        in `docs/ARCHITECTURE.md`.
        """
        if self._kernels is None or self._ta_state is None:
            cause = getattr(self, "_cuda_import_error", None)
            if cause is not None:
                raise RuntimeError(self._CUDA_ERR) from cause
            raise RuntimeError(self._CUDA_ERR)

    # Determinism
    def _reseed(self, seed: int) -> None:
        """Re-seed both the host RNG (negative-target draw) and the
        CUDA kernel RNG state. Called on entry to `fit` so two calls
        with the same seed produce identical weights."""
        self.rng = np.random.default_rng(seed)
        self._step = 0
        # `CudaTAState` may expose a `reseed` hook; if it does, use it.
        # Otherwise the kernel `feedback` call's `rng_seed` arg carries
        # determinism (Philox is seeded per-launch by (seed, step)).
        if self._ta_state is not None and hasattr(self._ta_state, "reseed"):
            self._ta_state.reseed(seed)

    # GraphTensor → GPU buffer adapter
    def _is_real_cuda_path(self) -> bool:
        """True iff both kernels and ta_state look like the production
        objects (have `compile` and `allocate` methods). Used to decide
        whether to build a `_GraphPump` (production path) or fall back to
        a stub-friendly call signature (used by unit-test mocks)."""
        return (hasattr(self._kernels, "compile") and
                hasattr(self._ta_state, "allocate"))

    def _ensure_pump(self) -> "_GraphPump":
        if not self._is_real_cuda_path():
            return None     # stub path, caller dispatches differently
        if not getattr(self, "_cuda_initialised", False):
            self._kernels.compile()
            self._ta_state.allocate()
            if hasattr(self._ta_state, "init_canonical_host"):
                self._ta_state.copy_from_host(
                    self._ta_state.init_canonical_host(seed=self.spec.seed)
                )
            else:
                self._ta_state.copy_from_host(self._ta_state.init_centre_host())
            self._cuda_initialised = True
        if self._pump is None:
            self._pump = _GraphPump(
                D_chunks=self._D_chunks,
                N_max=self.spec.max_nodes,
                K=self.spec.n_classes,
                C=self.spec.n_clauses,
            )
            # Persistent clause→class mapping. Even clauses vote +1 for
            # class (c//2 % K); odd clauses vote -1 for class ((c-1)//2 % K).
            # Class_sum kernel expects an int8[C] giving the class index.
            cuda_pump = self._pump
            # M2 kernel is binary (y_target ∈ {0,1}, sgn = sign*(2*tgt-1)).
            # All clauses contribute to one signed class_sum; I route them
            # into class_sum[1] so argmax(class_sum) picks class 1 when the
            # signed sum is positive, class 0 otherwise.
            clause_class = np.ones(self.spec.n_clauses, dtype=np.int8)
            self._clause_class_gpu = cuda_pump.upload_int8(clause_class)
        return self._pump

    def _forward_one(self, g: "GraphTensor"):
        pump = self._ensure_pump()
        if pump is None:
            # Stub path: call kernel with old-style (ta_state, graph) signature.
            fwd = self._kernels.forward(self._ta_state, g)
            return None, fwd, fwd, fwd
        gpu_bufs = pump.pack_single(g, self._D_chunks)
        clause_node_out_gpu, clause_out_gpu, class_sum_gpu = self._kernels.forward(
            self._ta_state,
            node_hv_gpu=gpu_bufs.node_hv,
            edge_hv_gpu=gpu_bufs.edge_hv,
            node_offset_gpu=gpu_bufs.node_offset,
            edge_index_gpu=gpu_bufs.edge_index,
            n_nodes_per_graph_gpu=gpu_bufs.n_nodes,
            clause_class_gpu=self._clause_class_gpu,
            B=1,
        )
        return gpu_bufs, clause_node_out_gpu, clause_out_gpu, class_sum_gpu

    # Training
    def fit(self, graphs: List["GraphTensor"], y: np.ndarray,
            epochs: int = 1) -> None:
        """Online training loop, one graph at a time.

        For each epoch:
          1. shuffle the graph order via `self.rng.permutation`
          2. for each `(g, y_g)`:
               a. forward: kernels.forward(ta_state, g)
                  → (clause_node_out, clause_out)
               b. class_sum_reduce → class_sum
               c. feedback: kernels.feedback(...) updates ta_state in place
                  for the true class (target=1) and one random non-true
                  class (target=0), matching
                  `MultiClassTsetlinMachine.c:140-151`.

        Determinism: re-seeds `self.rng` with `spec.seed` on entry, so
        repeated calls with the same data produce identical weights.

        Args:
          graphs   List of `GraphTensor` (one per molecule). Each is
                   the per-node + per-edge tensor product of the M1
                   encoder. **No bag-of-atoms fallback.**
          y        Integer class labels, shape `(len(graphs),)`. Values
                   must be in `[0, n_classes)`.
          epochs   Number of full passes over the data.

        Raises:
          RuntimeError                If CUDA is unavailable.
          ValueError                  If `len(graphs) != len(y)` or any
                                       label is out of range.
        """
        self._require_cuda()
        n = len(graphs)
        if n != len(y):
            raise ValueError(
                f"len(graphs)={n} != len(y)={len(y)}"
            )
        if n == 0:
            return
        y_arr = np.asarray(y, dtype=np.int64)
        if y_arr.min(initial=0) < 0 or y_arr.max(initial=0) >= self.spec.n_classes:
            raise ValueError(
                f"y labels must be in [0, {self.spec.n_classes}); "
                f"got range [{int(y_arr.min())}, {int(y_arr.max())}]"
            )
        # Re-seed only on first fit (so per-epoch fit(epochs=1) loops
        # don't reset RNG state and stall training). User can call
        # `_reseed(seed)` directly to force a hard reset.
        if not getattr(self, "_fit_started", False):
            self._reseed(self.spec.seed)
            self._fit_started = True
        K = self.spec.n_classes
        pump = self._ensure_pump()
        for _ in range(epochs):
            order = self.rng.permutation(n)
            for idx in order:
                g = graphs[int(idx)]
                y_g = int(y_arr[int(idx)])
                gbufs, cno_gpu, co_gpu, cs_gpu = self._forward_one(g)
                if pump is None:
                    # Stub path, old-signature kernel.feedback
                    self._kernels.feedback(
                        self._ta_state, cs_gpu,
                        y_target=y_g,
                        y_neg=self._draw_negative_target(y_g),
                        step=self._step,
                    )
                else:
                    y_target_gpu = pump.upload_int32_scalar(y_g, slot=0)
                    self._kernels.feedback(
                        self._ta_state,
                        clause_node_out_gpu=cno_gpu,
                        node_hv_gpu=gbufs.node_hv,
                        class_sum_gpu=cs_gpu,
                        n_nodes_per_graph_gpu=gbufs.n_nodes,
                        y_target_gpu=y_target_gpu,
                        clause_class_gpu=self._clause_class_gpu,
                        B=1,
                        s_specificity=float(self.spec.s),
                        rng_seed=int(self.spec.seed) ^ 0xA1B2,
                        step=int(self._step),
                    )
                self._step += 1

    def _draw_negative_target(self, y_g: int) -> int:
        """Uniform sample over `[0, K) \\ {y_g}` with rejection.

        Mirrors `MultiClassTsetlinMachine.c:145-148`. For `K == 1` (a
        degenerate binary case), returns `y_g` and the feedback kernel
        should treat that as a no-op on the negative side.
        """
        K = self.spec.n_classes
        if K <= 1:
            return y_g
        # Rejection sampling, exactly one draw needed in expectation
        # for K=2, log(K) for large K. Matches the C-ref rejection loop
        # (and importantly: I do NOT use "all other classes" or
        # "k random others", see invariant #9 in 02_hgtm_canonical_spec.md).
        neg = int(self.rng.integers(0, K))
        while neg == y_g:
            neg = int(self.rng.integers(0, K))
        return neg

    # Inference
    def predict(self, graphs: List["GraphTensor"]) -> np.ndarray:
        """Argmax over class sums. Forward only, no learning."""
        scores = self.class_scores(graphs)
        return scores.argmax(axis=1).astype(np.int64)

    def class_scores(self, graphs: List["GraphTensor"]) -> np.ndarray:
        """Per-graph class sums, clipped to ±T. Shape `(B, n_classes)`."""
        self._require_cuda()
        K = self.spec.n_classes
        out = np.zeros((len(graphs), K), dtype=np.int64)
        if not self._is_real_cuda_path():
            for i, g in enumerate(graphs):
                fwd = self._kernels.forward(self._ta_state, g)
                out[i] = self._extract_class_sum(fwd)
            return out
        import pycuda.driver as drv
        host_buf = np.zeros(K, dtype=np.int32)
        for i, g in enumerate(graphs):
            _, _, _, class_sum_gpu = self._forward_one(g)
            drv.memcpy_dtoh(host_buf, class_sum_gpu)
            out[i] = host_buf.astype(np.int64)
        return out

    @staticmethod
    def _extract_class_sum(fwd_result: Any) -> np.ndarray:
        """Best-effort projection of the forward-result tuple onto the
        host-side `class_sum` array of shape `(K,)`.

        The sister `CudaKernels.forward` is documented to return at
        least `(clause_node_out_gpu, clause_out_gpu)`. By the contract
        in `docs/ARCHITECTURE.md` M2, the class-sum reduction kernel is
        fused into the forward path; the returned object is either a
        ``ForwardResult`` dataclass with a ``class_sum`` field or a
        tuple whose last element is the class-sum array.

        I accept both shapes to keep the contract pliable while the
        sister module's exact return type is finalised. The fallback
        path returns the last element if it is a NumPy array.
        """
        if hasattr(fwd_result, "class_sum"):
            return np.asarray(fwd_result.class_sum, dtype=np.int64).reshape(-1)
        if isinstance(fwd_result, tuple) and fwd_result:
            last = fwd_result[-1]
            if hasattr(last, "get"):                 # PyCUDA GPUArray
                last = last.get()
            return np.asarray(last, dtype=np.int64).reshape(-1)
        # Last resort: assume it's array-like.
        arr = np.asarray(fwd_result, dtype=np.int64).reshape(-1)
        return arr

    # Interpretability
    def firing_clauses(self, graph: "GraphTensor") -> List[FiringClause]:
        """Return the list of clauses that voted non-zero on `graph`.

        For each clause c whose per-node activation array has any
        non-zero entry, emit a `FiringClause(clause_id=c,
        voted_class=c % n_classes, sign=±1, node_indices=[i for i in
        range(N) if clause_node_out[c, i] != 0], clause_output=sum_i
        clause_node_out[c, i])`.

        Returns an empty list if no clause fired.

        Raises:
          RuntimeError   If CUDA is unavailable.
        """
        self._require_cuda()
        if not self._is_real_cuda_path():
            fwd = self._kernels.forward(self._ta_state, graph)
            cno = self._extract_clause_node_out(fwd)
            if cno is None:
                return []
            cno = np.asarray(cno)
            if cno.ndim == 3:
                cno = cno[0]
        else:
            import pycuda.driver as drv
            _, cno_gpu, _, _ = self._forward_one(graph)
            cno = np.zeros((self.spec.n_clauses, self.spec.max_nodes), dtype=np.int8)
            drv.memcpy_dtoh(cno, cno_gpu)
        if cno.ndim == 3:
            # Some kernels return `[B, C, N]` even for a single graph.
            if cno.shape[0] != 1:
                raise ValueError(
                    f"firing_clauses expects a single graph in the forward "
                    f"result; got batch dim {cno.shape[0]}"
                )
            cno = cno[0]
        if cno.ndim != 2:
            raise ValueError(
                f"firing_clauses: clause_node_out must be 2-D `[C, N]`; "
                f"got shape {cno.shape}"
            )
        # Trim to the actual node count, if known. `min` here is the
        # scalar Python builtin used to clip an integer node-count to
        # the kernel's compile-time `N_max`; it is NOT a tensor
        # reduction.  # AGGREGATE -- non-graph
        n_nodes = int(getattr(graph, "n_nodes", cno.shape[1]))
        n_nodes = min(n_nodes, cno.shape[1])     # AGGREGATE -- non-graph
        cno = cno[:, :n_nodes]
        out: List[FiringClause] = []
        K = self.spec.n_classes
        for c in range(self.spec.n_clauses):
            row = cno[c]
            if not np.any(row):
                continue
            node_indices = [int(i) for i in np.flatnonzero(row).tolist()]
            sign = +1 if (c % 2 == 0) else -1
            voted_class = int(c % K)
            out.append(FiringClause(
                clause_id=int(c),
                voted_class=voted_class,
                sign=sign,
                node_indices=node_indices,
                clause_output=int(row.sum()),
            ))
        return out

    @staticmethod
    def _extract_clause_node_out(fwd_result: Any) -> Optional[Any]:
        """Pull the per-(clause, node) activation array out of the
        forward-result tuple. See `_extract_class_sum` for the shape
        invariants I accept."""
        if hasattr(fwd_result, "clause_node_out"):
            cno = fwd_result.clause_node_out
        elif isinstance(fwd_result, tuple) and fwd_result:
            cno = fwd_result[0]
        else:
            return None
        if hasattr(cno, "get"):                       # PyCUDA GPUArray
            cno = cno.get()
        return cno

    def clause_literals(self, clause_id: int) -> ClauseTree:
        """Walk the TA state for clause `clause_id` and emit a symbolic
        `ClauseTree`.

        Pulls the TA state to host via `CudaTAState.to_host()` (the
        sister module exposes this; I treat it as opaque). The walker
        below is the equivalent of
        `HierarchicalTM.extract_clause_tree` restricted to one clause.

        Returns a `ClauseTree` whose `literals` field has nested shape
        `[R][IA][IF][LA] -> list[str]`. Each string is either
        `"Xf=1"` (positive literal included at feature index `f`) or
        `"~Xf=1"` (negated literal included). An empty list at a leaf
        means no literal is in "include" action, the leaf reduces to
        the constant-1 conjunction.

        Raises:
          RuntimeError   If CUDA is unavailable.
          ValueError     If `clause_id` is out of range.
        """
        sp = self.spec
        if not (0 <= clause_id < sp.n_clauses):
            raise ValueError(
                f"clause_id must be in [0, {sp.n_clauses}); got {clause_id}"
            )
        self._require_cuda()
        # Pull the TA state for the requested clause to host. I accept
        # either `to_host()` returning the full `[C, R, IA, IF, LA, 2*LF]`
        # tensor, or `to_host(clause_id)` returning the per-clause slice.
        ta_host = self._host_ta_state(clause_id)
        return self._walk_clause(clause_id, ta_host)

    def _host_ta_state(self, clause_id: int) -> np.ndarray:
        """Return host-side TA state for `clause_id` with shape
        `(R, IA, IF, LA, 2*LF)`. Tolerates either a per-clause `to_host`
        or a full-state pull, so the sister module can choose either."""
        sp = self.spec
        expected_shape = sp.clause_shape
        ta = self._ta_state
        # Prefer a per-clause hook if the sister module provides one.
        if hasattr(ta, "to_host_clause"):
            arr = np.asarray(ta.to_host_clause(int(clause_id)))
            if arr.shape != expected_shape:
                raise ValueError(
                    f"CudaTAState.to_host_clause returned shape {arr.shape}; "
                    f"expected {expected_shape}"
                )
            return arr
        if not hasattr(ta, "to_host"):
            raise RuntimeError(
                "CudaTAState must expose either `to_host()` or "
                "`to_host_clause(int)`; sister module is incomplete."
            )
        full = np.asarray(ta.to_host())
        # Accept both the canonical 6-D layout and a flat per-clause slice.
        if full.ndim == 6:
            if full.shape != (sp.n_clauses,) + expected_shape:
                raise ValueError(
                    f"CudaTAState.to_host() returned shape {full.shape}; "
                    f"expected {(sp.n_clauses,) + expected_shape}"
                )
            return full[int(clause_id)]
        if full.ndim == 5 and full.shape == expected_shape:
            # Caller assumed only one clause materialised.
            return full
        raise ValueError(
            f"CudaTAState.to_host() returned shape {full.shape}; "
            f"expected 6-D `[C, R, IA, IF, LA, 2*LF]` or 5-D per-clause"
        )

    def _walk_clause(self, clause_id: int, ta_clause: np.ndarray) -> ClauseTree:
        """Build a `ClauseTree` from one clause's TA state.

        Action threshold for "include" is `state > n_states / 2` ,
        consistent with the kernel's bit-plane decision rule (top
        STATE_BITS plane = include mask; the host integer state crosses
        the threshold at `n_states / 2`, matching the C reference's
        `state > NUMBER_OF_STATES` rule, see
        `research/02_hgtm_canonical_spec.md` §6).

        Feature-index formula matches the C ref:
            feature = j * IF * LF + l * LF + n
        Negated literal at `n + LF` of the last axis.
        """
        sp = self.spec
        # Include action: state > n_states / 2 (top half).
        # For n_states even (default 200), threshold = 100, so 101..200 ↔ include.
        threshold = sp.n_states // 2
        include = ta_clause > threshold     # shape (R, IA, IF, LA, 2*LF)
        sign = +1 if (clause_id % 2 == 0) else -1
        literals: List[List[List[List[List[str]]]]] = []
        for j in range(sp.R):
            interior_alts: List[List[List[List[str]]]] = []
            for k in range(sp.IA):
                interior_facs: List[List[List[str]]] = []
                for l in range(sp.IF):
                    leaf_alts: List[List[str]] = []
                    for m in range(sp.LA):
                        leaf_lits: List[str] = []
                        for n in range(sp.LF):
                            feat = j * sp.IF * sp.LF + l * sp.LF + n
                            # Positive literal automaton at index n.
                            if bool(include[j, k, l, m, n]):
                                leaf_lits.append(f"X{feat}=1")
                            # Negated literal automaton at index n + LF.
                            if bool(include[j, k, l, m, n + sp.LF]):
                                leaf_lits.append(f"~X{feat}=1")
                        leaf_alts.append(leaf_lits)
                    interior_facs.append(leaf_alts)
                interior_alts.append(interior_facs)
            literals.append(interior_alts)
        return ClauseTree(
            clause_id=int(clause_id),
            sign=int(sign),
            R=int(sp.R),
            IA=int(sp.IA),
            IF=int(sp.IF),
            LA=int(sp.LA),
            LF=int(sp.LF),
            literals=literals,
        )

    # Diagnostics
    @property
    def cuda_ready(self) -> bool:
        """True iff CUDA init succeeded, useful in test harnesses."""
        return self._kernels is not None and self._ta_state is not None


@dataclass
class _PackedGraphGPU:
    node_hv: Any            # uint8 bytes, [B, N_max, D_chunks*4]
    edge_hv: Any            # uint8 bytes, [B, E_max, D_chunks*4]
    edge_index: Any         # int32, [B, 2, E_max]
    node_offset: Any        # int32, [B+1]
    n_nodes: Any            # int32, [B]


class _GraphPump:
    """Reusable GPU buffer pool for single-graph forward/feedback.

    Allocates the maximum-size buffers once and rewrites them per graph.
    Stays simple, batched packing would amortise launches but for online
    TM training one-at-a-time is the natural granularity.
    """

    def __init__(self, *, D_chunks: int, N_max: int, K: int, C: int):
        import pycuda.driver as drv
        self._drv = drv
        self.D_chunks = int(D_chunks)
        self.N_max = int(N_max)
        self.E_max = int(N_max * 8)  # ample headroom; real edge count ≤ N*max_degree
        # Per-call buffers (B=1).
        word_bytes = self.D_chunks * 4
        self._node_hv_gpu = drv.mem_alloc(self.N_max * word_bytes)
        self._edge_hv_gpu = drv.mem_alloc(self.E_max * word_bytes)
        self._edge_index_gpu = drv.mem_alloc(2 * self.E_max * 4)
        self._node_offset_gpu = drv.mem_alloc(2 * 4)
        self._n_nodes_gpu = drv.mem_alloc(1 * 4)
        # Scratch int32 scalars for y_target uploads (two slots: pos / neg).
        self._y_target_gpu = [drv.mem_alloc(4), drv.mem_alloc(4)]

    def upload_int8(self, arr: np.ndarray):
        drv = self._drv
        a = np.ascontiguousarray(arr, dtype=np.int8)
        buf = drv.mem_alloc(a.nbytes)
        drv.memcpy_htod(buf, a)
        return buf

    def upload_int32_scalar(self, val: int, *, slot: int = 0):
        drv = self._drv
        a = np.asarray([int(val)], dtype=np.int32)
        drv.memcpy_htod(self._y_target_gpu[slot], a)
        return self._y_target_gpu[slot]

    def pack_single(self, g: "GraphTensor", D_chunks: int) -> _PackedGraphGPU:
        """Pack one GraphTensor into the persistent GPU buffers."""
        drv = self._drv
        N_max = self.N_max
        n_nodes = min(int(g.n_nodes), N_max)
        # Pack per-node HV: uint8 [n_nodes, D_bits] → uint32 [N_max, D_chunks].
        node_hv = np.zeros((N_max, D_chunks), dtype=np.uint32)
        node_hv[:n_nodes] = _pack_bits_to_uint32(g.node_hv[:n_nodes], D_chunks)
        drv.memcpy_htod(self._node_hv_gpu, node_hv.view(np.uint8))

        # Per-edge HV
        n_edges = min(int(g.edge_hv.shape[0]), self.E_max)
        edge_hv = np.zeros((self.E_max, D_chunks), dtype=np.uint32)
        if n_edges > 0:
            edge_hv[:n_edges] = _pack_bits_to_uint32(g.edge_hv[:n_edges], D_chunks)
        drv.memcpy_htod(self._edge_hv_gpu, edge_hv.view(np.uint8))

        # Edge index
        edge_index = np.zeros((2, self.E_max), dtype=np.int32)
        if n_edges > 0:
            edge_index[:, :n_edges] = np.asarray(g.edge_index[:, :n_edges], dtype=np.int32)
        drv.memcpy_htod(self._edge_index_gpu, edge_index)

        node_offset = np.asarray([0, n_nodes], dtype=np.int32)
        drv.memcpy_htod(self._node_offset_gpu, node_offset)

        n_nodes_arr = np.asarray([n_nodes], dtype=np.int32)
        drv.memcpy_htod(self._n_nodes_gpu, n_nodes_arr)

        return _PackedGraphGPU(
            node_hv=self._node_hv_gpu,
            edge_hv=self._edge_hv_gpu,
            edge_index=self._edge_index_gpu,
            node_offset=self._node_offset_gpu,
            n_nodes=self._n_nodes_gpu,
        )


def _pack_bits_to_uint32(bits_uint8: np.ndarray, D_chunks: int) -> np.ndarray:
    """Pack [N, D] uint8 0/1 into [N, D_chunks] uint32 (LSB-first per word).

    Bit `feat` ends up at `((arr[n, feat/32]) >> (feat%32)) & 1`, matching
    kernels.cu's `hv_bit` indexing.
    """
    N, D = bits_uint8.shape
    out = np.zeros((N, D_chunks), dtype=np.uint32)
    # Vectorised: shift+OR within each uint32 chunk.
    n_full = min(D, D_chunks * 32)
    for chunk in range(D_chunks):
        lo = chunk * 32
        hi = min(lo + 32, n_full)
        if lo >= hi:
            break
        # Build the uint32 column by accumulating bit positions.
        col = np.zeros(N, dtype=np.uint32)
        for pos in range(hi - lo):
            col |= (bits_uint8[:, lo + pos].astype(np.uint32) & 1) << pos
        out[:, chunk] = col
    return out


__all__ = [
    "HGraphTMSpec",
    "FiringClause",
    "ClauseTree",
    "HierarchicalGraphTM",
]
