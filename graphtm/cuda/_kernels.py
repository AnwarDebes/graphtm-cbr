"""graphtm/cuda/_kernels.py, PyCUDA SourceModule loader & dispatcher for M2.

Loads `kernels.cu` with compile-time `#define`s, exposes a `CudaKernels`
class with `forward(...)` and `feedback(...)` methods, plus a
`quick_smoke()` entry-point used by M8 to validate the build pipeline.

The module DOES NOT attempt CUDA imports at import time. Calling any
method that hits the GPU triggers a clear `RuntimeError` if PyCUDA is
unusable on this host.

External dependencies: this module needs `pycuda` (not currently in
requirements.txt; M8 owns the install line). Pin: `pycuda>=2023.1`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .memory import CudaTAState, TAStateShape, _lazy_cuda


# path to the .cu source
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNEL_CU_PATH = os.path.join(_THIS_DIR, "kernels.cu")


def _read_kernels_source() -> str:
    """Load kernels.cu into a string for SourceModule compilation."""
    with open(_KERNEL_CU_PATH, "r", encoding="utf-8") as f:
        return f.read()


# compile-time parameter block
@dataclass(frozen=True)
class KernelDims:
    """All compile-time `#define`s expected by kernels.cu.

    Set ONCE per training run (matches cair's pattern of recompiling per
    model config; see vendors/.../tm.py:437-464). Tree dims come from the
    canonical HTM macros (research/02 §1); N_max and D_chunks from M1's
    encoding settings.
    """
    C: int                  # CLAUSES
    R: int                  # ROOT_FACTORS
    IA: int                 # INTERIOR_ALTERNATIVES
    IF_: int                # INTERIOR_FACTORS (renamed; 'IF' is a Python keyword-ish)
    LA: int                 # LEAF_ALTERNATIVES
    LF: int                 # LEAF_FACTORS
    K: int                  # number of classes
    T: int                  # class-sum clip threshold
    N_max: int              # compile-time per-graph node cap
    D_chunks: int           # node hypervector chunks (D = 2*FEATURES; D_chunks = D/32)
    state_bits: int = 8
    boost_true_positive_feedback: int = 0

    def __post_init__(self):
        # FEATURES = R*IF*LF; D_chunks must hold 2*FEATURES bits.
        feats = self.R * self.IF_ * self.LF
        min_d_chunks = (2 * feats + 31) // 32
        if self.D_chunks < min_d_chunks:
            raise ValueError(
                f"D_chunks ({self.D_chunks}) too small for "
                f"FEATURES={feats}; need at least {min_d_chunks} "
                f"(2*FEATURES bits)."
            )

    def parameters_block(self) -> str:
        # Use HGTM_* prefix to avoid name collisions with CUDA's curand
        # template params (curand_kernel.h uses bare `T`, `R`, etc.).
        return (
            f"#define HGTM_CLAUSES {self.C}\n"
            f"#define HGTM_R {self.R}\n"
            f"#define HGTM_IA {self.IA}\n"
            f"#define HGTM_IF {self.IF_}\n"
            f"#define HGTM_LA {self.LA}\n"
            f"#define HGTM_LF {self.LF}\n"
            f"#define HGTM_K {self.K}\n"
            f"#define HGTM_T {self.T}\n"
            f"#define HGTM_N_MAX {self.N_max}\n"
            f"#define HGTM_D_CHUNKS {self.D_chunks}\n"
            f"#define HGTM_STATE_BITS {self.state_bits}\n"
            f"#define HGTM_BOOST_TRUE_POSITIVE_FEEDBACK "
            f"{self.boost_true_positive_feedback}\n"
        )


# CudaKernels: SourceModule loader + Python launch wrappers
class CudaKernels:
    """Compile-once-launch-many CUDA kernel manager for HGTM.

    Lifecycle:
        cuda_k = CudaKernels(dims=KernelDims(...))
        cuda_k.compile()            # builds SourceModule, fetches __global__s
        ta = CudaTAState(...)
        ta.allocate(); ta.init_centre()
        clause_out, class_sum = cuda_k.forward(ta, batch_inputs)
        cuda_k.feedback(ta, clause_out, ...)
    """

    def __init__(self,
                 *,
                 # canonical contract names (match docs/ARCHITECTURE.md)
                 C: Optional[int] = None,
                 N_max: Optional[int] = None,
                 D_chunks: Optional[int] = None,
                 R: int = 2,
                 IA: int = 2,
                 IF: int = 2,
                 LA: int = 10,
                 LF: int = 2,
                 K: int = 2,
                 T: int = 200,
                 n_states: int = 100,
                 state_bits: int = 8,
                 boost_true_positive_feedback: int = 0,
                 dims: Optional[KernelDims] = None):
        if dims is None:
            if C is None or N_max is None or D_chunks is None:
                raise ValueError(
                    "CudaKernels requires either `dims=KernelDims(...)` or "
                    "all of (C, N_max, D_chunks)."
                )
            dims = KernelDims(
                C=C, R=R, IA=IA, IF_=IF, LA=LA, LF=LF,
                K=K, T=T, N_max=N_max, D_chunks=D_chunks,
                state_bits=state_bits,
                boost_true_positive_feedback=boost_true_positive_feedback,
            )
        self.dims = dims
        self.n_states = n_states                 # canonical HTM macro
        self._module = None
        self._fn_forward_pernode = None
        self._fn_or_across_nodes = None
        self._fn_class_sum_reduce = None
        self._fn_select_clause_node = None
        self._fn_feedback = None
        self._fn_init_ta_state = None

    # --- TA-state shape derived from dims (for convenience) ---

    @property
    def ta_shape(self) -> TAStateShape:
        d = self.dims
        return TAStateShape(C=d.C, R=d.R, IA=d.IA, IF=d.IF_,
                            LA=d.LA, LF=d.LF, state_bits=d.state_bits)

    # --- compile ---

    def compile(self) -> None:
        """Compile kernels.cu via PyCUDA SourceModule. Raises RuntimeError
        with a clear message if the host has no working CUDA / PyCUDA.
        """
        cuda = _lazy_cuda()       # may raise RuntimeError on no-CUDA hosts
        try:
            from pycuda.compiler import SourceModule
        except Exception as e:
            raise RuntimeError(
                f"Failed to import pycuda.compiler.SourceModule: {e!r}"
            )
        params = self.dims.parameters_block()
        source = params + _read_kernels_source()
        # `no_extern_c=True` because kernels.cu has its own extern "C" block
        # (matches vendors/.../tm.py:466).
        self._module = SourceModule(source, no_extern_c=True)
        self._fn_forward_pernode = self._module.get_function(
            "clause_forward_pernode")
        self._fn_or_across_nodes = self._module.get_function(
            "clause_or_across_nodes")
        self._fn_class_sum_reduce = self._module.get_function(
            "class_sum_reduce")
        self._fn_select_clause_node = self._module.get_function(
            "select_clause_node")
        self._fn_feedback = self._module.get_function("clause_feedback")
        self._fn_init_ta_state = self._module.get_function("init_ta_state")

    def _ensure_compiled(self) -> None:
        if self._module is None:
            self.compile()

    # --- forward ---

    def forward(self,
                ta_state: CudaTAState,
                *,
                node_hv_gpu,
                edge_hv_gpu,
                node_offset_gpu,
                edge_index_gpu,
                n_nodes_per_graph_gpu,
                clause_class_gpu,
                B: int,
                clause_node_out_gpu=None,
                clause_out_gpu=None,
                class_sum_gpu=None,
                stream=None):
        """Run forward pass: per-(graph, clause, node) AND-OR tree → per-
        (graph, clause) OR-reduction → per-(graph, class) signed sum.

        Allocates output buffers lazily; caller may pass preallocated
        ones to amortise alloc cost across batches.

        Returns (clause_node_out_gpu, clause_out_gpu, class_sum_gpu) as
        pycuda DeviceAllocations. Caller owns them.
        """
        self._ensure_compiled()
        cuda = _lazy_cuda()
        d = self.dims

        # Allocate output buffers if not supplied.
        cno_bytes = B * d.C * d.N_max          # int8
        co_bytes  = B * d.C                    # int8
        cs_bytes  = B * d.K * 4                # int32

        if clause_node_out_gpu is None:
            clause_node_out_gpu = cuda.mem_alloc(int(cno_bytes))
        if clause_out_gpu is None:
            clause_out_gpu = cuda.mem_alloc(int(co_bytes))
        if class_sum_gpu is None:
            class_sum_gpu = cuda.mem_alloc(int(cs_bytes))

        # Initialise class_sum to 0, class_sum_reduce overwrites it, but
        # zero-init keeps semantics clean if caller pre-allocated.
        cuda.memset_d8(class_sum_gpu, 0, int(cs_bytes))

        # --- launch clause_forward_pernode: grid=(B, C, 1), block=(N_max,) ---
        threads_per_block = min(d.N_max, 128) or 1
        self._fn_forward_pernode(
            ta_state.gpu_ptr,
            node_hv_gpu,
            edge_hv_gpu,
            node_offset_gpu,
            edge_index_gpu,
            clause_node_out_gpu,
            np.int32(B),
            np.int32(d.N_max),
            np.int32(d.D_chunks),
            block=(threads_per_block, 1, 1),
            grid=(B, d.C, 1),
            stream=stream,
        )

        # --- launch clause_or_across_nodes: grid=(B, C, 1), block=(1,) ---
        self._fn_or_across_nodes(
            clause_node_out_gpu,
            n_nodes_per_graph_gpu,
            clause_out_gpu,
            np.int32(B),
            np.int32(d.N_max),
            block=(1, 1, 1),
            grid=(B, d.C, 1),
            stream=stream,
        )

        # --- launch class_sum_reduce: grid=(B, K, 1), block=(1,) ---
        self._fn_class_sum_reduce(
            clause_out_gpu,
            clause_class_gpu,
            class_sum_gpu,
            np.int32(B),
            block=(1, 1, 1),
            grid=(B, d.K, 1),
            stream=stream,
        )

        return clause_node_out_gpu, clause_out_gpu, class_sum_gpu

    # --- feedback ---

    def feedback(self,
                 ta_state: CudaTAState,
                 *,
                 clause_node_out_gpu,
                 node_hv_gpu,
                 class_sum_gpu,
                 n_nodes_per_graph_gpu,
                 y_target_gpu,
                 clause_class_gpu,
                 B: int,
                 s_specificity: float,
                 rng_seed: int,
                 step: int,
                 chosen_node_gpu=None,
                 stream=None):
        """Run feedback (Type Ia/Ib/II) per the canonical HTM rules.

        Two-kernel sequence:
            (1) select_clause_node, pick a fired node per (graph,clause)
            (2) clause_feedback  , apply gated TA updates

        Mirrors the cair feedback dispatch (select_clause_node →
        select_clause_updates → update), with my HTM-specific gating
        in clause_feedback.
        """
        self._ensure_compiled()
        cuda = _lazy_cuda()
        d = self.dims

        # Allocate chosen_node buffer if not supplied: int32[B, C].
        cn_bytes = B * d.C * 4
        if chosen_node_gpu is None:
            chosen_node_gpu = cuda.mem_alloc(int(cn_bytes))

        # --- (1) select_clause_node: grid=(B, C, 1) ---
        self._fn_select_clause_node(
            clause_node_out_gpu,
            n_nodes_per_graph_gpu,
            chosen_node_gpu,
            np.uint64(rng_seed),
            np.uint64(step),
            np.int32(B),
            np.int32(d.N_max),
            block=(1, 1, 1),
            grid=(B, d.C, 1),
            stream=stream,
        )

        # --- (2) clause_feedback: grid=(B, C, 1) ---
        self._fn_feedback(
            ta_state.gpu_ptr,
            clause_node_out_gpu,
            chosen_node_gpu,
            node_hv_gpu,
            class_sum_gpu,
            y_target_gpu,
            clause_class_gpu,
            np.float32(s_specificity),
            np.int32(self.n_states),
            np.uint64(rng_seed),
            np.uint64(step),
            np.int32(d.N_max),
            np.int32(B),
            block=(1, 1, 1),
            grid=(B, d.C, 1),
            stream=stream,
        )
        return chosen_node_gpu

    # --- init helpers (device-side) ---

    def init_ta_state_on_device(self, ta_state: CudaTAState) -> None:
        """Run the on-device `init_ta_state` kernel (centre state). The
        host-side helper `CudaTAState.init_centre()` is equivalent and
        faster for one-shot calls; this kernel exists primarily for
        completeness with cair's pattern (`prepare`).
        """
        self._ensure_compiled()
        d = self.dims
        self._fn_init_ta_state(
            ta_state.gpu_ptr,
            block=(128, 1, 1),
            grid=(max(1, (d.C + 127) // 128), 1, 1),
        )


# smoke test
def quick_smoke(verbose: bool = True) -> dict:
    """Allocate a tiny 2-graph 4-node 16-clause state and run forward.

    Returns a dict with status and any output values. Used by M8 to
    sanity-check the build pipeline.

    Behaviour on a host without working CUDA / PyCUDA:
        The function returns {"status": "skipped-no-cuda", "error": ...}.
        It does NOT raise, explicit caller is responsible for treating
        "skipped" as a soft pass.
    """
    out = {"status": "unknown"}
    # Try to import PyCUDA; fast-skip if unavailable.
    try:
        cuda = _lazy_cuda()
    except RuntimeError as e:
        out["status"] = "skipped-no-cuda"
        out["error"] = repr(e)
        if verbose:
            print(f"[quick_smoke] skipped, {e}")
        return out

    # Tiny dims: 2 graphs × 4 nodes × 16 clauses
    # Canonical tree defaults (R=2, IA=2, IF=2, LA=10, LF=2 → 320 lits/clause)
    # → LA_CHUNKS = 10
    # FEATURES = R*IF*LF = 2*2*2 = 8; D_bits = 16; D_chunks = 1
    d = KernelDims(
        C=16, R=2, IA=2, IF_=2, LA=10, LF=2,
        K=2, T=200, N_max=4, D_chunks=1,
    )
    try:
        kernels = CudaKernels(dims=d, n_states=100)
        kernels.compile()
    except RuntimeError as e:
        out["status"] = "skipped-no-cuda"
        out["error"] = repr(e)
        if verbose:
            print(f"[quick_smoke] compile failed, {e}")
        return out

    # Allocate & init TA state.
    ta = CudaTAState(kernels.ta_shape)
    ta.allocate()
    ta.init_centre()

    # Build a tiny batch on host then push.
    B = 2
    N = d.N_max
    D_chunks = d.D_chunks
    # node_hv [B, N_max, D_chunks*4] uint8, random bits
    rng = np.random.default_rng(0)
    node_hv = rng.integers(0, 256, size=(B, N, D_chunks * 4),
                            dtype=np.uint8)
    # No edges at depth-0 forward, but contract requires the pointers.
    edge_hv = np.zeros((B, 1, D_chunks * 4), dtype=np.uint8)
    edge_index = np.zeros((B, 2, 1), dtype=np.int32)
    # CSR-style node offsets: graph 0 has 4 nodes, graph 1 has 3.
    node_offset = np.array([0, 4, 7], dtype=np.int32)
    n_nodes_per_graph = np.array([4, 3], dtype=np.int32)
    # Round-robin clause-to-class assignment.
    clause_class = np.array([c % d.K for c in range(d.C)], dtype=np.int8)
    # Per-graph binary target.
    y_target = np.array([1, 0], dtype=np.int32)

    # Upload all inputs.
    node_hv_gpu = cuda.mem_alloc(node_hv.nbytes)
    cuda.memcpy_htod(node_hv_gpu, node_hv)
    edge_hv_gpu = cuda.mem_alloc(edge_hv.nbytes)
    cuda.memcpy_htod(edge_hv_gpu, edge_hv)
    edge_index_gpu = cuda.mem_alloc(edge_index.nbytes)
    cuda.memcpy_htod(edge_index_gpu, edge_index)
    node_offset_gpu = cuda.mem_alloc(node_offset.nbytes)
    cuda.memcpy_htod(node_offset_gpu, node_offset)
    n_nodes_gpu = cuda.mem_alloc(n_nodes_per_graph.nbytes)
    cuda.memcpy_htod(n_nodes_gpu, n_nodes_per_graph)
    clause_class_gpu = cuda.mem_alloc(clause_class.nbytes)
    cuda.memcpy_htod(clause_class_gpu, clause_class)
    y_target_gpu = cuda.mem_alloc(y_target.nbytes)
    cuda.memcpy_htod(y_target_gpu, y_target)

    # Forward.
    cno_gpu, co_gpu, cs_gpu = kernels.forward(
        ta,
        node_hv_gpu=node_hv_gpu,
        edge_hv_gpu=edge_hv_gpu,
        node_offset_gpu=node_offset_gpu,
        edge_index_gpu=edge_index_gpu,
        n_nodes_per_graph_gpu=n_nodes_gpu,
        clause_class_gpu=clause_class_gpu,
        B=B,
    )

    # Pull class_sum back to confirm forward executed and bounds are sane.
    class_sum = np.empty(B * d.K, dtype=np.int32)
    cuda.memcpy_dtoh(class_sum, cs_gpu)

    # Quick feedback round.
    chosen_node_gpu = kernels.feedback(
        ta,
        clause_node_out_gpu=cno_gpu,
        node_hv_gpu=node_hv_gpu,
        class_sum_gpu=cs_gpu,
        n_nodes_per_graph_gpu=n_nodes_gpu,
        y_target_gpu=y_target_gpu,
        clause_class_gpu=clause_class_gpu,
        B=B,
        s_specificity=3.9,
        rng_seed=42,
        step=0,
    )

    cuda.Context.synchronize()

    out["status"] = "ok"
    out["class_sum"] = class_sum.reshape(B, d.K).tolist()
    out["bounds_ok"] = bool(((-d.T <= class_sum) & (class_sum <= d.T)).all())
    out["la_chunks"] = kernels.ta_shape.la_chunks
    out["literals_per_clause"] = kernels.ta_shape.literals_per_clause

    if verbose:
        print(f"[quick_smoke] OK, class_sum = "
              f"{class_sum.reshape(B, d.K).tolist()}, "
              f"bounds_ok={out['bounds_ok']}, "
              f"LA_CHUNKS={out['la_chunks']}, "
              f"LITERALS/clause={out['literals_per_clause']}")
    return out


if __name__ == "__main__":
    quick_smoke(verbose=True)
