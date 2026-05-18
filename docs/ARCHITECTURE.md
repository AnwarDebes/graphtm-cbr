# graphtm-cbr: Architecture Contract

Single source of truth for the parallel build. Every module agent works against the interfaces below; integration is gated on these contracts.

## Pillar

A **graph-walking** Hierarchical Tsetlin Machine. Clauses are evaluated at every node of the molecular graph and OR-aggregated across nodes. Edges enter clause evaluation via VSA binding. The student literally walks the graph each forward pass. No bag-of-atoms fingerprint.

## Stack

- **CUDA-C kernels** (`.cu` via PyCUDA `SourceModule`): forward, feedback, class-sum reduction, bit-packed TA updates.
- **Python orchestration**: PyTorch + PyTorch Geometric (teacher), RDKit (chemistry/validity), NumPy host glue.
- **Reference CPU**: `graphtm/core/hierarchical_tm.py` (canonical Granmo & Saha port) for parity tests.

## Target (pre-build contract; achieved numbers in `paper/paper.md` §4)

- Dataset: TDC AMES (Hansen 2009 + Kazius 2005, ~7k SMILES, scaffold split).
- Metric (pre-build target): ≥ 0.85 AUROC vs Morgan-FP MLP baseline 0.794 (research/05); headline gap ≥ +0.05.
  Achieved (`paper.md` §4.1): 0.685 distilled vs Morgan-FP RF 0.790, gap −0.105.
- Regulatory hook: ICH M7(R2) §6.1 expert-rule leg + §7.5 purging strategy.
- Recourse coverage on Kazius toxicophore held-out: 21/21 alerts present in test set covered (saturation caveat, `paper.md` §4.4).
- Training wall-clock ≤ 30 min on one GPU (A100/V100/RTX 4090).

## Module map (8 build modules)

```
graphtm/
  encoding/            (M1) graph -> BSC hypervectors + per-node feature tensors
  cuda/                (M2) CUDA-C kernels: forward, feedback, class-sum
  core/                (M3) Python wrapper: HierarchicalGraphTM class, training loop
  distill/             (M4) GIN teacher + distillation driver
  recourse/            (M5) clause-walk + greedy edit search + RDKit validity
  data/                (M6) TDC AMES + Kazius + Tox21 loaders, scaffold split
experiments/           (M7) entrypoint scripts: train, eval, recourse
tests/ + benchmarks/   (M8) per-module tests, parity tests, throughput benches
```

## Interface contracts (frozen)

### M1: encoding/

```python
# graphtm/encoding/hypervectors.py  (extend; basic ops already present)
def xor_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray: ...
def majority_bundle(stack: np.ndarray) -> np.ndarray: ...
def sparse_bsc(d: int, sparsity: float, rng) -> np.ndarray: ...   # 10% sparse

# graphtm/encoding/codebook.py
@dataclass
class AtomBondCodebook:
    atom_hv: np.ndarray   # [n_atom_types, D]   (uint8 0/1)
    bond_hv: np.ndarray   # [n_bond_types, D]
    role_hv: np.ndarray   # [k_hop+1, D]
    D: int = 8192
    sparsity: float = 0.10

# graphtm/encoding/graph_features.py
@dataclass
class GraphTensor:
    n_nodes: int
    atom_type: np.ndarray   # [n_nodes] int32
    edge_index: np.ndarray  # [2, n_edges] int32  (undirected: both directions)
    bond_type: np.ndarray   # [n_edges] int32
    node_hv: np.ndarray     # [n_nodes, D] uint8  (precomputed: atom HV ⊕ optional self-loop)
    edge_hv: np.ndarray     # [n_edges, D] uint8 (atom(u) ⊕ bond(b) ⊕ atom(v))

def encode_graph(mol, codebook: AtomBondCodebook, k_hop: int = 2) -> GraphTensor: ...
```

### M2: cuda/

```cuda
// graphtm/cuda/kernels.cu  (loaded via PyCUDA SourceModule)
// All kernels are __global__. Block/thread layout documented in kernels.py.

// ---------- forward ----------
__global__ void clause_forward_pernode(
    const uint32* ta_state,        // [C, LA_CHUNKS, STATE_BITS] bit-packed
    const uint8*  node_hv,         // [B, N_max, D_chunks] per-node hypervectors
    const uint8*  edge_hv,         // [B, E_max, D_chunks] per-edge hypervectors
    const int*    node_offset,     // [B+1] csr-style node ranges per graph
    const int*    edge_index,      // [B, 2, E_max]
    int8_t*       clause_node_out, // [B, C, N_max] per-(graph,clause,node) 0/1
    int C, int N_max, int D_chunks, int R, int IA, int IF, int LA, int LF
);

// OR across nodes within a graph → per-(graph,clause) scalar
__global__ void clause_or_across_nodes(
    const int8_t* clause_node_out,  // [B, C, N_max]
    const int*    n_nodes_per_graph,
    int8_t*       clause_out,       // [B, C]
    int C, int N_max
);

// alternating-sign weighted class-sum
__global__ void class_sum_reduce(
    const int8_t* clause_out,       // [B, C]
    const int8_t* clause_class,     // [C]   which class each clause votes for
    int*          class_sum,        // [B, K] clipped to ±T
    int C, int K, int T
);

// ---------- feedback ----------
__global__ void clause_feedback(
    uint32* ta_state,
    const int8_t* clause_node_out,  // path-conditional gating uses this
    const int*    class_sum,
    const int*    y_target,
    const int8_t* clause_class,
    int8_t*       fired_class_mask,
    int C, int K, int T, float s, int n_states,
    uint64_t rng_seed, uint64_t step
);
```

```python
# graphtm/cuda/_kernels.py: PyCUDA SourceModule loader, compile-time defines
class CudaKernels:
    def __init__(self, *, C: int, N_max: int, D_chunks: int,
                 R: int, IA: int, IF: int, LA: int, LF: int,
                 K: int, T: int, n_states: int): ...
    def forward(self, ta_state_gpu, batch: 'CudaBatch') -> Tuple[gpu_arr, gpu_arr]: ...
    def feedback(self, ta_state_gpu, clause_out_gpu, y_gpu, ...) -> None: ...

# graphtm/cuda/memory.py
class CudaTAState:
    """bit-plane uint32[C, LA_CHUNKS, STATE_BITS]: 32-way packed automaton state."""
```

### M3: core/

```python
# graphtm/core/hierarchical_graph_tm.py
@dataclass
class HGraphTMSpec:
    n_classes: int
    n_clauses: int
    threshold: int          # T
    s: float
    n_states: int = 200
    R: int = 2              # root factors
    IA: int = 2             # interior alternatives
    IF: int = 5             # interior factors
    LA: int = 15            # leaf alternatives
    LF: int = 3             # leaf factors per leaf
    D_bits: int = 8192      # hypervector dim
    n_atom_types: int = 9
    n_bond_types: int = 4
    k_hop: int = 2
    max_nodes: int = 80     # compile-time cap
    seed: int = 42

class HierarchicalGraphTM:
    def __init__(self, spec: HGraphTMSpec, device: str = "cuda"): ...
    def fit(self, graphs: List[GraphTensor], y: np.ndarray, epochs: int) -> None: ...
    def predict(self, graphs: List[GraphTensor]) -> np.ndarray: ...
    def class_scores(self, graphs: List[GraphTensor]) -> np.ndarray: ...
    def firing_clauses(self, graph: GraphTensor) -> List[FiringClause]:
        """Returns clauses that voted ≠ 0 and at which nodes they fired."""
    def clause_literals(self, clause_id: int) -> ClauseTree:
        """Walk TA state and emit the AND-OR-AND-OR-AND tree for clause `clause_id`."""
```

### M4: distill/

```python
# graphtm/distill/teacher.py
class GINTeacher(torch.nn.Module): ...   # 3-layer GIN, mean-pool, 32-d hidden
def train_teacher(graphs, y, epochs=80, lr=1e-3) -> Tuple[GINTeacher, np.ndarray]: ...

# graphtm/distill/student.py
def distill(student: HierarchicalGraphTM, graphs, y_teacher: np.ndarray,
            y_true: np.ndarray, epochs: int) -> dict: ...
```

### M5: recourse/

```python
# graphtm/recourse/candidates.py
@dataclass
class GraphEdit:
    op: Literal["remove_bond","add_bond","swap_atom","swap_bond_order"]
    indices: Tuple[int, ...]
    new_value: Optional[int]

def candidates_from_firing_clauses(graph: GraphTensor, firing: List[FiringClause],
                                    max_candidates: int = 50) -> List[GraphEdit]: ...

# graphtm/recourse/search.py
def greedy_minimal_edit(model: HierarchicalGraphTM, graph: GraphTensor,
                          candidates: List[GraphEdit], max_flips: int = 3
                         ) -> Optional[List[GraphEdit]]: ...

# graphtm/recourse/validity.py
def validate(mol_after_edit) -> ValidityReport:
    """RDKit sanitize + Lipinski Ro5 + SAscore (Ertl 2009). Returns flags."""
```

### M6: data/

```python
# graphtm/data/ames.py
def load_tdc_ames(split: str = "scaffold") -> Tuple[List[GraphTensor], np.ndarray, dict]: ...
# graphtm/data/kazius.py
def load_kazius_toxicophores() -> List[Toxicophore]: ...
# graphtm/data/tox21.py
def load_tox21(task: str = "NR-AhR", split: str = "scaffold") -> Tuple[...]: ...
```

### M7: experiments/

```
experiments/train_teacher.py     ← GIN teacher, save predictions
experiments/train_student.py     ← HGraphTM student, distillation, save model
experiments/eval_recourse.py     ← per-molecule recourse, validity, latency stats
experiments/full_pipeline.py     ← orchestrates all four phases end-to-end
```

### M8: tests/ + benchmarks/

- `tests/test_core/`: CPU-reference HTM parity (Granmo C-ref XOR, leak-trace).
- `tests/test_cuda/`: CUDA forward = CPU forward (numerical parity), feedback applies same delta.
- `tests/test_encoding/`: codebook round-trip via cleanup, BSC sparsity, k-hop topology preservation.
- `tests/test_recourse/`: candidates non-empty for firing clauses, validity filter removes invalid SMILES.
- `tests/test_integration/`: full pipeline on a 200-molecule micro-AMES.
- `benchmarks/`: kernel throughput, full-train wall-clock, scaling-with-clauses.
