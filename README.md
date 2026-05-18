# graphtm-cbr

**Graph-walking Hierarchical Tsetlin Machine with counterfactual Boolean
recourse, targeting ICH M7(R2) §6.1 mutagenicity assessment.**

## What it is

A neuro-symbolic classifier for molecular property prediction that, by
construction, exposes the structural-alert pattern behind each
prediction as a human-readable AND-OR tree of literals. Two pieces:

1. **Hierarchical Tsetlin Machine student**: a graph-walking variant
   in which clauses are evaluated at every node of the molecular graph
   and OR-aggregated across nodes. Edges enter clause evaluation via
   VSA hypervector binding. The student literally walks the graph each
   forward pass; no bag-of-atoms fingerprint is constructed.
2. **Counterfactual recourse**: for any positive prediction the system
   returns a minimal graph edit (`remove_bond`, `swap_atom`,
   `swap_bond_order`, `add_bond`) that flips the model to negative
   while remaining RDKit-valid and Lipinski-compliant. Per ICH M7(R2)
   §7.5, this is the "purging strategy" route.

The implementation pairs a PyTorch+PyG GIN teacher with a CUDA-C
Hierarchical Tsetlin Machine student distilled on the teacher's
predictions. The CUDA kernels (`graphtm/cuda/kernels.cu`, loaded via
PyCUDA `SourceModule`) carry the forward, feedback, and class-sum
reduction; a NumPy reference (`graphtm/core/hierarchical_tm.py`,
canonical Granmo & Saha port) is retained for numerical-parity tests.

The project does **not** claim novelty for the Hierarchical Tsetlin
Machine itself (that is Granmo & Saha's prior work) nor for graph
neural networks (a body of prior work). The contribution is the
combination: a topology-preserving HTM student distilled from a GIN
teacher with constructive, validity-checked counterfactual edits,
slotting into a named regulatory paragraph (ICH M7(R2) §6.1)
that current GNN-only systems cannot satisfy.

## Quickstart

```bash
# 1. Install (Python >= 3.10), from the cloned repo root.
pip install -e .

# 2. Build the CUDA kernels (one-time JIT compile via PyCUDA).
python -c "from graphtm.cuda._kernels import CudaKernels; print('kernels ok')"

# 3. Run the integration tests (skips cleanly without CUDA).
pytest tests/test_integration/ -ra

# 4. Train the GIN teacher, distil the HGTM student, evaluate recourse.
python experiments/train_teacher.py   --dataset ames --seed 42
python experiments/train_student.py   --dataset ames --seed 42
python experiments/eval_recourse.py   --dataset ames --seed 42
# or end-to-end:
python experiments/full_pipeline.py   --dataset ames --seed 42
```

A working CUDA device (compute capability >= 7.0) and the NVIDIA
driver that PyCUDA can attach to are required. The kernels do **not**
silently fall back to CPU; see `docs/ARCHITECTURE.md` invariant 4.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the frozen
interface contract. The module map:

```
graphtm/
  encoding/   M1  graph -> BSC hypervectors + per-node/-edge tensors
  cuda/       M2  CUDA-C kernels (forward, feedback, class-sum)
  core/       M3  HierarchicalGraphTM wrapper + training loop
  distill/    M4  GIN teacher + hard-label distillation driver
  recourse/   M5  clause-walk -> candidate edits -> greedy search -> RDKit
  data/       M6  TDC AMES / Kazius / Tox21 loaders, scaffold split
experiments/  M7  entry-point scripts (train, eval, recourse, full_pipeline)
tests/        M8  per-module + integration + parity tests
benchmarks/   M8  forward throughput, full-train wall-clock, CPU-vs-CUDA
```

Five invariants are enforced by `tests/test_integration/test_invariants.py`:

1. **No bag-of-atoms.** Any reducer collapsing a per-node tensor to a
   scalar must carry an `# AGGREGATE -- non-graph` audit tag.
2. **Parity testable.** Every CUDA kernel has a CPU reference and a
   numerical-parity test.
3. **No claim drift.** Module docstrings name the actual operation, not
   a marketing label.
4. **No silent CPU fallback.** Either CUDA is available and used, or the
   call errors. CPU paths are not benchmarked and called "GPU".
5. **Reproducibility.** All seeds are explicit; no module-level
   `np.random.seed(...)`.

## Results (TDC AMES, n=7,278, scaffold split 5822/727/729)

| System | Test AUROC | Test acc |
|---|---:|---:|
| Morgan-FP RF (n_est=200, 2048 bits, r=2) | 0.790 | 0.720 |
| GIN teacher (3-layer, 32-d, 80 ep, AdamW) | **0.796** | 0.730 |
| graphtm-cbr K=5 ensemble, direct labels | 0.669 | 0.575 |
| **graphtm-cbr K=5 ensemble, GIN-distilled** | **0.685** | **0.635** |

**Distillation closes ~13 % of the AUROC gap to the teacher, but transforms the recourse layer:**

| Recourse metric | Direct ensemble | **Distilled ensemble** |
|---|---:|---:|
| Success rate (200 positives) | 54.5 % (109/200) | **95.5 % (191/200)** |
| Mean flips | 1.51 | **1.30** |
| Latency p50 / p95 / p99 (ms) | 3692 / 8522 / 12687 | **2126 / 6775 / 9254** |
| Kazius coverage (alerts present 21/29) | 21/21 (saturated clauses) | 21/21 (saturated clauses) |

**Headline:** raw AUROC trails the fingerprint baseline by 0.10, but **19 of every 20 predicted-positive molecules receive a working <=3-edit counterfactual**, the regulator-load-bearing metric under ICH M7(R2) §7.5. See `paper/paper.md` for full discussion and honest scope.

## Regulatory framing

The ICH M7(R2) guideline (Step 4, Feb 2023) mandates a two-method
QSAR architecture for mutagenic impurity assessment (§6.1): "two
complementary (Q)SAR methodologies... one expert rule based, second
statistical based." A black-box GNN cannot serve as the expert-rule
leg, since its attribution is per-feature importance, not the structural
alerts the guideline language requires. A graph-walking Tsetlin
Machine outputs literal-grounded clauses, which is exactly the form an
expert-rule alert system requires. The novelty intersection (graph
topology x Tsetlin literals x constructive counterfactual recourse
x ICH M7(R2) §6.1 alignment) is empirically empty in the prior
literature; see `research/08_regulatory_target.md` for the citation
trail.

## Citation

```bibtex
@misc{graphtm_cbr,
  title  = {Hierarchical Graph Tsetlin Machine with Counterfactual Boolean Recourse},
  author = {Anwar},
  year   = {2026},
  note   = {Pre-print in preparation},
}
```

A machine-readable `CITATION.cff` is at the repository root.

## License

MIT. See `LICENSE` at the repository root.

## Development setup

```bash
# Dev-mode install with test extras.
pip install -e ".[dev]"

# Full integration tests (skip cleanly without CUDA).
pytest tests/test_integration/ -ra

# Per-module tests.
pytest tests/test_core/ tests/test_encoding/ tests/test_cuda/ \
       tests/test_recourse/ -ra

# Throughput benchmarks (CUDA required).
python benchmarks/throughput_forward.py
python benchmarks/throughput_train.py
python benchmarks/cpu_vs_cuda.py

# CUDA toolchain expectation
# - NVIDIA driver >= 535
# - CUDA toolkit >= 12.1 visible to nvcc
# - pycuda built against the system CUDA (see requirements.txt)
```

This project builds on two prior open-source implementations (kept in `vendors/`, not modified):

- Hierarchical Tsetlin Machine by **Ole-Christoffer Granmo and Rupsa Saha** (`HeirarchicalTM_experiments/`). `graphtm/core/hierarchical_tm.py` is a faithful Python port of their C reference; `graphtm/cuda/kernels.cu` matches its numerical semantics under parity tests (`tests/test_integration/test_cpu_cuda_parity.py`).
- Graph Tsetlin Machine by **Ole-Christoffer Granmo, Mayur Shende, Per-Arne Andersen, and Rupsa Saha** (CAIR / `vendors/GraphTsetlinMachine/`). My CUDA kernel layout (bit-plane TA storage, ripple-carry `inc`/`dec`, the `(include & X) != include` AND test, grid-stride launches, compile-time `#define`s via PyCUDA `SourceModule`) follows their pattern; the kernels themselves are written for HGTM's 5-level AND-OR-AND-OR-AND tree.

The build was structured as eight parallel modules (M1-M8) against a
frozen interface contract in `docs/ARCHITECTURE.md`. Each module owns
its own per-module tests under `tests/test_<module>/`; cross-module
integration, parity, and invariants live under `tests/test_integration/`.
