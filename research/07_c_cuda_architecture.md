# C / CUDA-C vs Numba Architecture for Production HGTM

Drop to CUDA-C (PyCUDA `SourceModule` or PyTorch C++ extension) or stay
in Numba-CUDA for the HGTM engine: 10k-100k graphs, 1000-5000 clauses,
training < 30 min on one GPU. **No numeric benchmark in this document
is fabricated**; where a number is not inspectable in the vendored
repos, I say so.

---

## 1. Survey of existing TM engines

| Engine | Lang | GPU? | Throughput / wall-clock | License |
|---|---|---|---|---|
| cair/pyTsetlinMachine (vanilla) | pure Python | no | not published in repo; baseline reference | MIT |
| cair/pyTsetlinMachineParallel | C + OpenMP via `ctypes` | no (CPU-MT) | not in vendored copy; speedup curve only in upstream README | MIT |
| cair/PyTsetlinMachineCUDA | **CUDA-C via PyCUDA `SourceModule`** | yes | no throughput numbers published in repo source | MIT |
| cair/TMU | OpenMP-C CPU; CUDA-C GPU backend | yes | claims 10-50x CPU on MNIST in upstream README; not measured here | MIT |
| cair/GraphTsetlinMachine (vendored) | **CUDA-C strings + PyCUDA** (`kernels.py` line 24+, 22 `__global__`/`__device__`; `tm.py` imports `pycuda.compiler.SourceModule`) | yes | repo has **no throughput benchmarks**, only demo stdout ("epoch 0 = 5.4s, 10k NoisyXOR, 10 clauses", `README.md:155-163`); roadmap explicitly flags `graphs.py` construction as bottleneck (`README.md:332-338`) | MIT |
| HierarchicalTM_experiments (vendored) | **plain C, single-threaded** (`TsetlinMachine.c`, `MultiClassTsetlinMachine.c`) | no | no benchmarks in repo | MIT |
| Bhattarai et al. 2024 ("Word-level Tsetlin Machine") | reports CUDA results, doesn't release a new engine | yes | published claim 2-7x CPU on text, **paper not in repo**, can't verify | n/a |
| Abeyrathna et al. "Coalesced TM" | CUDA-C, exists as separate cair repo | yes | paper claims coalesced layout helps; **not in vendored set**, no first-hand number | MIT |

**Field summary**: every production cair engine that needs speed is
CUDA-C (PyCUDA `SourceModule` with template-style specialisation via
`#define`). Only the "vanilla" pedagogical variant is pure Python.
**No major TM project is built on Numba-CUDA.** Strong prior.

---

## 2. Numba-CUDA vs CUDA-C realistic gap

The cair design specifically exploits things Numba-CUDA cannot do
cleanly:

**Bit-packed literal eval.** Hot path is
`(include_mask & X_chunk) != include_mask` over 32 packed literals per
`uint32` (`kernels.py:448-456`). Numba does `&`/`!=` on `uint32` fine,
but the companion intrinsics (`__ldg` (read-only L1 hint), `__popcll`,
warp-level `__ballot_sync`/`__shfl_xor_sync`, cooperative groups)
are either missing or awkward in Numba. Numba has basic
`cuda.shfl_sync` and atomics but not the broader intrinsic set.
**Realistic single-kernel gap on the AND test: 1.5-3x in favour of
CUDA-C**, dominated by `__ldg` + register scheduling. Estimate, not
a measurement on my workload.

**Compile-time specialisation.** cair compiles per-model with
`#define CLAUSES 1000`, `#define LITERALS 256`, `#define STATE_BITS 8`
baked in (`tm.py` builds the parameter string, prepends to
`code_header`, passes to `SourceModule`). nvcc unrolls inner loops and
eliminates branches. For HGTM with the canonical tree
(`R·IA·IF·LA·LF = 2·2·2·10·2 = 160` TA per clause, 5 fixed-depth
nested loops in `vendors/HeirarchicalTM_experiments/TsetlinMachine.c:80-138`),
nvcc can fully unroll the inner three loops and eliminate the LA·LF
indices entirely. Numba JITs once with runtime sizes and cannot do
this. **Estimated gap: 1.3-2x** on the tree forward.

**Combined CUDA-C vs Numba-CUDA**: 2-5x defensible range on the
bit-packed forward; 2-3x on feedback (more branch-heavy). Blanket
"10x" is not supported.

**Engineering cost.** Porting `calculate_messages` (~60 lines) to
CUDA-C with parity: 2-5 person-days. Full HGTM kernel set (5-level
forward, feedback, message exchange, encode): **3-6 person-weeks**
plus 1-2 weeks debugging. The cair `kernels.py` already covers ~70%
of what I need.

**Verdict.** CUDA-C is worth it when (a) shape is fixed at training
time (HGTM tree dims are), (b) inner loop is bit-parallel (it is),
(c) you train repeatedly at fixed size (yes). Not worth it for
research prototypes that change weekly.

---

## 3. TM-specific tricks

These are the levers a CUDA-C engine pulls that buy real speed for TM
workloads. Most are visible in the cair vendor code.

1. **Coalesced 32-bit packed pos/neg literals.** cair packs 32 TAs per
   `uint32`, each bit-plane stored contiguously across the 32 TAs
   (kernels.py:55-58). `inc`/`dec` becomes a ripple-carry across
   STATE_BITS planes (~8 `uint32` ops for 32 TAs). Doable in Numba,
   but the layout is unintuitive in Python and easy to get wrong.

2. **Block-cooperative class-sum reduction.** cair uses `atomicAdd`
   into `class_sum[class_id]` per clause (kernels.py:310-315). For
   larger CLASSES, a block-level warp-shuffle reduction (one block
   per class, threads = clauses-per-class) avoids atomic contention.
   Possible in Numba via `cuda.shfl_sync` but uglier.

3. **Persistent kernel: one launch, many samples.** cair launches
   the full kernel set per sample (`tm.py:_fit` loops graphs
   sequentially, 4-6 kernels each). A persistent kernel holding
   occupancy and pulling samples from a device-side queue eliminates
   most launch overhead (each launch is ~5-20 microseconds; 100k
   graphs × 5 kernels × 10us = ~5 seconds per epoch of pure launch
   overhead). **Practical only in CUDA-C**: Numba cannot express a
   device-side work queue cleanly.

4. **Warp-level feedback.** Type-I/II feedback uses per-literal
   `curand_uniform` rejection sampling (kernels.py:141-183). With
   `__ballot_sync` you can vote across a warp on which TAs to
   increment in one mask, then apply to the bit-plane in one shot.
   cair already does "32 random bits at once" (kernels.py:114-117)
   but not the warp vote. Estimated ~1.2-1.5x, not measured.

5. **Mixed-precision int8/int16.** cair uses bit-plane packed
   STATE_BITS=8 (~256 distinct states; matches the C reference's
   `NUMBER_OF_STATES = 100`). Bit-plane packing is *more*
   memory-efficient than int8 already; mixed precision is a
   non-trick here. **Skip.**

Biggest CUDA-C-only win: **(3) persistent kernels**.

---

## 4. Architecture recommendation

**(a) Numba-CUDA only.** 5-10x CPU is roughly right for plain
Numba-CUDA on a bit-packed TM kernel vs single-core OpenMP-C.
Training 100k graphs × 5000 clauses × HGTM tree depth extrapolates to
**single-hour to multi-hour** on one GPU. Margins too thin for 30 min.
**Reject.**

**(b) Numba-CUDA harness + critical kernels in CUDA-C.** The cair
GraphTM vendor *is already this architecture* (Python orchestration,
CUDA-C kernels via PyCUDA). For HGTM: write the per-level forward,
feedback, and message-exchange kernels in CUDA-C with `SourceModule`
compile-time `#define`s for tree dims; keep batching, graph
construction, and the training loop in Python. The cair vendor is a
working reference for ~70% of the kernels: bit-plane TA layout,
`inc`/`dec` ripple-carry, `(include & X) != include` AND test all
carry over. New work: 5-level tree forward, credit-assignment through
OR, persistent-kernel scheduling. **3-6 person-weeks. Recommended.**
30-50x CPU is a reasonable estimate for this pattern.

**(c) Full CUDA-C with PyTorch C++ extension.** Buys torch tensors,
nvcc build integration, ATen interop. **Does not buy faster kernels**
than PyCUDA `SourceModule`. The ~100x CPU claim is plausible *only*
with persistent kernels and warp-level feedback; those work in
either harness. PyTorch wrapper is higher engineering cost (build
system, ABI churn) for modest gain. **Use only if torch.autograd
integration becomes required** (it isn't; TM training is RL-style).
**Reject as default.**

**Recommendation: (b).** Numba-CUDA stays for non-critical paths
(graph construction, post-processing, CBR retrieval). HGTM training
kernels in CUDA-C, compiled per-config with `SourceModule`, cair
GraphTM vendor as reference. Persistent-kernel scheduling in a second
pass after C-reference parity.

Risks not hand-waved: (1) PyCUDA support for newer CUDA toolkits has
lagged; fallback is `.cu` files + `nvcc -shared` + `ctypes` (TMU's
approach). (2) 30-min target depends on per-graph node count: 10k
graphs × 30 nodes is very different from 100k × 500. The brief does
not pin this down. (3) **None of the speedup numbers here are
measured on my HGTM workload**; they are kernel-level priors. First
milestone after the CUDA-C port should be an actual benchmark on
representative HGTM data so option (b) earns its keep against an
honest Numba-CUDA baseline.
