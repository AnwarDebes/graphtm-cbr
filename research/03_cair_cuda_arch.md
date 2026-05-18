# cair/GraphTsetlinMachine CUDA Architecture

All paths relative to `vendors/GraphTsetlinMachine/`. Package files under
`GraphTsetlinMachine/`.

## 1. Top-level pipeline, API, GPU data layout

User code: build `Graphs`, instantiate a TM, call `fit/score/predict`
(`examples/NoisyXORDemo.py:39-83`).

`Graphs` (`graphs.py:26-86`) takes `(N, symbols, hypervector_size,
hypervector_bits, double_hashing, one_hot_encoding)`. Each symbol gets a random
hv index set (`graphs.py:72-76`). Construction sequence:
`set_number_of_graph_nodes` -> `prepare_node_configuration` (CSR-style
prefix-sum `node_index`, plus flat `node_type`,
`number_of_graph_node_edges`, `edge_index`, `X[total_nodes, hv_chunks]`,
`graphs.py:100-113`) -> `add_graph_node` -> `prepare_edge_configuration` (flat
`edge[edge_id]=(dest,type)`, `graphs.py:127-129`) -> `add_graph_node_edge` ->
`add_graph_node_property`. Properties OR the symbol's hv bits into the first
half of the node's `X` chunks and clear the corresponding bits in the second
(negated) half (`graphs.py:148-162`); the negated half is pre-initialised to
all-ones (`graphs.py:91-98`). `encode()` SHA-256-hashes `X|edge` into a
signature used for GPU caching (`graphs.py:209-225`).

The TM front-end is `MultiClassGraphTsetlinMachine` (also
`MultiOutput`/`GraphTsetlinMachine`) in `tm.py:903-1072`. `fit` encodes Y to
`{+T,-T}` per class (`tm.py:949-953`) and loops graphs sequentially:
`_evaluate` then four learning kernels (`tm.py:664-752`).

GPU allocations (`tm.py:515-561`, `tm.py:111-124`):
- `ta_state_gpu`: `uint32[clauses * LA_CHUNKS * STATE_BITS]`,
- `message_ta_state_gpu[depth]`: same with `MESSAGE_CHUNKS`,
- `clause_weights_gpu`: `int32[outputs * clauses]`,
- `encoded_X_*_gpu` from `graphs.X` (node-major uint32 chunks),
- `current/next_clause_node_output_gpu`: bit-packed
  `clauses * max_node_chunks * 4` (clause-fires-at-node),
- `clause_X_int_gpu` (unpacked msg literals), `clause_X_gpu[depth]` (packed),
- `hypervectors_gpu`: `clauses * message_bits * 4` of message-bit indices.

## 2. CUDA kernel inventory

Every `__global__` in `kernels.py`:

| Kernel | Module | Lines | Purpose |
|---|---|---|---|
| `prepare` | `code_prepare` | 638-664 | Init TAs to "just-excluded" (lower bits=1, top=0); init weights to +/-1 |
| `prepare_message_ta_state` | `code_prepare` | 622-636 | Same init for message TAs |
| `evaluate` | `code_evaluate` | 279-317 | OR clause output over nodes; atomicAdd weights into class_sum |
| `select_clause_node` | `code_evaluate` | 319-354 | Per clause pick one random node where it fired |
| `select_clause_updates` | `code_evaluate` | 356-405 | Choose (class,clause) feedback signs; update weights |
| `calculate_messages` | `code_evaluate` | 407-468 | Layer-0 clause-AND eval per (clause, node_chunk) |
| `calculate_messages_conditional` | `code_evaluate` | 470-530 | Same, AND-masked by previous layer (depth>0) |
| `prepare_messages` | `code_evaluate` | 532-547 | Reset per-node message bits |
| `exchange_messages` | `code_evaluate` | 549-590 | Push clause output along edges, with edge-type binding |
| `encode_messages` | `code_evaluate` | 592-615 | Pack per-node message bits into 32-bit chunks |
| `update` | `code_update` | 213-242 | Per-clause TA feedback on input literals |
| `update_message` | `code_update` | 185-211 | Per-clause TA feedback on message literals |
| `transform` | `code_transform` | 671-691 | Binary clause activation per sample |
| `transform_nodewise` | `code_transform` | 693-709 | Per-node activation |
| `get_ta_states` | `code_clauses` | 718-757 | Reconstruct integer TA state |
| `get_hyperliterals` | `code_clauses` | 761-796 | Extract include/exclude bit (top STATE_BITS plane) |

Launch config: `grid=(16*13*4,1,1)`, `block=(128,1,1)` on DGX H100; `16*13`
blocks on A100 (`README.md:312-329`, `tm.py:53-54`). All kernels use grid-stride
loops; no shared memory.

## 3. Forward kernel design

Per-clause AND lives in `calculate_messages` (`kernels.py:407-468`). Parallel
unit = one thread per **(clause, node_chunk)** pair:

- Iterates `clause_node_chunk in [0, CLAUSES*NODE_CHUNKS)` (`kernels.py:433`).
- `clause = idx % CLAUSES`, `node_chunk = idx / CLAUSES` (`kernels.py:434-435`).
  Note: `calculate_messages_conditional` *reverses* this:
  `clause = idx / NODE_CHUNKS`, `node_chunk = idx % NODE_CHUNKS`
  (`kernels.py:496-497`). Different memory access patterns per layer.
- Node-type filter: clause c only matches nodes where
  `node_type[graph_index + node] == c % number_of_node_types`
  (`kernels.py:447`) - round-robin clause-to-type assignment.
- AND test (`kernels.py:448-456`): for each `la_chunk`,
  `(include_mask & X_chunk) != include_mask` implies a missing required literal
  -> clear `clause_node_output` bit for that node. The "include mask" is the
  **top STATE_BITS plane**: `ta_state[la_chunk*STATE_BITS + STATE_BITS-1]`. Tail
  chunk masked with `FILTER` for non-multiple-of-32 literals.
- Output: 32 nodes -> 1 uint32 at
  `global_clause_node_output[clause*NODE_CHUNKS + node_chunk]`.

So 32 literals are evaluated per `&` instruction. There is no "block per
clause" structure; everything is flat grid-stride.

OR-across-nodes is in `evaluate` (`kernels.py:279-317`): one thread per clause
sweeps `NODE_CHUNKS`, breaks on first nonzero chunk, then loops classes and
`atomicAdd(&class_sum[class_id], clause_weight)` (`kernels.py:310-315`) - the
only atomic in the forward path.

## 4. Feedback kernel design

Three-step sequence per training example (`tm.py:692-731`):

(a) `select_clause_node` (`kernels.py:319-354`). One thread per clause walks
all `number_of_nodes`, collects nodes where it fired into local array
`clause_true_node[MAX_NODES]` (`kernels.py:331`), picks one uniformly using
`curand`. `MAX_NODES` is a compile-time macro - hard cap on graph size.
`clause_node[clause] = -1` when no node fired.

(b) `select_clause_updates` (`kernels.py:356-405`). One thread per clause,
loops over CLASSES. Clips `class_sum` to +/-THRESHOLD (`kernels.py:374-378`).
Two rejection samples decide whether to feedback: `Q/max(1,CLASSES-1)` and
`|y - sum|/(2*THRESHOLD)` (`kernels.py:384`). Writes
`class_clause_update[class_id*CLAUSES + clause]` in `{-1,0,+1}`. Updates
`clause_weights` non-atomically (`kernels.py:390-398`) - safe since one thread
per clause.

(c) `update` (`kernels.py:213-242`) and `update_message`
(`kernels.py:185-211`). Grid-stride over clauses; each loops classes calling
`update_clause` / `update_clause_message` (`kernels.py:141-183`, `98-139`)
implementing Type-I and Type-II feedback. Random bits sampled 32-at-a-time
against `1/s` (`kernels.py:114-117, 156-162`). Boosted true-positive feedback
toggled at compile time (`kernels.py:121-125`).

The bitwise TA update primitives are `inc`/`dec` (`kernels.py:55-96`): a
ripple-carry across STATE_BITS bit-planes via XOR and AND, saturating at top by
ORing carry into every plane. Operates on 32 TAs in parallel via bitwise ops.

**Reductions/atomics**: per-class sum reduced via the single
`atomicAdd(&class_sum[class_id], clause_weight)` in `evaluate`
(`kernels.py:313`). All clause threads re-read `class_sum[class_id]` in
`select_clause_updates` (`kernels.py:373`). Feedback kernels have no atomics
(one thread owns a clause).

## 5. Memory layout

TA state is **bit-plane packed**: `uint32[clauses * LA_CHUNKS * STATE_BITS]`
with STATE_BITS=8 default (`tm.py:47, 111`). Layout per clause is
chunk-major then bit-plane-minor: `ta_state[la_chunk*STATE_BITS + b]` holds
bit-plane `b` for 32 packed TAs at literals `la_chunk*32..la_chunk*32+31`
(`kernels.py:55-58`). One TA's full state is reconstructed by gathering bit
`(literal % 32)` from STATE_BITS consecutive ints (`kernels.py:746-754`). This
layout is what makes `inc/dec` operate on 32 TAs simultaneously and the AND
test consume one int per 32 literals.

`clause_weights` and `class_clause_update`: `int32[CLASSES, CLAUSES]`
row-major-over-classes (`tm.py:117, 555`; access at
`clause_weights[class_id*CLAUSES + clause]`, `kernels.py:312`).

Message TA state: `uint32[clauses * MESSAGE_CHUNKS * STATE_BITS]`, one tensor
per depth (`tm.py:113-115`).

Coalescing: `calculate_messages` uses `clause = idx % CLAUSES`
(`kernels.py:434`), so adjacent threads hit adjacent clauses, but
`ta_state[clause*LA_CHUNKS*STATE_BITS + ...]` is widely strided across threads
- not coalesced. The same line is reused 32 times (once per node in the chunk)
so L1 amortises. `X[node*LA_CHUNKS + la_chunk]` (`kernels.py:449`) is sequential
within a thread but adjacent threads (different clauses) touch the same X.

## 6. Graph-specific tricks

1. **Node-type filter via clause indexing**: clause `c` only matches nodes with
   `node_type == c % number_of_node_types` (`kernels.py:447, 509`). No learned
   type selection; assignment is round-robin.

2. **Hypervector edge-binding**: each clause has `MESSAGE_BITS` random message
   indices (`tm.py:103-106`; variants for double-hashing and one-hot at
   `tm.py:90-101`). In `exchange_messages` (`kernels.py:549-590`), when clause
   `c` fires at source `s`, for each outgoing edge `(d, edge_type)` the
   destination `d` gets bits
   `shifted = (clause_bit[i] + edge_type) mod MESSAGE_SIZE` set to true in
   `clause_X_int` and the negated-literal half cleared (`kernels.py:580-584`).
   Binding is **addition modulo MESSAGE_SIZE**, not XOR. `encode_messages`
   (`kernels.py:592-615`) then packs to 32-bit chunks for the next layer.

3. **Multi-hop / deeper reasoning**: `depth` parameter controls layers
   (`tm.py:597-650`). Each layer runs
   `prepare_messages` -> `exchange_messages` -> `encode_messages` ->
   `calculate_messages_conditional`. The conditional kernel ANDs the new
   clause output with `global_clause_node_output_condition` from the prior
   layer (`kernels.py:525-527`) - so a clause must remain true at node `n`
   across every layer. Sequential per-layer, not fused.

4. **Serial-within-clause messaging**: `exchange_messages` loops all
   `number_of_nodes` sources in a single thread (`kernels.py:571`); message
   passing is per-clause O(nodes * avg_degree) and not parallelised across
   nodes/edges.

## 7. What I can/cannot reuse for HTM 5-level tree

Cair is semantically two-level: per-node AND (literals -> clause-at-node,
`kernels.py:407-468`) and per-graph OR across nodes (`evaluate:299-308`). The
depth loop adds one conditional AND tier per layer (`tm.py:597-650`,
`kernels.py:525-527`) but always alternates "AND on literals -> AND with prior
output".

Reusable:
- Bit-plane TA layout, `inc`/`dec` ripple primitives (`kernels.py:55-96`).
- AND test `(include & X) != include` (`kernels.py:448-456`) per AND level.
- Grid-stride launch pattern across all kernels.
- `evaluate`'s atomicAdd reduction (`kernels.py:310-315`).
- Hypervector binding (`kernels.py:580-584`) for level-tagged binding.

Needs extension for HTM AND-OR-AND-OR-AND:
- `update`/`update_clause` (`kernels.py:141-183`) carry two-level Type-I/II
  semantics only. Credit assignment through inner OR nodes needs new kernels
  computing "which inner AND fired in which OR slot" before `inc`/`dec`.
- OR-across-nodes (`evaluate:299-308`) is hardwired to graph-node dimension;
  HTM ORs operate over learned sub-clauses - new kernel needed.
- `select_clause_node`'s stack array `clause_true_node[MAX_NODES]`
  (`kernels.py:331`) caps width at compile time.
- `calculate_messages_conditional` does masking, not OR-aggregation - HTM
  needs an OR-reduction between AND tiers.
- Hv binding uses mod-add not XOR (`kernels.py:580-584`) - fine; apply
  per-tier offset.

## 8. Performance notes

The repo reports **no benchmarks**: no throughput numbers, no GFLOPS, no
scaling plots in README or source. Only hardware hints are launch configs
(`README.md:312-329`):
- DGX-2 / A100: `grid=(16*13,1,1), block=(128,1,1)` -> 208 blocks * 128 =
  26,624 threads.
- DGX H100: `grid=(16*13*4,1,1), block=(128,1,1)` -> 832 blocks * 128 =
  106,496 threads.

The `16*13` factor implies SM-count tuning but is not measured anywhere in
code. The roadmap (`README.md:332-338`) explicitly flags
`graphs.py` (CPU graph construction) as the rewrite target, suggesting it is
the known bottleneck rather than CUDA. Demo printouts (`README.md:155-163`)
show 10k-graph NoisyXOR / 10 clauses / T=100 at 5.4s epoch 0 then ~1.5s/epoch
- demo timings only, not benchmarks.
