/* graphtm/cuda/kernels.cu — CUDA-C kernels for Hierarchical Graph TM (Option B).
 *
 * Loaded by graphtm/cuda/_kernels.py via PyCUDA SourceModule with the
 * following compile-time #defines prepended to this source. We prefix
 * with HGTM_ to avoid colliding with CUDA's own template parameters
 * (curand_kernel.h uses `T`, `R`, etc.):
 *
 *   HGTM_CLAUSES        — clauses per machine                      (C)
 *   HGTM_R              — ROOT_FACTORS (outer AND arity)
 *   HGTM_IA             — INTERIOR_ALTERNATIVES (OR arity)
 *   HGTM_IF             — INTERIOR_FACTORS (inner AND arity)
 *   HGTM_LA             — LEAF_ALTERNATIVES (OR arity at leaves)
 *   HGTM_LF             — LEAF_FACTORS (literals-per-leaf / 2)
 *   HGTM_K              — number of classes
 *   HGTM_T              — class-sum clip threshold
 *   HGTM_N_MAX          — compile-time per-graph node cap
 *   HGTM_D_CHUNKS       — node hypervector uint32 chunks (D = 2 * FEATURES)
 *   HGTM_STATE_BITS     — bit-planes per packed-32-TA cell           (8)
 *   HGTM_BOOST_TRUE_POSITIVE_FEEDBACK                                (0)
 *
 * Numerical semantics MUST match research/02_hgtm_canonical_spec.md
 * (= vendors/HeirarchicalTM_experiments/TsetlinMachine.{h,c}). The
 * companion CPU oracle is graphtm/core/hierarchical_tm.py — parity is
 * checked in tests/test_cuda (M8).
 *
 * Derived dims (via macros below): HGTM_FEATURES = R*IF*LF;
 * HGTM_LITERALS_PER_CLAUSE = R*IA*IF*LA*2*LF; HGTM_LA_CHUNKS = ceil(./32).
 *
 * TA state physical layout (matches contract):
 *   uint32 ta_state[HGTM_CLAUSES][HGTM_LA_CHUNKS][HGTM_STATE_BITS]
 * The "include action" bit for literal `lit` of clause `c` lives at
 *   ta_state[c][lit/32][HGTM_STATE_BITS-1] & (1u << (lit%32)) != 0.
 * The full state value of TA `lit` of clause `c` is reconstructed by
 *   state = sum over b in [0, HGTM_STATE_BITS) of
 *           ((ta_state[c][lit/32][b] >> (lit%32)) & 1u) << b.
 * Literal index `lit` for clause coord (j,k,l,m,n) (n in [0, 2*LF)):
 *   lit = ((j*IA + k)*IF + l)*LA*2*LF + m*2*LF + n.
 *
 * Node-HV physical layout (matches contract — M1 owns encoding):
 *   uint8 node_hv[B][HGTM_N_MAX][HGTM_D_CHUNKS*4] (uint8-typed, 32 bits per chunk).
 * Bit `feat` of node `n` of graph `b` is
 *   ((uint32*)node_hv)[(b*HGTM_N_MAX + n)*HGTM_D_CHUNKS + feat/32] >> (feat%32) & 1.
 * D = 2*HGTM_FEATURES, with the second half holding the negated bits
 * (M1's responsibility — see invariant 10 in research/02).
 */

#include <stdint.h>
#include <curand_kernel.h>

#ifndef HGTM_STATE_BITS
#define HGTM_STATE_BITS 8
#endif

#ifndef HGTM_BOOST_TRUE_POSITIVE_FEEDBACK
#define HGTM_BOOST_TRUE_POSITIVE_FEEDBACK 0
#endif

#define HGTM_FEATURES             (HGTM_R * HGTM_IF * HGTM_LF)
#define HGTM_LITERALS_PER_LEAF    (2 * HGTM_LF)
#define HGTM_LEAVES_PER_CLAUSE    (HGTM_R * HGTM_IA * HGTM_IF * HGTM_LA)
#define HGTM_LITERALS_PER_CLAUSE  (HGTM_LEAVES_PER_CLAUSE * HGTM_LITERALS_PER_LEAF)
#define HGTM_LA_CHUNKS            (((HGTM_LITERALS_PER_CLAUSE) + 31) / 32)
#define HGTM_INT_SIZE             32

/* ── Action / state read helpers ──────────────────────────────────────
 *
 * ta_action — read the top bit-plane bit for literal `lit` of a clause.
 * Returns 0/1. Hot path: called once per literal per leaf-eval.
 */
__device__ __forceinline__ int ta_action(const uint32_t* clause_ta,
                                          int lit)
{
    int chunk = lit / 32;
    int pos   = lit % 32;
    return (int)((clause_ta[chunk * HGTM_STATE_BITS + (HGTM_STATE_BITS - 1)] >> pos) & 1u);
}

/* Reconstruct full state value (0..2^HGTM_STATE_BITS-1). Diagnostics only. */
__device__ __forceinline__ int ta_state_value(const uint32_t* clause_ta,
                                                int lit)
{
    int chunk = lit / 32;
    int pos   = lit % 32;
    int v = 0;
    for (int b = 0; b < HGTM_STATE_BITS; ++b) {
        v |= (int)((clause_ta[chunk * HGTM_STATE_BITS + b] >> pos) & 1u) << b;
    }
    return v;
}

/* Bit of node hypervector at position `feat`. */
__device__ __forceinline__ int hv_bit(const uint32_t* node_chunks, int feat)
{
    int chunk = feat / 32;
    int pos   = feat % 32;
    return (int)((node_chunks[chunk] >> pos) & 1u);
}

/* ── inc / dec ripple-carry primitives (32 TAs per call) ──────────────
 *
 * Direct port of vendors/GraphTsetlinMachine/.../kernels.py:55-96. Each
 * call advances/retreats 32 packed automata by 1 across HGTM_STATE_BITS
 * bit-planes; the `active` mask says which of the 32 to update. Saturates
 * at the top plane by OR-ing remaining carry into every plane, and at
 * the bottom plane by AND-ing it out. State range mapped onto
 * HGTM_STATE_BITS is [0, 2^HGTM_STATE_BITS - 1] with action threshold at
 * the top bit (state >= 2^(HGTM_STATE_BITS-1)). For HGTM_STATE_BITS=8
 * this is [0,255] with action threshold 128.
 */
__device__ __forceinline__ void ripple_inc(uint32_t* clause_ta,
                                             int chunk,
                                             uint32_t active)
{
    uint32_t carry, carry_next;
    int id = chunk * HGTM_STATE_BITS;
    carry = active;
    for (int b = 0; b < HGTM_STATE_BITS; ++b) {
        if (carry == 0u) break;
        carry_next = clause_ta[id + b] & carry;
        clause_ta[id + b] = clause_ta[id + b] ^ carry;
        carry = carry_next;
    }
    if (carry > 0u) {
        for (int b = 0; b < HGTM_STATE_BITS; ++b) {
            clause_ta[id + b] |= carry;
        }
    }
}

__device__ __forceinline__ void ripple_dec(uint32_t* clause_ta,
                                             int chunk,
                                             uint32_t active)
{
    uint32_t carry, carry_next;
    int id = chunk * HGTM_STATE_BITS;
    carry = active;
    for (int b = 0; b < HGTM_STATE_BITS; ++b) {
        if (carry == 0u) break;
        carry_next = (~clause_ta[id + b]) & carry;
        clause_ta[id + b] = clause_ta[id + b] ^ carry;
        carry = carry_next;
    }
    if (carry > 0u) {
        for (int b = 0; b < HGTM_STATE_BITS; ++b) {
            clause_ta[id + b] &= ~carry;
        }
    }
}

/* Linearise canonical coord (j,k,l,m,n) (n in [0, 2*LF)) → literal idx. */
__device__ __forceinline__ int lit_index(int j, int k, int l, int m, int n)
{
    return ((j * HGTM_IA + k) * HGTM_IF + l) * HGTM_LA * HGTM_LITERALS_PER_LEAF
           + m * HGTM_LITERALS_PER_LEAF + n;
}

extern "C" {

/* ─────────────────────────────────────────────────────────────────────
 *  Kernel 1: clause_forward_pernode
 *
 *  Per (graph b, clause c, node n) compute the HTM AND-OR-AND-OR-AND
 *  tree output as the C reference does (TsetlinMachine.c:80-138).
 *
 *  Grid: dim3(B, C, 1)              one block per (graph, clause)
 *  Block: dim3(threads,  1, 1)      threads stride over nodes
 *
 *  Each thread loops over a stride of nodes within `n_nodes_per_graph`
 *  and evaluates the full nested-loop tree against that node's HV.
 *
 *  Output `clause_node_out[b, c, n]` ∈ {0, 1}:
 *      1 iff clause_output_at_that_node > 0
 *
 *  Note: the canonical clause_output is a vote-multiplicity (int).
 *  But for graph-walking we then OR across nodes, so collapsing to
 *  {0,1} per-node is the natural reduction. The class-sum below uses
 *  this {0,1} per-clause output (signed by the alternating clause
 *  parity).
 * ───────────────────────────────────────────────────────────────────── */
__global__ void clause_forward_pernode(
    const uint32_t* __restrict__ ta_state,        /* [C, HGTM_LA_CHUNKS, HGTM_STATE_BITS] */
    const uint8_t*  __restrict__ node_hv,         /* [B, N_max, D_chunks*4]    */
    const uint8_t*  __restrict__ edge_hv,         /* [B, E_max, D_chunks*4] (unused at depth 0) */
    const int*      __restrict__ node_offset,     /* [B+1] csr-style           */
    const int*      __restrict__ edge_index,      /* [B, 2, E_max] (unused at depth 0) */
    int8_t*         __restrict__ clause_node_out, /* [B, C, N_max]             */
    int n_graphs,
    int N_max_runtime,
    int D_chunks_runtime
)
{
    int b = blockIdx.x;
    int c = blockIdx.y;
    if (b >= n_graphs || c >= HGTM_CLAUSES) return;

    int n_nodes = node_offset[b + 1] - node_offset[b];
    if (n_nodes > N_max_runtime) n_nodes = N_max_runtime;

    const uint32_t* clause_ta = ta_state + (size_t)c * HGTM_LA_CHUNKS * HGTM_STATE_BITS;

    (void)edge_hv;
    (void)edge_index;

    for (int n = threadIdx.x; n < N_max_runtime; n += blockDim.x) {
        if (n >= n_nodes) {
            /* zero-pad nodes beyond the graph's actual count */
            clause_node_out[(size_t)b * HGTM_CLAUSES * N_max_runtime
                            + (size_t)c * N_max_runtime + n] = 0;
            continue;
        }

        const uint32_t* x_chunks = (const uint32_t*)(
            node_hv + ((size_t)b * N_max_runtime + n) * (size_t)D_chunks_runtime * 4
        );

        /* ── canonical AND-OR-AND-OR-AND tree, per the C reference ── */
        int clause_output = 1;             /* AND-product over root j  */
        for (int j = 0; j < HGTM_R; ++j) {
            int interior_vote_sums_j = 0;  /* OR-sum over k             */
            for (int k = 0; k < HGTM_IA; ++k) {
                int interior_vote_products_jk = 1;  /* AND-product over l */
                for (int l = 0; l < HGTM_IF; ++l) {
                    int leaf_vote_sum_jkl = 0;      /* OR-sum over m    */
                    for (int m = 0; m < HGTM_LA; ++m) {
                        int out = 1;
                        /* per-leaf AND of HGTM_LITERALS_PER_LEAF literals.
                         * Canonical feat partition: pos lit n at HV bit
                         * (j*IF*LF + l*LF + n); neg lit n at the same
                         * bit + HGTM_FEATURES. */
                        for (int n_lit = 0; n_lit < HGTM_LF; ++n_lit) {
                            int feat = j * HGTM_IF * HGTM_LF + l * HGTM_LF + n_lit;
                            /* positive literal */
                            int lp = lit_index(j, k, l, m, n_lit);
                            if (ta_action(clause_ta, lp) == 1
                                && hv_bit(x_chunks, feat) == 0) {
                                out = 0; break;
                            }
                            /* negated literal */
                            int ln = lit_index(j, k, l, m, n_lit + HGTM_LF);
                            if (ta_action(clause_ta, ln) == 1
                                && hv_bit(x_chunks, feat + HGTM_FEATURES) == 0) {
                                out = 0; break;
                            }
                        }
                        leaf_vote_sum_jkl += out;   /* "OR" = +1 if alt fired */
                    }
                    interior_vote_products_jk *= leaf_vote_sum_jkl;
                }
                interior_vote_sums_j += interior_vote_products_jk;
            }
            clause_output *= interior_vote_sums_j;
        }
        clause_node_out[(size_t)b * HGTM_CLAUSES * N_max_runtime
                        + (size_t)c * N_max_runtime + n] = (int8_t)(clause_output > 0 ? 1 : 0);
    }
}

/* ─────────────────────────────────────────────────────────────────────
 *  Kernel 2: clause_or_across_nodes
 *
 *  Per (graph b, clause c) OR the per-node {0,1} outputs into a single
 *  per-(graph,clause) byte. Mirrors `evaluate` (kernels.py:279-317)
 *  without the atomicAdd — the class-sum is split into a separate
 *  kernel for clarity with alt-sign weighted sums.
 *
 *  Grid: dim3(B, C, 1); one thread per (b, c).
 * ───────────────────────────────────────────────────────────────────── */
__global__ void clause_or_across_nodes(
    const int8_t* __restrict__ clause_node_out,   /* [B, C, N_max] */
    const int*    __restrict__ n_nodes_per_graph, /* [B]           */
    int8_t*       __restrict__ clause_out,        /* [B, C]        */
    int B_runtime,
    int N_max_runtime
)
{
    int b = blockIdx.x;
    int c = blockIdx.y;
    if (b >= B_runtime || c >= HGTM_CLAUSES) return;

    int n_nodes = n_nodes_per_graph[b];
    if (n_nodes > N_max_runtime) n_nodes = N_max_runtime;

    int any = 0;
    for (int n = 0; n < n_nodes; ++n) {
        if (clause_node_out[(size_t)b * HGTM_CLAUSES * N_max_runtime
                            + (size_t)c * N_max_runtime + n]) {
            any = 1; break;
        }
    }
    clause_out[(size_t)b * HGTM_CLAUSES + c] = (int8_t)any;
}

/* ─────────────────────────────────────────────────────────────────────
 *  Kernel 3: class_sum_reduce
 *
 *  For each (graph b, class k_id), sum clause votes weighted by the
 *  alternating clause sign (1 - 2*(c & 1)) for clauses with
 *  clause_class[c] == k_id. Clip to ±HGTM_T.
 *
 *  Canonical class-sum: research/02 §2 (TsetlinMachine.c:209-221).
 *
 *  Grid: dim3(B, K, 1); one thread per (b, k_id).
 * ───────────────────────────────────────────────────────────────────── */
__global__ void class_sum_reduce(
    const int8_t* __restrict__ clause_out,    /* [B, C]   */
    const int8_t* __restrict__ clause_class,  /* [C]      */
    int*          __restrict__ class_sum,     /* [B, K]   */
    int B_runtime
)
{
    int b   = blockIdx.x;
    int kid = blockIdx.y;
    if (b >= B_runtime || kid >= HGTM_K) return;

    int acc = 0;
    for (int c = 0; c < HGTM_CLAUSES; ++c) {
        if ((int)clause_class[c] != kid) continue;
        int sign = 1 - 2 * (c & 1);
        acc += sign * (int)clause_out[(size_t)b * HGTM_CLAUSES + c];
    }
    if (acc >  HGTM_T) acc =  HGTM_T;
    if (acc < -HGTM_T) acc = -HGTM_T;
    class_sum[(size_t)b * HGTM_K + kid] = acc;
}

/* ─────────────────────────────────────────────────────────────────────
 *  Kernel 4: clause_feedback
 *
 *  Per-clause path-conditional feedback (Type Ia / Ib / II) per the
 *  canonical HTM spec (research/02 §3,
 *  vendors/HeirarchicalTM_experiments/TsetlinMachine.c:233-279).
 *
 *  We recompute clause_component_output[c,j,k,l,m] and
 *  interior_vote_products[c,j,k] from the per-node clause outputs at
 *  the chosen node (set by select_clause_node, mirroring cair).
 *
 *  Per-clause Bernoulli prob (canonical, line 314):
 *      p = (T + (1 - 2*target) * class_sum) / (2T)
 *  Per-literal Bernoullis: `1/s` and `(s-1)/s` (Type Ia recognize) or
 *  `1/s` (Type Ib forget).
 *
 *  Grid: dim3(B, CLAUSES, 1); one thread per (b, c).
 * ───────────────────────────────────────────────────────────────────── */
__global__ void clause_feedback(
    uint32_t*       __restrict__ ta_state,        /* [C, LA_CHUNKS, STATE_BITS] (RW) */
    const int8_t*   __restrict__ clause_node_out, /* [B, C, N_max] */
    const int*      __restrict__ chosen_node,     /* [B, C] */
    const uint8_t*  __restrict__ node_hv,         /* [B, N_max, D_chunks*4] */
    const int*      __restrict__ class_sum,       /* [B, K] (clipped) */
    const int*      __restrict__ y_target,        /* [B] 0/1 */
    const int8_t*   __restrict__ clause_class,    /* [C] */
    float           s_specificity,
    int             n_states_canonical,           /* canonical macro (100); diagnostics only */
    uint64_t        rng_seed,
    uint64_t        step,
    int             N_max_runtime,
    int             B_runtime
)
{
    int b = blockIdx.x;
    int c = blockIdx.y;
    if (b >= B_runtime || c >= HGTM_CLAUSES) return;

    (void)clause_node_out;
    (void)n_states_canonical;

    /* Per-clause polarity from canonical alternating-sign rule. */
    int sign = 1 - 2 * (c & 1);
    int tgt  = y_target[b];
    int sgn  = sign * (2 * tgt - 1);     /* >0 → Type I, <0 → Type II */
    int kc   = (int)clause_class[c];
    int csum = class_sum[(size_t)b * HGTM_K + kc];

    /* Per-clause Bernoulli prob (research/02 §3, C-ref line 314). */
    float p_clause = ((float)HGTM_T + (float)((1 - 2 * tgt)) * (float)csum)
                     / (2.0f * (float)HGTM_T);
    if (p_clause < 0.0f) p_clause = 0.0f;
    if (p_clause > 1.0f) p_clause = 1.0f;

    int node = chosen_node[(size_t)b * HGTM_CLAUSES + c];
    int clause_output = (node >= 0) ? 1 : 0;

    /* RNG: per-(b,c) Philox stream. */
    curandStatePhilox4_32_10_t rng;
    uint64_t seq = (uint64_t)b * HGTM_CLAUSES + (uint64_t)c;
    curand_init(rng_seed ^ step, seq, 0, &rng);

    const uint32_t* x_chunks = nullptr;
    if (clause_output == 1) {
        x_chunks = (const uint32_t*)(
            node_hv + ((size_t)b * N_max_runtime + node)
                       * (size_t)HGTM_D_CHUNKS * 4
        );
    }

    uint32_t* clause_ta = ta_state + (size_t)c * HGTM_LA_CHUNKS * HGTM_STATE_BITS;

    /* Walk the tree, recompute the gating fields, apply feedback. */
    for (int j = 0; j < HGTM_R; ++j) {
        for (int k = 0; k < HGTM_IA; ++k) {
            int ivp_jk = 1;
            int component[HGTM_IF][HGTM_LA];
            for (int l = 0; l < HGTM_IF; ++l) {
                int leaf_vote_sum_jkl = 0;
                for (int m = 0; m < HGTM_LA; ++m) {
                    int cval = (clause_output == 1) ? 1 : 0;
                    if (clause_output == 1) {
                        for (int n_lit = 0; n_lit < HGTM_LF; ++n_lit) {
                            int feat = j * HGTM_IF * HGTM_LF + l * HGTM_LF + n_lit;
                            int lp = lit_index(j, k, l, m, n_lit);
                            int ln = lit_index(j, k, l, m, n_lit + HGTM_LF);
                            if (ta_action(clause_ta, lp) == 1
                                && hv_bit(x_chunks, feat) == 0) {
                                cval = 0; break;
                            }
                            if (ta_action(clause_ta, ln) == 1
                                && hv_bit(x_chunks, feat + HGTM_FEATURES) == 0) {
                                cval = 0; break;
                            }
                        }
                    }
                    component[l][m] = cval;
                    leaf_vote_sum_jkl += cval;
                }
                ivp_jk *= leaf_vote_sum_jkl;
            }

            for (int l = 0; l < HGTM_IF; ++l) {
                for (int m = 0; m < HGTM_LA; ++m) {
                    float u = curand_uniform(&rng);
                    int fb_active = (u <= p_clause) ? 1 : 0;
                    if (!fb_active) continue;

                    int cco = component[l][m];

                    if (sgn > 0) {
                        /* Type I */
                        int path_alive = (clause_output > 0)
                                          && (ivp_jk > 0)
                                          && (cco != 0);
                        if (!path_alive) {
                            /* Type Ib: stochastic decrement of every TA
                             * in this leaf with prob 1/s. */
                            for (int n_lit = 0; n_lit < HGTM_LF * 2; ++n_lit) {
                                if (curand_uniform(&rng) <= 1.0f / s_specificity) {
                                    int lit = lit_index(j, k, l, m, n_lit);
                                    int chunk = lit / 32;
                                    uint32_t bit = (1u << (lit % 32));
                                    ripple_dec(clause_ta, chunk, bit);
                                }
                            }
                        } else {
                            /* Type Ia */
                            for (int n_lit = 0; n_lit < HGTM_LF; ++n_lit) {
                                int feat = j * HGTM_IF * HGTM_LF + l * HGTM_LF + n_lit;
                                int xp   = hv_bit(x_chunks, feat);
                                int xn   = hv_bit(x_chunks, feat + HGTM_FEATURES);
                                int lp = lit_index(j, k, l, m, n_lit);
                                int ln = lit_index(j, k, l, m, n_lit + HGTM_LF);

                                /* pos lit */
                                {
                                    int chunk = lp / 32;
                                    uint32_t bit = (1u << (lp % 32));
                                    if (xp == 1) {
                                        float u2 = curand_uniform(&rng);
#if HGTM_BOOST_TRUE_POSITIVE_FEEDBACK == 1
                                        ripple_inc(clause_ta, chunk, bit);
                                        (void)u2;
#else
                                        if (u2 <= (s_specificity - 1.0f) / s_specificity) {
                                            ripple_inc(clause_ta, chunk, bit);
                                        }
#endif
                                    } else {
                                        float u2 = curand_uniform(&rng);
                                        if (u2 <= 1.0f / s_specificity) {
                                            ripple_dec(clause_ta, chunk, bit);
                                        }
                                    }
                                }
                                /* neg lit */
                                {
                                    int chunk = ln / 32;
                                    uint32_t bit = (1u << (ln % 32));
                                    if (xn == 1) {
                                        float u2 = curand_uniform(&rng);
#if HGTM_BOOST_TRUE_POSITIVE_FEEDBACK == 1
                                        ripple_inc(clause_ta, chunk, bit);
                                        (void)u2;
#else
                                        if (u2 <= (s_specificity - 1.0f) / s_specificity) {
                                            ripple_inc(clause_ta, chunk, bit);
                                        }
#endif
                                    } else {
                                        float u2 = curand_uniform(&rng);
                                        if (u2 <= 1.0f / s_specificity) {
                                            ripple_dec(clause_ta, chunk, bit);
                                        }
                                    }
                                }
                            }
                        }
                    } else if (sgn < 0) {
                        /* Type II — only on firing leaves with live
                         * ancestor path (C-ref:268). Deterministic
                         * include-push for excluded literals on X==0. */
                        int path_alive = (clause_output > 0)
                                          && (ivp_jk > 0)
                                          && (cco == 1);
                        if (!path_alive) continue;
                        for (int n_lit = 0; n_lit < HGTM_LF; ++n_lit) {
                            int feat = j * HGTM_IF * HGTM_LF + l * HGTM_LF + n_lit;
                            int xp   = hv_bit(x_chunks, feat);
                            int xn   = hv_bit(x_chunks, feat + HGTM_FEATURES);
                            int lp = lit_index(j, k, l, m, n_lit);
                            int ln = lit_index(j, k, l, m, n_lit + HGTM_LF);
                            if (ta_action(clause_ta, lp) == 0 && xp == 0) {
                                int chunk = lp / 32;
                                uint32_t bit = (1u << (lp % 32));
                                ripple_inc(clause_ta, chunk, bit);
                            }
                            if (ta_action(clause_ta, ln) == 0 && xn == 0) {
                                int chunk = ln / 32;
                                uint32_t bit = (1u << (ln % 32));
                                ripple_inc(clause_ta, chunk, bit);
                            }
                        }
                    }
                    /* sgn == 0 → skip */
                }
            }
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
 *  Kernel 5: select_clause_node
 *
 *  Per (graph, clause) pick a random node where the clause fired.
 *  Mirrors vendors/.../kernels.py:319-354. -1 if none fired (Type Ib
 *  decay still runs in clause_feedback).
 *  Grid: dim3(B, C, 1); one thread per block.
 * ───────────────────────────────────────────────────────────────────── */
__global__ void select_clause_node(
    const int8_t* __restrict__ clause_node_out,
    const int*    __restrict__ n_nodes_per_graph,
    int*          __restrict__ chosen_node,
    uint64_t      rng_seed,
    uint64_t      step,
    int           B_runtime,
    int           N_max_runtime
)
{
    int b = blockIdx.x;
    int c = blockIdx.y;
    if (b >= B_runtime || c >= HGTM_CLAUSES) return;

    int n_nodes = n_nodes_per_graph[b];
    if (n_nodes > N_max_runtime) n_nodes = N_max_runtime;

    int fired[HGTM_N_MAX];
    int n_fired = 0;
    for (int n = 0; n < n_nodes; ++n) {
        if (clause_node_out[(size_t)b * HGTM_CLAUSES * N_max_runtime
                            + (size_t)c * N_max_runtime + n]) {
            fired[n_fired++] = n;
        }
    }
    int picked = -1;
    if (n_fired > 0) {
        curandStatePhilox4_32_10_t rng;
        uint64_t seq = (uint64_t)b * HGTM_CLAUSES + (uint64_t)c;
        curand_init(rng_seed ^ step ^ 0xa5a5a5a5ULL, seq, 0, &rng);
        unsigned int r = curand(&rng) % (unsigned int)n_fired;
        picked = fired[r];
    }
    chosen_node[(size_t)b * HGTM_CLAUSES + c] = picked;
}

/* ─────────────────────────────────────────────────────────────────────
 *  Kernel 6: init_ta_state
 *
 *  Initialise TA state to "centre" (lower planes all-ones, top plane
 *  zero ⇒ state = 2^(HGTM_STATE_BITS-1) - 1, action = 0). Mirrors
 *  cair's `prepare` (kernels.py:638-664). M3 may override with the
 *  canonical pos/neg coin-flip init for tighter parity to the C-ref;
 *  the host-side variant lives in graphtm/cuda/memory.py.
 * ───────────────────────────────────────────────────────────────────── */
__global__ void init_ta_state(uint32_t* __restrict__ ta_state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int c = index; c < HGTM_CLAUSES; c += stride) {
        uint32_t* clause_ta = ta_state + (size_t)c * HGTM_LA_CHUNKS * HGTM_STATE_BITS;
        for (int la = 0; la < HGTM_LA_CHUNKS; ++la) {
            for (int b = 0; b < HGTM_STATE_BITS - 1; ++b) {
                clause_ta[la * HGTM_STATE_BITS + b] = ~0u;
            }
            clause_ta[la * HGTM_STATE_BITS + HGTM_STATE_BITS - 1] = 0u;
        }
    }
}

}  /* extern "C" */
