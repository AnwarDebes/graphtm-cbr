# Hierarchical Tsetlin Machine: Canonical Engineering Spec

Source of truth: `vendors/HeirarchicalTM_experiments/{TsetlinMachine.h, TsetlinMachine.c, MultiClassTsetlinMachine.h, MultiClassTsetlinMachine.c}` (Granmo et al., 2026 commit). All claims below cite specific lines. Anything not in those files is omitted.

---

## 1. Clause structure (AND-OR-AND-OR-AND tree)

The clause tree has five hierarchical levels, top-down (`TsetlinMachine.c:80-138` for forward; `NoisyParityDemo_withClauses.c:71-130` for symbolic print confirming operator order):

```
Clause   = AND over Root-factors j
Root     = OR  over Interior-alternatives k
Interior = AND over Interior-factors l
Group    = OR  over Leaf-alternatives m
Leaf     = AND over LITERALS_PER_GROUP literals n  (the leaf conjunction)
```

Confirmed by the export-to-JSON nesting `AND > OR > AND > OR > AND` in `NoisyParityDemo_withClauses.c:146-185`.

**Canonical TA tensor** (`TsetlinMachine.h:48`):

```c
int ta_state[CLAUSES][ROOT_FACTORS][INTERIOR_ALTERNATIVES]
            [INTERIOR_FACTORS][LEAF_ALTERNATIVES][LITERALS_PER_GROUP];
```

Axis index meaning, with default sizes from `TsetlinMachine.h:28-39`:

| Axis | Symbol  | Size                              | Role                                                   |
|------|---------|-----------------------------------|--------------------------------------------------------|
| 0    | C       | `CLAUSES = 16`                    | clause id; alternating-sign at sum-time                |
| 1    | j       | `ROOT_FACTORS = 2`                | top-level AND-conjuncts                                |
| 2    | k       | `INTERIOR_ALTERNATIVES = 2`       | OR-disjuncts inside each root conjunct                 |
| 3    | l       | `INTERIOR_FACTORS = 2`            | AND-conjuncts inside each interior alternative         |
| 4    | m       | `LEAF_ALTERNATIVES = 10`          | OR-disjuncts inside each interior factor (leaves)      |
| 5    | 2·LF    | `LITERALS_PER_GROUP = LEAF_FACTORS*2 = 4` | per-leaf TA, first half pos lit, second half neg |

Total automata per clause = `R·IA·IF·LA·2·LF = 2·2·2·10·4 = 320`; per machine = `16·320 = 5120`; per multi-class machine = `8·5120 = 40 960` (`MultiClassTsetlinMachine.h:30`).

**Feature-to-leaf partitioning function** (`TsetlinMachine.c:108`):

```c
int feature = j * INTERIOR_FACTORS * LEAF_FACTORS + l * LEAF_FACTORS + n;
```

`FEATURES = R·IF·LF = 8` (`TsetlinMachine.h:38`); `LITERALS = 2·FEATURES` (`TsetlinMachine.h:39`); negated literal at `feature + FEATURES` (`TsetlinMachine.c:117`). Disjoint block partition: features grouped by `(j,l)`; `k` and `m` are alternatives sharing the same block. `k` does NOT remap features; every k re-evaluates the same block under independent TAs, so the OR over `k` widens expressivity, not coverage.

Caller pre-constructs the negated half (`X[i][j + FEATURES] = 1 - X[i][j]`), see `additionDemo.c:52`.

## 2. Forward pass

Single-clause pseudocode, mirroring `TsetlinMachine.c:83-138` exactly (variable names preserved):

```
for clause i:
  clause_output[i] = 1
  for root j:
    interior_vote_sums[i][j] = 0
    for interior-alt k:
      interior_vote_products[i][j][k] = 1
      for interior-fac l:
        leaf_vote_sum[i][j][k][l] = 0
        for leaf-alt m:
          out = 1
          for n in 0..LEAF_FACTORS-1:
            feat = j*IF*LF + l*LF + n
            if action(ta[i,j,k,l,m,n])         == 1 and X[feat]            == 0: out=0; break
            if action(ta[i,j,k,l,m,n+LF])      == 1 and X[feat+FEATURES]   == 0: out=0; break
          clause_component_output[i,j,k,l,m] = out
          leaf_vote_sum[i,j,k,l] += out            # OR of leaf alternatives = SUM (line 124)
        interior_vote_products[i,j,k] *= leaf_vote_sum[i,j,k,l]   # AND of factors = PRODUCT (128)
      interior_vote_sums[i,j] += interior_vote_products[i,j,k]    # OR over k = SUM (132)
    clause_output[i] *= interior_vote_sums[i,j]   # AND over root = PRODUCT (136)
```

`action(state) = state > NUMBER_OF_STATES` (`TsetlinMachine.c:75-78`). Note "OR" is implemented as integer **sum** and "AND" as **product**: values are non-negative counts, not booleans, so `clause_output[i]` is an integer (a "vote multiplicity"), not 0/1. This is load-bearing for the class-sum.

**Class-sum with alternating signs** (`TsetlinMachine.c:209-221`):

```c
int class_sum = 0;
for (int i = 0; i < CLAUSES; i++) {
    int sign = 1 - 2 * (i & 1);            // even i → +1, odd i → −1
    class_sum += clause_output[i] * sign;
}
class_sum = (class_sum >  THRESHOLD) ?  THRESHOLD : class_sum;
class_sum = (class_sum < -THRESHOLD) ? -THRESHOLD : class_sum;
```

Clip range `[-T, +T]` with `T = THRESHOLD = 2000` (`TsetlinMachine.h:28`). Sign is structural (clause parity), not learned.

## 3. Feedback rules

For each clause, every leaf-position `(i,j,k,l,m)` gets a feedback value drawn ONCE per `tm_update` call (`TsetlinMachine.c:307-319`):

```c
int sign = 1 - 2 * (i & 1);
feedback_to_components[i][j][k][l][m] =
    sign * (2*target-1) *
    (1.0*rand()/RAND_MAX <= (1.0/(THRESHOLD*2))*(THRESHOLD + (1 - 2*target)*class_sum));
```

Per-clause Bernoulli probability:
- `target=1`: `p = (T − class_sum) / (2T)`
- `target=0`: `p = (T + class_sum) / (2T)`

Sign decides type: `>0 → Type I` (line 331), `<0 → Type II` (line 333), `==0 → skip`. Combined with alternating `(i&1)` sign and `(2·target−1)`, positive clauses on `target=1` receive Type I and negative clauses Type II (mirrored for `target=0`). This is the standard TM polarity, derived structurally with no stored polarity field.

**Path-conditional gating.** Type Ia (boost include on `X=1`) fires only when `clause_output[i] != 0 && interior_vote_products[i][j][k] != 0 && clause_component_output[i][j][k][l][m] != 0` (else-branch of `TsetlinMachine.c:235`). Every ancestor on the path root→leaf-alt must be "alive". The OR-level gate uses `interior_vote_products[i][j][k]` (the AND-chain inside *this* k), not the OR over all k, so each path is gated independently.

Type Ib (if-branch, `TsetlinMachine.c:236-240`) runs when any ancestor is dead:

```c
ta[...][n]    -= (state > 1) && (rand <= 1/s);
ta[...][n+LF] -= (state > 1) && (rand <= 1/s);
```

Both pos and neg literal TAs decay with prob `1/s`, clamped at state 1.

Type Ia (else-branch, `TsetlinMachine.c:242-256`), per leaf factor `n`:
- `X[feat+n] == 1`: increment pos-lit TA with prob `(s−1)/s` (or always if `BOOST_TRUE_POSITIVE_FEEDBACK==1`). Cap at `NUMBER_OF_STATES*2 = 200`.
- `X[feat+n] == 0`: decrement pos-lit TA with prob `1/s`.
- Same logic on neg-lit TA via `X[feat+n+FEATURES]`.

`BOOST_TRUE_POSITIVE_FEEDBACK = 0` (`TsetlinMachine.h:42`); production uses the probabilistic `(s−1)/s` path.

Type II (`TsetlinMachine.c:265-279`) fires only when `clause_output[i] > 0 && interior_vote_products[i][j][k] > 0 && clause_component_output[i][j][k][l][m] == 1`:

```c
if action==0 && state<200 && X[feat+n]==0:           state++   // pos lit
if action==0 && state<200 && X[feat+n+FEATURES]==0:  state++   // neg lit
```

Deterministic increment (no `1/s` gate); drives excluded literals on 0-valued features into include, suppressing the false-positive firing.

**Two RNG layers**: clause-level "does this path receive feedback" (line 314) drawn per `(i,j,k,l,m)`; then per-literal `1/s` or `(s−1)/s` Bernoullis (lines 237, 246, 248, 252, 254).

## 4. Multi-class wrapper

`mc_tm_update` (`MultiClassTsetlinMachine.c:140-151`):

```c
tm_update(mc_tm->tsetlin_machines[target_class], Xi, 1, s);
unsigned int negative_target_class = (unsigned int)CLASSES * 1.0*rand()/((unsigned int)RAND_MAX+1);
while (negative_target_class == target_class)
    negative_target_class = (unsigned int)CLASSES * 1.0*rand()/((unsigned int)RAND_MAX+1);
tm_update(mc_tm->tsetlin_machines[negative_target_class], Xi, 0, s);
```

Per example: **exactly two** per-class updates: true class with `target=1`, one uniformly-random non-true class with `target=0` (pairwise contrastive). The other 6 machines are untouched. No softmax, no per-class weights; each class is an independent `TsetlinMachine` with its own 5120 TAs (`MultiClassTsetlinMachine.h:33-35`). Inference: `argmax_c tm_score(c)` (`MultiClassTsetlinMachine.c:75-83`). Clause polarity is per-clause-id and identical across classes (`TsetlinMachine.c:213`).

## 5. Hyperparameters

**Compile-time / structural** (`TsetlinMachine.h:28-42`, must be known at allocator time):

| Macro | Default | Role |
|-------|---------|------|
| `CLAUSES` | 16 | clauses per class |
| `ROOT_FACTORS` | 2 | top-AND arity |
| `INTERIOR_ALTERNATIVES` | 2 | OR arity |
| `INTERIOR_FACTORS` | 2 | inner-AND arity |
| `LEAF_ALTERNATIVES` | 10 | OR arity at leaves |
| `LEAF_FACTORS` | 2 | literals-per-leaf / 2 |
| `NUMBER_OF_STATES` | 100 | TA decision threshold = `NUMBER_OF_STATES`; saturation at `2·NUMBER_OF_STATES=200` |
| `THRESHOLD` | 2000 | T in feedback prob and class-sum clip |
| `BOOST_TRUE_POSITIVE_FEEDBACK` | 0 | force deterministic Type Ia increment |
| `CLASSES` | 8 | `MultiClassTsetlinMachine.h:30` |

**Per-call / training-time** (function arg):
- `s` (float): feedback specificity; per-call to `tm_update`/`mc_tm_fit` (`TsetlinMachine.c:288`, `MultiClassTsetlinMachine.c:157`). Used: parity demo `s=32.1` epochs=500 (`NoisyParityDemo.c:80`), addition demo `s=2.1` epochs=100 (`additionDemo.c:167-168`).
- `target` (0/1): set by `mc_tm_update` per per-class call.

For a CUDA port: every structural macro fixes a kernel grid axis. The natural launch is `<<<dim3(CLASSES, CLAUSES), dim3(...)>>>`, with the innermost six-deep loop unrolled or tiled per warp. `s` and `target` are scalar kernel args.

## 6. Memory layout & RNG

**TA state** is `int` (`TsetlinMachine.h:48`), magnitude = automaton position, include action = `state > NUMBER_OF_STATES` (`TsetlinMachine.c:75-78`). Range **[1, 200]**: `state++` gated `state < 200` (lines 246, 252, 273, 276); `state--` gated `state > 1` (237, 248, 254). **No signed-int convention**: states ≤100 mean "exclude", >100 mean "include".

Init (`TsetlinMachine.c:59-65`): per leaf factor `n`, a coin flip puts the pos-lit TA at 100 or 101; the neg-lit TA gets the complement. Half the population starts "include" but exactly one of each (pos, neg) pair is on each side, never both.

**Randomness**: global `rand()` seeded once with `srand(time(NULL))` (`NoisyParityDemo.c:69`). Uniform draws via `1.0*rand()/RAND_MAX` (init + Bernoulli) and `(unsigned int)CLASSES * 1.0*rand()/((unsigned int)RAND_MAX+1)` for class selection (`MultiClassTsetlinMachine.c:145`). A CUDA port must replace this with a per-thread PRNG (Philox/curand); bit-exact equivalence is lost, only distributional equivalence is preserved.

Scratch tensors `leaf_vote_sum`, `interior_vote_products`, `interior_vote_sums`, `clause_output`, `clause_component_output`, `feedback_to_components` (`TsetlinMachine.h:49-58`) are per-machine forward/backward cache. In a batched CUDA port these must become per-`(class, example)` to avoid clobbering.

## 7. Invariants the port must preserve

1. **Type II only on firing leaves**: gated by `clause_output > 0 && interior_vote_products > 0 && clause_component_output == 1` (`TsetlinMachine.c:268`). Catches false positives only.
2. **Type Ia path-conditional**: same gate as Type II, applied as the else-branch of `TsetlinMachine.c:235`. If any ancestor is 0, only Type Ib decay runs. Without this, the hierarchy collapses to a flat clause.
3. **Per-leaf-alt independence**: feedback at `(i,j,k,l,m)` consults only `clause_component_output[i,j,k,l,m]` and `interior_vote_products[i,j,k]`. Sibling `m` leaves do not see each other's outputs except via the OR-sum into `clause_output[i]`.
4. **Pos and neg TAs are independent automata**: at indices `n` and `n+LEAF_FACTORS` (`TsetlinMachine.h:36`), updated by independent Bernoulli draws (`TsetlinMachine.c:111-117, 244-255, 271-276`).
5. **Clause sign is structural, not stored**: `sign = 1 - 2*(i&1)` computed inline at score-time (`TsetlinMachine.c:213`) and feedback-time (`TsetlinMachine.c:308`). Do not add a stored polarity field.
6. **Class-sum clip is *before* feedback prob**: clip at lines 217-218, feedback at 314 uses the clipped value. Don't reorder.
7. **State bounds [1, 200]**: never reach 0 or 201. Decrement gate `state > 1`, increment gate `state < 200`.
8. **`feedback_to_components` is drawn unconditionally for every `(i,j,k,l,m)`** (lines 307-319). Porting an "early exit" optimization changes the RNG consumption pattern and breaks reproducibility.
9. **Negative-target class sampled exactly once per example, with rejection** (`MultiClassTsetlinMachine.c:145-148`). Do not change to "all other classes" or "k random others"; that rescales the gradient.
10. **Caller pre-encodes negation**: `Xi` has shape `LITERALS = 2·FEATURES`, with `Xi[f+FEATURES] = 1 - Xi[f]`. The TM core never recomputes this; a port accepting `FEATURES`-wide input must enforce the bool-pair invariant before kernel launch.

---

Every constant and structural claim is cited to `vendors/HeirarchicalTM_experiments/`.
