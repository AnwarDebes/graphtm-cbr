"""Numba-JIT hot paths for `tsetlin.hierarchical_tm`.

These are direct ports of `calculate_clause_output` and `_apply_feedback`
from explicit nested loops, matching the C reference at
`vendors/HeirarchicalTM_experiments/TsetlinMachine.c` line-for-line.

The Python module imports these and uses them when `numba` is
available; if numba is missing, the pure-NumPy fallback in the parent
module is used. Architecture is identical; only the inner loops are
compiled.

Speedup vs the NumPy mask-broadcast version: 30-80× typical on
MNIST-sized configs.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:                                       # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):                          # pragma: no cover
        def deco(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return deco


@njit(cache=True)
def seed_rng(seed: int) -> None:
    """Seed Numba's intrinsic RNG. Must be called from a JIT function;
    `np.random.seed()` from regular Python does NOT affect Numba's RNG.
    Call this once per `HierarchicalTM.fit` to make multi-seed runs
    reproducible."""
    np.random.seed(seed)


@njit(cache=True)
def update_one(ta_state, X, target, ns, s, threshold, boost_tpf,
                 clause_component_output, leaf_vote_sum,
                 interior_vote_products, interior_vote_sums,
                 clause_output):
    """One end-to-end sample update: forward + class-sum + per-leaf
    feedback decision + apply feedback. Pure JIT, zero Python overhead.

    Returns the clipped class sum.
    """
    n_clauses = ta_state.shape[0]
    R = ta_state.shape[1]
    Aint = ta_state.shape[2]
    Fint = ta_state.shape[3]
    Aleaf = ta_state.shape[4]
    LF2 = ta_state.shape[5]
    LF = LF2 // 2
    # Forward pass (inlined)
    for c in range(n_clauses):
        clause_output[c] = 1
        for j in range(R):
            interior_vote_sums[c, j] = 0
            for k in range(Aint):
                interior_vote_products[c, j, k] = 1
                for l in range(Fint):
                    leaf_vote_sum[c, j, k, l] = 0
                    for m in range(Aleaf):
                        out = 1
                        for n_idx in range(LF):
                            feat = j * Fint * LF + l * LF + n_idx
                            x_pos = X[feat]
                            if ta_state[c, j, k, l, m, n_idx] > ns:
                                if x_pos == 0:
                                    out = 0
                                    break
                            if ta_state[c, j, k, l, m, n_idx + LF] > ns:
                                if x_pos == 1:
                                    out = 0
                                    break
                        clause_component_output[c, j, k, l, m] = out
                        leaf_vote_sum[c, j, k, l] += out
                    interior_vote_products[c, j, k] *= leaf_vote_sum[c, j, k, l]
                interior_vote_sums[c, j] += interior_vote_products[c, j, k]
            clause_output[c] *= interior_vote_sums[c, j]
    # Class sum with alternating ±
    class_sum = 0
    for c in range(n_clauses):
        sign_c = 1 if (c & 1) == 0 else -1
        class_sum += sign_c * clause_output[c]
    if class_sum > threshold:
        class_sum = threshold
    elif class_sum < -threshold:
        class_sum = -threshold
    # Feedback probability
    prob = (threshold + (1 - 2 * target) * class_sum) / (2.0 * threshold)
    if prob < 0.0:
        prob = 0.0
    elif prob > 1.0:
        prob = 1.0
    p_rec = (s - 1) / s
    p_for = 1.0 / s
    state_max = 2 * ns
    # Per-leaf feedback decision + application
    for c in range(n_clauses):
        sign_c = 1 if (c & 1) == 0 else -1
        co = clause_output[c]
        for j in range(R):
            for k in range(Aint):
                ivp = interior_vote_products[c, j, k]
                for l in range(Fint):
                    feature_base = j * Fint * LF + l * LF
                    for m in range(Aleaf):
                        # Sample whether this leaf gets feedback.
                        if np.random.random() > prob:
                            continue
                        fb = sign_c * (2 * target - 1)
                        cco = clause_component_output[c, j, k, l, m]
                        if fb > 0:
                            if co == 0 or ivp == 0 or cco == 0:
                                # Type Ib: forget
                                for n_idx in range(LF):
                                    if (ta_state[c, j, k, l, m, n_idx] > 1
                                            and np.random.random() <= p_for):
                                        ta_state[c, j, k, l, m, n_idx] -= 1
                                    nn = n_idx + LF
                                    if (ta_state[c, j, k, l, m, nn] > 1
                                            and np.random.random() <= p_for):
                                        ta_state[c, j, k, l, m, nn] -= 1
                            else:
                                # Type Ia: recognise
                                for n_idx in range(LF):
                                    feat = feature_base + n_idx
                                    x_pos = X[feat]
                                    if x_pos == 1:
                                        if (ta_state[c, j, k, l, m, n_idx] < state_max
                                                and (boost_tpf == 1
                                                     or np.random.random() <= p_rec)):
                                            ta_state[c, j, k, l, m, n_idx] += 1
                                    else:
                                        if (ta_state[c, j, k, l, m, n_idx] > 1
                                                and np.random.random() <= p_for):
                                            ta_state[c, j, k, l, m, n_idx] -= 1
                                    nn = n_idx + LF
                                    if x_pos == 0:
                                        if (ta_state[c, j, k, l, m, nn] < state_max
                                                and (boost_tpf == 1
                                                     or np.random.random() <= p_rec)):
                                            ta_state[c, j, k, l, m, nn] += 1
                                    else:
                                        if (ta_state[c, j, k, l, m, nn] > 1
                                                and np.random.random() <= p_for):
                                            ta_state[c, j, k, l, m, nn] -= 1
                        else:
                            # Type II
                            if co > 0 and ivp > 0 and cco == 1:
                                for n_idx in range(LF):
                                    feat = feature_base + n_idx
                                    x_pos = X[feat]
                                    state = ta_state[c, j, k, l, m, n_idx]
                                    if (state <= ns and x_pos == 0
                                            and state < state_max):
                                        ta_state[c, j, k, l, m, n_idx] = state + 1
                                    nn = n_idx + LF
                                    state = ta_state[c, j, k, l, m, nn]
                                    if (state <= ns and x_pos == 1
                                            and state < state_max):
                                        ta_state[c, j, k, l, m, nn] = state + 1
    return class_sum


@njit(cache=True)
def forward_pass(ta_state, X, ns,
                  clause_component_output, leaf_vote_sum,
                  interior_vote_products, interior_vote_sums,
                  clause_output):
    """Bottom-up tree evaluation, modifies the buffers in place.

    Faithful port of `calculate_clause_output` in TsetlinMachine.c:83-139.
    """
    n_clauses = ta_state.shape[0]
    R = ta_state.shape[1]
    Aint = ta_state.shape[2]
    Fint = ta_state.shape[3]
    Aleaf = ta_state.shape[4]
    LF2 = ta_state.shape[5]
    LF = LF2 // 2
    for c in range(n_clauses):
        clause_output[c] = 1
        for j in range(R):
            interior_vote_sums[c, j] = 0
            for k in range(Aint):
                interior_vote_products[c, j, k] = 1
                for l in range(Fint):
                    leaf_vote_sum[c, j, k, l] = 0
                    for m in range(Aleaf):
                        out = 1
                        for n_idx in range(LF):
                            feat = j * Fint * LF + l * LF + n_idx
                            x_pos = X[feat]
                            # positive literal automaton at index n_idx
                            if ta_state[c, j, k, l, m, n_idx] > ns:
                                if x_pos == 0:
                                    out = 0
                                    break
                            # negated-literal automaton at n_idx + LF
                            if ta_state[c, j, k, l, m, n_idx + LF] > ns:
                                if x_pos == 1:
                                    out = 0
                                    break
                        clause_component_output[c, j, k, l, m] = out
                        leaf_vote_sum[c, j, k, l] += out
                    interior_vote_products[c, j, k] *= leaf_vote_sum[c, j, k, l]
                interior_vote_sums[c, j] += interior_vote_products[c, j, k]
            clause_output[c] *= interior_vote_sums[c, j]


@njit(cache=True)
def apply_feedback(ta_state, X, feedback_to_components,
                     clause_output, interior_vote_products,
                     clause_component_output,
                     ns, s, boost_tpf):
    """Apply Type Ia / Ib / II feedback per the C reference.

    Randoms are drawn ONLY where the feedback path actually needs
    them, via numba's intrinsic `np.random.random()`. Eliminates the
    per-update `np.random.random(ta_state.shape)` allocation that
    dominated the NumPy baseline.

    Faithful port of `type_i_feedback` (TsetlinMachine.c:233-258) and
    `type_ii_feedback` (TsetlinMachine.c:265-279).
    """
    n_clauses = ta_state.shape[0]
    R = ta_state.shape[1]
    Aint = ta_state.shape[2]
    Fint = ta_state.shape[3]
    Aleaf = ta_state.shape[4]
    LF2 = ta_state.shape[5]
    LF = LF2 // 2
    p_rec = (s - 1) / s
    p_for = 1.0 / s
    state_max = 2 * ns
    for c in range(n_clauses):
        co = clause_output[c]
        for j in range(R):
            for k in range(Aint):
                ivp = interior_vote_products[c, j, k]
                for l in range(Fint):
                    feature_base = j * Fint * LF + l * LF
                    for m in range(Aleaf):
                        cco = clause_component_output[c, j, k, l, m]
                        fb = feedback_to_components[c, j, k, l, m]
                        if fb == 0:
                            continue
                        if fb > 0:
                            if co == 0 or ivp == 0 or cco == 0:
                                # Type Ib: forget
                                for n_idx in range(LF):
                                    if (ta_state[c, j, k, l, m, n_idx] > 1
                                            and np.random.random() <= p_for):
                                        ta_state[c, j, k, l, m, n_idx] -= 1
                                    nn = n_idx + LF
                                    if (ta_state[c, j, k, l, m, nn] > 1
                                            and np.random.random() <= p_for):
                                        ta_state[c, j, k, l, m, nn] -= 1
                            else:
                                # Type Ia: recognise
                                for n_idx in range(LF):
                                    feat = feature_base + n_idx
                                    x_pos = X[feat]
                                    if x_pos == 1:
                                        if (ta_state[c, j, k, l, m, n_idx] < state_max
                                                and (boost_tpf == 1
                                                     or np.random.random() <= p_rec)):
                                            ta_state[c, j, k, l, m, n_idx] += 1
                                    else:
                                        if (ta_state[c, j, k, l, m, n_idx] > 1
                                                and np.random.random() <= p_for):
                                            ta_state[c, j, k, l, m, n_idx] -= 1
                                    nn = n_idx + LF
                                    if x_pos == 0:
                                        if (ta_state[c, j, k, l, m, nn] < state_max
                                                and (boost_tpf == 1
                                                     or np.random.random() <= p_rec)):
                                            ta_state[c, j, k, l, m, nn] += 1
                                    else:
                                        if (ta_state[c, j, k, l, m, nn] > 1
                                                and np.random.random() <= p_for):
                                            ta_state[c, j, k, l, m, nn] -= 1
                        else:
                            # Type II
                            if co > 0 and ivp > 0 and cco == 1:
                                for n_idx in range(LF):
                                    feat = feature_base + n_idx
                                    x_pos = X[feat]
                                    state = ta_state[c, j, k, l, m, n_idx]
                                    if (state <= ns and x_pos == 0
                                            and state < state_max):
                                        ta_state[c, j, k, l, m, n_idx] = state + 1
                                    nn = n_idx + LF
                                    state = ta_state[c, j, k, l, m, nn]
                                    if (state <= ns and x_pos == 1
                                            and state < state_max):
                                        ta_state[c, j, k, l, m, nn] = state + 1
