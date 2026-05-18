"""HierarchicalTM, Tsetlin Machine with internal AND-OR tree clauses.

This is the TRUE Hierarchical Tsetlin Machine as specified by Granmo &
Saha (vendors/HeirarchicalTM_experiments/TsetlinMachine.{h,c}). It is
fundamentally different from "stacked TMs". One TM per class. Each
clause is internally a five-level AND-OR tree:

    clause_i = AND over j∈[0..R)         # ROOT_FACTORS root factors
       root_j = OR over k∈[0..A_int)     # INTERIOR_ALTERNATIVES
         alt_k = AND over l∈[0..F_int)   # INTERIOR_FACTORS
           grp_l = OR over m∈[0..A_leaf) # LEAF_ALTERNATIVES
             leaf_m = AND over LITERALS  # leaf clause component
                       (include/exclude
                        of LEAF_FACTORS features)

Each (j, l, n) triple deterministically addresses ONE input feature
via `feature_idx = j*F_int*LF + l*LF + n`, so features are partitioned
across the tree's leaves. Different `m` (leaf alternatives) and `k`
(interior alternatives) provide redundancy, multiple alternative
AND-rules can match at any position.

Inference (`forward`): bottom-up. Each leaf clause-component outputs
1 iff every included literal in its 2·LEAF_FACTORS automaton-pair
matches the input; otherwise 0. ORs are sums (votes), ANDs are
products. clause_output[c] is the AND-product of root-factor
vote-sums.

Class sum: Σ sign(c) · clause_output[c] with alternating sign over
clauses (Granmo's standard ± clause symmetry).

Feedback: Type I (toward target=1) or Type II (toward target=0)
selected stochastically per leaf based on (target − 2·sign·current).
The Type I path further branches into "Type Ia / recognise" when ALL
ancestors of the leaf evaluated to 1 (clause output, interior
product, clause component were all positive), or "Type Ib / forget"
otherwise. Type II only acts when the leaf is firing on a clause
output that should not be high.

This is the path-conditional feedback that makes HTM hierarchical ,
the feedback to a literal depends on whether its TREE PATH was active
in the forward pass, not just on the clause label.

The implementation is NumPy-only, CPU. Vectorisation is applied where
it doesn't sacrifice the C reference's exact semantics. Per-sample
training is the natural fit (online RL), the reference C code also
trains one sample at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from . import _hgtm_jit
    _HAVE_JIT = _hgtm_jit.NUMBA_AVAILABLE
except Exception:
    _HAVE_JIT = False
    _hgtm_jit = None


@dataclass
class HTMArchSpec:
    """Tree-clause architecture. Matches the C reference's macros.

    `n_features` must equal `root_factors * interior_factors * leaf_factors`.
    """
    n_features: int
    n_clauses: int = 16
    root_factors: int = 2
    interior_alternatives: int = 2
    interior_factors: int = 2
    leaf_alternatives: int = 10
    leaf_factors: int = 2
    n_states: int = 100             # TA states per side
    threshold: int = 200            # T (class-sum saturation)
    s: float = 3.9                  # specificity
    boost_true_positive_feedback: int = 0
    seed: int = 0

    def __post_init__(self):
        expected = self.root_factors * self.interior_factors * self.leaf_factors
        if self.n_features != expected:
            raise ValueError(
                f"n_features ({self.n_features}) must equal "
                f"root*interior_factors*leaf_factors "
                f"({self.root_factors}*{self.interior_factors}*"
                f"{self.leaf_factors} = {expected})"
            )


class HierarchicalTM:
    """Single-class binary TM with tree-structured clauses (target ∈ {0,1}).

    For multi-class, wrap K of these one-vs-rest and pick argmax of
    class-sums. See `HGTMMultiClass`.
    """

    def __init__(self, spec: HTMArchSpec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)
        # Seed Numba's intrinsic RNG so multi-run experiments are
        # reproducible. Numba's RNG is independent of NumPy's global
        # state, I must seed it via a JIT function.
        if _HAVE_JIT:
            _hgtm_jit.seed_rng(int(spec.seed))
        # TA states: shape [C, R, A_int, F_int, A_leaf, 2·LF]
        # Last axis: first LF are "include positive literal" automata,
        # remaining LF are "include negated literal" automata.
        shape = (spec.n_clauses, spec.root_factors,
                 spec.interior_alternatives, spec.interior_factors,
                 spec.leaf_alternatives, 2 * spec.leaf_factors)
        # Initialise on the "exclude" side of the state boundary
        # (state ≤ n_states = exclude action). Half above, half below,
        # randomly assigned per literal pair as in the C reference.
        s = spec.n_states
        ta_state = np.full(shape, s, dtype=np.int32)
        # For each pair (positive, negated) flip a coin: one becomes
        # NUMBER_OF_STATES (exclude side), the other NUMBER_OF_STATES+1
        # (include side). This matches `tm_initialize` in
        # vendors/HeirarchicalTM_experiments/TsetlinMachine.c.
        for c in range(spec.n_clauses):
            for j in range(spec.root_factors):
                for k in range(spec.interior_alternatives):
                    for l in range(spec.interior_factors):
                        for m in range(spec.leaf_alternatives):
                            for n in range(spec.leaf_factors):
                                if self.rng.random() <= 0.5:
                                    ta_state[c, j, k, l, m, n] = s
                                    ta_state[c, j, k, l, m,
                                              n + spec.leaf_factors] = s + 1
                                else:
                                    ta_state[c, j, k, l, m, n] = s + 1
                                    ta_state[c, j, k, l, m,
                                              n + spec.leaf_factors] = s
        self.ta_state = ta_state

        # Forward-pass buffers (reused; sized for one sample).
        self._clause_component_output = np.zeros(
            (spec.n_clauses, spec.root_factors,
             spec.interior_alternatives, spec.interior_factors,
             spec.leaf_alternatives), dtype=np.int32)
        self._leaf_vote_sum = np.zeros(
            (spec.n_clauses, spec.root_factors,
             spec.interior_alternatives, spec.interior_factors), dtype=np.int32)
        self._interior_vote_products = np.zeros(
            (spec.n_clauses, spec.root_factors,
             spec.interior_alternatives), dtype=np.int32)
        self._interior_vote_sums = np.zeros(
            (spec.n_clauses, spec.root_factors), dtype=np.int32)
        self._clause_output = np.zeros(spec.n_clauses, dtype=np.int32)

    # Inference
    def _action(self, state: int) -> int:
        return int(state > self.spec.n_states)

    def calculate_clause_output(self, X: np.ndarray) -> None:
        """Bottom-up tree evaluation. Populates internal buffers.

        X is a 1D array of length n_features (Boolean 0/1).
        Side effect: writes _clause_component_output, _leaf_vote_sum,
        _interior_vote_products, _interior_vote_sums, _clause_output.
        Uses the Numba-JIT fast path when available; otherwise falls
        back to the pure-NumPy mask-broadcast path. Both produce
        identical numerical output.
        """
        sp = self.spec
        ns = sp.n_states
        if _HAVE_JIT:
            X_arr = np.ascontiguousarray(X, dtype=np.int32)
            _hgtm_jit.forward_pass(
                self.ta_state, X_arr, ns,
                self._clause_component_output, self._leaf_vote_sum,
                self._interior_vote_products, self._interior_vote_sums,
                self._clause_output,
            )
            return
        # Action mask: action_include = (state > n_states)
        # Shape after thresholding: same as ta_state.
        action_include = (self.ta_state > ns)
        # X tile-indexed: for each leaf cell at (j, l, n) I need X[feat_idx].
        # Build a lookup of shape [R, F_int, LF] → feature index.
        feat_idx = (
            np.arange(sp.root_factors)[:, None, None] * sp.interior_factors * sp.leaf_factors
            + np.arange(sp.interior_factors)[None, :, None] * sp.leaf_factors
            + np.arange(sp.leaf_factors)[None, None, :]
        )                                            # shape [R, F_int, LF]
        # Build broadcasted X view: shape [R, F_int, LF]
        x_at_leaf = X[feat_idx].astype(bool)         # positive literal value
        x_neg_at_leaf = ~x_at_leaf                   # negated literal value

        # For each (c, j, k, l, m), leaf component output = AND over n of:
        #   (¬action_include_pos[n] OR x_at_leaf[j,l,n])
        # AND
        #   (¬action_include_neg[n] OR x_neg_at_leaf[j,l,n])
        # i.e. each included literal must match.
        ai_pos = action_include[..., :sp.leaf_factors]   # [C,R,A_int,F_int,A_leaf,LF]
        ai_neg = action_include[..., sp.leaf_factors:]   # [C,R,A_int,F_int,A_leaf,LF]
        # Broadcast x_at_leaf [R,F_int,LF] → [1,R,1,F_int,1,LF]
        xpos = x_at_leaf[None, :, None, :, None, :]
        xneg = x_neg_at_leaf[None, :, None, :, None, :]
        # Component output: AND over LF (axis -1) of (¬ai_pos | xpos) ∧ (¬ai_neg | xneg)
        pos_ok = (~ai_pos) | xpos
        neg_ok = (~ai_neg) | xneg
        component = (pos_ok & neg_ok).all(axis=-1)       # [C,R,A_int,F_int,A_leaf]
        self._clause_component_output[...] = component.astype(np.int32)

        # leaf_vote_sum[c, j, k, l] = sum over m of component[c,j,k,l,m]
        leaf_vote_sum = component.sum(axis=-1).astype(np.int32)  # [C,R,A_int,F_int]
        self._leaf_vote_sum[...] = leaf_vote_sum

        # interior_vote_products[c, j, k] = product over l of leaf_vote_sum
        interior_vote_products = leaf_vote_sum.prod(axis=-1).astype(np.int32)  # [C,R,A_int]
        self._interior_vote_products[...] = interior_vote_products

        # interior_vote_sums[c, j] = sum over k of interior_vote_products
        interior_vote_sums = interior_vote_products.sum(axis=-1).astype(np.int32)  # [C,R]
        self._interior_vote_sums[...] = interior_vote_sums

        # clause_output[c] = product over j of interior_vote_sums
        clause_output = interior_vote_sums.prod(axis=-1).astype(np.int32)         # [C]
        self._clause_output[...] = clause_output

    def sum_up_class_votes(self) -> int:
        """Signed sum with alternating ± and threshold clipping."""
        signs = np.where(np.arange(self.spec.n_clauses) % 2 == 0, 1, -1)
        s = int((self._clause_output * signs).sum())
        T = self.spec.threshold
        return max(-T, min(T, s))

    def score(self, X: np.ndarray) -> int:
        """Forward pass, returns clipped class sum."""
        self.calculate_clause_output(X)
        return self.sum_up_class_votes()

    # Interpretability
    def extract_clause_tree(self, X: Optional[np.ndarray] = None,
                              feature_names: Optional[list[str]] = None
                              ) -> list[dict]:
        """Walk every clause's AND-OR tree and emit it as a JSON-friendly
        nested dict, the human-readable explanation of what the HGTM
        learned. Faithful to the format of
        `vendors/HeirarchicalTM_experiments/parityclauses_1000.json`.

        Each clause is rendered as:
            {"type": "AND", "sign": ±1, "fired": bool, "children": [
                {"type": "OR", "fired": bool, "children": [
                    {"type": "AND", "fired": bool, "children": [
                        {"type": "OR", "fired": bool, "children": [
                            # leaf clause components, AND of literals
                            ["X3=1", "~X5=1"],
                            ["X3=1", "X5=0"], ...
                        ]}, ...
                    ]}, ...
                ]}, ...
            ]}

        If X is provided, `fired` flags are populated from the most
        recent forward pass over X (use `calculate_clause_output(X)`
        first or just pass X here). If X is None, only the LEARNED
        structure is emitted (no `fired` flags).

        `feature_names` (optional) labels each input bit. Defaults to
        `X0..Xn`.
        """
        sp = self.spec
        if X is not None:
            self.calculate_clause_output(X)
        names = feature_names or [f"X{i}" for i in range(sp.n_features)]
        ns = sp.n_states
        action = self.ta_state > ns                # bool array, "include" mask
        ap_pos = action[..., :sp.leaf_factors]     # [C,R,Aint,Fint,Aleaf,LF]
        ap_neg = action[..., sp.leaf_factors:]
        clauses: list[dict] = []
        for c in range(sp.n_clauses):
            root_children: list[dict] = []
            sign = +1 if (c % 2 == 0) else -1
            for j in range(sp.root_factors):
                interior_children: list[dict] = []
                for k in range(sp.interior_alternatives):
                    alt_children: list[dict] = []
                    for l in range(sp.interior_factors):
                        leaf_children: list[list[str]] = []
                        for m in range(sp.leaf_alternatives):
                            literals: list[str] = []
                            for n in range(sp.leaf_factors):
                                feat_i = (j * sp.interior_factors * sp.leaf_factors
                                          + l * sp.leaf_factors + n)
                                if ap_pos[c, j, k, l, m, n]:
                                    literals.append(f"{names[feat_i]}=1")
                                if ap_neg[c, j, k, l, m, n]:
                                    literals.append(f"~{names[feat_i]}=1")
                            if literals:
                                leaf_children.append(literals)
                        leaf_node = {"type": "OR", "children": leaf_children}
                        if X is not None:
                            leaf_node["fired"] = bool(
                                self._leaf_vote_sum[c, j, k, l] > 0)
                        alt_children.append(leaf_node)
                    interior_node = {"type": "AND", "children": alt_children}
                    if X is not None:
                        interior_node["fired"] = bool(
                            self._interior_vote_products[c, j, k] > 0)
                    interior_children.append(interior_node)
                root_node = {"type": "OR", "children": interior_children}
                if X is not None:
                    root_node["fired"] = bool(
                        self._interior_vote_sums[c, j] > 0)
                root_children.append(root_node)
            clause: dict = {
                "type": "AND", "sign": sign, "children": root_children,
            }
            if X is not None:
                clause["fired"] = bool(self._clause_output[c] > 0)
                clause["clause_output"] = int(self._clause_output[c])
            clauses.append(clause)
        return clauses

    def explain(self, X: np.ndarray,
                  feature_names: Optional[list[str]] = None,
                  top_k: int = 3) -> dict:
        """Return a high-level decision explanation for input X.

        Output:
            {
              "class_sum": int,
              "decision": +1 if class_sum > 0 else -1 if class_sum < 0 else 0,
              "active_clauses": [{clause_index, sign, output, contribution,
                                   tree}, ...] sorted by |contribution|,
            }

        Walks every fired clause, sorts by |sign × clause_output|, returns
        the top_k contributors plus the class-sum total. This is what a
        thesis chapter on "interpretable AGI through HGTM" would show
        as a worked example.
        """
        self.calculate_clause_output(X)
        class_sum = self.sum_up_class_votes()
        names = feature_names or [f"X{i}" for i in range(self.spec.n_features)]
        full_tree = self.extract_clause_tree(X=None, feature_names=names)
        # Score each clause by its signed contribution.
        contributors = []
        for c in range(self.spec.n_clauses):
            output = int(self._clause_output[c])
            if output == 0:
                continue
            sign = +1 if (c % 2 == 0) else -1
            contributors.append({
                "clause_index": c,
                "sign": sign,
                "output": output,
                "contribution": sign * output,
                "tree": full_tree[c],
            })
        contributors.sort(key=lambda d: -abs(d["contribution"]))
        return {
            "class_sum": class_sum,
            "decision": (+1 if class_sum > 0
                         else -1 if class_sum < 0 else 0),
            "active_clauses": contributors[:top_k],
            "n_active_clauses": len(contributors),
        }

    # Training
    def reseed(self, seed: int) -> None:
        """Re-seed both NumPy RNG and Numba's intrinsic RNG. Use this
        before `fit` to make a training run reproducible."""
        self.rng = np.random.default_rng(seed)
        if _HAVE_JIT:
            _hgtm_jit.seed_rng(int(seed))

    def update(self, X: np.ndarray, target: int) -> None:
        """Online update on one sample (target ∈ {0, 1})."""
        sp = self.spec
        if _HAVE_JIT:
            X_arr = np.ascontiguousarray(X, dtype=np.int32)
            _hgtm_jit.update_one(
                self.ta_state, X_arr, int(target), sp.n_states, sp.s,
                sp.threshold, sp.boost_true_positive_feedback,
                self._clause_component_output, self._leaf_vote_sum,
                self._interior_vote_products, self._interior_vote_sums,
                self._clause_output,
            )
            return
        # pure-NumPy fallback path
        self.calculate_clause_output(X)
        class_sum = self.sum_up_class_votes()
        T = sp.threshold
        signs = np.where(np.arange(sp.n_clauses) % 2 == 0, 1, -1)
        prob = (T + (1 - 2 * target) * class_sum) / (2.0 * T)
        prob = max(0.0, min(1.0, prob))
        rand_mask = (self.rng.random(
            (sp.n_clauses, sp.root_factors, sp.interior_alternatives,
             sp.interior_factors, sp.leaf_alternatives)) <= prob)
        per_clause_signed = (signs * (2 * target - 1))[
            :, None, None, None, None]
        feedback_to_components = (rand_mask.astype(np.int32)
                                   * per_clause_signed.astype(np.int32))
        self._apply_feedback(X, feedback_to_components)

    def _apply_feedback(self, X: np.ndarray,
                          feedback_to_components: np.ndarray) -> None:
        if _HAVE_JIT:
            X_arr = np.ascontiguousarray(X, dtype=np.int32)
            sp = self.spec
            _hgtm_jit.apply_feedback(
                self.ta_state, X_arr, feedback_to_components,
                self._clause_output, self._interior_vote_products,
                self._clause_component_output,
                sp.n_states, sp.s, sp.boost_true_positive_feedback,
            )
            return
        return self._apply_feedback_numpy(X, feedback_to_components)

    def _apply_feedback_numpy(self, X: np.ndarray,
                          feedback_to_components: np.ndarray) -> None:
        """Apply Type I / Type II feedback at each leaf clause-component.

        feedback_to_components shape: [C, R, A_int, F_int, A_leaf]
            > 0  → Type I  (push the leaf toward firing on this sample)
            < 0  → Type II (push the leaf away from firing on this sample)
            = 0  → no update

        Type I path-conditional rule (from the C reference,
        type_i_feedback): if clause_output==0 OR interior_vote_products==0
        OR clause_component_output==0  →  "Type Ib" (forget, stochastic
        decrement of TA states). Else  →  "Type Ia" (recognise, increment
        TA states for matching literals, decrement for non-matching).

        Type II rule (type_ii_feedback): if clause_output>0 AND
        interior_vote_products>0 AND clause_component_output==1 →
        push back (force include actions where Xi==0 to make the
        clause more discriminative).
        """
        sp = self.spec
        ns = sp.n_states
        s = sp.s
        rng = self.rng

        # Build the feature-index lookup [R, F_int, LF]
        feat_idx = (
            np.arange(sp.root_factors)[:, None, None] * sp.interior_factors * sp.leaf_factors
            + np.arange(sp.interior_factors)[None, :, None] * sp.leaf_factors
            + np.arange(sp.leaf_factors)[None, None, :]
        )

        # Broadcast helpers.
        clause_out = self._clause_output[:, None, None, None, None]            # [C,1,1,1,1]
        ivp = self._interior_vote_products[:, :, :, None, None]                # [C,R,A_int,1,1]
        cco = self._clause_component_output                                     # [C,R,A_int,F_int,A_leaf]

        type1_mask = feedback_to_components > 0
        type2_mask = feedback_to_components < 0

        # Path-conditional forget branch for Type I:
        # any ancestor = 0 ↔ clause_out==0 OR ivp==0 OR cco==0
        forget_branch = (clause_out == 0) | (ivp == 0) | (cco == 0)
        type1_forget = type1_mask & forget_branch
        type1_recognise = type1_mask & ~forget_branch

        # Apply per literal at each (c, j, k, l, m, n).
        # x_lit shape: [R, F_int, LF] -> broadcast to [C,R,A_int,F_int,A_leaf,LF]
        xpos = X[feat_idx].astype(np.int32)[None, :, None, :, None, :]  # [1,R,1,F_int,1,LF]
        xneg = (1 - xpos)

        # - Type I forget: stochastic decrement of EVERY automaton
        #    in the leaf (both positive and negated literals).
        if type1_forget.any():
            mask = type1_forget[..., None]                            # [C,R,A_int,F_int,A_leaf,1]
            mask_pos = np.broadcast_to(mask, self.ta_state[..., :sp.leaf_factors].shape)
            mask_neg = np.broadcast_to(mask, self.ta_state[..., sp.leaf_factors:].shape)
            rand_pos = rng.random(mask_pos.shape) <= 1.0 / s
            rand_neg = rng.random(mask_neg.shape) <= 1.0 / s
            # Decrement (clamped to >= 1) where mask & gt-1 & rand.
            ta_pos = self.ta_state[..., :sp.leaf_factors]
            ta_neg = self.ta_state[..., sp.leaf_factors:]
            ta_pos -= (mask_pos & (ta_pos > 1) & rand_pos).astype(np.int32)
            ta_neg -= (mask_neg & (ta_neg > 1) & rand_neg).astype(np.int32)

        # - Type I recognise: encourage matching literals, discourage
        #    non-matching ones (standard TM Type Ia).
        if type1_recognise.any():
            mask = type1_recognise[..., None]                         # [C,R,A_int,F_int,A_leaf,1]
            mask_b = np.broadcast_to(mask, self.ta_state[..., :sp.leaf_factors].shape)
            rand_high = rng.random(mask_b.shape) <= (s - 1) / s if sp.boost_true_positive_feedback == 0 else None
            rand_low = rng.random(mask_b.shape) <= 1.0 / s
            xpos_b = np.broadcast_to(xpos, mask_b.shape).astype(bool)
            ta_pos = self.ta_state[..., :sp.leaf_factors]
            # Where literal is 1 and ta below max: increment with prob (s-1)/s
            incr_pos = mask_b & xpos_b & (ta_pos < sp.n_states * 2)
            if sp.boost_true_positive_feedback == 1:
                ta_pos += incr_pos.astype(np.int32)
            else:
                ta_pos += (incr_pos & rand_high).astype(np.int32)
            # Where literal is 0 and ta above 1: decrement with prob 1/s
            decr_pos = mask_b & (~xpos_b) & (ta_pos > 1) & rand_low
            ta_pos -= decr_pos.astype(np.int32)

            # Same for negated literals
            xneg_b = np.broadcast_to(xneg, mask_b.shape).astype(bool)
            ta_neg = self.ta_state[..., sp.leaf_factors:]
            rand_high2 = rng.random(mask_b.shape) <= (s - 1) / s if sp.boost_true_positive_feedback == 0 else None
            rand_low2 = rng.random(mask_b.shape) <= 1.0 / s
            incr_neg = mask_b & xneg_b & (ta_neg < sp.n_states * 2)
            if sp.boost_true_positive_feedback == 1:
                ta_neg += incr_neg.astype(np.int32)
            else:
                ta_neg += (incr_neg & rand_high2).astype(np.int32)
            decr_neg = mask_b & (~xneg_b) & (ta_neg > 1) & rand_low2
            ta_neg -= decr_neg.astype(np.int32)

        # - Type II: when the leaf fires AND its ancestors fire,
        #    push exclude → include for literals whose X==0 (so future
        #    similar samples cause the leaf to drop, reducing false +).
        if type2_mask.any():
            # Eligibility: only leaves where ancestor path is true
            eligible = type2_mask & (clause_out > 0) & (ivp > 0) & (cco == 1)
            if eligible.any():
                mask = eligible[..., None]
                mask_b = np.broadcast_to(mask, self.ta_state[..., :sp.leaf_factors].shape)
                xpos_b = np.broadcast_to(xpos, mask_b.shape).astype(bool)
                xneg_b = np.broadcast_to(xneg, mask_b.shape).astype(bool)
                ta_pos = self.ta_state[..., :sp.leaf_factors]
                ta_neg = self.ta_state[..., sp.leaf_factors:]
                # For positive literals: state currently exclude (state ≤ ns)
                # AND X[feat]==0 → push toward include
                push_pos = mask_b & (ta_pos <= ns) & (~xpos_b) & (ta_pos < 2 * sp.n_states)
                ta_pos += push_pos.astype(np.int32)
                push_neg = mask_b & (ta_neg <= ns) & (~xneg_b) & (ta_neg < 2 * sp.n_states)
                ta_neg += push_neg.astype(np.int32)

    # Diagnostics
    @property
    def clause_output(self) -> np.ndarray:
        return self._clause_output.copy()

    def n_clauses(self) -> int:
        return self.spec.n_clauses


class HierarchicalTMMultiClass:
    """One-vs-rest wrapper: one HierarchicalTM per class.

    Negative-target sampling matches the C reference's `mc_tm_update`:
    train the target class with target=1, and a randomly chosen other
    class with target=0.
    """

    def __init__(self, n_classes: int, spec: HTMArchSpec):
        self.n_classes = n_classes
        self.spec = spec
        self.machines = [HierarchicalTM(
            HTMArchSpec(**{**spec.__dict__, "seed": spec.seed + i})
        ) for i in range(n_classes)]
        self.rng = np.random.default_rng(spec.seed)

    def reseed(self, seed: int) -> None:
        """Reseed every wrapped TM and the meta-class RNG."""
        for i, m in enumerate(self.machines):
            m.reseed(seed + i)
        self.rng = np.random.default_rng(seed)

    def update(self, X: np.ndarray, target: int) -> None:
        self.machines[target].update(X, 1)
        if self.n_classes > 1:
            neg = int(self.rng.integers(self.n_classes - 1))
            if neg >= target:
                neg += 1
            self.machines[neg].update(X, 0)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        # Re-seed Numba's RNG at the start of every fit so runs are
        # reproducible even if multiple HierarchicalTM instances share
        # the JIT module.
        self.reseed(self.spec.seed)
        n = len(X)
        for _ in range(epochs):
            order = self.rng.permutation(n)
            for i in order:
                self.update(X[i], int(y[i]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X), dtype=np.int64)
        for idx in range(len(X)):
            scores = np.array([m.score(X[idx]) for m in self.machines])
            out[idx] = int(np.argmax(scores))
        return out

    def class_scores(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), self.n_classes), dtype=np.int64)
        for idx in range(len(X)):
            for c, m in enumerate(self.machines):
                out[idx, c] = m.score(X[idx])
        return out
