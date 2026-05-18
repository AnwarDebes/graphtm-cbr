# Graph Counterfactual Recourse for HGTM-CBR

Goal: given an HGTM prediction over a binary vector that VSA-binds nodes/edges,
return the minimal *graph edit* (not bit flip) that flips the prediction,
subject to chemical validity. Sections 1-3 survey the field; section 4 fixes
the recommended algorithm.

---

## 1. Graph counterfactual methods (2021-2026)

Columns: edits | "minimal" objective | validity filter | metrics | dataset.

| Method | Edits | Minimal | Validity | Metrics | Datasets |
|--------|-------|---------|----------|---------|----------|
| **CF-GNNExplainer** (Lucic et al., AISTATS 2022, [arXiv:2102.03322](https://arxiv.org/abs/2102.03322)) | edge **deletion only**, via a learnable mask `P` on `A`; `A_v = P ⊙ A` | `L = L_pred(f(A_v)) + β · ‖A − A_v‖_1`, prefer `≤3` edges | none (graphs are abstract; no chem sanity) | accuracy of flip, fidelity, sparsity, mean explanation size | Tree-Cycles, Tree-Grid, BA-shapes |
| **MEG** (Numeroso & Bacciu, IJCNN 2021, [arXiv:2104.08060](https://arxiv.org/abs/2104.08060)) | RL agent acts on RDKit-valid action set: add atom, add bond, remove bond, change bond type | reward = `Δpred − λ · (1 − Tanimoto(s,s'))` | RDKit validity hard-coded in action mask (only legal RDKit ops emitted) | validity, similarity, prediction shift; QED gain | Tox21, ESOL |
| **MMACE** (Wellawatte, Seshadri, White, *Chem. Sci.* 2022, [DOI:10.1039/D1SC05259D](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05259d)) | SELFIES string mutations via STONED (insert / delete / replace tokens) | rank candidates by Tanimoto-ECFP4 to source, keep top-k that flip | SELFIES grammar guarantees 100 % parsable molecules; downstream RDKit sanitize | validity, Tanimoto sim, % flipped | BBBP, HIV, solubility, scent (Aditi et al. 2022) |
| **CLEAR** (Ma et al., NeurIPS 2022, [arXiv:2210.08443](https://arxiv.org/abs/2210.08443)) | VGAE samples whole new `(A', X')`; not edit-local | ELBO + flip-loss + causal consistency via auxiliary variable | none chemistry-specific (Community, ogbg, IMDB-M) | validity 0.85 on Community; proximity, causality | Community, Ogbg-Molhiv, IMDB-M |
| **GCFExplainer** (Huang, Kosan et al., WSDM 2023, [arXiv:2210.11695](https://arxiv.org/abs/2210.11695)) | vertex-reinforced random walk over the graph-edit-distance neighborhood (add/remove edge, add/remove node) | **global**: minimise size of CF set covering all input graphs at GED ≤ θ | none built-in | coverage, cost (GED), summary size ≤100 | Mutagenicity, NCI1, AIDS |
| **RLHEX** (Wang et al., KDD 2024, [arXiv:2406.13869](https://arxiv.org/abs/2406.13869)) | latent edits via VAE generator; PPO-trained adapter aligns latents to human-defined motif principles | PPO reward = flip + similarity + human-principle alignment | learnt validity head + RDKit sanitize at decode | coverage +4.12 %, dist −0.47 % | AIDS, Mutagenicity, Dipole |
| **MMGCF** (Cheng et al., *J. Comp. Comm.* 2025, [DOI:10.4236/jcc.2025.131011](https://doi.org/10.4236/jcc.2025.131011)) | **BRICS-motif-level** add/remove on a hierarchical motif tree; RGCN encoder | objective combines fidelity, sparsity (motif count), validity | BRICS retrosynthesis rules + extra 2 fragmentation rules ensure each piece is a real fragment | fidelity, sparsity, % valid | BBBP, BACE, ClinTox, Mutagenicity |
| **COMRECGC** (Fournier & Medya, ICML 2025, [arXiv:2505.07081](https://arxiv.org/abs/2505.07081)) | small *shared* set of edits that turn many rejected graphs into accepted ones (find-common-recourse, FCR) | min `|R|` s.t. `∀ g ∈ Reject, ∃ r ∈ R, f(g ⊕ r) = accept` | none built-in | coverage, recourse-set size | Mutagenicity, AIDS, NCI1, Proteins |
| **XPlore** (Beyond Edge Deletion, *arXiv* March 2026, [arXiv:2603.04209](https://arxiv.org/abs/2603.04209)) | edge add **and** delete, plus node-feature perturbations, gradient-guided through oracle GNN | unified L1 + flip loss, oracle-gradient guided | none built-in | validity +62.3 vs CF-GNNExpl on node-class | citation graphs |

Note on dates: only XPlore bears a 2026 stamp; others are pre-2026.

---

## 2. Boolean/tabular counterfactual baselines (for contrast)

| Method | Form | Minimal | Action constraints |
|--------|------|---------|---------------------|
| **Wachter et al. 2017** ([arXiv:1711.00399](https://arxiv.org/abs/1711.00399)) | `min_x' d(x,x') s.t. f(x')=y'`, single CF, gradient descent on input | weighted L1/L2 in feature space | none (purely geometric) |
| **DiCE** (Mothilal, Sharma, Tan, FAccT 2020, [arXiv:1905.07697](https://arxiv.org/abs/1905.07697)) | k diverse CFs via DPP-regularised loss = proximity + diversity + validity | combines L1 proximity with DPP diversity | per-feature `mutable=True/False` flag, ranges, categorical sets |
| **MACE / Karimi 2020** (AISTATS, [arXiv:1905.11190](https://arxiv.org/abs/1905.11190)) | encode `f` and `d` as logic formulas, hand to **SAT/SMT** solver | provably optimal under L0/L1/L∞ | logical constraints on integer / categorical features; model-agnostic |
| **AReS** (Rawal & Lakkaraju, NeurIPS 2020, [arXiv:2009.07165](https://arxiv.org/abs/2009.07165)) | **global** if-then summaries (two-level decision sets) of recourses across the population | jointly optimises correctness + interpretability + cost | actionable / immutable feature masks per subgroup |

Group recourse and actionable masks are subsumed by AReS (group) and
DiCE/MACE (instance).

---

## 3. Tsetlin-machine recourse

Targeted search ("Tsetlin machine" + "counterfactual" / "recourse" / "clause
flip") returns nothing. Adjacent but distinct:

- Clause-literal pruning ([arXiv:2301.08190](https://arxiv.org/abs/2301.08190)).
- Probabilistic-TM uncertainty ([arXiv:2410.17851](https://arxiv.org/abs/2410.17851)).
- Closed-form TM interpretation (Blakely & Granmo 2021).

None compute minimal input edits that flip the class-sum sign. The layer
below is therefore the first targeted at HGTM clauses.

---

## 4. Recommendation for HGTM-CBR: **HGTM-RECOURSE**

Single algorithm, four layers. Goal: minimal RDKit-valid graph edit that flips
the class-sum sign; runtime ≤100 ms on a single CUDA stream.

### 4.1 Operate on graph edits, not bits

The VSA binding `H(node) = bind(atom_sym, ...)` and `H(edge) = bind(src, dst,
bond)` (see `02_hgtm_canonical_spec.md` §1, `03_cair_cuda_arch.md` §1) defines
an **edit catalogue** `E`:

```
E = { remove_edge(i,j), add_edge(i,j,b),
      change_atom(i, sym -> sym'),
      remove_node(i), add_node(sym, attach_to=i, bond=b) }
```

Each `e ∈ E` deterministically maps to a delta `ΔX_e` on the binary
feature/literal vector (a small set of bit flips because each symbol's
hypervector touches only `hypervector_bits` of the chunks). The recourse
operates on `E`, never on raw bits; this keeps every candidate a real graph.

### 4.2 Clause-driven candidate generation (no BFS, no enumeration)

For input `x` predicted class `c`, with class-sum `S_c(x) > S_{c'}(x)`:

1. **Identify the supporting clauses**: positive-polarity clauses with
   `clause_output[i] > 0` for class `c`. Call this set `C+`.
2. **Identify the suppressing clauses**: positive-polarity clauses
   currently 0 for class `c'`. Call this set `C-`.
3. For each clause `c ∈ C+ ∪ C-`, run `attribute_to_edits(c)`: walk the
   AND/OR/AND/OR/AND tree from `02_hgtm_canonical_spec.md` §1; for every leaf
   with `action(ta)=1`, look up which graph element `(node, edge)` contributed
   that bit through the VSA binding (kept as an inverse index built once at
   encode time). This produces the **candidate edit set** `K(x)` of size
   `O(active_literals)`, typically <50 per molecule.

No graph enumeration; I only consider edits that touch a literal currently
deciding the prediction.

### 4.3 Greedy minimal-edit search with class-sum margin

Objective (Wachter-style, in graph-edit space):

```
min_{S ⊆ K(x)}   |S|
s.t.             sign(S_c(x ⊕ S) − S_{c'}(x ⊕ S))  flips
                  validity(x ⊕ S) = True
```

Algorithm (each step is a CUDA kernel call against the existing inference
path in `03_cair_cuda_arch.md` §3):

```
S ← ∅;  remaining ← K(x)
while not flipped:
    e* ← argmax_{e ∈ remaining}  Δmargin(e | S)        # one TM forward per e, batched
    if validity(x ⊕ S ⊕ {e*}):
        S ← S ∪ {e*}
    remaining ← remaining \ {e*}
return S
```

`Δmargin` is the change in `S_c − S_{c'}` after applying `e`; since literal
bindings are fixed, the per-edit forward is just the affected clauses, not the
whole machine. With `|K(x)|≈50` and `|C+ ∪ C-|≈50`, the inner loop is
~2500 clause-forwards, well under 100 ms on one GPU stream when batched as
one CUDA call per outer step (≤8 outer steps in practice for K=10 ensembles).

### 4.4 Chemistry-validity filter

After each candidate `e*`, before commit:

1. **RDKit sanitize**: `Chem.SanitizeMol(mol, catchErrors=True)`. Fails on
   valence / kekulization errors; reject.
2. **Lipinski Ro5** (Lipinski et al. 1997): MW ≤500, logP ≤5, HBD ≤5,
   HBA ≤10; computed by `rdkit.Chem.Descriptors`. Reject if any breach.
3. **Synthesizability**: SAscore (Ertl & Schuffenhauer, *J. Cheminform.* 2009,
   [DOI:10.1186/1758-2946-1-8](https://doi.org/10.1186/1758-2946-1-8)).
   Reject if `SA > 6`.
4. (Optional) BRICS check from MMGCF: every retained bond is BRICS-cleavable
   somewhere in the dataset; rules out hallucinated linkers.

All four are CPU-light (<2 ms/molecule via batched RDKit calls), so the GPU
search loop is the bottleneck.

### 4.5 Why this design over the alternatives

- **vs CF-GNNExplainer / XPlore**: those backprop a continuous mask; HGTM has
  no gradient (TA states are discrete). Clause-driven candidate generation
  is the natural analogue.
- **vs MEG / RLHEX**: RL needs a policy net; the clause tree *is* the policy.
- **vs MMACE**: SELFIES mutations ignore the model and miss the 100 ms target.
- **vs MMGCF**: BRICS motifs are too coarse for `change_atom`; I keep BRICS
  only as a validity *filter*.
- **vs CLEAR / generative**: those produce de-novo molecules, not minimal edits.
- **vs COMRECGC**: global-set recourse; useful as a *next* step.
- **vs DiCE / MACE / AReS**: tabular only; cannot express edge ops without
  the graph-edit catalogue.

### 4.6 Outputs (the user-facing contract)

For molecule `x` classified mutagenic, HGTM-RECOURSE returns:

```
edits = [
  remove_edge(atom_3, atom_7, "single"),
  change_atom(atom_3, "N" -> "C"),
]
validity = {rdkit: ok, lipinski: ok, sa: 3.4}
margin_before = +2.1   # class-sum for "mutagenic"
margin_after  = -0.4   # flipped
clauses_touched = [C+ #4, #11; C- #3]
latency_ms = 73
```

Matches the K=10 soft-class-sum / 100 % recourse contract in
`project_axiom_coi_unified.md`. Citations are inline as DOI / arXiv links.
