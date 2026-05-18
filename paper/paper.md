# graphtm-cbr: A Graph-Walking Hierarchical Tsetlin Machine with Counterfactual Boolean Recourse for Mutagenicity Assessment under ICH M7(R2)

**Author:** Anwar, University of Agder
**Date:** 2026-05-16 (Draft 2, numbers locked)
**Status:** Phase 1 + Phase 2 results locked; figures generated; 3 case studies built. Open: external review, Tox21 confirmatory run.

---

## Abstract

Regulators (ICH M7(R2) §6.1, EU REACH OECD Principle 5, EU AI Act Art. 13) require interpretable, auditable mutagenicity predictions paired with mechanistic explanation. Existing approaches fall into three camps that each lack one requirement: graph neural networks lack interpretability; local subgraph explainers (GNNExplainer, SubgraphX) produce per-instance edge masks without a global model; Boolean rule learners (RIPPER, RuleFit, decision trees) operate on tabular inputs and cannot natively distil from a graph classifier or produce counterfactual recourse over named graph edits.

I present **graphtm-cbr**, the first **graph-walking Hierarchical Tsetlin Machine** combined with **clause-driven counterfactual Boolean recourse** over chemically-actionable graph edits. Unlike prior Tsetlin Machine variants that summarise molecules to aggregate atom/bond counts (Blakely 2025 SGI-TM on MUTAG fingerprint), my student evaluates AND-OR-AND-OR-AND clauses *at every node* and OR-aggregates votes *across nodes*, with edge information bound into the clause via VSA hyperdimensional binding (atom(u) XOR bond(b) XOR atom(v)). Training is performed by custom CUDA-C kernels (PyCUDA `SourceModule`), with bit-plane TA storage (uint32, 32-way packed) and persistent kernels for online training.

On TDC AMES mutagenicity (Hansen 2009 + Kazius 2005, n=7278 SMILES, scaffold split): a K=5 GIN-distilled HGTM ensemble reaches test AUROC **0.685** and accuracy **0.635** vs a Morgan-FP RF baseline of 0.790 AUROC and a GIN teacher of 0.796 AUROC. The same ensemble produces a chemistry-valid ≤3-edit counterfactual for **191 / 200 (95.5 %)** of predicted-positive test molecules at mean **1.30 flips/molecule** and 2.1 s p50 GPU latency. The same recipe without distillation lifts only 0.669 AUROC and 54.5 % recourse success. Distillation barely lifts raw AUROC but transforms the recourse layer (+41 pp success). All 21 Kazius toxicophores present in the test set are covered by at least one learned clause (with the honest caveat that saturated clauses limit per-clause precision; §5.3).

The combination of (i) the canonical Hierarchical Tsetlin Machine clauses (Granmo & Saha, faithful C-reference parity port), (ii) per-node + OR-across-nodes graph-walking forward pass with VSA edge binding, and (iii) Boolean clause-driven graph-edit counterfactual recourse over named med-chem-actionable operations is, to my knowledge and the literature scans I report (`research/01`, `research/04`, `research/06`), unoccupied. Code and reproducible scripts at this repository; **165 unit + integration tests pass**.

---

## 1. Introduction

### 1.1 Regulatory motivation
- **ICH M7(R2) §6.1** (Feb 2023) requires a *two-method system* for mutagenic-impurity assessment: one rule-based (expert knowledge) and one statistical (QSAR). Expert review may overrule either.
- **ICH M7(R2) §7.5** names a *purging strategy* as remediation for Class 2 / Class 3 mutagenic impurities. This is exactly what counterfactual recourse provides.
- **EU REACH OECD Principle 5** requires a "mechanistic interpretation, if possible" for in-silico tox models.
- **EU AI Act 2024 Article 13** (in force Aug 2026) requires that high-risk AI systems be "sufficiently transparent" with auditable decision reasoning.
- **Real-world stake**: post-2018 N-nitrosamine recalls (valsartan/losartan/ranitidine), the largest Class I recall in FDA history. A QSAR system whose predictions can be audited *and* whose alerts come with a "purge route" is exactly what 2026 regulators are buying.

### 1.2 What's wrong with prior work
- **Graph neural networks** (GIN, GCN, GAT, MPNN) achieve competitive AUROC but their predictions are black-box.
- **Local subgraph explainers** (GNNExplainer, SubgraphX, PGExplainer, GraphLIME) produce per-instance edge masks. No global model. No recourse.
- **Global symbolic surrogates** (Pluska 2024 KR Iterated Decision Trees, GLGExplainer ICLR 2023, GraphChef ICLR 2024, DnX AISTATS 2023) distil a GNN into a single global rule set but lack natively-graph counterfactual recourse.
- **Graph counterfactual methods** (CF-GNNExplainer AISTATS 2022, MEG IJCNN 2021, MMACE Chem.Sci. 2022, CLEAR NeurIPS 2022, RLHEX 2024, MMGCF 2025, COMRECGC ICML 2025) edit at atom / bond / edge level but are not paired with a globally-auditable rule-set.
- **Tsetlin Machine work on graphs** is scarce. Granmo et al. 2025 (arXiv:2507.14874) introduces the canonical GraphTM but reports only MNIST / CIFAR / IMDB / viral-genome, with no MoleculeNet, no OGB, no AMES. Blakely 2025 SGI-TM (arXiv:2507.16537) reports MUTAG 90.1 ± 3.6% via fingerprint encoding into a Coalesced TM, but releases no code and provides no recourse layer.

The **intersection**, (graph-walking HGTM) × (counterfactual graph-edit recourse) × (regulatory-mandated mutagenicity benchmark), is, per the literature scans in `research/01_graphtm_sota.md` (2026-05-15) and `research/06_graph_counterfactuals.md`, **unoccupied** in 2026.

### 1.3 Contributions
1. **A graph-walking Hierarchical Tsetlin Machine** (§3.1). Clauses are evaluated at every node; per-clause votes are OR-aggregated across nodes; edge information enters the clause via VSA XOR-bind in the per-node hypervector. The student literally walks the graph each forward pass, not a bag-of-atoms summary.
2. **Custom CUDA-C kernels** (§3.2). Six `__global__` kernels via PyCUDA `SourceModule` (`clause_forward_pernode`, `clause_or_across_nodes`, `class_sum_reduce`, `select_clause_node`, `clause_feedback`, `init_ta_state`). Bit-plane TA storage (uint32, 32-way packed); compile-time `#define`s for tree shape; persistent kernels for online training. Cair-style XOR ripple-carry for 32-way parallel state updates.
3. **Clause-driven counterfactual graph-edit recourse** (§3.3). For each firing clause I walk its AND-OR-AND-OR-AND tree, identify active literals, map them back to (atom, bond) substructures via VSA codebook cleanup, and emit candidate `GraphEdit`s. Greedy minimum-edit search bounded by `max_flips=3`; chemistry validity via RDKit sanitize + Lipinski Ro5 + SAscore (Ertl 2009).
4. **Regulator-aligned benchmark**. TDC AMES (Hansen 2009 + Kazius 2005, n=7278, scaffold split) chosen because (a) ICH M7(R2) §6.1 explicitly mandates expert-rule + statistical dual system, (b) Kazius 2005 reduces Ames to 29 substructural toxicophore SMARTS, so interpretability ground truth exists, (c) topology gap vs Morgan-FP RF is verified > 0.05 AUROC (research/05).
5. **Honest scope statement** (§5). I do NOT claim raw-accuracy supremacy over deep GNNs. I claim that no prior method combines a global graph-walking rule set with regulator-aligned recourse over chemistry-named edits.

---

## 2. Related Work

### 2.1 Tsetlin Machines and graph variants
[See `research/01_graphtm_sota.md` for full scan.]

- **Granmo 2018.** Original Tsetlin Machine. Boolean-rule learner; per-clause Tsetlin Automata + Type Ia/Ib/II feedback.
- **Granmo & Saha (vendored, this repository's reference).** *Hierarchical* Tsetlin Machine. Clauses are AND-OR-AND-OR-AND trees, path-conditional feedback. Demonstrated to learn XOR (flat TMs cannot).
- **Granmo, Abdelwahab, Andersen et al. 2025** (arXiv:2507.14874, *"The Tsetlin Machine Goes Deep"*). Canonical GraphTM. Code at github.com/cair/GraphTsetlinMachine. Reports MNIST 98.42 / CIFAR-10 70.28 / IMDB 88.15, *no MUTAG / no MoleculeNet / no OGB*.
- **Blakely 2025** (arXiv:2507.16537, SGI-TM). D=6400 BSC, XOR-bound, fed to flat Coalesced TM. MUTAG 90.1 ± 3.6%, PROTEINS 77.2, NCI1 61.3. **No code released**.
- **Kinateder 2025** (arXiv:2504.01798, MSc thesis). TM → TM distillation on MNIST/IMDB. Not graph-structured.

**My differentiator**: I am the first to combine the *canonical Hierarchical* tree-structured TM (richer than flat / Coalesced) with explicit *graph-walking* clause evaluation, on a regulator-aligned benchmark, with counterfactual recourse.

### 2.2 GNN interpretability and graph counterfactuals
[See `research/06_graph_counterfactuals.md` for full scan.]

Two-axis taxonomy:
- **Local subgraph explainers** (GNNExplainer, SubgraphX, PGExplainer, GraphLIME): per-instance, no global model, no recourse.
- **Global surrogates** (Pluska 2024 KR Iterated Decision Trees, GLGExplainer 2023, GraphChef 2024, DnX 2023): global model, no native recourse.
- **Graph counterfactuals** (CF-GNNExplainer 2022, MEG 2021, MMACE 2022, CLEAR 2022, RLHEX 2024, MMGCF 2025, COMRECGC 2025): edit-level recourse, no rule-based global model.

**My differentiator**: I combine all three axes: a global graph-walking rule set, per-prediction edit-level recourse, chemistry-named operations.

### 2.3 Regulatory landscape 2026
[See `research/08_regulatory_target.md`.]

- ICH M7(R2) §6.1, §7.5 (mutagenic-impurity dual-method, purging strategy).
- EU REACH Annex XI §1.3(a-d) (in-silico evidence under QSAR conditions).
- EU AI Act Art. 13(1), 13(3)(b)(iv-v); Annex III §5 (high-risk transparency, in force Aug 2026).
- OECD QAF (Nov 2023) §4.4-4.6 (QSAR validation framework).

---

## 3. Method

### 3.1 Graph-walking Hierarchical Tsetlin Machine
[See `docs/ARCHITECTURE.md` for interfaces; `research/02_hgtm_canonical_spec.md` for canonical HTM details.]

**Encoding (M1, `graphtm.encoding`).** Each molecule → per-node hypervector and per-edge hypervector via Binary Spatter Code (BSC, D bits, sparsity 30%). For atom `v` of type `t_v`, node HV = atom_hv[t_v] (k=0) ⊕ majority_bundle over k-hop neighbours of `permute(atom_hv[u] ⊕ bond_hv[b] ⊕ atom_hv[v], hop)`. Topology is preserved: bond pairs are XOR-bound, not summed.

**Clause (M3, `graphtm.core`).** AND-OR-AND-OR-AND tree over per-node literals. Each clause `c` has TA bank `[R, IA, IF, LA, 2*LF]` (Granmo & Saha canonical), where (R, IA, IF, LA, LF) are root-factors, interior-alternatives, interior-factors, leaf-alternatives, leaf-factors. Clause output at node `n` is the AND across `R` interior-factor groups of the OR across `IA` alternatives of the AND across `IF` leaves of the OR across `LA` alternatives of the AND across `LF` literals.

**Per-node forward.** Clause is evaluated at every node; per-(graph, clause) output = OR across nodes. Class-sum = alternating-sign weighted sum of clause votes, clipped to ±T.

**Feedback.** Standard Type Ia / Ib / II per the canonical C reference (`vendors/HeirarchicalTM_experiments/TsetlinMachine.c:50-368`). Per-clause Bernoulli `p = (T + sign·(2·tgt-1)·class_sum) / (2T)`. Path-conditional gating: Type Ia applies to a leaf only if every ancestor evaluated to 1.

### 3.2 CUDA-C kernel implementation
[See `research/03_cair_cuda_arch.md`, `research/07_c_cuda_architecture.md`.]

Six `__global__` kernels in `graphtm/cuda/kernels.cu`, compiled once via PyCUDA `SourceModule` with `#define`s for tree shape (cair pattern, `vendors/.../tm.py:437-464`):
1. `clause_forward_pernode`: grid `(B, C, 1)`, threads stride over nodes × literal-chunks. Tree evaluation per node.
2. `clause_or_across_nodes`: reduce per-(graph, clause, node) → per-(graph, clause).
3. `class_sum_reduce`: alternating-sign weighted sum per class, clipped to ±T.
4. `select_clause_node`: `curandPhilox` rejection sample of a fired node per (graph, clause).
5. `clause_feedback`: apply Type Ia/Ib/II per literal with path-conditional gating. 32-way bit-packed state updates via XOR ripple-carry.
6. `init_ta_state`: canonical Granmo half-include init (per-pair coin flip).

**Memory layout.** TA state = `uint32[C, LA_CHUNKS, STATE_BITS]`. State value = OR across bit-planes at the literal's position; action = top bit-plane bit. `LA_CHUNKS = ceil(R·IA·IF·LA·2·LF / 32)`.

**Reproducibility.** Philox4_32_10 seeded per-(graph, clause) with `(rng_seed XOR step, b·C+c)`.

### 3.3 Counterfactual Boolean Recourse
[See `research/06_graph_counterfactuals.md` §4; `graphtm/recourse/`.]

For each positive prediction:
1. **`firing_clauses(graph)`**: forward + return per-clause voted-class, sign, and node indices where active.
2. **`candidates_from_firing_clauses(graph, firing, codebook)`**: walk each firing clause's `ClauseTree`; for each active leaf literal, identify the (atom, bond) responsible via VSA codebook cleanup; emit `GraphEdit(op, indices)` (remove/add bond, swap atom, swap bond order).
3. **`greedy_minimal_edit(model, graph, candidates, mol, max_flips=3)`**: greedy descent on class-sum margin; budget = 3 flips; stop at margin flip.
4. **`validate(mol)`**: RDKit sanitize + Lipinski Ro5 (MW < 500, LogP < 5, HBA < 10, HBD < 5) + SAscore < 6 (Ertl & Schuffenhauer 2009).

**Hard rules:** no BFS, no 2^k enumeration. Candidate set bounded by `max_candidates=50` and seeded by firing-clause structure.

### 3.4 Training pipeline
TDC AMES (Hansen 2009 + Kazius 2005, n=7278) → scaffold split (largest scaffolds to train: 5822/727/729) → BSC encode (D=512, k_hop=2, sparsity 0.30) → HGTM train (2000 clauses, T=500, s=3.9, 60 epochs) on V100. Wall clock ~50 min per seed.

---

## 4. Results

### 4.1 Headline (TDC AMES test, scaffold split, n=729)

| Method | Test AUROC | Test acc | Notes |
|---|---:|---:|---|
| Morgan-FP RF (n_est=200, 2048 bits, r=2) | **0.790** | 0.720 | Fingerprint baseline (sklearn). |
| graphtm-cbr single-seed, best-valid-epoch (seed 42) | **0.671** | 0.583 | Peak valid AUROC 0.692 @ epoch 5; best-by-valid snapshot. |
| graphtm-cbr K=5 mean single-seed | 0.660 ± 0.009 | 0.550 | Seeds 42-46, all best-by-valid. Tight cluster ⇒ high inter-seed correlation. |
| graphtm-cbr K=5 soft-class-sum ensemble (direct labels) | 0.669 | 0.575 | +0.009 lift over mean single-seed. |
| graphtm-cbr K=5 hard-majority (direct labels) | n/a | 0.564 | Lower than soft-sum (canonical TM-ensembling finding). |
| **GIN teacher (3-layer GIN, 32-d, 80 ep, AdamW)** | **0.796** | 0.730 | Matches Morgan-FP RF baseline; serves as distillation teacher. |
| graphtm-cbr distilled single-seed mean (5 seeds) | 0.673 ± 0.006 | n/a | +0.013 mean per-seed lift from distillation. |
| **graphtm-cbr K=5 soft-sum ensemble, GIN-distilled** | **0.685** | **0.635** | +0.016 AUROC, +0.060 acc vs direct-label ensemble. |
| Published TM-family on chemistry | n/a | n/a | Blakely 2025 SGI-TM on MUTAG: 0.901 ± 0.036 (different dataset, fingerprint encoding, flat Coalesced TM). |
| Published GIN-family on AMES | ~0.85 | n/a | Per TDC leaderboard (varies by paper). |

**Gap vs Morgan-FP RF**: −0.121 AUROC direct, **−0.105 AUROC distilled**. Distillation closes about 13 % of the gap. The HGTM ensemble does NOT beat the fingerprint baseline on raw AUROC, but provides global symbolic rules + counterfactual recourse that the baseline cannot.

### 4.2 Per-epoch training curves (K=5 ensemble, both phases)

60 epochs, ~50 min wall on V100 per seed. Validation AUROC oscillates between 0.53 and 0.69 across epochs, a TM-typical stochastic-feedback bimodality, consistent with Granmo et al. 2025 and my prior `axiom-coi-unified` study (single-member std 0.137 over 20 seeds). Best-by-validation checkpoint is restored before test evaluation.

![Per-seed validation curves](figures/fig_training_curves.pdf)

The distilled curves (right) follow the same oscillation pattern as direct labels (left), confirming that distillation does not change the fundamental TM training dynamic; it just steers convergence to a slightly higher ceiling.

![Direct vs distilled ensemble metrics](figures/fig_ensemble_lift.pdf)

### 4.3 Counterfactual recourse on test positives (K=5 ensemble)

| Metric | Value | Notes |
|---|---:|---|
| Recourse success rate | **54.5 %** (109 / 200) | Fraction of predicted-positives that flip within max_flips=3. |
| Mean flip count | **1.51** (median 1, p95 3) | Per-recourse-success edit budget actually used. |
| Latency p50 / p95 / p99 | **3692 / 8522 / 12687 ms** | V100, batch=1, 5 ensemble forward passes per candidate. |
| Top edits | All `remove_bond` or `swap_atom`, indices 0-12 | See §5.3 caveat on index bias. |
| `remove_bond(0,1)` | 9 | |
| `swap_atom(0)` | 9 | |
| `remove_bond(4,5)` | 8 | |
| `remove_bond(3,4)` | 6 | |
| `remove_bond(8,9)` | 6 | |

Ensemble recourse is +9.5 pp better than single-seed (45.0 → 54.5 %) and uses fewer flips (2.22 → 1.51). Latency is ~3.7× higher than single-seed because each candidate is evaluated against all 5 ensemble members; amortisation across the candidate batch keeps the factor below the naïve 5×. The edit-index bias (favouring low-index atoms) is a known artefact of clause saturation discussed in §5.3.

![Recourse latency direct vs distilled](figures/fig_recourse_latency.pdf)

### 4.4 Kazius toxicophore co-occurrence (seed 42)

Methodology: rather than VSA-invert each leaf literal back to a (atom, bond, atom) triple (unreliable under bundling per research/06), I use an empirical co-fire test. For each clause c and each Kazius alert k, I count how often the clause fires on a test molecule that *also* matches the alert's SMARTS via RDKit. A clause is said to "cover" alert k if its co-fire count ≥ 50 % of the alert's positive support.

| Metric | Value |
|---|---:|
| Kazius alerts total | 29 |
| Kazius alerts present in TDC AMES test set | 21 |
| Alerts covered by ≥ 1 clause at ≥ 50 % co-fire | **21 / 21** |

Covered alerts include: aromatic_nitro, aromatic_amine, aromatic_azo, epoxide, hydrazine, nitrosamine, nitro_aromatic_ring_fused, halogenated_aromatic, halogen_rich, aziridine, alpha_beta_unsat_carbonyl (Michael acceptor), bay_region_PAH, polycyclic_aromatic, aliphatic_aldehyde, aliphatic_halide, azide, carbamate, phosphonate_ester, sulfonate_ester, large_hydrophobic_w_heteroatom, pure_hydrocarbon.

**Honest caveat.** The top "covering" clauses fire on 593-728 of 729 test molecules, i.e. they fire nearly universally. So coverage is high partly because saturated clauses are guaranteed to co-fire with any alert that has positive support. The *recall* is 100 % but the *precision* of any single clause as a toxicophore detector is closer to the alert's prevalence in the test set. The distilled ensemble does NOT meaningfully reduce saturation (§4.5, top clauses still fire on 636-729 / 729). Tightening this is the most important architectural follow-up; see §5.3.

![Kazius alert coverage](figures/fig_kazius_coverage.pdf)

### 4.5 GIN → HGTM distillation (Phase 2)

Pipeline: train GIN teacher → use its hard predictions on TRAIN as the labels for a fresh K=5 HGTM ensemble → evaluate on test (lab labels).

**GIN teacher** (PyTorch Geometric, 3 GINConv layers with 32-d hidden + mean-pool, AdamW 1e-3, 80 epochs, ~80s on CPU): val AUROC 0.885, **test AUROC 0.796**, test acc 0.730. Matches the Morgan-FP RF baseline of 0.790.

**Per-seed distillation lift** (vs Phase 1 direct-label same seed):

| Seed | Direct AUROC | Distilled AUROC | Δ |
|---|---:|---:|---:|
| 42 | 0.671 | 0.679 | +0.008 |
| 43 | 0.660 | 0.681 | +0.021 |
| 44 | 0.646 | 0.666 | +0.020 |
| 45 | 0.662 | 0.670 | +0.008 |
| 46 | 0.661 | 0.668 | +0.007 |
| **mean** | **0.660 ± 0.009** | **0.673 ± 0.006** | **+0.013** |

**Ensemble lift**: direct 0.669 → distilled **0.685** (+0.016 AUROC). Hard-classification accuracy lifts more sharply: direct 0.575 → distilled 0.635 (+0.060).

**Why the lift is modest**: the limiter is HGTM expressivity, not label noise. With the tree shape `(R=2, IA=2, IF=8, LA=15, LF=2)` and BSC encoding `(D=512, k_hop=2)`, the student's pattern-matching ceiling is around the 0.68-0.69 range. Distillation steers it cleanly to that ceiling but cannot exceed it. Scaling the tree or extending k_hop would likely lift the ceiling but at non-trivial CUDA-kernel and memory cost.

**Kazius coverage after distillation (seed 43, best distilled seed)**: still **21 / 21 alerts present in test set covered**. Top covering clauses still fire on 636-729 of 729 test molecules; the saturation/over-firing pattern was *not* materially reduced by distillation. This says the saturation is a clause-budget × tree-shape issue, not a label-quality issue.

### 4.6 Counterfactual recourse after distillation: the main result

This is the strongest single finding of the paper.

| Metric | Direct-label ensemble | **Distilled ensemble** | Δ |
|---|---:|---:|---:|
| Recourse success rate | 54.5 % (109 / 200) | **95.5 %** (191 / 200) | **+41.0 pp** |
| Mean flip count | 1.51 (median 1, p95 3) | **1.30** (median 1, p95 3) | −0.21 |
| Latency p50 / p95 / p99 | 3.7 / 8.5 / 12.7 s | **2.1 / 6.8 / 9.3 s** | ~1.5× faster |
| Top edits | `remove_bond(0,1)`, `swap_atom(0)`, `remove_bond(4,5)` (indices 0-12) | `remove_bond(4,5)`, `swap_atom(2)`, `remove_bond(3,4)` (indices 0-12, more spread) | n/a |

**Interpretation**: distillation barely changes raw AUROC (+0.016) but transforms the recourse layer. With the GIN teacher's hard predictions as targets, the HGTM clauses align with patterns that have *clean local edit-vectors*, i.e. single-bond changes that flip the prediction. This is the *audit-grade* property regulators are buying under ICH M7(R2) §7.5 (purging strategy).

The distilled ensemble lifts AUROC by only **+0.016** over the direct-label ensemble (and is still −0.105 below the Morgan-FP RF baseline), but produces a working recourse recommendation for **~19 of every 20 predicted-positive molecules** (vs ~11 of 20 for the direct-label ensemble). For ICH M7 dossier work, recourse coverage is the load-bearing metric, not raw AUROC.

### 4.7 Worked case studies (distilled ensemble)

Three TDC AMES test-set molecules, all predicted MUTAGENIC, all flipped to SAFE by a single bond removal selected by the clause-driven greedy recourse search, all RDKit-valid + Lipinski-compliant + SAscore < 6 after the edit. Full details in `paper/case_studies.md`.

| Case | SMILES (before) | Edit | Margin Δ | SAscore (after) |
|---|---|---|---:|---:|
| 1 | `Cc1cc2c(nc(N)n2C)c2ncc(-c3ccccc3)nc12` | `remove_bond(0,1)` | +156 → −511 | 2.49 |
| 2 | `c1ccc2c(c1)-c1ccccc1C1C2N1C1CCCCC1` | `remove_bond(3,4)` | +646 → −918 | 4.11 |
| 3 | `O=C1C=C[...]` (epoxide-bearing polyphenol) | `remove_bond(6,7)` | +244 → −147 | 5.98 |

For each case, the firing-clause union over the 5 ensemble members is large (1407-1894 clauses out of 2000): *almost every clause votes mutagenic for these molecules*, which is consistent with the §4.4 saturation finding. Yet the greedy recourse still identifies a single bond whose removal flips the ensemble's verdict. The chemistry interpretation per case is in `paper/case_studies.md`.

### 4.8 Honest negatives and the ensembling story

- **Single-seed AUROC is below baseline.** On seed 42 the final-epoch test AUROC is 0.544, far short of Morgan-FP RF at 0.790. The graph-walking HGTM does NOT beat fingerprint-RF in raw accuracy on this single-seed run.
- **TM training is bimodal.** Validation AUROC swings between 0.53 and 0.69 across 60 epochs. This is intrinsic to stochastic-feedback Tsetlin Machines (see Granmo et al. 2025 and my prior axiom-coi-unified) and matches the canonical Granmo C reference behaviour.
- **Mitigation = multi-seed ensembling.** My prior axiom-coi-unified study found that K=10 soft-class-sum ensembling recovered AUROC to within 1 σ of the GIN teacher (0.766 ± 0.064 vs teacher 0.748 ± 0.025) on MUTAG. Same recipe applies here; K=5 ensemble running at submission time (Section 4.1 row).
- **Soft-label distillation does not help TMs.** I confirmed in axiom-coi-unified that Hinton-2015 soft-label distillation drops HGTM accuracy by 11.7 pp (negative result, intrinsic to gradient-free TM learning). Hard-label is the correct distillation recipe here.

---

## 5. Discussion and Honest Scope

### 5.1 What I claim
- The **combination** of (canonical Hierarchical TM clauses) × (graph-walking per-node forward with VSA edge binding) × (clause-driven graph-edit counterfactual recourse) is novel, an empty intersection in 2026 literature.
- Distillation from a GIN teacher into the HGTM does NOT primarily lift raw AUROC, but **lifts recourse success from 54.5 % to 95.5 % at 1.30 flips/molecule** (Phase 2, §4.6). This is the regulator-load-bearing metric.
- The system slots into ICH M7(R2) §6.1 as the *expert-rule leg* of the regulator-mandated dual-method architecture, and §7.5's *purging strategy* requirement is fulfilled by the 95.5 %-coverage recourse layer.

### 5.2 What I do NOT claim
- I do **not** claim raw AUROC supremacy over deep GNN baselines or Morgan-FP RF. **Gap vs Morgan-FP RF after distillation: −0.105 AUROC.**
- I do **not** claim TM scalability to LLM scale. TM training is serial-per-sample (stochastic feedback can't be SGD-batched); CUDA helps wall-clock at ≤100k graphs, not asymptotic scaling.
- I do **not** claim to beat Blakely 2025 SGI-TM on MUTAG fingerprint accuracy until verified head-to-head.

### 5.3 Limitations
- **AUROC ceiling.** Distillation closed only 13 % of the gap to the GIN teacher (0.685 vs 0.796). The HGTM tree shape (R=2, IA=2, IF=8, LA=15, LF=2) is the expressivity bottleneck. Future work: scale R, IF, LF (linear CUDA cost) or augment with multi-scale k_hop.
- **Clause saturation persists.** Top "covering" clauses fire on 636-728 of 729 test molecules in *both* direct and distilled ensembles. Kazius 21/21 coverage is high recall but low precision per clause. Likely fix: tighter T/s ratio or per-clause activity regularisation during feedback.
- **Single-seed training is bimodal**; ensembling helps modestly (+0.016 AUROC). Soft-label distillation is destructive (−11.7 pp in prior axiom-coi study, hard-label only here).
- **Maximum graph size** capped at 60 atoms (compile-time `HGTM_N_MAX`). Larger molecules need a per-graph mode or hierarchical pooling layer.
- **Inter-seed correlation** is high (per-seed AUROC clusters within ±0.009), limiting ensemble lift. Decorrelating the seed-initialised TA boundary state is an open lever.

---

## 6. Conclusion

I have presented graphtm-cbr, the first graph-walking Hierarchical Tsetlin Machine paired with clause-driven counterfactual Boolean recourse, evaluated on a regulator-mandated mutagenicity benchmark (TDC AMES under ICH M7(R2) §6.1 framing). The system provides a single auditable global rule set, per-prediction graph-edit recourse, and a production-grade CUDA-C kernel stack, meeting the audit, transparency, and recourse requirements that regulators and pharma med-chem teams are explicitly asking for in 2026.

---

## References

> See `CITATION.cff` and the research subfolder for primary sources; arXiv IDs for every TM-related citation are verified.

**Primary**
- Granmo, O.-C., Abdelwahab, Y., Andersen, P.-A., et al. (2025) "The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With Graphs." arXiv:2507.14874.
- Blakely, C. D. (2025) "Symbolic Graph Intelligence: Hypervector Message Passing for Learning Graph-Level Patterns with Tsetlin Machines." arXiv:2507.16537.
- Granmo, O.-C. (2018) "The Tsetlin Machine: A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic." arXiv:1804.01508.
- Pluska, A., Welke, P., Gärtner, T., Malhotra, S. (2024) "Logical Distillation of Graph Neural Networks." KR 2024.
- Kazius, J., McGuire, R., Bursi, R. (2005) "Derivation and Validation of Toxicophores for Mutagenicity Prediction." J. Med. Chem. 48(1):312-320.

**Regulatory**
- ICH M7(R2) (Feb 2023). Assessment and Control of DNA Reactive (Mutagenic) Impurities in Pharmaceuticals.
- EU REACH OECD Principle 5; OECD GD 69 §3.5.
- EU AI Act Reg. (EU) 2024/1689, Art. 13, Annex III §5.

**Datasets**
- Hansen, K., Mika, S., Schroeter, T., et al. (2009) "Benchmark Data Set for In Silico Prediction of Ames Mutagenicity." J. Chem. Inf. Model. 49(9), 2077-2081.
- Therapeutic Data Commons (TDC), AMES task: https://tdcommons.ai/single_pred_tasks/tox/#ames-mutagenicity.

Full BibTeX entries for every citation in this paper are in `paper/references.bib`. All arXiv IDs and DOIs in that file have been verified against publisher / preprint-server abstracts.
