# Graph-Native Tsetlin Machine SOTA Scan (2022-2026)

Scope: Tsetlin Machine (TM) papers that consume input graphs *with topology* (adjacency / edges / message passing). Bag-of-atoms or flat-feature TM variants are excluded.

## Verified papers

### 1. Granmo et al. 2025 - "The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With Graphs" (GraphTM)
- Citation: O.-C. Granmo, M. Shende, P.-A. Andersen, R. Saha, et al. 2025. arXiv:2507.14874. https://arxiv.org/abs/2507.14874
- Encoding: directed/labeled multigraph -> sparse Boolean hypervectors. Bundle operator (sum, thresholded) for node properties; bind operator (component product / XOR-style) for messages bound to edge types. Symbol set S = P (properties) cup M (messages) cup T (edge types).
- Message passing: depth parameter D (1 or 2 used in experiments). Each layer i propagates messages along incoming edges, binds with edge-type hypervector, then bundles into the receiving node's hypervector. TM clauses are evaluated over the resulting per-node Boolean literals -> "deep clauses" nesting clauses across hops.
- Benchmarks (mean +/- std over runs; original splits where available):
  - MNIST (k-NN spatial graph): 98.42 +/- 0.05
  - Fashion-MNIST: 87.07 +/- 0.08
  - CIFAR-10 (patch graph): 70.28 +/- 0.17 (+3.86 pp over CoTM)
  - MNIST Superpixel: 89.24 +/- 1.34, 60k clauses
  - IMDB sentiment (token co-occurrence graph): 88.15 +/- 2.16 (D=2)
  - Yelp: 85.24 +/- 1.45; MPQA: 81.77 +/- 1.15
  - Action coreference (5 utterances): 57.92 +/- 1.02
  - Recommendation (noise=0.1): 89.86 (vs GCN 70.87)
  - Viral genome 5-class: 95.14 (D=2; ~2.5x faster training than GCN)
  - No TUDataset / OGB / ZINC / QM9 results reported.
- Code/CUDA: yes. https://github.com/cair/GraphTsetlinMachine (PyPI GraphTsetlinMachine 0.3.3, April 2025). CUDA kernels included; tuned for DGX-2, A100, H100. Example scripts in repo: NoisyXORDemo, Vanilla MNIST, Convolutional MNIST, Sequence Classification, Noisy XOR MNIST. NO molecular benchmark scripts ship with the library.
- Differentiator: First TM that performs explicit message passing across labeled multigraphs and nests clauses across hops via vector-symbolic binding/bundling.

### 2. Blakely 2025 - "Symbolic Graph Intelligence: Hypervector Message Passing for Learning Graph-Level Patterns with Tsetlin Machines" (SGI-TM)
- Citation: Christian D. Blakely. 2025. arXiv:2507.16537. ISTM'25 submission. https://arxiv.org/abs/2507.16537
- Encoding: 6400-bit sparse binary hypervectors (stored as 100 x uint64), ~20% density (~1280 active bits). Binding = element-wise XOR; bundling = top-K majority preserving K active bits.
- Message passing: 2 layers core, optional 3rd. Layer 1 binds node categorical label + linear attributes + interval-encoded importance. Layer 2 binds [source node HV] tensor [edge metadata] tensor [target node HV] across all edges and bundles into a single graph-level hypervector. Optional layer 3 propagates to second-hop neighbors.
- Benchmarks (mean +/- std, 10 runs, TUDataset):
  - MUTAG: 90.1 +/- 3.6
  - PROTEINS: 77.2 +/- 1.8
  - NCI1: 61.3 +/- 2.0
  - AIDS: 71.2 +/- 3.4
  - DHFR: 79.0 +/- 4.0; DHFR_MD: 89.0 +/- 2.2
  - ER_MD: 84.0 +/- 5.1
  - ENZYMES, OGBG-MolHIV, ZINC, QM9: not reported.
- Code: NOT released as of paper PDF (no URL in manuscript). No CUDA path.
- Differentiator (paper's own words): "the Graph TM assumes a fixed graph and learns over evolving node/edge states, whereas our method encodes and learns over fully dynamic graph instances" - i.e. inductive graph-level classification across varying-structure inputs.

### 3. Halenka et al. 2024 - "Exploring Effects of Hyperdimensional Vectors for Tsetlin Machines"
- Citation: V. Halenka, A. K. Kadhim, P. F. A. Clarke, B. Bhattarai, R. Saha, O.-C. Granmo, L. Jiao, P.-A. Andersen. 2024. arXiv:2406.02648. https://arxiv.org/abs/2406.02648
- Status: PRECURSOR, not a graph TM. Introduces HV-based Booleanization for arbitrary concepts (images / text / chemical compounds); no adjacency, no message passing. Included because GraphTM (paper 1) directly inherits its HV machinery. No TUDataset.

## Confirmed NOT graph-with-topology (rejected from this list)

- Kinateder 2025, arXiv:2504.01798 - knowledge distillation for TM. Image + text only, no graph topology. Reject.
- Hnilov 2025 "Fuzzy-Pattern Tsetlin Machine", arXiv:2508.08350 - clause-level fuzzy logic. Only references GraphTM as a *baseline number* on Amazon Sales (78.17 vs FPTM 85.22, 20% noise). Not itself a graph model. Reject.
- Saha & Granmo 2021 "Relational Tsetlin Machine", arXiv:2102.10952 - first-order logic over Herbrand terms for NLU. Operates over relations as logical predicates, not over graphs with adjacency/message-passing. Reject as graph-topology paper.
- Dumbre, Jiao, Granmo 2025 "Scalable Bayesian Network Structure Learning Using TM", arXiv:2511.19273 - *outputs* a DAG from tabular data; input is non-graph. Reject.
- Blakely (earlier) "Generating Bayesian Network Models from Data Using TM", arXiv:2305.10538 - same: structure-learning over tabular input. Reject.

## Other TM work checked and excluded as off-topic

Probabilistic TM (2410.17851), Uncertainty-quantification TM (2507.04175), Reasoning-by-elimination TM (2407.09162), Multigranular Clauses TM (1909.07310), HV-TM for sequences (2408.16620), Convolutional TM hardware accelerators (2501.19347, 2510.15519, 2510.24282) - none ingests graph topology.

## Gap analysis

What the field has actually demonstrated, as of May 2026:

1. **Two graph-topology TM papers exist**, both 2025. Granmo et al. 2507.14874 is the canonical one (with CUDA + active repo), Blakely 2507.16537 is the only one that reports TUDataset numbers (MUTAG, PROTEINS, NCI1, AIDS, DHFR, ER_MD).
2. **Granmo GraphTM does NOT report any standard molecular graph benchmark.** Its CIFAR-10 / MNIST-superpixel / IMDB results use k-NN spatial or co-occurrence graphs, not MUTAG/PROTEINS/NCI1/OGB. The shipping code has no MUTAG/PROTEINS/NCI1 example.
3. **Blakely SGI-TM reports TUDataset but no public code.** Cannot be reproduced without re-implementation. No OGB-MolHIV, no ZINC, no QM9 anywhere in the TM literature.
4. **No TM paper reports ENZYMES, OGBG-MolHIV, ZINC, QM9, or DD.** These remain open targets.
5. **No comparison to modern GNNs on a unified TUDataset split** (e.g., 10-fold CV per Errica et al. 2020 protocol) from a TM paper. Blakely uses 10 runs but split methodology is not standard-published.
6. **No paper reports per-clause subgraph extraction with chemical interpretation** (e.g., aromatic-NO2 motif on MUTAG). Granmo claims "subgraph clauses" but exhibits only on synthetic XOR / image patches.
7. **Inductive vs transductive split is sharp.** Granmo GraphTM = fixed graph / evolving features (transductive-leaning); Blakely SGI-TM = inductive graph classification. No TM paper does both with a shared codebase.
8. **No CBR-style retrieval or case-based reasoning on top of a graph TM.** This is precisely the empty slot the present project targets (HGTM-CBR), and the literature offers no prior art to compete against on that axis.

Implication for my project: the predecessor pipeline's flat HTM on aggregate atom/bond counts is two steps behind. The minimum credible upgrade is (a) topology-aware HV encoding a la Granmo or Blakely, (b) standard TUDataset 10-fold numbers, (c) released code/CUDA. The differentiating slot still open: chemically-interpretable per-clause subgraph extraction + CBR retrieval, on MUTAG + PROTEINS + NCI1 + one MoleculeNet/OGB target untouched by the TM community.

## Sources

- [arXiv:2507.14874 - GraphTM](https://arxiv.org/abs/2507.14874) | [HTML](https://arxiv.org/html/2507.14874v1)
- [arXiv:2507.16537 - SGI-TM](https://arxiv.org/abs/2507.16537) | [HTML](https://arxiv.org/html/2507.16537v1)
- [arXiv:2406.02648 - HV-TMs](https://arxiv.org/abs/2406.02648)
- [arXiv:2504.01798 - KD-TM (rejected)](https://arxiv.org/abs/2504.01798)
- [arXiv:2508.08350 - Fuzzy-Pattern TM (rejected)](https://arxiv.org/abs/2508.08350)
- [arXiv:2511.19273 - BN via TM (rejected)](https://arxiv.org/abs/2511.19273)
- [arXiv:2305.10538 - BN-Gen TM (rejected)](https://arxiv.org/abs/2305.10538)
- [arXiv:2102.10952 - Relational TM (rejected)](https://arxiv.org/abs/2102.10952)
- [GitHub: cair/GraphTsetlinMachine](https://github.com/cair/GraphTsetlinMachine) | [PyPI](https://pypi.org/project/GraphTsetlinMachine/)
- [ISTM 2025](https://istm.no/expanded-program/)
