# VSA / Hypervector Encoding for Graph-Native Tsetlin Machines

Scope: design a Boolean fixed-size feature vector that encodes atoms, bonds and k-hop
topology of a molecular graph for a flat Tsetlin Machine (literals = single bits).
Topology must survive so that Hamming-flip recourse maps back to chemical edits.

---

## 1. VSA / Hypervector survey

VSA models all share a triple {atomic vectors, **binding** ⊗, **bundling** ⊕}; binding
is approximately invertible and produces a vector dissimilar to its operands,
bundling is similarity-preserving superposition. Survey reference: Kleyko, Rachkovskij,
Osipov, Rahimi, *A Survey on Hyperdimensional Computing aka Vector Symbolic
Architectures*, Part I (ACM CSUR 55(6), 2022)
[arXiv:2111.06077](https://arxiv.org/abs/2111.06077) and Part II (ACM CSUR 55(9), 2023)
[arXiv:2112.15424](https://arxiv.org/abs/2112.15424).

| Model | Atoms | Binding | Bundling | Boolean-friendly? |
|---|---|---|---|---|
| **HRR**, Plate 1995 ([IEEE TNN 6(3)](https://ieeexplore.ieee.org/document/377968/), [PDF](https://www2.fiit.stuba.sk/~kvasnicka/CognitiveScience/6.prednaska/plate.ieee95.pdf)) | real, N(0,1/d) | circular convolution | sum + normalise | No, requires reals/FFT |
| **FHRR**, Plate 2003; phasor/complex | unit complex phasors | element-wise complex multiply | complex sum, renorm | Only via qFHRR quantisation ([arXiv:2604.25939](https://arxiv.org/abs/2604.25939)) |
| **BSC**, Kanerva 1994/1997 (Spatter Code, [PDF](http://www.cap-lore.com/RWC97-kanerva.pdf), [NIPS98 PDF](https://people.engr.tamu.edu/choe/choe/mirror/kanerva.NIPS98-kanerva.pdf)) | dense {0,1}^d | XOR | majority + tie-break | **Yes, natively** |
| **MAP**, Gayler 1998 ([Redwood mirror](https://redwood.berkeley.edu/wp-content/uploads/2021/08/GallantOkaywe2013.pdf), survey Part I §3) | {-1,+1}^d | element-wise multiply | element-wise sum, sign | Bipolar; binary by mapping {-1→0,+1→1} but binding becomes XNOR |
| **SBDR / CDT**, Rachkovskij & Kussul 2001 (survey Part I; [Begell scalars](https://www.dl.begellhouse.com/journals/2b6239406278e43e,4637b2492464cd1e,1000f1ff5080938f.html)) | sparse {0,1}^d (k≪d ones) | context-dependent thinning | bitwise OR + thinning | **Yes**, sparse, energy-cheap |

Capacity (Plate 1995; survey Part I §4): with d ≈ 10^4 a single hypervector can
faithfully bundle ~50–200 atomic vectors before retrieval accuracy collapses;
HRR/MAP/BSC are within constant factors of each other.

Recent (2022–2026) graph-on-HDC work:

* GraphHD (Nunes et al., DATE 2022, [arXiv:2205.07826](https://arxiv.org/abs/2205.07826)):
  BSC, d ≈ 10^4. Vertex hypervectors from PageRank-binned ranks; edge HV = bind(u,v);
  graph HV = bundle of edge HVs. 14.6× faster training than GCN.
* MoleHD (Ma et al., ICCAD 2022, [arXiv:2106.02894](https://arxiv.org/abs/2106.02894)):
  SMILES tokens, BSC binding+permutation; CPU-only, no backprop; matches/exceeds GNN
  ROC-AUC on Clintox/BBBP/SIDER.
* HDBind (Jones et al., *Sci. Reports* 2024,
  [arXiv:2303.15604](https://arxiv.org/abs/2303.15604),
  [PMC11584749](https://pmc.ncbi.nlm.nih.gov/articles/PMC11584749/),
  [code](https://github.com/LLNL/hdbind)): introduces "Direct ECFP", where each ECFP bit
  is one slot in a binary hypervector whose bit corresponds to a real molecular
  subgraph. d ≈ 10^4 ; binding = Hadamard, bundling = sum-threshold; first HDC
  graph encoder for drug screening (LIT-PCBA).
* HDGL (Kang et al., WSDM 2025,
  [arXiv:2402.17073](https://arxiv.org/abs/2402.17073)): permutation by hop distance,
  bundle k-hop neighborhoods, approximates Weisfeiler–Lehman.
* HyperGraphX (2025/2026,
  [arXiv:2510.23980](https://arxiv.org/abs/2510.23980)):
  weightless graph convolution; binary aggregation via element-wise OR
  (`h_v^{(l+1)} = ⋁_{(u,v)∈E} h_u^{(l)}`); 144× faster than HDGL.
* Symbolic Graph Intelligence (Tsetlin lab, Granmo et al.),
  [arXiv:2507.16537](https://arxiv.org/html/2507.16537): sparse binary HVs,
  D = 6400 with K = 20 % active, XOR-bind + top-K bundling, fed bit-for-bit as
  literals to a Coalesced Tsetlin Machine. Currently the closest prior art.
* Deep Graph Tsetlin Machine,
  [arXiv:2507.14874](https://arxiv.org/abs/2507.14874): sparse hypervectors for
  node properties, edge-type-annotated message-passing inboxes, layer depth = k-hop.

---

## 2. Boolean message passing: how to keep topology

The recipe shared by GraphHD, MoleHD, HDBind, HDGL and the Granmo lab work is:

1. **Atomic codebook.** Sample fixed random binary HVs for every atom-type symbol
   (C, N, O, F ...), every bond-type (single, double, aromatic, triple), and every
   hop-distance role (`ρ_0, ρ_1, ..., ρ_K`).
2. **Per-edge binding.** For edge (u,v,b): `m_{u→v} = atom(u) ⊗ bond(b) ⊗ atom(v)`
   with ⊗ = XOR (BSC). Because XOR is self-inverse, the same bit pattern recurs
   whenever the *triple* (atom_u, bond, atom_v) appears; that is what gives the
   TM a stable literal handle on bonds.
3. **k-hop role assignment.** For a centre atom c and a neighbour u at hop r, bind
   `atom(u) ⊗ ρ_r`. Permutation (cyclic shift by r positions, used in HDGL and the
   Granmo paper) is an equivalent role marker and is cheap on uint32 words.
4. **Neighbourhood bundling.** Aggregate all per-hop messages into one fixed-size
   binary vector via majority (BSC) or top-K (sparse SBDR). The result is
   noise-tolerant and stays in {0,1}^d.

This is exactly the Weisfeiler–Lehman one-step refinement, but executed with binary
operators only (see survey Part II §5, Nunes 2022 §III, HDGL §3). Topology is
preserved up to k-WL equivalence at hop depth k.

---

## 3. Design recommendation: single concrete scheme

I pick **Binary Spatter Code with sparsity** (10 % ones) to combine BSC's clean
XOR/majority algebra with SBDR's energy/sparsity friendliness and HDBind's
"bit ≡ real subgraph" interpretability.

### Hyperparameters

| Symbol | Value | Reason |
|---|---|---|
| `d` (HV dim) | 8192 (= 128 × 64-bit words) | Granmo lab uses 6400; capacity ~150 bundled items; word-aligned on GPU |
| `K` active per HV | 819 (10 % density) | SBDR regime; matches Rachkovskij 2001 |
| `K_hop` | 2 | ECFP4 equivalent; matches Morgan radius 2 |
| Atom alphabet | 9 types (C,N,O,F,P,S,Cl,Br,I) → 9 atom HVs | Standard ToxBenchmark vocabulary |
| Bond alphabet | 4 types (single, double, triple, aromatic) → 4 bond HVs | RDKit defaults |
| Role HVs | `ρ_0, ρ_1, ρ_2` | One per hop |

### Encoding pipeline (per molecule)

```
1. For each atom v in molecule:
     bag(v) = ∅
     For each hop r ∈ {0,1,...,K_hop}:
       For each neighbour u at exactly r bonds from v along path π:
         w = atom(v)
         For (a_i, b_i, a_{i+1}) in path π:
           w ← w  XOR  bond(b_i)  XOR  atom(a_{i+1})    # rolling bind
         bag(v) ∪= { w  XOR  ρ_r }                       # role-tag

2. node_HV(v) = top-K majority-bundle( bag(v) )

3. molecule_HV = top-K majority-bundle( { node_HV(v) : v ∈ V } )

4. Tsetlin literals = bits of molecule_HV  (and their negations)
```

Binding ⊗ = **XOR** (BSC); bundling ⊕ = **majority + top-K threshold** (SBDR).
Top-K is implemented by counting per-bit votes, picking the K highest counts,
breaking ties with a deterministic per-bit hash. Output is a Boolean vector of
length d = 8192 that drops straight into the Hierarchical TM literal array
(`tsetlin/hypervectors.py:bind/bundle/permute` already implements this trio).

### Capacity / collisions

With d = 8192 and ≤ 150 bundled items per molecule_HV, expected
crosstalk on retrieval is < 5 % (Plate 1995 Eq. 6; survey Part I §4.2). For 100-atom
molecules this is comfortable; for proteins, raise d to 16384.

### Counterfactual recourse over Boolean Hamming flips

The classifier is monotone in literals, so a counterfactual is a minimum set
`F ⊂ {0,...,d−1}` of bit flips such that the predicted class changes. Question:
does a flip map back to a real chemical edit?

* **Per-edge bits are recoverable.** Each edge contributes the triple HV
  `T_{u,b,v} = atom(u) ⊗ bond(b) ⊗ atom(v)`. Because XOR-bind is exactly invertible
  on raw triples and approximately invertible after one majority bundle (Kanerva
  2009; survey Part I §3.3), I can run a clean-up step against the atom-and-bond
  codebook: for each candidate triple `T`, compute Hamming-similarity to
  `molecule_HV`; triples present in the molecule cluster above chance, absent
  triples cluster at chance. A bit flip that *removes* support for `T_{u,b,v}`
  decodes to "delete bond b between atoms u and v"; a flip that *adds* support
  decodes to "add bond b between atoms u and v": both are chemically actionable
  (RDKit `RWMol.RemoveBond` / `AddBond`).
* **Role-permuted bits localise hop distance.** Because `ρ_r` is fixed, flipping a
  bit inside the ρ_r-rotated region tells me the edit affects an r-hop relation,
  not an isolated atom property.
* **Per-atom bits decode through the same codebook.** Flipping a bit that
  contributes mainly to `atom(C) ⊗ ρ_0` (degree-1 carbon) corresponds to an atom
  substitution (`RWMol.ReplaceAtom`).

Recourse loop: (1) TM produces minimum flip set `F` over `molecule_HV`;
(2) clean up each flipped bit against the triple codebook; (3) emit RDKit edit
ops; (4) re-encode and confirm boundary crossed. Chain TM-flip → triple-HV →
RDKit edit is end-to-end auditable, the property HGTM-CBR needs.

---

## References

- Plate, *Holographic Reduced Representations*, IEEE TNN 6(3), 1995.
  [IEEE](https://ieeexplore.ieee.org/document/377968/),
  [PDF mirror](https://www2.fiit.stuba.sk/~kvasnicka/CognitiveScience/6.prednaska/plate.ieee95.pdf).
- Kanerva, *The Spatter Code for Encoding Concepts at Many Levels*, 1994.
  [Semantic Scholar](https://www.semanticscholar.org/paper/940f29f161666a673abd79ce80021474cd6118ec);
  *Fully Distributed Representation* (RWCP 1997) [PDF](http://www.cap-lore.com/RWC97-kanerva.pdf);
  *Large Patterns Make Great Symbols* (NIPS 1998) [PDF](https://people.engr.tamu.edu/choe/choe/mirror/kanerva.NIPS98-kanerva.pdf).
- Gayler, *MAP coding*, 1998. Referenced in
  Gallant & Okaywe 2013 [Redwood PDF](https://redwood.berkeley.edu/wp-content/uploads/2021/08/GallantOkaywe2013.pdf)
  and survey Part I.
- Kleyko, Rachkovskij, Osipov, Rahimi, *Survey on HDC/VSA Parts I/II*, ACM CSUR 2022/2023.
  [Part I arXiv](https://arxiv.org/abs/2111.06077),
  [Part II arXiv](https://arxiv.org/abs/2112.15424),
  [ACM Part I](https://dl.acm.org/doi/10.1145/3538531),
  [ACM Part II](https://dl.acm.org/doi/10.1145/3558000).
- Rachkovskij & Kussul, sparse binary encoding.
  *Sparse Binary Distributed Encoding of Scalars*, Begell 2005
  [Begell](https://www.dl.begellhouse.com/journals/2b6239406278e43e,4637b2492464cd1e,1000f1ff5080938f.html).
- Nunes et al., *GraphHD*, DATE 2022.
  [arXiv:2205.07826](https://arxiv.org/abs/2205.07826).
- Ma et al., *MoleHD*, ICCAD 2022.
  [arXiv:2106.02894](https://arxiv.org/abs/2106.02894),
  [IEEE](https://ieeexplore.ieee.org/document/9995708/).
- Jones et al., *HDBind*, Sci. Reports 2024.
  [arXiv:2303.15604](https://arxiv.org/abs/2303.15604),
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11584749/),
  [code](https://github.com/LLNL/hdbind).
- Kang et al., *HDGL*, WSDM 2025.
  [arXiv:2402.17073](https://arxiv.org/abs/2402.17073).
- *HyperGraphX*, 2026. [arXiv:2510.23980](https://arxiv.org/abs/2510.23980).
- *Symbolic Graph Intelligence* (Granmo et al.), 2025.
  [arXiv:2507.16537](https://arxiv.org/html/2507.16537).
- *Deep Graph Tsetlin Machine*, 2025.
  [arXiv:2507.14874](https://arxiv.org/abs/2507.14874).
- Rogers & Hahn, *Extended-Connectivity Fingerprints*, J. Chem. Inf. Model. 2010.
  [ACS](https://pubs.acs.org/doi/10.1021/ci100050t).
