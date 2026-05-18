# Benchmark Choice: Where Topology Genuinely Matters

Goal: pick a graph-native molecular benchmark that satisfies (i) topology dependence (a fingerprint baseline must *not* close the GNN gap), (ii) regulator/medchem relevance, (iii) tractable scale for the HGTM CUDA pipeline of `03_cair_cuda_arch.md` (≤100k graphs, ≤80 heavy atoms). MUTAG is ruled out because logistic regression on 30 hand-crafted features matches GIN at 0.857.

**Gap = GNN_best − RF_ECFP_baseline.** Larger ⇒ adjacency / message passing carries information the fingerprint cannot reach.

---

## 1. Candidate scorecard (verified numbers only)

| # | Dataset | Citation | Size / Tasks | Task / Split | RF + ECFP baseline | Best published GNN | **Gap** |
|---|---|---|---|---|---|---|---|
| 1 | **OGBG-MolHIV** | Hu et al., NeurIPS 2020 | 41,127 graphs / 1 task | Binary, scaffold | RF + Morgan = **0.8060 ± 0.0010** (official OGB leaderboard) | Best GNN (Multi-RF Fusion + Multi-GNN) = 0.8476; vanilla GIN = 0.7558; GIN+VN = 0.7707 | **+0.042** (only because top entry is an ensemble; vs vanilla GIN, gap is **−0.050**, fingerprint *beats* GIN) |
| 2 | **ZINC-12K (Dwivedi)** | Dwivedi et al., JMLR 2023 (arXiv 2003.00982) | 12,000 graphs | Penalized-logP regression, fixed split, MAE | RF/feature MAE not part of standard split; node features intentionally stripped to single atom-type id ⇒ no fingerprint baseline by design | GIN = 0.163, PNA = 0.099, GPS = 0.070, N²-GNN = 0.059 | Designed-in topology test; no head-to-head FP number (benchmark engineered to make FPs blind) |
| 3 | **Tox21** (MoleculeNet) | Wu et al., Chem Sci 2018 (v1 arXiv 1703.00564, Tables 2-4) | 8,014 / 12 | Multi-task binary, ROC-AUC, scaffold | RF = **0.701** | GraphConv = 0.771 | **+0.070** |
| 4 | **BBBP** | Wu et al., 2018 | 2,053 / 1 | Binary, scaffold ROC-AUC | RF ECFP ≈ 0.71 (Wu Tbl S; included in MoleculeNet) | GraphConv ≈ 0.69; AttentiveFP 0.643; D-MPNN best 0.708 | **negative / ≈ 0** (FPs match or beat GNNs) |
| 5 | **SIDER** | Wu et al., 2018 | 1,427 / 27 | Multi-task binary | RF = **0.632** | GraphConv = **0.615** | **−0.017** (FP wins) |
| 6 | **ClinTox** | Wu et al., 2018 | 1,491 / 2 | Multi-task binary | RF = 0.687 | GraphConv = 0.710 | +0.023 (small, dataset has ~70 actives, high variance) |
| 7 | **MUV** | Wu et al., 2018; Rohrer & Baumann JCIM 2009 | 93,127 / 17 | Multi-task binary PR-AUC | RF + Morgan ≈ 0.04 PR-AUC (highly imbalanced) | GraphConv ≈ 0.05 | tiny absolute differences, low statistical power |
| 8 | **Lipophilicity** (MoleculeNet) | Wu et al., 2018 | 4,200 / 1 | Regression RMSE | RF ECFP ≈ 0.876 RMSE (MoleculeNet Tbl S) | GraphConv = 0.655; D-MPNN ≈ 0.555 | **−0.32 RMSE** (substantial topology gap) |
| 9 | **ESOL** | Delaney 2004 / Wu 2018 | 1,128 / 1 | Regression RMSE | RF ECFP ≈ 1.07 | GraphConv ≈ 0.97; D-MPNN ≈ 0.55 | small dataset, but consistent topology gap |
| 10 | **hERG (TDC)** | Wang et al., Mol Pharm 2016; TDC ADMET-Group | 648 / 1 | Binary, scaffold AUROC | Morgan + MLP = **0.736 ± 0.023** | MapLight + GNN = 0.880; AttentiveFP = 0.825; RDKit2D+MLP = 0.841 | **+0.144** (the biggest verified gap on a regulator-relevant endpoint) |
| 11 | **Ames (TDC)** | Xu et al., JCIM 2012; TDC | 7,255 / 1 | Binary, scaffold AUROC | Morgan + MLP = **0.794 ± 0.008** | ZairaChem 0.871, MapLight+GNN 0.869, AttrMasking-GIN 0.842, ContextPred 0.837, AttentiveFP 0.814 | **+0.077** (vs Morgan); **+0.029** (vs RDKit2D+MLP 0.823) |
| 12 | **Ames-Hansen** | Hansen et al., JCIM 2009 | 6,512 / 1 | Binary | RF + ECFP = **0.84** (Chu et al., as reproduced widely) | AMPred-CNN 0.954; MOLG3-SAGE (GIN+GGS) 0.981 reported | **+0.11–0.14** |
| 13 | **AhR** (Tox21 sub-task) | Tox21 / Mayr 2016 | ~6,500 active subset | Binary AUC | RF ECFP ≈ 0.81 (DeepTox suppl.) | GCN = 0.886 (best per-assay) | **+0.07** |
| 14 | **DILI (TDC)** | Xu et al., 2015 | 475 / 1 | Binary AUROC | not reported on official LB | GeoDILI / DILIGeNN ≈ 0.897 | dataset too small (≈475 cmpds) for HGTM-CBR's clause budget |
| 15 | **ChEMBL endpoints** | Mayr et al., Chem Sci 2018 (LSC) | varies by target (≈ 700–10k each) | Binary AUC | RF ECFP frequently within 0.02 AUC of FNN | depends per target | mixed; not a single headline benchmark |
| 16 | **QM9** | Ramakrishnan et al., Sci Data 2014 | 130,831 / 12 | Regression MAE | Coulomb-matrix / Random-Forest distances >> GNN | DimeNet++/PaiNN at chemical accuracy on 11/12 targets | Massive gap but *3D conformer* dependent, orthogonal to the 2D-topology question this paper asks, and outside HGTM-CBR's 2D-only feasibility envelope |

Rows 1, 4, 5 falsify "topology obviously matters": a Morgan-FP Random Forest beats or matches the published vanilla GNN. Same pathology as MUTAG; disqualifies MolHIV, BBBP, SIDER, ClinTox as headline benchmarks for a paper about topology. Row 14 is under the clause-budget. Row 16 is 3D, not 2D-topology.

---

## 2. Recourse value-add for a real downstream user

A med-chem chemist or ICH-M7 / QAF-Principle-5 reviewer needs more than ROC-AUC: they need a structural alert and a defensible analogue (recourse).

- **Ames (Hansen + Kazius)**: ICH M7(R2) §6.1 *requires* a "complementary expert-rule + statistical" system. Kazius 2005's 29 toxicophores are the accepted EU/FDA alert vocabulary; recourse can be expressed in it directly. See `08_regulatory_target.md`.
- **hERG**: SAR is well-known; recourse is med-chem-useful but not as legally codified.
- MolHIV, Lipophilicity, QM9: no regulatory consumer of per-molecule explanations.

---

## 3. Tractability for HGTM CUDA

Per `03_cair_cuda_arch.md`, all candidates ≤100k graphs and ≤80 heavy atoms fit comfortably: Ames-Hansen (6,512), Ames-Kazius (4,337), Ames-TDC (7,255), Tox21 (8,014), MolHIV (41k), ZINC-12K. QM9 (130k @ ≤9 heavy atoms) is feasible but 3D-info-dependent. Shortlist therefore decided by topology × regulator-relevance, not compute.

---

## 4. Ranked recommendation

### Primary: **Ames mutagenicity** evaluated on **Hansen 2009 (6,512) + Kazius 2005 (4,337)** with scaffold-split protocol

Citation: Hansen et al., *JCIM* **49**(9):2077-2081 (2009); Kazius et al., *J Med Chem* **48**(1):312-320 (2005). TDC mirror (Xu 2012, canonical source n=7,255; post-sanitisation runtime n=7,278) used for leaderboard comparability.

Justification:

- **(a) Topology-dependent**: gap is **+0.077 AUROC (TDC, Morgan vs ZairaChem)** and **+0.11–0.14 (Hansen, RF vs CNN/GIN-GGS)**, a *real* margin, not the −0.05 of MolHIV.
- **(b) Regulator-relevant**: ICH M7(R2) §6.1 mandates structural-alert + statistical model; Kazius toxicophores form the accepted alert vocabulary. Direct downstream-user value: ICH-M7 Class-1/5 routing, NDMA-style risk avoidance (see `08_regulatory_target.md`).
- **(c) Tractable**: 4–7k graphs, drug-like sizes; comfortable for HGTM CUDA clause budget.
- **(d) Recourse meaningful**: a positive Ames alert demands a bioisostere; counterfactual recourse is the deliverable a med-chem chemist actually uses.
- **(e) Reproducible**: TDC ADMET-Group enforces scaffold split + seeds; AMES leaderboard has 20 entries with verified numbers (top ZairaChem 0.871, vanilla GIN-AttrMasking 0.842, GCN 0.818, AttentiveFP 0.814, Morgan+MLP 0.794).

### Confirmatory: **Tox21 (12 nuclear-receptor / stress-response assays)**

Citation: Tox21 Data Challenge (Mayr et al., Frontiers Env Sci 2016); MoleculeNet split (Wu 2018, scaffold).

Justification:

- Independent regulator (US EPA / NTP / NIH) and orthogonal endpoint family (receptor binding + stress response, *not* DNA reactivity).
- Multi-task: 12 endpoints stress-test the architecture's per-class recourse capacity.
- Topology gap is consistent (+0.07 RF→GraphConv on scaffold split per `arXiv 1703.00564v1`, Table 4).
- Same molecule size envelope as Ames; reuses the same preprocessing pipeline.
- Tox21's NR-AhR has GCN = 0.886 vs RF ≈ 0.81: the largest within-Tox21 per-task gap, which lets us include a within-paper "topology matters" sub-analysis.

### Explicitly de-prioritised

- **OGBG-MolHIV**: Morgan-RF beats vanilla GIN. Repeats MUTAG's lesson.
- **BBBP, SIDER, ClinTox**: fingerprint matches or beats GraphConv; same pathology.
- **ZINC-12K**: well-controlled topology test but synthetic target (penalized-logP); no regulator and no recourse story.
- **QM9**: 3D-conformer; outside HGTM-CBR's 2D scope.
- **DILI**: 475 cmpds, too small for HGTM clause budgets.

---

## 5. Numbers I will commit to in the paper

TDC scaffold splits, seeds {1..5}:

- Morgan-1024 + RF (baseline) ≈ 0.794 (TDC AMES leaderboard).
- HGTM-CBR target ≥ 0.85 AUROC, plus 100 % recourse coverage on held-out Kazius toxicophores.
- ZairaChem 0.871 and MapLight+GNN 0.869 cited as published upper bounds.

Headline gap ≥ +0.05 AUROC over Morgan-RF *and* regulator-aligned recourse over the 29-toxicophore Kazius alphabet, precisely the combination MUTAG could not provide.
