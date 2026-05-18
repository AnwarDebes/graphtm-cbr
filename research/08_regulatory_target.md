# Regulatory Target Selection: Interpretable, Recourse-Capable Molecular Property Predictor

Audience: EU/US/UK regulators and pharma med-chem teams. Selection criteria: (a) publicly datasetted, (b) regulator-mandated, (c) topology-dependent, (d) counterfactual recourse meaningful.

---

## 1. Regulatory landscape (2026 snapshot)

| Framework | Year | Required property | Article / paragraph | Gap left by GNN |
|---|---|---|---|---|
| **OECD Principle 5** (REACH adopts via Annex XI 1.3) | 2007; reaffirmed QAF ENV/CBC/MONO(2023)32 | "(Q)SAR should be associated with a mechanistic interpretation, if possible." Documented descriptor-endpoint link. | OECD GD 69 §3.5; QAF §4.5 | GNN attribution is not mechanistic - no toxicophore substructure. |
| **EU REACH** Reg. (EC) 1907/2006 | amended 2024 | Annex XI 1.3: QSAR usable iff (i) valid model, (ii) within AD, (iii) adequate, (iv) **documented method**. | Annex XI §1.3(a-d) | GNN AD ill-defined; no per-prediction audit trail. |
| **ICH M7(R2)** (FDA/EMA/PMDA/HC) | Step 4 Feb 2023 | "Two complementary (Q)SAR methodologies... one expert rule based, second statistical based... Absence of alerts from both sufficient for Class 5." | M7(R2) §6.1, §6.2, §7.4 | A black-box GNN cannot serve as the expert-rule leg, nor produces an auditable alert list. |
| **EU AI Act** Reg. (EU) 2024/1689 | In force 1 Aug 2024; high-risk obligations **2 Aug 2026** | Art. 13(1): "sufficiently transparent to enable deployers to interpret the system's output." Art. 13(3)(b)(iv): document accuracy metrics + foreseeable misuse. | Art. 13(1), 13(3)(b)(iv-v); Annex III §5 | Medical-device safety-component AI is high-risk; opaque output fails "interpret the output". |
| **ICH S1B(R1)** | Step 4 Aug 2022 | WoE carcinogenicity: 6 factors incl. in-silico genotoxicity integrated with target biology, hormones, secondary pharmacology. | S1B(R1) §2.2 (a)-(f) | GNN scalar can't decompose to per-factor evidence. |
| **US EPA NAMs / TSCA** | Strategic Vision 2021; NAMs Work Plan Update Sep 2024 | TSCA §4(h)(1)(B): "scientifically defensible and transparent" non-animal alternatives. | TSCA §4(h)(1)(B); WP 2024 §3.1 | Opaque embeddings fail "defensible/transparent". |
| **OECD QAF** | Nov 2023 (ENV/CBC/MONO(2023)32) | Per-prediction: AD (§4.4), mechanistic interp (§4.5), uncertainty (§4.6). | QAF §4.4-4.6 | GNNs lack training-distribution-grounded uncertainty. |
| **EURL ECVAM** in-silico validation | Hartung et al. 2004 modular principles; JRC 2023 ECVAM Status Report | Modules 1-7: definition, WLR, BLR, predictive capacity, **mechanistic relevance**, AD, performance. | Modules 1-7 | "Mechanistic relevance" module is the central gap. |
| **ICH M14** (PMDA) | Step 4 Aug 2024 | RWE pharmacoepi - not in scope for molecular property prediction. | N/A | N/A. |

**Cross-cutting gap**: every framework addressing in-silico methods (OECD P5, REACH Annex XI, M7, QAF) demands *mechanistic* + *AD-aware* output, not just a calibrated probability. GNN attribution (GNNExplainer, integrated gradients) gives node importance scores - not toxicophores, cannot satisfy M7(R2) §6.1 expert-rule leg.

---

## 2. Concrete pharma/tox tasks

| Task | (a) Regulator | (b) Cost of wrong call | (c) State of practice | (d) Why GNN-alone rejected | (e) Public dataset (size) |
|---|---|---|---|---|---|
| **Ames mutagenicity** | FDA+EMA+PMDA (M7); EPA; ECHA | Valsartan NDMA recall (Jul 2018) = largest Class I in FDA history; multi-billion USD + cancer-risk litigation. | DEREK Nexus + Sarah Nexus per M7(R2) §6.1. Sensitivities on Ames/QSAR Int. Challenge: DEREK 54.7%, Sarah 44.0%. | M7(R2) §6.1 mandates an expert-rule leg; black-box GNN cannot substitute. | Hansen 2009 (6,512); Kazius 2005 (4,337: 2,401 mut / 1,936 non-mut). |
| **Carcinogenicity** | FDA+EMA+PMDA (S1B(R1)) | 2-yr rat bioassay ~$2-4M, 3 yrs; Phase III failure ~$1B. | WoE per S1B(R1) §2.2 + 2-yr rat bioassay. | Requires per-factor WoE; scalar incompatible. | CPDB 2023 (1,591 cmpds, 6,227 studies); Tox21 ~12K. |
| **DILI** | FDA (LTKB) | #1 post-approval withdrawal cause (troglitazone, ximelagatran); $2.6B/cmpd cost-to-market 2024. | LTKB rules + in vitro hepatocyte panel + clinical hold. | Multi-pathway (mitochondrial, BSEP, reactive metab); GNN can't disentangle. | DILIrank 2.0 (1,336); DILIst (1,279). |
| **hERG** | FDA, EMA (ICH S7B) | Terfenadine, cisapride, astemizole withdrawn; QT litigation. | Patch-clamp + alerts + ADMET Predictor. | SAR is well-known (basic amine + lipophilic aromatic); GNN adds no mitigation guidance. | ChEMBL/Karim extracts ~9K; Tox21 ~7K. |
| **Skin sensitisation** | EU Reg. 1223/2009 (cosmetics ban Mar 2013); OECD TG 497 | Cosmetic recall, occupational dermatitis. | OECD TG 497 Defined Approaches (2o3, ITS, kDA) over DPRA/KeratinoSens/h-CLAT. | TG 497 §3.1 demands *defined* integration; GNN not defined. | NICEATM HPPT (2,277 tests/136 subst.); LLNA. |
| **Endocrine disruptor (ER)** | EPA EDSP (FFDCA §408(p)); ECHA | Atrazine, BPA litigation. | EDSP21 ToxCast ER 16-assay AOP model (Judson 2015). | AOP framework demands causal-chain decomposition. | EDSP21 ~1,800; ToxCast ER. |
| **Ecotox (Daphnia/FHM)** | EPA OPPT, ECHA, OECD | Pesticide deregistration. | ECOSAR + OECD Toolbox read-across. | REACH Annex XI demands documented similarity rationale; embedding ≠ category. | EPA FHM LC50 ~600; Daphnia EC50 ~258 (KATE); EFSA OpenFoodTox. |
| **Reactive metab / GSH** | FDA pre-IND | Idiosyncratic DILI. | GSH trapping + Lhasa Meteor Nexus. | Reactivity-driven; rules dominate. | Stepan 2011 ~200. |
| **Genotoxic impurity (CoC)** | M7(R2) §7.5 | NDMA/NDEA recalls USD bn; CoC AI ≪ 1.5 μg/day TTC. | Structural alerts + expert rules. | CoC is rule-defined (aflatoxin-like, N-nitroso, alkyl-azoxy). | Hansen subset + FDA Nitrosamine API list 2023. |
| **Repro / dev tox** | FDA (ICH S5(R3) Feb 2020), EMA | Thalidomide legacy. | EFD studies + ToxCast DevTox. | Multi-organ; data sparse. | ToxRefDB DevTox ~700. |

---

## 3. Recourse value-add for top-3 candidates

For each, "recourse" = a constructively-derived structural edit that flips prediction from *toxic* to *non-toxic* while staying within applicability domain.

**(A) Ames.** Today a chemist receives a QSAR alert (e.g. "aromatic nitro at atoms 3-5-7") and manually proposes a bioisostere. Recourse returns the nearest non-mutagenic analogue (e.g. "replace -NO2 with -CN at position 5") with predicted activity, AD distance, and a precedent matched-molecular-pair. Converts QSAR from blocker into design tool. Regulatory enablement: ICH M7(R2) §7.5 "purging strategy" - recourse is the purge route.

**(B) hERG.** The basic-amine + aromatic-lipophile pharmacophore is known; mitigation is non-obvious. Recourse suggests "reduce pKa via β-fluorination of piperidine N" with predicted IC50 shift. Formalises med-chem tribal knowledge.

**(C) DILI.** Recourse says "mitigate predicted BSEP inhibition by replacing carboxylate with tetrazole". Critical because DILI is multi-mechanism: recourse + which mechanism = clinical-hold avoidance. Harder to ground in topology since labels are systemic.

---

## 4. Final recommendation: **Ames mutagenicity (ICH M7-aligned)**

Primary target: Ames mutagenicity prediction with counterfactual recourse, evaluated on Hansen 2009 (6,512 cmpds) and Kazius 2005 (4,337 cmpds), with held-out evaluation aligned to the Ames/QSAR International Challenge protocol.

Why this wins on all four criteria:

- **(a) Publicly datasetted**: Hansen and Kazius are the de-facto Ames benchmarks; both ship as SMILES + binary label, no licensing barrier. Sizes (4-6K) match HGTM-CBR's training regime documented in `03_cair_cuda_arch.md`.
- **(b) Regulator-mandated**: ICH M7(R2) §6.1 *explicitly mandates* a two-QSAR architecture with an expert-rule leg. A topology-aware Tsetlin Machine outputs literal-grounded clauses, i.e. exactly the structural-alert form an "expert-rule" system requires. This is the cleanest regulatory fit of any task in the matrix - I am slotting into a named guideline paragraph, not arguing analogically.
- **(c) Topology-dependent**: Ames mutagenicity is driven by ~30 well-documented toxicophores (Kazius 2005 derived 29 substructural rules). These are graph substructures; clause-derived literals can map directly to them. Carcinogenicity (multi-organ, multi-mechanism) and DILI (systemic, multi-mechanism) are less topology-pure.
- **(d) Recourse-meaningful**: ICH M7(R2) §7.5 names "purging strategy" as the remediation path for Class 2 (mutagenic) and Class 3 (positive but no carcinogenicity data) impurities. Recourse *is* the purge route. Furthermore, the financial stake is concrete and recent: the post-2018 N-nitrosamine recalls (valsartan, losartan, ranitidine - largest Class I recall in FDA history) created urgent demand for QSAR systems that not only flag impurities but suggest synthetic-route changes that eliminate them.

**Defence vs runner-up (hERG)**. hERG is also a strong candidate - well-defined topology, recent papers, financial stake. It loses on (b): there is no equivalent of ICH M7(R2) §6.1 mandating a transparent two-system architecture. ICH S7B requires a patch-clamp assay, not a QSAR; in-silico hERG is best-practice but not regulatory-required. I would be arguing the value, not slotting into a named requirement. Ames lets the paper open with "the model satisfies ICH M7(R2) §6.1 by construction".

**Secondary task (optional Section 5 in paper)**: ecotoxicity on fathead minnow LC50 - small dataset (~600 cmpds), well-aligned to REACH Annex XI §1.3 read-across requirement, demonstrates generalisation beyond mutagenicity.

**Primary metric**: balanced accuracy + sensitivity on the Ames/QSAR Int. Challenge held-out set (current SoTA: Sarah Nexus 44.0%, DEREK 54.7%, deep models ~73%). HGTM-CBR target: ≥80% sensitivity at ≥75% specificity, with every positive prediction accompanied by (i) a literal-grounded toxicophore (clause output), (ii) a nearest non-mutagenic counterfactual analogue (CBR retrieval), (iii) applicability-domain distance (clause-coverage score per Hansen §3.4).
