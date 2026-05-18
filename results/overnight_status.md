# Overnight run status: DONE

Final update: 2026-05-16 07:03

## Phase 1: direct-label K=5 HGTM ensemble: DONE

| Seed | Best valid AUROC | Test AUROC | Test acc |
|---|---|---|---|
| 42 | 0.692 (ep5) | 0.671 | 0.583 |
| 43 | 0.678 (ep5) | 0.660 | 0.610 |
| 44 | n/a | 0.646 | 0.480 |
| 45 | 0.688 | 0.662 | 0.547 |
| 46 | 0.687 (ep7) | 0.661 | 0.529 |

- Mean per-seed: **0.660 ± 0.009**
- Ensemble soft-sum: **AUROC 0.669, acc 0.575**
- Recourse: **54.5 %** success, 1.51 mean flips, 3.7s p50 latency
- Kazius coverage: 21/21 (saturation artefact)

## Phase 2: GIN→HGTM distillation: DONE

- GIN teacher: val AUROC 0.885, **test AUROC 0.796** (matches Morgan-RF baseline 0.790)

| Seed | Direct AUROC | Distilled AUROC | Δ |
|---|---:|---:|---:|
| 42 | 0.671 | 0.679 | +0.008 |
| 43 | 0.660 | 0.681 | +0.021 |
| 44 | 0.646 | 0.666 | +0.020 |
| 45 | 0.662 | 0.670 | +0.008 |
| 46 | 0.661 | 0.668 | +0.007 |
| **mean** | **0.660** | **0.673** | **+0.013** |

- Ensemble: **AUROC 0.685, acc 0.635** (+0.016 AUROC, +0.060 acc over direct)
- Recourse: **95.5 %** success (+41 pp), 1.30 mean flips, 2.1s p50 latency
- Kazius coverage: 21/21 (saturation persists; not fixed by distillation)

## Headline finding

Distillation barely lifts raw AUROC (+0.016) **but transforms recourse** (+41 pp success). For ICH M7(R2) §7.5 (purging strategy), recourse coverage is the regulator-load-bearing metric, not raw AUROC.

## Gap to Morgan-FP RF baseline

- AUROC: −0.105 (distilled HGTM 0.685 vs 0.790 baseline)
- Recourse: baseline cannot provide recourse natively. **HGTM uniquely capable.**

## Artifacts updated

- `paper/paper.md`: §4.1, §4.3, §4.4, §4.5, §4.6, §4.7, §4.8, §5 all locked.
- `README.md`: Results table updated.
- `results/*.json` + `results/hgtm_ames*_seed{42-46}.npy`: full reproduction artefacts.

## No errors encountered.
