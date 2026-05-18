"""Kazius toxicophore coverage from learned HGTM clauses.

For each learned clause I check: does at least one TEST molecule in the
positive-prediction set, that matches a given Kazius SMARTS, also have
that clause fire? If yes, the clause "covers" that toxicophore family.

This is an empirical, post-hoc mapping, I don't try to invert the VSA
encoding analytically. Per research/06, that inversion is approximate
under bundling; the empirical co-fire test is the regulator-defensible
audit.
"""
from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/anward/project/graphtm-cbr")
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from graphtm.data.ames import load_tdc_ames
from graphtm.data.kazius import load_kazius_toxicophores
from graphtm.encoding.codebook import make_codebook
from graphtm.encoding.graph_features import encode_graph
from graphtm.core.hierarchical_graph_tm import HierarchicalGraphTM, HGraphTMSpec

results_dir = PROJECT_ROOT / "results"
log_path = results_dir / f"kazius_coverage_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def main(seed: int = 42):
    D_bits = 512
    k_hop = 2
    max_nodes = 60
    spec = HGraphTMSpec(
        n_classes=2, n_clauses=2000, threshold=500, s=3.9, n_states=100,
        R=2, IA=2, IF=8, LA=15, LF=2,
        D_bits=D_bits, n_atom_types=20, n_bond_types=5, k_hop=k_hop,
        max_nodes=max_nodes, seed=seed,
    )
    npy = results_dir / f"hgtm_ames_distilled_seed{seed}.npy"
    if not npy.exists():
        log.error("Missing %s", npy); return 1

    log.info("Loading model seed=%d", seed)
    m = HierarchicalGraphTM(spec)
    m._ensure_pump()
    m._ta_state.copy_from_host(np.load(npy))

    log.info("Loading AMES + Kazius alerts")
    mols, y, meta = load_tdc_ames(split="scaffold", seed=42, encode=False)
    cb = make_codebook(n_atom_types=20, n_bond_types=5, k_hop=k_hop,
                       D=D_bits, sparsity=0.30, seed=42)
    alerts = load_kazius_toxicophores()
    log.info("loaded %d Kazius toxicophores", len(alerts))

    # Build full test set encoded
    test_idx = meta["split_indices"]["test"]
    encoded = []
    encoded_mols = []
    encoded_y = []
    for i in test_idx:
        try:
            g = encode_graph(mols[int(i)], cb, k_hop=k_hop)
            if g.n_nodes <= max_nodes:
                encoded.append(g)
                encoded_mols.append(mols[int(i)])
                encoded_y.append(int(y[int(i)]))
        except Exception:
            pass
    encoded_y = np.array(encoded_y)
    log.info("encoded %d test mols", len(encoded))

    # SMARTS-match each mol against every alert
    alert_match = np.zeros((len(encoded), len(alerts)), dtype=bool)
    for i, mol in enumerate(encoded_mols):
        for j, a in enumerate(alerts):
            try:
                pat = Chem.MolFromSmarts(a.smarts)
                if pat is None: continue
                if mol.HasSubstructMatch(pat):
                    alert_match[i, j] = True
            except Exception:
                pass
    n_per_alert = alert_match.sum(axis=0)
    log.info("Alerts hit in test set: %s",
             [(alerts[j].name, int(n_per_alert[j])) for j in range(len(alerts)) if n_per_alert[j] > 0][:10])

    # For each mol, get firing clauses
    log.info("Collecting firing clauses per molecule...")
    fire_matrix = np.zeros((len(encoded), spec.n_clauses), dtype=bool)
    for i, g in enumerate(encoded):
        if i % 100 == 0:
            log.info("  %d / %d", i, len(encoded))
        try:
            firing = m.firing_clauses(g)
            for fc in firing:
                fire_matrix[i, fc.clause_id] = True
        except Exception:
            pass

    # For each clause + alert, count co-occurrence
    covered_alerts = set()
    clause_alert = {}     # alert_name -> list of (clause_id, n_co_occur, n_alert, n_clause)
    for j, a in enumerate(alerts):
        mask_a = alert_match[:, j]
        if mask_a.sum() == 0:
            continue
        for c in range(spec.n_clauses):
            mask_c = fire_matrix[:, c]
            if mask_c.sum() == 0:
                continue
            co = int((mask_a & mask_c).sum())
            # Coverage criterion: clause fires on a majority of alert-positive
            # mols (>= 0.5 of those that have this Kazius motif).
            if co >= max(1, int(0.5 * mask_a.sum())):
                clause_alert.setdefault(a.name, []).append(
                    (int(c), co, int(mask_a.sum()), int(mask_c.sum()))
                )
                covered_alerts.add(a.name)

    log.info("=== KAZIUS COVERAGE ===")
    log.info("Alerts present in test set: %d / %d",
             int((n_per_alert > 0).sum()), len(alerts))
    log.info("Alerts COVERED by ≥1 clause (≥50%% co-fire): %d / %d",
             len(covered_alerts), int((n_per_alert > 0).sum()))
    for name in sorted(covered_alerts):
        ent = clause_alert[name][:3]
        log.info("  %s : top clauses %s", name, ent)

    out = {
        "seed": seed,
        "n_alerts_total": len(alerts),
        "n_alerts_present_in_test": int((n_per_alert > 0).sum()),
        "n_alerts_covered": len(covered_alerts),
        "covered_alerts": sorted(covered_alerts),
        "alert_clause_map": {
            k: [{"clause_id": c, "co_fire": co, "alert_pos": n_a, "clause_fire": n_c}
                for c, co, n_a, n_c in v[:5]]
            for k, v in clause_alert.items()
        },
    }
    p = results_dir / f"kazius_coverage_distilled_seed{seed}_{int(time.time())}.json"
    p.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 42))
