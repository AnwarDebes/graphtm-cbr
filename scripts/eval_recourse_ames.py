"""Recourse evaluation on TDC AMES test positives.

Loads the trained HGTM (TA state on disk), runs M5 recourse on every test
molecule predicted positive (mutagenic), records:
  - recourse success rate
  - mean / median / p95 flip count
  - validity rate
  - per-recourse latency (p50 / p95 / p99)
  - top-K most-recommended edits
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

from graphtm.data.ames import load_tdc_ames
from graphtm.encoding.codebook import make_codebook
from graphtm.encoding.graph_features import encode_graph
from graphtm.core.hierarchical_graph_tm import HierarchicalGraphTM, HGraphTMSpec
from graphtm.recourse.candidates import candidates_from_firing_clauses, apply_edit
from graphtm.recourse.search import greedy_minimal_edit
from graphtm.recourse.validity import validate
from graphtm.recourse.output import recourse_report

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)
log_path = results_dir / f"eval_recourse_ames_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def main():
    # config, must match train_full_ames.py
    D_bits = 512
    k_hop = 2
    max_nodes = 60
    spec = HGraphTMSpec(
        n_classes=2, n_clauses=2000, threshold=500, s=3.9, n_states=100,
        R=2, IA=2, IF=8, LA=15, LF=2,
        D_bits=D_bits, n_atom_types=20, n_bond_types=5, k_hop=k_hop,
        max_nodes=max_nodes, seed=42,
    )

    ta_path = results_dir / "hgtm_ames_final.npy"
    if not ta_path.exists():
        log.error("No trained TA state at %s, run scripts/train_full_ames.py first.", ta_path)
        return 1

    log.info("Loading AMES + encoding...")
    mols, y, meta = load_tdc_ames(split="scaffold", seed=42, encode=False)
    splits = meta["split_indices"]
    cb = make_codebook(n_atom_types=20, n_bond_types=5, k_hop=k_hop,
                       D=D_bits, sparsity=0.30, seed=42)
    encoded = {}
    for i in splits["test"]:
        try:
            g = encode_graph(mols[int(i)], cb, k_hop=k_hop)
            if g.n_nodes <= max_nodes:
                encoded[int(i)] = g
        except Exception:
            pass
    log.info("test encoded=%d", len(encoded))

    log.info("Restoring HGTM from disk...")
    m = HierarchicalGraphTM(spec)
    m._ensure_pump()  # forces compile + allocate
    m._ta_state.copy_from_host(np.load(ta_path))
    log.info("HGTM ready.")

    # Predict all test, focus recourse on positives.
    test_idx = list(encoded.keys())
    G_te = [encoded[i] for i in test_idx]
    y_te = y[np.array(test_idx)]
    log.info("Predicting test set...")
    scores = m.class_scores(G_te)
    margin = scores[:, 1].astype(np.float64) - scores[:, 0].astype(np.float64)
    pred = (margin > 0).astype(np.int32)
    log.info("test_pred_pos_rate=%.3f y_pos_rate=%.3f", pred.mean(), y_te.mean())

    pos_indices = np.where(pred == 1)[0]
    log.info("Running recourse on %d predicted-positive molecules...", len(pos_indices))

    model_for_recourse = m   # HierarchicalGraphTM exposes predict + class_scores

    def encode_fn(mol):
        return encode_graph(mol, cb, k_hop=k_hop)

    reports = []
    successes = 0
    flip_counts = []
    latencies = []
    edit_counter: dict = {}

    for n, p_idx in enumerate(pos_indices[:200]):   # cap at 200 for tractable eval
        orig_g = G_te[p_idx]
        orig_mol = mols[test_idx[p_idx]]
        try:
            firing = m.firing_clauses(orig_g)
        except Exception as e:
            log.warning("firing_clauses failed for #%d: %s", p_idx, e)
            continue
        if not firing:
            continue
        cands = candidates_from_firing_clauses(orig_g, firing, cb, max_candidates=50)
        if not cands:
            continue
        t0 = time.time()
        try:
            edits = greedy_minimal_edit(
                model_for_recourse, orig_g, cands, orig_mol,
                max_flips=3, encode_fn=encode_fn, validity_fn=validate,
            )
        except Exception as e:
            log.warning("recourse search failed: %s", e)
            continue
        dt = time.time() - t0
        latencies.append(dt)
        if edits is None:
            continue
        successes += 1
        flip_counts.append(len(edits))
        for e in edits:
            key = (e.op, *e.indices)
            edit_counter[key] = edit_counter.get(key, 0) + 1
        if (n + 1) % 25 == 0:
            log.info("  recourse done %d / %d, success_so_far=%d", n + 1, len(pos_indices[:200]), successes)

    n_eval = min(200, len(pos_indices))
    success_rate = successes / max(1, n_eval)
    log.info("RECOURSE RESULTS")
    log.info("  evaluated=%d successes=%d success_rate=%.3f", n_eval, successes, success_rate)
    if flip_counts:
        log.info("  mean_flips=%.2f median=%d p95=%d",
                 float(np.mean(flip_counts)), int(np.median(flip_counts)),
                 int(np.percentile(flip_counts, 95)))
    if latencies:
        log.info("  latency_ms p50=%.1f p95=%.1f p99=%.1f",
                 1000 * float(np.percentile(latencies, 50)),
                 1000 * float(np.percentile(latencies, 95)),
                 1000 * float(np.percentile(latencies, 99)))
    top_edits = sorted(edit_counter.items(), key=lambda kv: -kv[1])[:10]
    log.info("  TOP-10 most-recommended edits (op, indices, count):")
    for k, v in top_edits:
        log.info("    %s -> %d", k, v)

    out = {
        "n_evaluated": n_eval,
        "successes": successes,
        "success_rate": success_rate,
        "mean_flips": float(np.mean(flip_counts)) if flip_counts else None,
        "latency_ms_p50": 1000 * float(np.percentile(latencies, 50)) if latencies else None,
        "latency_ms_p95": 1000 * float(np.percentile(latencies, 95)) if latencies else None,
        "top_edits": [{"key": str(k), "count": v} for k, v in top_edits],
    }
    out_path = results_dir / f"eval_recourse_ames_{int(time.time())}.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
