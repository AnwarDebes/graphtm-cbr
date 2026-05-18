"""Ensemble recourse: K=5 HGTMs vote via soft class-sum aggregation.

Loads all hgtm_ames_seed*.npy files, builds K HierarchicalGraphTM instances
sharing the same spec/codebook/encoding, and exposes an EnsembleModel with
.predict and .class_scores that the M5 recourse stack expects.
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
from graphtm.recourse.candidates import candidates_from_firing_clauses
from graphtm.recourse.search import greedy_minimal_edit
from graphtm.recourse.validity import validate

results_dir = PROJECT_ROOT / "results"
log_path = results_dir / f"eval_recourse_distilled_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


SEEDS = [42, 43, 44, 45, 46]


class EnsembleHGTM:
    """K HGTMs sharing spec; soft-sum class scores."""

    def __init__(self, members):
        self.members = members  # list of HierarchicalGraphTM

    def class_scores(self, graphs):
        stacked = np.stack([m.class_scores(graphs) for m in self.members], axis=0)
        return stacked.sum(axis=0).astype(np.int64)

    def predict(self, graphs):
        s = self.class_scores(graphs)
        margin = s[:, 1].astype(np.float64) - s[:, 0].astype(np.float64)
        return (margin > 0).astype(np.int64)

    def firing_clauses(self, graph):
        # union of firing clauses across members for richer candidate set
        out = []
        for m in self.members:
            try:
                out.extend(m.firing_clauses(graph))
            except Exception:
                pass
        return out


def main():
    D_bits = 512
    k_hop = 2
    max_nodes = 60
    spec_template = dict(
        n_classes=2, n_clauses=2000, threshold=500, s=3.9, n_states=100,
        R=2, IA=2, IF=8, LA=15, LF=2,
        D_bits=D_bits, n_atom_types=20, n_bond_types=5, k_hop=k_hop,
        max_nodes=max_nodes,
    )

    log.info("Loading test set...")
    mols, y, meta = load_tdc_ames(split="scaffold", seed=42, encode=False)
    cb = make_codebook(n_atom_types=20, n_bond_types=5, k_hop=k_hop,
                       D=D_bits, sparsity=0.30, seed=42)
    test_idx = meta["split_indices"]["test"]
    encoded = {}
    for i in test_idx:
        try:
            g = encode_graph(mols[int(i)], cb, k_hop=k_hop)
            if g.n_nodes <= max_nodes:
                encoded[int(i)] = g
        except Exception:
            pass
    log.info("test encoded=%d", len(encoded))

    members = []
    for seed in SEEDS:
        npy = results_dir / f"hgtm_ames_distilled_seed{seed}.npy"
        if not npy.exists():
            log.warning("Missing seed %d npy at %s, skipping.", seed, npy)
            continue
        m = HierarchicalGraphTM(HGraphTMSpec(seed=seed, **spec_template))
        m._ensure_pump()
        m._ta_state.copy_from_host(np.load(npy))
        members.append(m)
    log.info("Ensemble members loaded: %d", len(members))
    if not members:
        log.error("No members available."); return 1

    ens = EnsembleHGTM(members)

    test_keys = list(encoded.keys())
    G_te = [encoded[i] for i in test_keys]
    y_te = y[np.array(test_keys)]

    log.info("Ensemble test scores...")
    scores = ens.class_scores(G_te)
    margin = scores[:, 1].astype(np.float64) - scores[:, 0].astype(np.float64)
    pred = (margin > 0).astype(np.int32)

    def auroc(yt, ys):
        yt = np.asarray(yt).astype(np.int32); ys = np.asarray(ys, dtype=np.float64)
        pos = ys[yt == 1]; neg = ys[yt == 0]
        if len(pos) == 0 or len(neg) == 0: return float("nan")
        return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) /
                     (len(pos) * len(neg)))

    ens_acc = float((pred == y_te).mean())
    ens_au = auroc(y_te, margin)
    log.info("ENSEMBLE TEST acc=%.3f auroc=%.3f", ens_acc, ens_au)

    pos_indices = np.where(pred == 1)[0]
    log.info("Running ensemble recourse on %d predicted-positives (capped 200)...",
             min(len(pos_indices), 200))

    def encode_fn(mol):
        return encode_graph(mol, cb, k_hop=k_hop)

    n_eval = min(200, len(pos_indices))
    successes = 0
    flip_counts = []
    latencies = []
    edit_counter = {}

    for n, p_idx in enumerate(pos_indices[:n_eval]):
        orig_g = G_te[p_idx]
        orig_mol = mols[test_keys[p_idx]]
        try:
            firing = ens.firing_clauses(orig_g)
        except Exception as e:
            log.warning("firing_clauses failed for #%d: %s", p_idx, e); continue
        if not firing:
            continue
        cands = candidates_from_firing_clauses(orig_g, firing, cb, max_candidates=50)
        if not cands:
            continue
        t0 = time.time()
        try:
            edits = greedy_minimal_edit(ens, orig_g, cands, orig_mol,
                                          max_flips=3, encode_fn=encode_fn,
                                          validity_fn=validate)
        except Exception as e:
            log.warning("recourse fail: %s", e); continue
        latencies.append(time.time() - t0)
        if edits is None:
            continue
        successes += 1
        flip_counts.append(len(edits))
        for e in edits:
            key = (e.op, *e.indices)
            edit_counter[key] = edit_counter.get(key, 0) + 1
        if (n + 1) % 25 == 0:
            log.info("  %d/%d, success=%d", n + 1, n_eval, successes)

    rate = successes / max(1, n_eval)
    log.info("ENSEMBLE RECOURSE: success=%d/%d (%.3f)", successes, n_eval, rate)
    if flip_counts:
        log.info("  mean_flips=%.2f median=%d p95=%d",
                 float(np.mean(flip_counts)), int(np.median(flip_counts)),
                 int(np.percentile(flip_counts, 95)))
    if latencies:
        log.info("  latency_ms p50=%.0f p95=%.0f p99=%.0f",
                 1000 * np.percentile(latencies, 50),
                 1000 * np.percentile(latencies, 95),
                 1000 * np.percentile(latencies, 99))
    top = sorted(edit_counter.items(), key=lambda kv: -kv[1])[:10]
    for k, v in top:
        log.info("  %s -> %d", k, v)

    out = {
        "ensemble_size": len(members),
        "test_acc": ens_acc, "test_auroc": ens_au,
        "recourse_success_rate": rate,
        "mean_flips": float(np.mean(flip_counts)) if flip_counts else None,
        "latency_ms_p50": float(np.percentile(latencies, 50) * 1000) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95) * 1000) if latencies else None,
        "top_edits": [{"key": str(k), "count": v} for k, v in top],
    }
    p = results_dir / f"eval_recourse_distilled_{int(time.time())}.json"
    p.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
