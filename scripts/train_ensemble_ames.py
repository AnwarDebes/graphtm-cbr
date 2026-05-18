"""K-seed HGTM ensemble on TDC AMES.

Trains K independent HGTMs with different seeds, snapshots best-by-validation
TA state for each, then evaluates the ensemble via soft class-sum aggregation
(per the prior `axiom-coi-unified` finding that K=10 soft sum recovers within
1 std of teacher).
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

# config
K_SEEDS = 5
SEEDS = [42, 43, 44, 45, 46]
D_bits = 512
k_hop = 2
max_nodes = 60
n_clauses = 2000
T = 500
s_param = 3.9
epochs = 60

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)
log_path = results_dir / f"ensemble_ames_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def auroc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    pos2 = pos[:, None]; neg2 = neg[None, :]
    return float(((pos2 > neg2).sum() + 0.5 * (pos2 == neg2).sum()) / (len(pos) * len(neg)))


def main():
    log.info("=== K=%d ensemble on TDC AMES ===", K_SEEDS)
    mols, y, meta = load_tdc_ames(split="scaffold", seed=42, encode=False)
    splits = meta["split_indices"]
    log.info("n_mols=%d splits tr=%d va=%d te=%d", len(mols),
             len(splits["train"]), len(splits["valid"]), len(splits["test"]))

    cb = make_codebook(n_atom_types=20, n_bond_types=5, k_hop=k_hop,
                       D=D_bits, sparsity=0.30, seed=42)
    log.info("Encoding...")
    t0 = time.time()
    encoded = {}
    for i, m_ in enumerate(mols):
        try:
            g = encode_graph(m_, cb, k_hop=k_hop)
            if g.n_nodes <= max_nodes:
                encoded[i] = g
        except Exception:
            pass
    log.info("encoded=%d (%.1fs)", len(encoded), time.time() - t0)

    def take(indices):
        kept = [int(i) for i in indices if int(i) in encoded]
        return [encoded[i] for i in kept], y[np.array(kept)]

    G_tr, y_tr = take(splits["train"])
    G_va, y_va = take(splits["valid"])
    G_te, y_te = take(splits["test"])
    log.info("tr=%d va=%d te=%d", len(G_tr), len(G_va), len(G_te))

    member_scores_te = []     # K × (n_test, 2) class_sum arrays
    member_summaries = []

    for k_idx, seed in enumerate(SEEDS):
        log.info("--- Member %d / %d (seed=%d) ---", k_idx + 1, K_SEEDS, seed)
        spec = HGraphTMSpec(
            n_classes=2, n_clauses=n_clauses, threshold=T, s=s_param, n_states=100,
            R=2, IA=2, IF=8, LA=15, LF=2,
            D_bits=D_bits, n_atom_types=20, n_bond_types=5, k_hop=k_hop,
            max_nodes=max_nodes, seed=seed,
        )
        model = HierarchicalGraphTM(spec)
        best_au = -1.0
        best_state = None
        best_epoch = 0
        t_member_start = time.time()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.fit(G_tr, y_tr, epochs=1)
            scores_va = model.class_scores(G_va)
            margin_va = scores_va[:, 1].astype(np.float64) - scores_va[:, 0].astype(np.float64)
            pred_va = (margin_va > 0).astype(np.int32)
            au_va = auroc(y_va, margin_va)
            acc_va = float((pred_va == y_va).mean())
            log.info("  seed=%d ep=%2d val_acc=%.3f val_auroc=%.3f wall=%.1fs",
                     seed, epoch, acc_va, au_va, time.time() - t0)
            if au_va > best_au:
                best_au = au_va; best_epoch = epoch
                best_state = model._ta_state.copy_to_host().copy()
        log.info("  best ep=%d val_auroc=%.3f", best_epoch, best_au)

        # restore best, eval test
        model._ta_state.copy_from_host(best_state)
        scores_te = model.class_scores(G_te)
        margin_te = scores_te[:, 1].astype(np.float64) - scores_te[:, 0].astype(np.float64)
        pred_te = (margin_te > 0).astype(np.int32)
        acc_te = float((pred_te == y_te).mean())
        au_te = auroc(y_te, margin_te)
        log.info("  seed=%d TEST acc=%.3f auroc=%.3f", seed, acc_te, au_te)
        member_scores_te.append(scores_te.copy())
        # save best state per seed
        np.save(results_dir / f"hgtm_ames_seed{seed}.npy", best_state)
        member_summaries.append({
            "seed": seed, "best_epoch": best_epoch, "best_valid_auroc": best_au,
            "test_acc": acc_te, "test_auroc": au_te,
            "wall_min": (time.time() - t_member_start) / 60,
        })

    # Ensemble: soft class-sum aggregation
    log.info("=== Ensembling K=%d ===", K_SEEDS)
    stacked = np.stack(member_scores_te, axis=0)   # [K, n_test, 2]
    ens_soft_sum = stacked.sum(axis=0)
    margin_ens = ens_soft_sum[:, 1].astype(np.float64) - ens_soft_sum[:, 0].astype(np.float64)
    pred_ens = (margin_ens > 0).astype(np.int32)
    acc_ens = float((pred_ens == y_te).mean())
    au_ens = auroc(y_te, margin_ens)
    log.info("ENSEMBLE soft-class-sum: acc=%.3f auroc=%.3f", acc_ens, au_ens)
    # also report hard-majority and best-single
    hard_preds = np.array([(s[:, 1] > s[:, 0]).astype(np.int32) for s in member_scores_te])  # [K, n]
    hard_vote = (hard_preds.sum(axis=0) > K_SEEDS / 2).astype(np.int32)
    acc_hard = float((hard_vote == y_te).mean())
    log.info("ENSEMBLE hard-majority: acc=%.3f", acc_hard)

    out = {
        "config": {
            "K": K_SEEDS, "seeds": SEEDS, "epochs": epochs,
            "n_clauses": n_clauses, "T": T, "s": s_param,
            "D_bits": D_bits, "max_nodes": max_nodes,
        },
        "members": member_summaries,
        "ensemble_soft_sum": {"test_acc": acc_ens, "test_auroc": au_ens},
        "ensemble_hard_majority": {"test_acc": acc_hard},
        "n_train": len(G_tr), "n_valid": len(G_va), "n_test": len(G_te),
    }
    out_path = results_dir / f"ensemble_ames_{int(time.time())}.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
