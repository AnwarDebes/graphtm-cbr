"""Phase 2: GIN teacher → K=5 HGTM ensemble distillation on TDC AMES.

Pipeline:
  1. Encode all AMES mols via M1 (same BSC config as Phase 1).
  2. Train GIN teacher on TRAIN split (M4 train_teacher). Save GIN soft+hard
     predictions on TRAIN, plus VAL+TEST AUROC.
  3. Train K=5 HGTM ensemble on the TEACHER's hard predictions (not lab labels).
  4. Evaluate each member + ensemble on test (lab labels).
  5. Save trained TA states as hgtm_ames_distilled_seed*.npy.
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
from graphtm.distill.teacher import train_teacher

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)
log_path = results_dir / f"distill_ensemble_ames_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

SEEDS = [42, 43, 44, 45, 46]
D_bits = 512
k_hop = 2
max_nodes = 60
n_clauses = 2000
T = 500
s_param = 3.9
epochs = 60


def auroc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]; neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() +
                  0.5 * (pos[:, None] == neg[None, :]).sum()) /
                 (len(pos) * len(neg)))


def main():
    log.info("=== Phase 2: GIN→HGTM distillation on TDC AMES ===")
    mols, y, meta = load_tdc_ames(split="scaffold", seed=42, encode=False)
    splits = meta["split_indices"]
    log.info("n_mols=%d splits tr=%d va=%d te=%d", len(mols),
             len(splits["train"]), len(splits["valid"]), len(splits["test"]))

    log.info("Encoding...")
    cb = make_codebook(n_atom_types=20, n_bond_types=5, k_hop=k_hop,
                       D=D_bits, sparsity=0.30, seed=42)
    encoded = {}
    t0 = time.time()
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
        return [encoded[i] for i in kept], y[np.array(kept)], np.array(kept)

    G_tr, y_tr, idx_tr = take(splits["train"])
    G_va, y_va, idx_va = take(splits["valid"])
    G_te, y_te, idx_te = take(splits["test"])
    log.info("tr=%d va=%d te=%d", len(G_tr), len(G_va), len(G_te))

    # --- Step 1: GIN teacher ---
    log.info("Training GIN teacher (80 epochs, lr=1e-3)...")
    t0 = time.time()
    teacher, soft_train, hard_train, val_au = train_teacher(
        G_tr, y_tr, epochs=80, lr=1e-3, batch_size=32, seed=42,
        n_atom_types=20, n_bond_types=5, hidden_dim=32,
    )
    log.info("GIN teacher: val_auroc=%.3f train_wall=%.1fs", val_au, time.time() - t0)

    # GIN test eval
    import torch
    from torch_geometric.loader import DataLoader as PyGLoader
    from graphtm.distill.teacher import graphs_to_pyg_list
    teacher.eval()
    device = next(teacher.parameters()).device
    test_pyg = graphs_to_pyg_list(G_te, n_atom_types=20, n_bond_types=5)
    loader = PyGLoader(test_pyg, batch_size=64, shuffle=False)
    test_soft = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = teacher(batch)
            test_soft.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    test_soft = np.concatenate(test_soft)
    gin_test_au = auroc(y_te, test_soft)
    gin_test_acc = float(((test_soft >= 0.5).astype(np.int32) == y_te).mean())
    log.info("GIN TEST: acc=%.3f auroc=%.3f", gin_test_acc, gin_test_au)

    # --- Step 2: K=5 HGTM ensemble on TEACHER hard labels ---
    log.info("Training K=%d HGTM ensemble on teacher hard labels...", len(SEEDS))
    member_scores = []
    member_summaries = []
    for k_idx, seed in enumerate(SEEDS):
        log.info("--- Distilled member %d / %d (seed=%d) ---",
                 k_idx + 1, len(SEEDS), seed)
        spec = HGraphTMSpec(
            n_classes=2, n_clauses=n_clauses, threshold=T, s=s_param, n_states=100,
            R=2, IA=2, IF=8, LA=15, LF=2,
            D_bits=D_bits, n_atom_types=20, n_bond_types=5, k_hop=k_hop,
            max_nodes=max_nodes, seed=seed,
        )
        model = HierarchicalGraphTM(spec)
        best_au = -1.0; best_state = None; best_epoch = 0
        t_start = time.time()
        for ep in range(1, epochs + 1):
            t0 = time.time()
            model.fit(G_tr, hard_train.astype(np.int64), epochs=1)
            sv = model.class_scores(G_va)
            mg = sv[:, 1].astype(np.float64) - sv[:, 0].astype(np.float64)
            au_v = auroc(y_va, mg)   # eval on lab labels
            log.info("  seed=%d ep=%2d val_auroc=%.3f wall=%.1fs",
                     seed, ep, au_v, time.time() - t0)
            if au_v > best_au:
                best_au = au_v; best_epoch = ep
                best_state = model._ta_state.copy_to_host().copy()
        log.info("  seed=%d best ep=%d val_auroc=%.3f", seed, best_epoch, best_au)
        model._ta_state.copy_from_host(best_state)
        st = model.class_scores(G_te)
        mg = st[:, 1].astype(np.float64) - st[:, 0].astype(np.float64)
        au_t = auroc(y_te, mg)
        acc_t = float(((mg > 0).astype(np.int32) == y_te).mean())
        log.info("  seed=%d TEST acc=%.3f auroc=%.3f", seed, acc_t, au_t)
        member_scores.append(st.copy())
        np.save(results_dir / f"hgtm_ames_distilled_seed{seed}.npy", best_state)
        member_summaries.append({
            "seed": seed, "best_epoch": best_epoch, "best_valid_auroc": best_au,
            "test_acc": acc_t, "test_auroc": au_t,
            "wall_min": (time.time() - t_start) / 60,
        })

    # --- Step 3: Ensemble ---
    stacked = np.stack(member_scores, axis=0)
    ens_sum = stacked.sum(axis=0)
    mg_ens = ens_sum[:, 1].astype(np.float64) - ens_sum[:, 0].astype(np.float64)
    pr_ens = (mg_ens > 0).astype(np.int32)
    acc_ens = float((pr_ens == y_te).mean())
    au_ens = auroc(y_te, mg_ens)
    log.info("DISTILLED ENSEMBLE soft-sum: acc=%.3f auroc=%.3f", acc_ens, au_ens)

    out = {
        "config": {"K": len(SEEDS), "seeds": SEEDS, "epochs": epochs,
                   "n_clauses": n_clauses, "T": T, "s": s_param,
                   "D_bits": D_bits, "max_nodes": max_nodes},
        "gin_teacher": {"test_acc": gin_test_acc, "test_auroc": gin_test_au,
                         "val_auroc": val_au},
        "members": member_summaries,
        "ensemble_soft_sum": {"test_acc": acc_ens, "test_auroc": au_ens},
        "n_train": len(G_tr), "n_valid": len(G_va), "n_test": len(G_te),
    }
    p = results_dir / f"distill_ensemble_ames_{int(time.time())}.json"
    p.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", p)


if __name__ == "__main__":
    main()
