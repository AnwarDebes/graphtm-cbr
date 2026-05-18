"""Full TDC AMES training: Morgan-FP RF baseline + GIN teacher + HGTM student.

Saves all artefacts under results/.
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


# logging
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)
log_path = results_dir / f"train_full_ames_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# helpers
def auroc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    pos2 = pos[:, None]; neg2 = neg[None, :]
    return float(((pos2 > neg2).sum() + 0.5 * (pos2 == neg2).sum()) / (len(pos) * len(neg)))


# data
def load_and_encode(D_bits: int, k_hop: int, max_nodes: int):
    log.info("Loading TDC AMES (Hansen 2009)...")
    mols, y, meta = load_tdc_ames(split="scaffold", seed=42, encode=False)
    splits = meta["split_indices"]
    log.info("n_mols=%d y_pos=%.3f splits tr=%d va=%d te=%d",
             len(mols), y.mean(), len(splits["train"]), len(splits["valid"]), len(splits["test"]))

    log.info("Encoding (BSC D=%d k_hop=%d max_nodes=%d sparsity=0.30)...", D_bits, k_hop, max_nodes)
    cb = make_codebook(n_atom_types=20, n_bond_types=5, k_hop=k_hop,
                       D=D_bits, sparsity=0.30, seed=42)
    t0 = time.time()
    encoded = {}
    dropped_too_big = 0
    dropped_failed = 0
    for i, m in enumerate(mols):
        try:
            g = encode_graph(m, cb, k_hop=k_hop)
            if g.n_nodes <= max_nodes:
                encoded[i] = g
            else:
                dropped_too_big += 1
        except Exception:
            dropped_failed += 1
    log.info("encoded=%d dropped_big=%d dropped_failed=%d (%.1fs)",
             len(encoded), dropped_too_big, dropped_failed, time.time() - t0)

    def take(indices):
        kept = [int(i) for i in indices if int(i) in encoded]
        return ([encoded[i] for i in kept], y[np.array(kept)], np.array(kept), [mols[i] for i in kept])

    G_tr, y_tr, idx_tr, mols_tr = take(splits["train"])
    G_va, y_va, idx_va, mols_va = take(splits["valid"])
    G_te, y_te, idx_te, mols_te = take(splits["test"])
    log.info("After encoding: tr=%d va=%d te=%d", len(G_tr), len(G_va), len(G_te))
    return cb, (G_tr, y_tr, mols_tr), (G_va, y_va, mols_va), (G_te, y_te, mols_te)


# baseline: Morgan-FP + RandomForest
def morgan_rf_baseline(mols_tr, y_tr, mols_te, y_te):
    log.info("Morgan-FP + RF baseline...")
    from rdkit.Chem import AllChem
    from sklearn.ensemble import RandomForestClassifier

    def fp(m, n_bits=2048, radius=2):
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        return np.frombuffer(bv.ToBitString().encode(), 'u1') - ord('0')

    X_tr = np.array([fp(m) for m in mols_tr], dtype=np.uint8)
    X_te = np.array([fp(m) for m in mols_te], dtype=np.uint8)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    proba = rf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(np.int32)
    out = {
        "acc": float((pred == y_te).mean()),
        "auroc": auroc(y_te, proba),
        "train_wall_s": time.time() - t0,
    }
    log.info("Morgan-RF baseline: acc=%.3f auroc=%.3f (%.1fs)",
             out["acc"], out["auroc"], out["train_wall_s"])
    return out


# HGTM student
def train_hgtm(G_tr, y_tr, G_va, y_va, *, D_bits: int, max_nodes: int,
                n_clauses: int, T: int, s: float, epochs: int):
    spec = HGraphTMSpec(
        n_classes=2,
        n_clauses=n_clauses,
        threshold=T,
        s=s,
        n_states=100,
        R=2, IA=2, IF=8, LA=15, LF=2,            # FEATURES=32, literals/clause=1920
        D_bits=D_bits,
        n_atom_types=20, n_bond_types=5, k_hop=2,
        max_nodes=max_nodes, seed=42,
    )
    log.info("HGTM spec: C=%d T=%d s=%.2f FEATURES=%d literals/clause=%d D_bits=%d",
             spec.n_clauses, spec.threshold, spec.s,
             spec.R * spec.IF * spec.LF,
             spec.R * spec.IA * spec.IF * spec.LA * 2 * spec.LF,
             spec.D_bits)

    m = HierarchicalGraphTM(spec)
    log.info("Training HGTM: tr=%d va=%d epochs=%d", len(G_tr), len(G_va), epochs)

    curve = []
    t_total = time.time()
    best_au = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        m.fit(G_tr, y_tr, epochs=1)
        dt = time.time() - t0

        scores_va = m.class_scores(G_va)
        margin_va = scores_va[:, 1].astype(np.float64) - scores_va[:, 0].astype(np.float64)
        pred_va = (margin_va > 0).astype(np.int32)
        acc_va = float((pred_va == y_va).mean())
        au_va = auroc(y_va, margin_va)
        curve.append({"epoch": epoch, "valid_acc": acc_va, "valid_auroc": au_va,
                      "epoch_wall_s": dt})
        log.info("epoch=%2d valid_acc=%.3f valid_auroc=%.3f wall=%.1fs",
                 epoch, acc_va, au_va, dt)
        if au_va > best_au:
            best_au = au_va
            best_epoch = epoch
            best_state = m._ta_state.copy_to_host().copy()
            log.info("  ** new best @ epoch %d auroc=%.3f", best_epoch, best_au)

    total_wall = time.time() - t_total
    log.info("HGTM total train wall: %.1f min", total_wall / 60)
    # Restore best-by-validation state for final test eval.
    if best_state is not None:
        m._ta_state.copy_from_host(best_state)
        log.info("Restored best-by-valid state from epoch %d (auroc=%.3f)", best_epoch, best_au)
    return m, spec, curve, total_wall, best_epoch, best_au


# main
def main():
    # config
    D_bits = 512
    k_hop = 2
    max_nodes = 60
    n_clauses = 2000
    T = 500
    s = 3.9
    epochs = 60

    cb, (G_tr, y_tr, mols_tr), (G_va, y_va, mols_va), (G_te, y_te, mols_te) = \
        load_and_encode(D_bits=D_bits, k_hop=k_hop, max_nodes=max_nodes)

    # Baseline
    baseline = morgan_rf_baseline(mols_tr, y_tr, mols_te, y_te)

    # HGTM student (direct on ground truth, distillation as next iteration)
    m, spec, curve, train_wall, best_epoch, best_au = train_hgtm(
        G_tr, y_tr, G_va, y_va,
        D_bits=D_bits, max_nodes=max_nodes,
        n_clauses=n_clauses, T=T, s=s, epochs=epochs,
    )

    # Final test eval
    log.info("Test eval...")
    scores_te = m.class_scores(G_te)
    margin_te = scores_te[:, 1].astype(np.float64) - scores_te[:, 0].astype(np.float64)
    pred_te = (margin_te > 0).astype(np.int32)
    hgtm_acc = float((pred_te == y_te).mean())
    hgtm_auroc = auroc(y_te, margin_te)
    log.info("HGTM TEST: acc=%.3f auroc=%.3f", hgtm_acc, hgtm_auroc)
    log.info("Morgan-FP RF baseline: acc=%.3f auroc=%.3f", baseline["acc"], baseline["auroc"])
    log.info("GAP (HGTM - Morgan-RF) AUROC: %.3f", hgtm_auroc - baseline["auroc"])

    # save
    out = {
        "config": {
            "D_bits": D_bits, "k_hop": k_hop, "max_nodes": max_nodes,
            "n_clauses": n_clauses, "T": T, "s": s, "epochs": epochs,
            "FEATURES": spec.R * spec.IF * spec.LF,
            "literals_per_clause": spec.R * spec.IA * spec.IF * spec.LA * 2 * spec.LF,
        },
        "morgan_rf_baseline": baseline,
        "hgtm": {
            "test_acc": hgtm_acc,
            "test_auroc": hgtm_auroc,
            "train_wall_s": train_wall,
            "best_epoch": best_epoch,
            "best_valid_auroc": best_au,
            "epoch_curve": curve,
        },
        "n_train": len(G_tr), "n_valid": len(G_va), "n_test": len(G_te),
    }
    out_json = results_dir / f"train_full_ames_{int(time.time())}.json"
    out_json.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_json)

    # Also save the trained HGTM TA state for later recourse evaluation
    ta_host = m._ta_state.copy_to_host()
    ta_path = results_dir / "hgtm_ames_final.npy"
    np.save(ta_path, ta_host)
    log.info("Wrote %s (shape=%s)", ta_path, ta_host.shape)


if __name__ == "__main__":
    main()
