"""End-to-end AMES smoke (subsampled for first run)."""
import time
import numpy as np
import sys
sys.path.insert(0, "/home/anward/project/graphtm-cbr")
from graphtm.data.ames import load_tdc_ames
from graphtm.encoding.codebook import make_codebook
from graphtm.encoding.graph_features import encode_graph
from graphtm.core.hierarchical_graph_tm import HierarchicalGraphTM, HGraphTMSpec


def auroc(y_true, y_score):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    pos2 = pos[:, None]; neg2 = neg[None, :]
    return float(((pos2 > neg2).sum() + 0.5 * (pos2 == neg2).sum()) / (len(pos) * len(neg)))


def main():
    print("[1/4] Load AMES")
    mols, y, meta = load_tdc_ames(split="scaffold", seed=42, encode=False)
    splits = meta["split_indices"]
    tr_idx = splits["train"]
    te_idx = splits["test"]
    n_train_target = 500
    n_test_target = 200
    rng = np.random.default_rng(42)
    tr_sub = rng.choice(tr_idx, size=min(n_train_target, len(tr_idx)), replace=False)
    te_sub = rng.choice(te_idx, size=min(n_test_target, len(te_idx)), replace=False)
    keep_idx = np.concatenate([tr_sub, te_sub])
    print(f"   total={len(mols)} subsampled train={len(tr_sub)} test={len(te_sub)}")

    print("[2/4] Encode (BSC d=256, k_hop=2)")
    cb = make_codebook(n_atom_types=20, n_bond_types=5, k_hop=2, D=256, sparsity=0.10, seed=42)
    t0 = time.time()
    encoded = {}
    for i in keep_idx:
        try:
            g = encode_graph(mols[int(i)], cb, k_hop=2)
            if g.n_nodes <= 60:
                encoded[int(i)] = g
        except Exception:
            pass
    tr_keep = [int(i) for i in tr_sub if int(i) in encoded]
    te_keep = [int(i) for i in te_sub if int(i) in encoded]
    G_tr = [encoded[i] for i in tr_keep]
    G_te = [encoded[i] for i in te_keep]
    y_tr = y[np.array(tr_keep)]
    y_te = y[np.array(te_keep)]
    print(f"   encoded train={len(G_tr)} test={len(G_te)} y_tr_pos={y_tr.mean():.3f} time={time.time()-t0:.1f}s")

    print("[3/4] Train HGTM (200 clauses, T=200, s=3.9, 10 epochs)")
    spec = HGraphTMSpec(
        n_classes=2, n_clauses=200, threshold=200, s=3.9, n_states=100,
        R=2, IA=2, IF=5, LA=15, LF=3,
        D_bits=256, n_atom_types=20, n_bond_types=5, k_hop=2, max_nodes=60, seed=42,
    )
    m = HierarchicalGraphTM(spec)
    t0 = time.time()
    m.fit(G_tr, y_tr, epochs=10)
    dt = time.time() - t0
    print(f"   train wall={dt:.1f}s ({dt*1000/(len(G_tr)*10):.1f} ms/sample/epoch)")

    print("[4/4] Eval")
    pred_te = m.predict(G_te)
    acc = (pred_te == y_te).mean()
    scores = m.class_scores(G_te)
    margin = scores[:, 1].astype(np.float64) - scores[:, 0].astype(np.float64)
    au = auroc(y_te, margin)
    print(f"   test_acc={acc:.3f} test_AUROC={au:.3f}")
    print(f"   class_sum sample: {scores[:3].tolist()}")
    print(f"   baseline (predict majority): {(y_te == int(y_te.mean() >= 0.5)).mean():.3f}")


if __name__ == "__main__":
    main()
