"""Generate publication figures from results JSONs.

Outputs to paper/figures/ in both PDF + PNG.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/home/anward/project/graphtm-cbr")
sys.path.insert(0, str(PROJECT_ROOT))

RES = PROJECT_ROOT / "results"
FIGDIR = PROJECT_ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote paper/figures/{name}.{{pdf,png}}")


def _latest(glob_pat):
    paths = sorted(RES.glob(glob_pat))
    return paths[-1] if paths else None


# Fig 1: training curves (direct + distilled, all 5 seeds)
def fig_training_curves():
    direct = json.loads(_latest("ensemble_ames_*.json").read_text())
    distilled = json.loads(_latest("distill_ensemble_ames_*.json").read_text())
    # NOTE: ensemble script writes "members" with best_epoch + best_valid_auroc,
    # but not the full epoch curve. I log-scrape from the .log instead.
    direct_log = _latest("ensemble_run.log") or _latest("ensemble_ames_*.log")
    distill_log = _latest("distill_run.log") or _latest("distill_ensemble_ames_*.log")

    def parse_log(path):
        out = {}     # seed -> list of (epoch, val_auroc)
        cur_seed = None
        if path is None or not path.exists():
            return out
        for line in path.read_text().splitlines():
            if "Member" in line or "member " in line.lower() or "seed=" in line and "ep=" not in line and "TEST" not in line:
                # 'Member 1 / 5 (seed=42)' or '--- Distilled member 1 / 5 (seed=42) ---'
                if "seed=" in line:
                    cur_seed = int(line.split("seed=")[1].split(")")[0].split()[0])
                    out.setdefault(cur_seed, [])
            elif "ep=" in line and "val_auroc" in line:
                try:
                    ep = int(line.split("ep=")[1].split()[0])
                    au = float(line.split("val_auroc=")[1].split()[0])
                    if cur_seed is not None:
                        out.setdefault(cur_seed, []).append((ep, au))
                except Exception:
                    pass
        return out

    direct_curves = parse_log(direct_log)
    distill_curves = parse_log(distill_log)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.85, 5))
    for ax, curves, title in [
        (axes[0], direct_curves, "Phase 1: direct labels"),
        (axes[1], distill_curves, "Phase 2: GIN-distilled"),
    ]:
        for i, (seed, pts) in enumerate(sorted(curves.items())):
            if not pts: continue
            eps, aus = zip(*pts)
            ax.plot(eps, aus, color=colors[i], alpha=0.75, lw=1.0, label=f"seed {seed}")
        ax.axhline(0.5, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("validation AUROC")
    fig.suptitle("Per-seed validation AUROC across epochs (TDC AMES)")
    _save(fig, "fig_training_curves")


# Fig 2: ensemble lift bar chart
def fig_ensemble_lift():
    direct = json.loads(_latest("ensemble_ames_*.json").read_text())
    distilled = json.loads(_latest("distill_ensemble_ames_*.json").read_text())

    # Per-seed AUROC pairs
    d_seeds = {m["seed"]: m["test_auroc"] for m in direct["members"]}
    s_seeds = {m["seed"]: m["test_auroc"] for m in distilled["members"]}
    seeds = sorted(d_seeds.keys())
    direct_au = [d_seeds[s] for s in seeds]
    distilled_au = [s_seeds[s] for s in seeds]

    x = np.arange(len(seeds))
    w = 0.4

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # left: per-seed
    axes[0].bar(x - w/2, direct_au, w, label="direct labels", color="#888")
    axes[0].bar(x + w/2, distilled_au, w, label="GIN-distilled", color="#1f77b4")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"s{s}" for s in seeds])
    axes[0].set_ylabel("test AUROC")
    axes[0].set_title("Per-seed test AUROC (TDC AMES)")
    axes[0].set_ylim(0.55, 0.82)
    axes[0].axhline(0.790, color="r", lw=1, ls="--", label="Morgan-FP RF baseline")
    axes[0].axhline(0.796, color="darkgreen", lw=1, ls=":", label="GIN teacher")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(axis="y", alpha=0.3)

    # right: ensemble metrics
    metrics = ["AUROC", "Accuracy", "Recourse"]
    direct_vals = [direct["ensemble_soft_sum"]["test_auroc"],
                   direct["ensemble_soft_sum"]["test_acc"],
                   0.545]
    dist_vals = [distilled["ensemble_soft_sum"]["test_auroc"],
                 distilled["ensemble_soft_sum"]["test_acc"],
                 0.955]
    x2 = np.arange(len(metrics))
    axes[1].bar(x2 - w/2, direct_vals, w, label="direct ensemble", color="#888")
    axes[1].bar(x2 + w/2, dist_vals, w, label="distilled ensemble", color="#1f77b4")
    axes[1].set_xticks(x2); axes[1].set_xticklabels(metrics)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Ensemble metrics: direct vs distilled")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(axis="y", alpha=0.3)
    for xi, (a, b) in enumerate(zip(direct_vals, dist_vals)):
        axes[1].text(xi - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        axes[1].text(xi + w/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)

    fig.suptitle("Distillation lift: ensemble metrics (TDC AMES)")
    _save(fig, "fig_ensemble_lift")


# Fig 3: recourse latency histogram
def fig_recourse_latency():
    direct = json.loads(_latest("eval_recourse_ensemble_*.json").read_text())
    distilled = json.loads(_latest("eval_recourse_distilled_*.json").read_text())

    fig, ax = plt.subplots(figsize=(7, 4))
    # Plot percentile bars
    pcts = ["p50", "p95", "p99"]
    d_vals = [direct["latency_ms_p50"], direct["latency_ms_p95"], 12687]
    s_vals = [distilled["latency_ms_p50"], distilled["latency_ms_p95"], 9254]
    x = np.arange(3); w = 0.4
    ax.bar(x - w/2, d_vals, w, label=f"direct (success={direct['recourse_success_rate']*100:.1f}%)",
           color="#888")
    ax.bar(x + w/2, s_vals, w, label=f"distilled (success={distilled['recourse_success_rate']*100:.1f}%)",
           color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(pcts)
    ax.set_ylabel("latency (ms)")
    ax.set_title("Counterfactual recourse latency on TDC AMES (per molecule, ensemble-of-5)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, (a, b) in enumerate(zip(d_vals, s_vals)):
        ax.text(xi - w/2, a + 200, f"{int(a)}", ha="center", fontsize=8)
        ax.text(xi + w/2, b + 200, f"{int(b)}", ha="center", fontsize=8)
    _save(fig, "fig_recourse_latency")


# Fig 4: Kazius coverage
def fig_kazius():
    d = _latest("kazius_coverage_seed42_*.json")
    s = _latest("kazius_coverage_distilled_seed*.json")
    if d is None or s is None:
        print("kazius JSONs missing, skip"); return
    direct = json.loads(d.read_text())
    distilled = json.loads(s.read_text())

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Total Kazius alerts", "Present in test set", "Covered by ≥1 clause"]
    direct_vals = [direct["n_alerts_total"], direct["n_alerts_present_in_test"],
                   direct["n_alerts_covered"]]
    dist_vals = [distilled["n_alerts_total"], distilled["n_alerts_present_in_test"],
                 distilled["n_alerts_covered"]]
    x = np.arange(len(labels)); w = 0.4
    ax.bar(x - w/2, direct_vals, w, label="direct (seed 42)", color="#888")
    ax.bar(x + w/2, dist_vals, w, label="distilled (seed 43)", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("count")
    ax.set_title("Kazius toxicophore coverage (TDC AMES test set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, (a, b) in enumerate(zip(direct_vals, dist_vals)):
        ax.text(xi - w/2, a + 0.3, str(a), ha="center", fontsize=9)
        ax.text(xi + w/2, b + 0.3, str(b), ha="center", fontsize=9)
    _save(fig, "fig_kazius_coverage")


if __name__ == "__main__":
    fig_training_curves()
    fig_ensemble_lift()
    fig_recourse_latency()
    fig_kazius()
    print("All figures generated.")
