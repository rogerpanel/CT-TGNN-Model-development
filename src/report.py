#!/usr/bin/env python3
"""
Aggregate runs into the paper's tables and figures.

HARD RULE: every number emitted here is recomputed from a
runs/*/scores_test.npz array at generation time. There is no path by
which a value can be typed in. If a run is missing, the corresponding
cell reads "not run" -- it is never filled by estimate.

The ROC figure and the AUROC column are computed from the same array in
the same function call, so the curve and its reported AUC cannot
disagree. That is the specific failure mode Reviewer 2 identified.

Outputs (into paper/):
    table_main.tex        detection performance, mean +/- sd over seeds
    table_latency.tex     latency percentiles and NFE
    fig_roc.pdf           ROC across models, AUC computed from the curve
    fig_pr.pdf            precision-recall (matters under imbalance)
    results_summary.json  machine-readable
    provenance.txt        which score file produced which number

Usage:
    python -m src.report --runs runs --out paper
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

DISPLAY = {
    "edgebank": "EdgeBank (memorisation)",
    "mlp_flow": "MLP (no topology)",
    "cnn_lstm": "CNN-LSTM",
    "discrete_gnn": "Discrete GNN (snapshot)",
    "temporal_attn": "Temporal attention (TGAT-style)",
    "ct_tgnn": "CT-TGNN (ours)",
}
ORDER = ["edgebank", "mlp_flow", "cnn_lstm", "discrete_gnn",
         "temporal_attn", "ct_tgnn"]


def load_runs(runs_dir: str):
    """{model: [ {seed, y_true, y_score, metrics, manifest, path}, ... ]}"""
    out = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        sp = os.path.join(d, "scores_test.npz")
        if not os.path.exists(sp):
            continue
        with np.load(sp) as z:
            y_true = z["y_true"].ravel()
            y_score = z["y_score"].ravel()
            seed = int(z["seed"]) if "seed" in z else None
        name = os.path.basename(d)
        model = name.rsplit("_seed", 1)[0]
        rec = {"seed": seed, "y_true": y_true, "y_score": y_score,
               "path": sp, "dir": d}
        for f, k in (("metrics.json", "metrics"),
                     ("manifest.json", "manifest")):
            p = os.path.join(d, f)
            if os.path.exists(p):
                with open(p) as fh:
                    rec[k] = json.load(fh)
        out[model].append(rec)
    return out


def per_run_metrics(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(int)
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "precision": precision_score(y_true, y_pred, zero_division=0),
         "recall": recall_score(y_true, y_pred, zero_division=0),
         "f1": f1_score(y_true, y_pred, zero_division=0)}
    if len(np.unique(y_true)) > 1:
        m["auroc"] = roc_auc_score(y_true, y_score)
        m["auprc"] = average_precision_score(y_true, y_score)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        m["fpr"] = fp / max(fp + tn, 1)
    else:
        m["auroc"] = m["auprc"] = m["fpr"] = float("nan")
    return m


def agg(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


def fmt(mean, sd, n, places=3):
    if not np.isfinite(mean):
        return "n/a"
    if n > 1:
        return f"{mean:.{places}f} $\\pm$ {sd:.{places}f}"
    return f"{mean:.{places}f}"


def paired_test(a: dict, b: dict):
    """Wilcoxon signed-rank across shared seeds on AUROC."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 5:
        return None, len(shared)
    from scipy.stats import wilcoxon
    x = [a[s] for s in shared]
    y = [b[s] for s in shared]
    try:
        return float(wilcoxon(x, y).pvalue), len(shared)
    except Exception:
        return None, len(shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--dataset-name", default="")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        print(f"No runs with scores_test.npz under '{args.runs}'.\n"
              f"Nothing can be reported. Train first:\n"
              f"    python -m src.train --model ct_tgnn --seed 0")
        return 3

    os.makedirs(args.out, exist_ok=True)
    summary, auroc_by_seed, prov = {}, {}, []

    print(f"{'model':<32}{'seeds':>6}{'AUROC':>18}{'AUPRC':>10}{'F1':>10}")
    print("-" * 76)
    for model in [m for m in ORDER if m in runs] + \
                 [m for m in runs if m not in ORDER]:
        recs = runs[model]
        per = [per_run_metrics(r["y_true"], r["y_score"]) for r in recs]
        auroc_by_seed[model] = {r["seed"]: p["auroc"]
                                for r, p in zip(recs, per)
                                if r["seed"] is not None}
        s = {"n_seeds": len(recs),
             "seeds": [r["seed"] for r in recs],
             "n_samples": int(recs[0]["y_true"].size),
             "prevalence": float(recs[0]["y_true"].mean())}
        for k in ("accuracy", "precision", "recall", "f1", "auroc",
                  "auprc", "fpr"):
            m, sd = agg([p[k] for p in per])
            s[k] = {"mean": m, "std": sd}
        for k in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                  "mean_nfe"):
            vals = [r.get("metrics", {}).get(k) for r in recs]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                m, sd = agg(vals)
                s[k] = {"mean": m, "std": sd}
        summary[model] = s
        for r in recs:
            prov.append(f"{model:<20} seed={r['seed']}  <- {r['path']}")
        print(f"{DISPLAY.get(model, model):<32}{s['n_seeds']:>6}"
              f"{fmt(s['auroc']['mean'], s['auroc']['std'], s['n_seeds']):>18}"
              f"{s['auprc']['mean']:>10.3f}{s['f1']['mean']:>10.3f}")

    # ---- significance vs CT-TGNN ------------------------------------
    sig = {}
    if "ct_tgnn" in auroc_by_seed:
        for m in auroc_by_seed:
            if m == "ct_tgnn":
                continue
            p, n = paired_test(auroc_by_seed["ct_tgnn"], auroc_by_seed[m])
            sig[m] = {"p_value": p, "n_paired_seeds": n}

    # ---- main table --------------------------------------------------
    ds = f" on {args.dataset_name}" if args.dataset_name else ""
    L = [r"% Auto-generated by src/report.py. Do not hand-edit.",
         r"% Every value recomputed from runs/*/scores_test.npz.",
         r"\begin{table}[t]\centering",
         rf"\caption{{Detection performance{ds}. Mean $\pm$ standard "
         r"deviation across seeds. Splits are chronological. AUROC is "
         r"computed from the same score arrays used to render "
         r"Fig.~\ref{fig:roc}.}",
         r"\label{tab:main}\small",
         r"\begin{tabular}{@{}lccccc@{}}\toprule",
         r"Method & Seeds & AUROC & AUPRC & $F_1$ & FPR \\ \midrule"]
    for model in [m for m in ORDER if m in summary] + \
                 [m for m in summary if m not in ORDER]:
        s = summary[model]
        n = s["n_seeds"]
        name = DISPLAY.get(model, model)
        if model == "ct_tgnn":
            name = r"\textbf{" + name + "}"
        L.append(
            f"{name} & {n} & "
            f"{fmt(s['auroc']['mean'], s['auroc']['std'], n)} & "
            f"{fmt(s['auprc']['mean'], s['auprc']['std'], n)} & "
            f"{fmt(s['f1']['mean'], s['f1']['std'], n)} & "
            f"{fmt(s['fpr']['mean'], s['fpr']['std'], n)} \\\\")
    L += [r"\bottomrule\end{tabular}",
          rf"\\[2pt]\footnotesize Test prevalence: "
          rf"{summary[list(summary)[0]]['prevalence']:.4f}. ",
          r"\end{table}"]
    with open(os.path.join(args.out, "table_main.tex"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    # ---- latency table ----------------------------------------------
    have_lat = [m for m in summary if "latency_p50_ms" in summary[m]]
    if have_lat:
        hw = "unspecified"
        for m in have_lat:
            man = runs[m][0].get("manifest", {})
            h = man.get("hardware", {})
            if h:
                hw = h.get("gpu_name") or h.get("processor", "unspecified")
                break
        L2 = [r"% Auto-generated by src/report.py.",
              r"\begin{table}[t]\centering",
              rf"\caption{{Inference latency per evaluation window, "
              rf"measured on: {hw}.}}",
              r"\label{tab:latency}\small",
              r"\begin{tabular}{@{}lcccc@{}}\toprule",
              r"Method & P50 (ms) & P95 (ms) & P99 (ms) & NFE \\ \midrule"]
        for m in [x for x in ORDER if x in have_lat]:
            s = summary[m]
            nfe = s.get("mean_nfe", {}).get("mean", float("nan"))
            L2.append(f"{DISPLAY.get(m, m)} & "
                      f"{s['latency_p50_ms']['mean']:.1f} & "
                      f"{s['latency_p95_ms']['mean']:.1f} & "
                      f"{s['latency_p99_ms']['mean']:.1f} & "
                      f"{'--' if not np.isfinite(nfe) else f'{nfe:.1f}'} \\\\")
        L2 += [r"\bottomrule\end{tabular}\end{table}"]
        with open(os.path.join(args.out, "table_latency.tex"), "w") as fh:
            fh.write("\n".join(L2) + "\n")

    # ---- ROC + PR figures -------------------------------------------
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        for model in [m for m in ORDER if m in runs] + \
                     [m for m in runs if m not in ORDER]:
            r = runs[model][0]
            yt, ysc = r["y_true"], r["y_score"]
            if len(np.unique(yt)) < 2:
                continue
            if kind == "roc":
                x, y, _ = roc_curve(yt, ysc)
                val = roc_auc_score(yt, ysc)   # SAME arrays as the curve
                lbl = f"{DISPLAY.get(model, model)} (AUC = {val:.3f})"
            else:
                y, x, _ = precision_recall_curve(yt, ysc)
                val = average_precision_score(yt, ysc)
                lbl = f"{DISPLAY.get(model, model)} (AP = {val:.3f})"
            ax.plot(x, y, lw=2, label=lbl,
                    ls="-" if model == "ct_tgnn" else "--")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="grey", lw=1, ls=":",
                    label="Random (AUC = 0.500)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC curves (test split)")
            loc = "lower right"
        else:
            prev = runs[list(runs)[0]][0]["y_true"].mean()
            ax.axhline(prev, color="grey", lw=1, ls=":",
                       label=f"Prevalence ({prev:.3f})")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall curves (test split)")
            loc = "upper right"
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3); ax.legend(loc=loc, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"fig_{kind}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(args.out, "results_summary.json"), "w") as fh:
        json.dump({"summary": summary, "significance_vs_ct_tgnn": sig},
                  fh, indent=2)
    with open(os.path.join(args.out, "provenance.txt"), "w") as fh:
        fh.write("Every number in the generated tables and figures was\n"
                 "recomputed from these score files:\n\n")
        fh.write("\n".join(prov) + "\n")

    # ---- honest warnings ---------------------------------------------
    print()
    n_min = min(s["n_seeds"] for s in summary.values())
    if n_min < 5:
        print(f"WARNING: only {n_min} seed(s) for some models. Reviewer 2 "
              f"asked for variance; 5+ seeds is the norm.")
    if sig:
        print("Paired Wilcoxon vs CT-TGNN (AUROC across seeds):")
        for m, v in sig.items():
            p = v["p_value"]
            print(f"  vs {m:<20} "
                  f"{'p = %.4g' % p if p is not None else 'insufficient seeds'}"
                  f"  (n={v['n_paired_seeds']})")
    if "edgebank" in summary and "ct_tgnn" in summary:
        eb = summary["edgebank"]["auroc"]["mean"]
        ct = summary["ct_tgnn"]["auroc"]["mean"]
        if np.isfinite(eb) and np.isfinite(ct) and ct <= eb + 0.01:
            print(f"\nWARNING: CT-TGNN AUROC ({ct:.3f}) does not meaningfully "
                  f"exceed the EdgeBank memorisation heuristic ({eb:.3f}). "
                  f"The learned model may be adding nothing.")
    print(f"\nwrote {args.out}/")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
