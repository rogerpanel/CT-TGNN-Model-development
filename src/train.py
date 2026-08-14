#!/usr/bin/env python3
"""
Train and evaluate. One model, one seed, one dataset per invocation.

Every run writes, into runs/<tag>/:
    scores_test.npz    y_true, y_score  <- the ONLY source of reported metrics
    metrics.json       computed from those arrays, nothing hand-entered
    manifest.json      seed, config, git commit, hardware, library versions
    curve.csv          per-epoch train/val loss

There is no code path that emits a metric not derived from scores_test.npz.
That property is the whole point of this file.

Usage:
    python -m src.train --config configs/default.yaml --model ct_tgnn --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml

from src.data.loader import prepare
from src.data.graph import lift_node_labels, adjacency_at
from src.models.ct_tgnn import CTTGNN, HAS_TORCHDIFFEQ
from src.models.baselines.models import build_baseline, EdgeBank


# ---------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unavailable"


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torchdiffeq": HAS_TORCHDIFFEQ,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        info["cpu_count"] = os.cpu_count()
    return info


# ---------------------------------------------------------------------
def make_windows(g, window: float, max_windows: int | None,
                 min_edges: int = 1):
    """Slice a temporal graph into evaluation windows."""
    y, edge_groups, t_ends = lift_node_labels(g, window=window)
    out = []
    for w, ks in enumerate(edge_groups):
        if len(ks) < min_edges:
            continue
        out.append({"w": w, "edges": ks, "t_end": float(t_ends[w]),
                    "y": y[w]})
        if max_windows and len(out) >= max_windows:
            break
    return out


def window_tensors(g, win, device):
    ks = win["edges"]
    ei = torch.tensor(
        np.stack([g.interaction_src[ks], g.interaction_dst[ks]]),
        dtype=torch.long, device=device)
    ea = torch.tensor(g.edge_features[ks], dtype=torch.float32, device=device)
    et = torch.tensor(g.interaction_times[ks] - g.interaction_times[ks].min(),
                      dtype=torch.float32, device=device)
    yy = torch.tensor(win["y"], dtype=torch.float32, device=device)
    return ei, ea, et, yy


def active_node_mask(g, win) -> np.ndarray:
    """Only score nodes that actually appear in the window."""
    ks = win["edges"]
    m = np.zeros(g.num_nodes, dtype=bool)
    m[g.interaction_src[ks]] = True
    m[g.interaction_dst[ks]] = True
    return m


# ---------------------------------------------------------------------
def run_epoch(model, g, windows, device, optimizer=None, pos_weight=None,
              t_span=(0.0, 1.0)):
    train = optimizer is not None
    model.train(train)
    tot, n = 0.0, 0
    for win in windows:
        ei, ea, et, yy = window_tensors(g, win, device)
        mask = torch.tensor(active_node_mask(g, win), device=device)
        if mask.sum() == 0:
            continue
        with torch.set_grad_enabled(train):
            out = model(g.num_nodes, ei, ea, edge_time=et, t_span=t_span)
            logits, yv = out["logits"][mask], yy[mask]
            if hasattr(model, "loss"):
                loss, _ = model.loss({"logits": logits, "intensity": None},
                                     yv, pos_weight=pos_weight)
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yv, pos_weight=pos_weight)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        tot += float(loss.detach()) * int(mask.sum())
        n += int(mask.sum())
    return tot / max(n, 1)


@torch.no_grad()
def collect_scores(model, g, windows, device, t_span=(0.0, 1.0)):
    """Return y_true, y_score, per-window latency (ms), mean NFE."""
    model.eval()
    ys, ss, lat, nfes = [], [], [], []
    for win in windows:
        ei, ea, et, yy = window_tensors(g, win, device)
        mask = torch.tensor(active_node_mask(g, win), device=device)
        if mask.sum() == 0:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(g.num_nodes, ei, ea, edge_time=et, t_span=t_span)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
        nfes.append(out.get("nfe", 0))
        ys.append(yy[mask].cpu().numpy())
        ss.append(torch.sigmoid(out["logits"][mask]).cpu().numpy())
    if not ys:
        raise RuntimeError("no scorable windows produced")
    return (np.concatenate(ys).astype(np.int64),
            np.concatenate(ss).astype(np.float32),
            np.array(lat), float(np.mean(nfes)) if nfes else 0.0)


def compute_metrics(y_true, y_score, latencies, threshold=0.5) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score,
                                 average_precision_score, confusion_matrix)
    y_pred = (y_score >= threshold).astype(int)
    m = {
        "n_samples": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        m["auroc"] = float(roc_auc_score(y_true, y_score))
        m["auprc"] = float(average_precision_score(y_true, y_score))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        m["fpr"] = float(fp / max(fp + tn, 1))
        m["tnr"] = float(tn / max(tn + fp, 1))
    else:
        m["auroc"] = m["auprc"] = m["fpr"] = float("nan")
        m["note"] = "test split contains a single class; AUROC undefined"
    if len(latencies):
        m.update({
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
        })
    return m


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="ct_tgnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.data_path:
        cfg["data"]["path"] = args.data_path
    if args.max_rows:
        cfg["data"]["max_rows"] = args.max_rows
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{args.model}_seed{args.seed}"
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== {tag} | device={device} ===")

    data = prepare(
        cfg["data"]["path"],
        cols_override=cfg["data"].get("columns"),
        max_rows=cfg["data"].get("max_rows"),
        splits=tuple(cfg["data"].get("splits", (0.7, 0.15, 0.15))),
        tau_idle=cfg["data"].get("tau_idle", 15.0),
        tau_active=cfg["data"].get("tau_active", 1800.0),
        max_features=cfg["data"].get("max_features", 64),
    )
    meta = data["meta"]
    F_dim = meta["edge_feat_dim"]
    win_s = cfg["data"].get("window_seconds", 60.0)
    mw = cfg["data"].get("max_windows")
    wins = {k: make_windows(data[k], win_s, mw)
            for k in ("train", "val", "test")}
    print({k: len(v) for k, v in wins.items()})
    if not wins["test"]:
        raise RuntimeError("no test windows; lower window_seconds or "
                           "raise max_rows")

    # --- EdgeBank is fitted, not trained -----------------------------
    if args.model == "edgebank":
        eb = EdgeBank(window=cfg["train"].get("edgebank_window")).fit(
            data["train"])
        ys, ss = [], []
        for w in wins["test"]:
            m = active_node_mask(data["test"], w)
            idx = np.flatnonzero(m)
            ys.append(w["y"][idx])
            ss.append(eb.predict_nodes(idx, w["t_end"]))
        y_true, y_score = np.concatenate(ys), np.concatenate(ss)
        latencies, nfe, curve = np.array([]), 0.0, []
    else:
        mcfg = cfg["model"]
        if args.model == "ct_tgnn":
            model = CTTGNN(
                edge_feat_dim=F_dim,
                hidden_dim=mcfg.get("hidden_dim", 128),
                n_blocks=mcfg.get("n_blocks", 2),
                taus=tuple(mcfg.get("taus", [1e-6, 1e-3, 1.0, 3600.0])),
                dropout=mcfg.get("dropout", 0.1),
                solver=mcfg.get("solver", "dopri5"),
                rtol=mcfg.get("rtol", 1e-5), atol=mcfg.get("atol", 1e-5),
                use_adjoint=mcfg.get("use_adjoint", True),
                rk4_steps=mcfg.get("rk4_steps", 4),
                spectral_norm=mcfg.get("spectral_norm", True),
                point_process=mcfg.get("point_process", True),
            ).to(device)
        else:
            model = build_baseline(
                args.model, edge_feat_dim=F_dim,
                hidden_dim=mcfg.get("hidden_dim", 128),
                dropout=mcfg.get("dropout", 0.1)).to(device)

        n_par = sum(p.numel() for p in model.parameters())
        print(f"parameters: {n_par:,}")

        pw = None
        af = meta["attack_frac"]["train"]
        if 0 < af < 1 and cfg["train"].get("class_weighting", True):
            pw = torch.tensor([(1 - af) / af], device=device).clamp(max=50.0)

        opt = torch.optim.Adam(model.parameters(),
                               lr=cfg["train"].get("lr", 1e-3),
                               weight_decay=cfg["train"].get("wd", 1e-4))
        best, best_state, patience = float("inf"), None, 0
        curve = []
        for ep in range(cfg["train"].get("epochs", 20)):
            tr = run_epoch(model, data["train"], wins["train"], device,
                           opt, pw)
            va = run_epoch(model, data["val"], wins["val"], device, None, pw)
            curve.append({"epoch": ep, "train_loss": tr, "val_loss": va})
            print(f"  epoch {ep:3d}  train {tr:.4f}  val {va:.4f}")
            if va < best - 1e-5:
                best, patience = va, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg["train"].get("patience", 5):
                    print(f"  early stop at epoch {ep}")
                    break
        if best_state:
            model.load_state_dict(best_state)
        y_true, y_score, latencies, nfe = collect_scores(
            model, data["test"], wins["test"], device)
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    # --- persist ------------------------------------------------------
    np.savez(os.path.join(out_dir, "scores_test.npz"),
             y_true=y_true, y_score=y_score, seed=args.seed)
    metrics = compute_metrics(y_true, y_score, latencies)
    metrics.update({"model": args.model, "seed": args.seed,
                    "mean_nfe": nfe})
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({"tag": tag, "timestamp": datetime.now().isoformat(),
                   "git_commit": git_commit(), "config": cfg,
                   "seed": args.seed, "hardware": hardware(),
                   "data_meta": {k: v for k, v in meta.items()
                                 if k != "feature_names"},
                   "n_windows": {k: len(v) for k, v in wins.items()}},
                  fh, indent=2, default=str)
    if curve:
        import csv
        with open(os.path.join(out_dir, "curve.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss",
                                               "val_loss"])
            w.writeheader()
            w.writerows(curve)

    print("\n--- test metrics (from scores_test.npz) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
