"""
Real dataset loader.

Reads the tabular NIDS files, resolves the semantic columns the graph
builder needs, engineers edge features, and produces CHRONOLOGICAL
train/val/test splits.

Chronological splitting is mandatory, not stylistic: random shuffling of
temporally ordered security data is data snooping (Arp et al., USENIX
Security 2022, pitfall P3; Pendlebury et al., TESSERACT, constraint C1)
and inflates results. All training data precedes all test data here.

No synthetic fallback exists in this module. If the data cannot be read,
it raises. That is deliberate.
"""
from __future__ import annotations

import glob
import os
import warnings

import numpy as np
import pandas as pd

from .graph import build_temporal_graph

warnings.filterwarnings("ignore", category=FutureWarning)


BENIGN_TOKENS = {"benign", "normal", "background", "0", "false", "none", "-"}


def _norm(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def resolve_columns(df: pd.DataFrame, overrides: dict | None = None) -> dict:
    """Map semantic roles to actual column names."""
    from .inspect import guess_roles
    roles = guess_roles(df.columns)
    out = {r: (hits[0] if hits else None) for r, hits in roles.items()}
    if overrides:
        out.update({k: v for k, v in overrides.items() if v})
    missing = [r for r in ("timestamp", "label") if not out.get(r)]
    if missing:
        raise ValueError(
            f"Could not resolve required column(s) {missing}. "
            f"Set them explicitly in the config under data.columns. "
            f"Available columns: {list(df.columns)[:40]}"
        )
    return out


def parse_timestamp(col: pd.Series) -> np.ndarray:
    """Return float seconds. Handles epoch numerics and datetime strings."""
    if pd.api.types.is_numeric_dtype(col):
        v = pd.to_numeric(col, errors="coerce").astype("float64")
        # heuristics for ms / us epochs
        finite = v[np.isfinite(v)]
        if len(finite):
            m = float(finite.median())
            if m > 1e14:
                v = v / 1e6
            elif m > 1e11:
                v = v / 1e3
        return v.to_numpy()
    dt = pd.to_datetime(col, errors="coerce", format="mixed", utc=True)
    return (dt.astype("int64") / 1e9).to_numpy()


def binarize_label(col: pd.Series) -> np.ndarray:
    """1 = malicious, 0 = benign."""
    if pd.api.types.is_numeric_dtype(col):
        return (pd.to_numeric(col, errors="coerce").fillna(0)
                .astype(float) > 0).astype(np.int64).to_numpy()
    s = col.astype(str).str.strip().str.lower()
    return (~s.isin(BENIGN_TOKENS)).astype(np.int64).to_numpy()


def engineer_edge_features(df: pd.DataFrame, cols: dict,
                           max_features: int = 64) -> tuple[np.ndarray, list]:
    """
    Build the per-flow edge feature matrix from numeric columns, excluding
    identity and label columns (which would leak).

    Excluding identifiers matters: leaving raw IPs or ports in the feature
    matrix lets a model memorise which host is the attacker rather than
    learning attack behaviour (Arp et al., pitfall P4, spurious
    correlations).
    """
    exclude = {cols.get(k) for k in
               ("src_ip", "dst_ip", "label", "timestamp")} - {None}
    # ports are weak identifiers; keep them but they are often informative.
    numeric = [c for c in df.columns
               if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("no numeric feature columns found after exclusions")

    # rank by variance, keep the most informative
    sub = df[numeric].replace([np.inf, -np.inf], np.nan)
    var = sub.var(numeric_only=True).fillna(0.0).sort_values(ascending=False)
    keep = list(var.index[:max_features])

    X = sub[keep].fillna(0.0).to_numpy(dtype=np.float32)
    X = np.clip(X, -1e9, 1e9)
    return X, keep


def standardize(train: np.ndarray, *others: np.ndarray):
    """Fit on train only; apply to all. Prevents test-set leakage."""
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return tuple((a - mu) / sd for a in (train, *others))


def load_flow_table(path: str, cols_override=None, max_rows=None,
                    verbose=True) -> pd.DataFrame:
    files = []
    if os.path.isdir(path):
        for ext in ("csv", "parquet"):
            files += glob.glob(os.path.join(path, "**", f"*.{ext}"),
                               recursive=True)
    else:
        files = [path]
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no csv/parquet files under {path}")

    frames, budget = [], max_rows
    for f in files:
        if budget is not None and budget <= 0:
            break
        if f.endswith(".parquet"):
            d = pd.read_parquet(f)
            if budget:
                d = d.head(budget)
        else:
            d = pd.read_csv(f, nrows=budget, low_memory=False)
        d.columns = [str(c).strip() for c in d.columns]
        d["__source_file"] = os.path.basename(f)
        frames.append(d)
        if budget is not None:
            budget -= len(d)
        if verbose:
            print(f"  read {os.path.basename(f)}: {len(d):,} rows, "
                  f"{d.shape[1]} cols")

    df = pd.concat(frames, ignore_index=True, sort=False)
    if verbose:
        print(f"  total: {len(df):,} rows")
    return df


def prepare(path: str, cols_override: dict | None = None,
            max_rows: int | None = None,
            splits=(0.70, 0.15, 0.15),
            tau_idle: float = 15.0, tau_active: float = 1800.0,
            max_features: int = 64, verbose: bool = True):
    """
    Full pipeline: read -> resolve -> sort by time -> chronological split
    -> standardise (train-fit) -> build one TemporalGraph per split.

    Returns dict with keys 'train', 'val', 'test' plus 'meta'.
    """
    if verbose:
        print(f"Loading from {path}")
    df = load_flow_table(path, max_rows=max_rows, verbose=verbose)
    cols = resolve_columns(df, cols_override)
    if verbose:
        print(f"  resolved columns: {cols}")

    t = parse_timestamp(df[cols["timestamp"]])
    y = binarize_label(df[cols["label"]])

    ok = np.isfinite(t)
    if ok.sum() < len(t):
        if verbose:
            print(f"  dropping {(~ok).sum():,} rows with unparseable "
                  f"timestamps")
        df, t, y = df[ok].reset_index(drop=True), t[ok], y[ok]
    if len(df) == 0:
        raise ValueError("no rows remain after timestamp parsing")

    order = np.argsort(t, kind="stable")
    df, t, y = df.iloc[order].reset_index(drop=True), t[order], y[order]

    X, feat_names = engineer_edge_features(df, cols, max_features)

    # endpoint identities; fall back to a single-node graph only if the
    # dataset genuinely lacks endpoints (then the graph is degenerate and
    # we say so loudly rather than pretending otherwise)
    if cols.get("src_ip") and cols.get("dst_ip"):
        src = df[cols["src_ip"]].astype(str).to_numpy()
        dst = df[cols["dst_ip"]].astype(str).to_numpy()
        degenerate = False
    else:
        raise ValueError(
            "No source/destination identity columns found. A temporal GRAPH "
            "cannot be constructed without endpoints. Set data.columns.src_ip "
            "and data.columns.dst_ip in the config, or use a dataset that "
            "retains endpoint identifiers."
        )

    dur = (pd.to_numeric(df[cols["duration"]], errors="coerce").fillna(0.0)
           .to_numpy() if cols.get("duration") else np.zeros(len(df)))
    dur = np.clip(np.nan_to_num(dur), 0, 3600)
    t_end = t + dur

    n = len(df)
    i1, i2 = int(n * splits[0]), int(n * (splits[0] + splits[1]))
    idx = {"train": slice(0, i1), "val": slice(i1, i2), "test": slice(i2, n)}

    Xtr, Xva, Xte = standardize(X[idx["train"]], X[idx["val"]],
                                X[idx["test"]])
    Xs = {"train": Xtr, "val": Xva, "test": Xte}

    # shared node index map so indices are consistent across splits
    node_ids: dict = {}
    out = {}
    for name in ("train", "val", "test"):
        s = idx[name]
        g = build_temporal_graph(
            src[s], dst[s], t[s], t_end[s], Xs[name], y[s],
            tau_idle=tau_idle, tau_active=tau_active, node_ids=node_ids,
        )
        out[name] = g
        if verbose:
            print(f"  {name}: {g}")

    out["meta"] = {
        "columns": cols,
        "feature_names": feat_names,
        "edge_feat_dim": int(X.shape[1]),
        "n_nodes": len(node_ids),
        "split_boundaries_time": [float(t[i1 - 1]), float(t[i2 - 1])],
        "split_sizes": {k: int((idx[k].stop - idx[k].start))
                        for k in ("train", "val", "test")},
        "attack_frac": {k: float(y[idx[k]].mean())
                        for k in ("train", "val", "test")},
        "tau_idle": tau_idle, "tau_active": tau_active,
    }
    if verbose:
        print(f"  attack fraction: {out['meta']['attack_frac']}")
    return out
