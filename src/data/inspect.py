#!/usr/bin/env python3
"""
Dataset schema inspector.

RUN THIS FIRST. It reports the structure of the Kaggle dataset so the
loader can be configured against the real columns rather than guesses.

    python -m src.data.inspect --path /kaggle/input/integrated-idps-security-3datasets

It prints, for every tabular file found: shape, dtypes, null counts,
candidate columns for the fields the graph builder needs (source
identity, destination identity, timestamp, label), label balance, and
a small sample. It writes schema_report.json alongside.

Nothing here modifies data or produces results.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Column-name fragments commonly used across NIDS datasets. Matching is
# case-insensitive and on normalised names (non-alphanumerics stripped).
CANDIDATES = {
    "src_ip":    ["srcip", "sourceip", "sourceipaddress", "ipvsrcaddr",
                  "srcaddr", "source", "src", "origh", "idorigh"],
    "dst_ip":    ["dstip", "destinationip", "destip", "ipvdstaddr",
                  "dstaddr", "destination", "dst", "resph", "idresph"],
    "src_port":  ["srcport", "sourceport", "l4srcport", "origp", "idorigp"],
    "dst_port":  ["dstport", "destinationport", "l4dstport", "respp",
                  "idrespp"],
    "timestamp": ["timestamp", "ts", "stime", "starttime", "flowstart",
                  "time", "date", "datetime", "firstseen"],
    "duration":  ["dur", "duration", "flowduration", "tottime"],
    "label":     ["label", "attack", "attackcat", "attackcategory",
                  "class", "target", "iscattack", "type", "category",
                  "malicious", "binarylabel"],
}


def norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def guess_roles(columns) -> dict:
    """Map semantic role -> list of matching column names, best first."""
    out = {}
    normed = {c: norm(c) for c in columns}
    for role, frags in CANDIDATES.items():
        hits = []
        for col, n in normed.items():
            for rank, frag in enumerate(frags):
                if n == frag:
                    hits.append((0, rank, col))
                    break
                if frag in n:
                    hits.append((1, rank, col))
                    break
        hits.sort()
        out[role] = [c for _, _, c in hits]
    return out


def describe_file(path: str, nrows: int) -> dict:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".csv", ".txt"):
            df = pd.read_csv(path, nrows=nrows, low_memory=False)
            total = None  # counting rows is expensive; done lazily below
        elif ext == ".parquet":
            df = pd.read_parquet(path)
            total = len(df)
            df = df.head(nrows)
        else:
            return {"path": path, "skipped": f"unsupported extension {ext}"}
    except Exception as e:
        return {"path": path, "error": f"{type(e).__name__}: {e}"}

    df.columns = [str(c).strip() for c in df.columns]
    roles = guess_roles(df.columns)

    info = {
        "path": path,
        "size_mb": round(os.path.getsize(path) / 1e6, 2),
        "rows_sampled": int(len(df)),
        "rows_total": total,
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_frac": {c: round(float(df[c].isna().mean()), 4)
                      for c in df.columns},
        "role_candidates": roles,
    }

    # Label balance, if a label column is identifiable
    if roles["label"]:
        lc = roles["label"][0]
        vc = df[lc].value_counts(dropna=False).head(20)
        info["label_column"] = lc
        info["label_counts"] = {str(k): int(v) for k, v in vc.items()}
        info["label_n_unique"] = int(df[lc].nunique(dropna=False))

    # Timestamp sanity: monotonic? parseable?
    if roles["timestamp"]:
        tc = roles["timestamp"][0]
        info["timestamp_column"] = tc
        col = df[tc]
        info["timestamp_dtype"] = str(col.dtype)
        info["timestamp_sample"] = [str(v) for v in col.head(3).tolist()]
        try:
            if np.issubdtype(col.dtype, np.number):
                parsed = pd.to_numeric(col, errors="coerce")
            else:
                parsed = pd.to_datetime(col, errors="coerce",
                                        format="mixed").astype("int64") / 1e9
            parsed = parsed.dropna()
            info["timestamp_parseable_frac"] = round(
                float(len(parsed) / max(len(col), 1)), 4)
            if len(parsed) > 1:
                info["timestamp_monotonic"] = bool(
                    parsed.is_monotonic_increasing)
                info["timestamp_span_s"] = float(parsed.max() - parsed.min())
        except Exception as e:
            info["timestamp_parse_error"] = str(e)[:200]

    info["head"] = df.head(3).astype(str).to_dict(orient="records")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True,
                    help="Dataset root (e.g. the kagglehub download path)")
    ap.add_argument("--nrows", type=int, default=5000)
    ap.add_argument("--out", default="schema_report.json")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"Path does not exist: {args.path}", file=sys.stderr)
        return 2

    files = []
    for ext in ("csv", "parquet", "txt"):
        files += glob.glob(os.path.join(args.path, "**", f"*.{ext}"),
                           recursive=True)
    files = sorted(set(files))

    if not files:
        print(f"No tabular files under {args.path}. Contents:")
        for root, dirs, fs in os.walk(args.path):
            for f in fs[:50]:
                print("   ", os.path.join(root, f))
        return 3

    print(f"Found {len(files)} tabular file(s) under {args.path}\n")
    report = []
    for p in files:
        info = describe_file(p, args.nrows)
        report.append(info)

        rel = os.path.relpath(p, args.path)
        print("=" * 72)
        print(f"{rel}   ({info.get('size_mb', '?')} MB)")
        print("=" * 72)
        if "error" in info or "skipped" in info:
            print("  ", info.get("error") or info.get("skipped"))
            continue
        print(f"  columns ({info['n_columns']}): "
              f"{', '.join(info['columns'][:14])}"
              f"{' ...' if info['n_columns'] > 14 else ''}")
        print("  detected roles:")
        for role, hits in info["role_candidates"].items():
            mark = "  " if hits else "!!"
            print(f"    {mark} {role:<10} -> "
                  f"{hits[0] if hits else 'NOT FOUND'}"
                  f"{'  (alts: ' + ', '.join(hits[1:4]) + ')' if len(hits) > 1 else ''}")
        if "label_counts" in info:
            print(f"  label '{info['label_column']}' "
                  f"({info['label_n_unique']} unique): "
                  f"{info['label_counts']}")
        if "timestamp_column" in info:
            print(f"  timestamp '{info['timestamp_column']}' "
                  f"dtype={info['timestamp_dtype']} "
                  f"parseable={info.get('timestamp_parseable_frac')} "
                  f"monotonic={info.get('timestamp_monotonic')} "
                  f"span={info.get('timestamp_span_s')}s")
            print(f"     sample: {info.get('timestamp_sample')}")
        print()

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"Wrote {args.out}")

    missing = [i for i in report
               if not i.get("role_candidates", {}).get("timestamp")
               or not i.get("role_candidates", {}).get("label")]
    if missing:
        print("\nNOTE: some files lack an auto-detected timestamp or label "
              "column. Set them explicitly in configs/default.yaml under "
              "data.columns before training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
