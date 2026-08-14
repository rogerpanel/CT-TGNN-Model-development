#!/usr/bin/env python3
"""
Export the interaction stream in DyGLib/TGB format so TGN, TGAT, DyRep,
JODIE and GraphMixer can be run in their own reference harness under the
same chronological splits.

Running the reference implementations rather than reimplementing them is
what makes the comparison credible to a reviewer.

    python scripts/export_dyglib.py --config configs/default.yaml --out dyglib_export
"""
import argparse, os, sys
import numpy as np, pandas as pd, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.loader import prepare

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--out", default="dyglib_export")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
d = prepare(cfg["data"]["path"], cols_override=cfg["data"].get("columns"),
            max_rows=cfg["data"].get("max_rows"),
            splits=tuple(cfg["data"].get("splits", (.7, .15, .15))))
os.makedirs(a.out, exist_ok=True)

rows, feats = [], []
for name in ("train", "val", "test"):
    g = d[name]
    for k in range(len(g.interaction_times)):
        rows.append({"u": int(g.interaction_src[k]),
                     "i": int(g.interaction_dst[k]),
                     "ts": float(g.interaction_times[k]),
                     "label": int(g.flow_labels[k]), "split": name})
        feats.append(g.edge_features[k])
df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
df["idx"] = np.arange(len(df))
df.to_csv(os.path.join(a.out, "ml_ctgnn.csv"), index=False)
np.save(os.path.join(a.out, "ml_ctgnn.npy"), np.stack(feats).astype(np.float32))
np.save(os.path.join(a.out, "ml_ctgnn_node.npy"),
        np.zeros((d["meta"]["n_nodes"] + 1, 172), dtype=np.float32))
print(f"wrote {a.out}/ : {len(df):,} events, {d['meta']['n_nodes']} nodes")
print("Split boundaries (time):", d["meta"]["split_boundaries_time"])
print("\nNext: place these in DyGLib/processed_data/ctgnn/ and run e.g.\n"
      "  python train_link_prediction.py --dataset_name ctgnn --model_name TGN \\\n"
      "      --negative_sample_strategy historical --num_runs 5")
