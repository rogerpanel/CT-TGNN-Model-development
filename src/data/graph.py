"""
Continuous-time dynamic graph construction from flow records.

Implements Algorithm 1 of the manuscript: a single pass over the
time-ordered flow stream producing

  * a topology-event sequence  T = [(t, i, j, ADD|REMOVE), ...]
  * an interaction-event stream X = [(t, i, j, edge_features), ...]

Design points (these answer Reviewer 1, Comment 1 directly):

  * Repeated invocation between the same pair does NOT create new edges.
    It emits interaction events. The adjacency carries connection STATE;
    the point process carries USAGE. Keeping these separate is what
    preserves the piecewise-constant topology the hybrid ODE needs.

  * Edge lifetime is an idle timeout on last activity, with a maximum
    active lifetime, mirroring NetFlow/IPFIX expiry. A long-lived
    connection with continuing requests is refreshed, not re-created,
    so it generates no spurious topology jumps.

  * Node labels are lifted from flow labels over a short window, with
    NO k-hop propagation (which inflates metrics; see KAIROS critique).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

ADD, REMOVE = 1, -1


@dataclass
class TemporalGraph:
    """Container for a constructed continuous-time dynamic graph."""
    node_ids: dict            # original identity -> integer index
    topology_events: np.ndarray   # (M, 4) float: t, i, j, delta
    interaction_times: np.ndarray # (K,) float
    interaction_src: np.ndarray   # (K,) int
    interaction_dst: np.ndarray   # (K,) int
    edge_features: np.ndarray     # (K, F) float
    flow_labels: np.ndarray       # (K,) int
    meta: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    def __repr__(self):
        return (f"TemporalGraph(nodes={self.num_nodes}, "
                f"topology_events={len(self.topology_events)}, "
                f"interactions={len(self.interaction_times)}, "
                f"edge_feat_dim={self.edge_features.shape[1]}, "
                f"attack_frac={self.flow_labels.mean():.4f})")


def build_temporal_graph(
    src, dst, t_start, t_end, features, labels,
    tau_idle: float = 15.0,
    tau_active: float = 1800.0,
    node_ids: dict | None = None,
) -> TemporalGraph:
    """
    Algorithm 1. All input arrays must be sorted by t_start ascending.

    Args:
        src, dst:   array-like of hashable endpoint identities
        t_start:    (K,) float seconds
        t_end:      (K,) float seconds
        features:   (K, F) float per-flow edge features
        labels:     (K,) int flow labels (1 = malicious)
        tau_idle:   idle timeout; edge expires this long after last activity
        tau_active: max lifetime; edge expires and re-creates after this
        node_ids:   optional fixed identity->index map (for consistent
                    indexing across train/val/test splits)

    Returns:
        TemporalGraph
    """
    src = np.asarray(src)
    dst = np.asarray(dst)
    t_start = np.asarray(t_start, dtype=np.float64)
    t_end = np.asarray(t_end, dtype=np.float64)
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    K = len(t_start)
    if not (len(src) == len(dst) == len(t_end) == len(features) ==
            len(labels) == K):
        raise ValueError("all inputs must have the same length")
    if K and np.any(np.diff(t_start) < 0):
        raise ValueError("flows must be sorted by t_start ascending")

    # ---- node index map -------------------------------------------------
    if node_ids is None:
        node_ids = {}
        for v in np.concatenate([src, dst]):
            if v not in node_ids:
                node_ids[v] = len(node_ids)
    unknown = 0

    def idx(v):
        nonlocal unknown
        if v not in node_ids:
            unknown += 1
            node_ids[v] = len(node_ids)
        return node_ids[v]

    src_i = np.fromiter((idx(v) for v in src), dtype=np.int64, count=K)
    dst_i = np.fromiter((idx(v) for v in dst), dtype=np.int64, count=K)

    # ---- single pass ----------------------------------------------------
    active: dict[tuple[int, int], float] = {}   # (i,j) -> last activity
    born: dict[tuple[int, int], float] = {}     # (i,j) -> creation time
    expiry: list[tuple[float, int, int]] = []   # min-heap of scheduled expiry
    topo: list[tuple[float, int, int, int]] = []

    for k in range(K):
        ts, te = t_start[k], t_end[k]
        i, j = int(src_i[k]), int(dst_i[k])

        # process expiries that fall due before this flow starts
        while expiry and expiry[0][0] <= ts:
            t_exp, ei, ej = heapq.heappop(expiry)
            key = (ei, ej)
            last = active.get(key)
            if last is None:
                continue
            # only expire if no activity refreshed it since scheduling,
            # or if the max active lifetime has elapsed
            idle_expired = (t_exp - last) >= tau_idle - 1e-9
            age_expired = (t_exp - born.get(key, t_exp)) >= tau_active - 1e-9
            if idle_expired or age_expired:
                del active[key]
                born.pop(key, None)
                topo.append((float(t_exp), ei, ej, REMOVE))

        key = (i, j)
        if key not in active:
            active[key] = te
            born[key] = ts
            topo.append((float(ts), i, j, ADD))
        else:
            active[key] = max(active[key], te)

        # schedule next candidate expiry
        cand = min(active[key] + tau_idle, born[key] + tau_active)
        heapq.heappush(expiry, (float(cand), i, j))

    # drain remaining expiries
    while expiry:
        t_exp, ei, ej = heapq.heappop(expiry)
        key = (ei, ej)
        if key in active and (t_exp - active[key]) >= tau_idle - 1e-9:
            del active[key]
            born.pop(key, None)
            topo.append((float(t_exp), ei, ej, REMOVE))

    topo.sort(key=lambda r: r[0])
    topology_events = (np.array(topo, dtype=np.float64)
                       if topo else np.zeros((0, 4)))

    return TemporalGraph(
        node_ids=node_ids,
        topology_events=topology_events,
        interaction_times=t_end.copy(),
        interaction_src=src_i,
        interaction_dst=dst_i,
        edge_features=features,
        flow_labels=labels,
        meta={"tau_idle": tau_idle, "tau_active": tau_active,
              "n_flows": int(K), "unknown_nodes_added": int(unknown)},
    )


def lift_node_labels(g: TemporalGraph, window: float = 1.0,
                     n_windows: int | None = None):
    """
    Lift per-flow labels to per-node labels over fixed windows.

    y[w, v] = 1 iff node v is an endpoint of at least one malicious flow
    ending in window w. No k-hop propagation.

    Returns:
        y:        (W, N) int8
        w_edges:  list of length W; each entry is an array of flow indices
                  falling in that window
        t_edges:  (W,) float window end times
    """
    t = g.interaction_times
    if len(t) == 0:
        return np.zeros((0, g.num_nodes), np.int8), [], np.zeros(0)

    t0, t1 = float(t.min()), float(t.max())
    W = n_windows or max(1, int(np.ceil((t1 - t0) / window)))
    edges_per_w = [[] for _ in range(W)]

    bins = np.clip(((t - t0) / window).astype(int), 0, W - 1)
    for k, b in enumerate(bins):
        edges_per_w[b].append(k)

    y = np.zeros((W, g.num_nodes), dtype=np.int8)
    mal = g.flow_labels == 1
    for w, ks in enumerate(edges_per_w):
        for k in ks:
            if mal[k]:
                y[w, g.interaction_src[k]] = 1
                y[w, g.interaction_dst[k]] = 1

    t_edges = t0 + window * (np.arange(W) + 1)
    return y, [np.asarray(e, dtype=np.int64) for e in edges_per_w], t_edges


def adjacency_at(g: TemporalGraph, t: float):
    """Edge list active at time t, by replaying topology events up to t."""
    live = set()
    for row in g.topology_events:
        if row[0] > t:
            break
        i, j, d = int(row[1]), int(row[2]), int(row[3])
        if d == ADD:
            live.add((i, j))
        else:
            live.discard((i, j))
    if not live:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(sorted(live), dtype=np.int64).T
