"""Milestone 3 Task 2, Phase 1: the reuse study's numbers (spec 007 US2).

Reads the Phase 1 engine runs' tables (``analysis/tables.py``), the cache
simulator sweep (``run_m3_task2 sim``), the identification, signals and
paper-check reports, and the plans' manifests, and writes the one
``summary.json`` that every figure and the results page read
(constitution V)::

    python -m analysis.m3t2_study --runs results/m3-task2 \\
        --sweep results/m3-sim/sweep.jsonl --reports results/m3-study \\
        --out results/m3-study/summary.json

What each section measures, and from which run:

- **Workload properties** (how soon and how much a workload reuses) come
  from the ceiling runs, which never evict, so no policy's choices are in
  them. Policy outcomes come from each policy's run of the same plan.
- **Window.** Requests planned in [``w0_s``, ``stop_s``) count (the
  warm-up is skipped); a block stay counts when the request that freed the
  block was planned in that window.
- **Usable reuse** is ``004``'s usable reused fraction: prompt tokens
  served from the cache before the first miss, over prompt tokens.
- **Reuse interval** is ``004``'s: seconds since the hash's previous access
  (insert or hit), by its source (same session, another session of the
  workload, another workload). **Idle age** is a block's time in the free
  pool, from its free to the next lookup of its hash.
- **P(usable hit | interval)** counts re-accesses that were not shared with
  a running request (``in_use``), which no policy can lose.
- **Lost reuse.** A request's ceiling usable prefix ``U_c`` and the
  policy's ``U_p``: blocks ``[U_p, U_c)`` are lost, either **evicted** (the
  policy missed them) or **cut off** (still cached, but behind a miss, so
  vLLM cannot use them).
- **Reuse distance** is the LRU stack distance (distinct blocks accessed
  since the block's previous access) over the ceiling run's accesses; an
  LRU cache of C blocks keeps a reuse whose distance is below C.

Simulator numbers (the sweep, Belady) are a guide, marked as such in the
summary; reported numbers come from the engine runs (constitution I).
"""
import argparse
import gzip
import json
import math
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq

from analysis import events as ev
from analysis import stackdist, survival

CONFIG_PATH = Path(__file__).with_name("m3t2_config.json")
Columns = Dict[str, np.ndarray]
Row = Dict[str, Any]
SOURCES = ("same_session", "other_session", "other_workload")


def load_config(path: Path = CONFIG_PATH) -> Row:
    return json.loads(path.read_text())


def run_name(mix: str,
             pacing: str,
             seed: int,
             policy: str,
             capacity: int,
             replicate: int = 0) -> str:
    name = (f"{mix}-{pacing}-s{seed}-ceiling" if policy == "ceiling" else
            f"{mix}-{pacing}-s{seed}-{policy}-cap{capacity}")
    return name + (f"-r{replicate}" if replicate else "")


def read(run_dir: Path, table: str, columns: Sequence[str]) -> Columns:
    """Columns of one table as numpy arrays (strings as object arrays)."""
    t = pq.read_table(Path(run_dir) / "tables" / f"{table}.parquet",
                      columns=list(columns))
    return {c: t.column(c).to_numpy() for c in columns}


def window(planned: np.ndarray, w0: float, stop: float) -> np.ndarray:
    return (planned >= w0) & (planned < stop)


def share(part: float, whole: float) -> Optional[float]:
    return part / whole if whole else None


def spread(values: Sequence[Optional[float]]) -> Optional[Row]:
    """Mean, min, max and the values of a per-seed metric."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "mean": float(np.mean(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "seeds": [None if v is None else float(v) for v in values],
    }


# Reuse portion -----------------------------------------------------------


def usable_by_workload(req: Columns, workloads: Sequence[str], w0: float,
                       stop: float, block_size: int) -> Row:
    """004's usable reused fraction per workload, token-weighted overall,
    and the macro mean (each workload counts once)."""
    m = window(req["planned_s"], w0, stop)
    usable = np.minimum(req["prompt_tokens"].astype(np.int64),
                        block_size * req["usable_hits"].astype(np.int64))
    out: Row = {}
    for w in workloads:
        k = m & (req["workload"] == w)
        prompt = int(req["prompt_tokens"][k].sum())
        out[w] = {
            "requests": int(k.sum()),
            "prompt_tokens": prompt,
            "usable_tokens": int(usable[k].sum()),
            "fraction": share(float(usable[k].sum()), prompt),
        }
    prompt = int(req["prompt_tokens"][m].sum())
    fractions = [
        out[w]["fraction"] for w in workloads if out[w]["fraction"] is not None
    ]
    out["all"] = {
        "requests": int(m.sum()),
        "prompt_tokens": prompt,
        "usable_tokens": int(usable[m].sum()),
        "fraction": share(float(usable[m].sum()), prompt),
    }
    out["macro"] = {
        "fraction": float(np.mean(fractions)) if fractions else None
    }
    return out


def overlap(lo: np.ndarray, hi: np.ndarray,
            edges: Sequence[float]) -> np.ndarray:
    """Per bin [edges[k], edges[k+1]), the total length of the intervals
    [lo, hi) inside it."""
    e = np.asarray(edges, dtype=float)
    a = np.maximum(lo[:, None], e[None, :-1])
    b = np.minimum(hi[:, None], e[None, 1:])
    return np.clip(b - a, 0, None).sum(axis=0)


def position_profile(blocks: np.ndarray, ceiling_usable: np.ndarray,
                     policy_usable: Dict[str, np.ndarray],
                     edges: Sequence[float]) -> Row:
    """By block position: the share of prompt blocks the ceiling reuses, and
    the share each policy loses (usable at the ceiling, not under it)."""
    zero = np.zeros(len(blocks))
    total = overlap(zero, blocks.astype(float), edges)
    hit = overlap(zero, ceiling_usable.astype(float), edges)
    out: Row = {
        "edges": list(edges),
        "blocks": total.tolist(),
        "ceiling_share": [share(h, t) for h, t in zip(hit, total)],
    }
    for policy, kept in policy_usable.items():
        lo = np.minimum(kept, ceiling_usable).astype(float)
        lost = overlap(lo, ceiling_usable.astype(float), edges)
        out[f"{policy}_lost_share"] = [
            share(x, t) for x, t in zip(lost, total)
        ]
    return out


def lost_reuse(lk: Columns, ceiling_usable: Dict[str, int],
               policy_usable: Dict[str, int], workloads: Sequence[str],
               w0: float, stop: float) -> Row:
    """Blocks the ceiling reuses but a policy does not, per workload:
    evicted (missed) or cut off (cached behind an earlier miss)."""
    m = window(lk["planned_s"], w0, stop) & (lk["prefill"] == 0)
    req, idx = lk["req"][m], lk["index"][m]
    hit, wl = lk["hit"][m], lk["workload"][m]
    uc = np.array([ceiling_usable.get(r, 0) for r in req], dtype=np.int64)
    up = np.array([policy_usable.get(r, 0) for r in req], dtype=np.int64)
    lost = (idx >= up) & (idx < uc)
    out: Row = {}
    for w in list(workloads) + ["all"]:
        k = lost if w == "all" else lost & (wl == w)
        kept = (idx < np.minimum(up, uc))
        kk = kept if w == "all" else kept & (wl == w)
        out[w] = {
            "kept": int(kk.sum()),
            "evicted": int((k & ~hit).sum()),
            "cut_off": int((k & hit).sum()),
        }
    return out


def source_shares(lk: Columns, workloads: Sequence[str], w0: float,
                  stop: float) -> Row:
    """Where the ceiling's reuse comes from: the share of reused blocks
    whose previous access was the same session, another session of the
    workload, or another workload; and the share read by a session's first
    request (turn 0)."""
    m = (window(lk["planned_s"], w0, stop) & (lk["prefill"] == 0) & lk["hit"])
    out: Row = {}
    for w in workloads:
        k = m & (lk["workload"] == w)
        n = int(k.sum())
        out[w] = {
            "hits": n,
            **{
                s: share(float((k & (lk["source"] == s)).sum()), n)
                for s in SOURCES
            },
            "first_turn": share(float((k & (lk["turn"] == 0)).sum()), n),
            "in_use": share(float((k & lk["in_use"]).sum()), n),
        }
    return out


# Reuse interval ----------------------------------------------------------


def cdf_points(values: np.ndarray, points: int) -> Row:
    """A CDF as (x, p) pairs at evenly spaced probabilities, plus the
    quantiles the text quotes."""
    v = np.sort(np.asarray(values, dtype=float))
    if not len(v):
        return {"n": 0}
    p = np.linspace(0, 1, points)
    x = np.quantile(v, p)
    q = {
        f"p{int(round(k * 100))}": float(np.quantile(v, k))
        for k in (0.1, 0.25, 0.5, 0.8, 0.9, 0.99)
    }
    within = {str(s): float(np.mean(v <= s)) for s in (1, 15, 60, 300)}
    return {
        "n": int(len(v)),
        "x": x.tolist(),
        "p": p.tolist(),
        "quantiles": q,
        "within": within
    }


def intervals(lk: Columns, w0: float,
              stop: float) -> Dict[Tuple[str, str], np.ndarray]:
    """004 reuse intervals of re-accesses, by (workload, source)."""
    m = (window(lk["planned_s"], w0, stop) & (lk["prefill"] == 0)
         & (lk["source"] != "first"))
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for w in np.unique(lk["workload"][m]):
        for s in SOURCES:
            k = m & (lk["workload"] == w) & (lk["source"] == s)
            out[(str(w), s)] = lk["interval_s"][k]
    return out


def hit_by_interval(lk: Columns, workloads: Sequence[str],
                    edges: Sequence[float], w0: float, stop: float) -> Row:
    """Per workload and interval bin: re-accesses (not shared with a running
    request) and the share that were usable hits."""
    m = (window(lk["planned_s"], w0, stop) & (lk["prefill"] == 0)
         & (lk["source"] != "first") & ~lk["in_use"])
    e = np.asarray(edges, dtype=float)
    out: Row = {"edges": list(edges)}
    for w in list(workloads) + ["all"]:
        k = m if w == "all" else m & (lk["workload"] == w)
        bins = np.clip(
            np.searchsorted(e, lk["interval_s"][k], "right") - 1, 0,
            len(e) - 2)
        n = np.bincount(bins, minlength=len(e) - 1)
        hits = np.bincount(bins,
                           weights=lk["usable"][k].astype(float),
                           minlength=len(e) - 1)
        out[w] = {
            "n": n.tolist(),
            "usable_share": [share(h, c) for h, c in zip(hits, n)]
        }
    return out


def residency_window(rs: Columns, w0: float, stop: float) -> np.ndarray:
    return window(rs["owner_planned_s"], w0, stop)


def come_back(rs: Columns, workloads: Sequence[str], grid: Sequence[float],
              w0: float, stop: float) -> Row:
    """Ceiling run: per workload, the Kaplan-Meier share of freed blocks
    looked up again within each idle age of `grid` (a never-evicting cache:
    every stay ends with its revive or the end of the run)."""
    m = residency_window(rs, w0, stop)
    out: Row = {"grid": list(grid)}
    for w in workloads:
        k = m & (rs["owner_workload"] == w)
        d = rs["idle_s"][k]
        e = rs["end"][k] == "revive"
        if not len(d):
            continue
        times, surv = survival.kaplan_meier(d, e)
        out[w] = {
            "stays":
            int(len(d)),
            "came_back":
            [float(1 - s) for s in survival.survival_at(times, surv, grid)],
            "revived_share":
            float(e.mean()),
        }
    return out


def eviction_ages(rs: Columns, workloads: Sequence[str], w0: float,
                  stop: float) -> Row:
    """Idle age at eviction (how long the policy kept a freed block), per
    workload and pooled, and the share of stays that ended evicted."""
    m = residency_window(rs, w0, stop) & (rs["end"] != "censored")
    out: Row = {}
    for w in list(workloads) + ["all"]:
        k = m if w == "all" else m & (rs["owner_workload"] == w)
        ev_ = k & (rs["end"] == "evict")
        ages = np.sort(rs["idle_s"][ev_])
        out[w] = {
            "evicted":
            int(ev_.sum()),
            "evicted_share":
            share(float(ev_.sum()), float(k.sum())),
            "quantiles":
            ({f"p{p}": float(np.percentile(ages, p))
              for p in (10, 50, 90)} if len(ages) else None),
        }
    return out


def frequency_outcomes(rs: Columns, workloads: Sequence[str], w0: float,
                       stop: float) -> Row:
    """Share of free-pool stays that ended revived (not evicted), by the
    hits the block had before the stay (0, 1, 2+): LFU's ordering key."""
    m = residency_window(rs, w0, stop) & (rs["end"] != "censored")
    bucket = np.minimum(rs["hits_before"], 2)
    out: Row = {}
    for w in list(workloads) + ["all"]:
        k = m if w == "all" else m & (rs["owner_workload"] == w)
        out[w] = {
            str(b) + ("+" if b == 2 else ""): {
                "stays":
                int((k & (bucket == b)).sum()),
                "revived_share":
                share(
                    float((k & (bucket == b) & (rs["end"] == "revive")).sum()),
                    float((k & (bucket == b)).sum())),
            }
            for b in (0, 1, 2)
        }
    return out


def hit_density(rs: Columns, workloads: Sequence[str], w0: float,
                stop: float) -> Row:
    """Per workload: its share of the free pool's block-seconds and of its
    revives, and revives per block-minute of pool time."""
    m = residency_window(rs, w0, stop)
    total_s = float(rs["idle_s"][m].sum())
    total_r = float((m & (rs["end"] == "revive")).sum())
    out: Row = {}
    for w in workloads:
        k = m & (rs["owner_workload"] == w)
        secs = float(rs["idle_s"][k].sum())
        rev = float((k & (rs["end"] == "revive")).sum())
        out[w] = {
            "pool_time_share": share(secs, total_s),
            "revive_share": share(rev, total_r),
            "revives_per_block_minute": share(rev, secs / 60.0),
        }
    return out


def evictions_matrix(rs: Columns, workloads: Sequence[str], w0: float,
                     stop: float) -> Row:
    """Who evicts whom: evictions by (evicting workload, victim's
    workload), and the share of each workload's evicted blocks that were
    looked up again later (regretted)."""
    m = residency_window(rs, w0, stop) & (rs["end"] == "evict")
    counts = [[
        int((m & (rs["evictor_workload"] == a)
             & (rs["owner_workload"] == b)).sum()) for b in workloads
    ] for a in workloads]
    regret = {}
    for w in workloads:
        k = m & (rs["owner_workload"] == w)
        regret[w] = share(float((k & np.isfinite(rs["next_lookup_t"])).sum()),
                          float(k.sum()))
    return {
        "rows_evictor": list(workloads),
        "counts": counts,
        "regretted_share": regret
    }


# Reuse distance -----------------------------------------------------------


def access_stream(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """The ceiling run's 004 accesses (insert or hit) in log order: hashes,
    whether each is a hit, and each access's client request id."""
    hashes, hits = array("q"), array("b")
    reqs: List[str] = []
    path = ev.log_path(run_dir)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line in f:
            if '"insert"' not in line and '"hit"' not in line:
                continue
            row = json.loads(line)
            if row["ev"] not in ("insert", "hit"):
                continue
            hashes.append(row["hash"])
            hits.append(row["ev"] == "hit")
            reqs.append(ev.client_request_id(row["req"]) if row["req"] else "")
    return (np.frombuffer(hashes, dtype=np.int64),
            np.frombuffer(hits, dtype=np.int8).astype(bool), reqs)


def reuse_distance(run_dir: Path, planned: Dict[str, float],
                   workloads: Sequence[str], capacity_blocks: int, w0: float,
                   stop: float) -> Dict[str, np.ndarray]:
    """Reuse distances of the hits of requests in the window, by workload,
    with the share beyond `capacity_blocks`."""
    hashes, hits, reqs = access_stream(run_dir)
    dist = stackdist.reuse_distances(hashes)
    out: Dict[str, List[int]] = defaultdict(list)
    for i in np.flatnonzero(hits & (dist >= 0)):
        r = reqs[i]
        t = planned.get(r)
        if t is None or not w0 <= t < stop:
            continue
        out[ev.workload_of(r)].append(int(dist[i]))
    return {w: np.asarray(out.get(w, []), dtype=np.int64) for w in workloads}


# The page's example request ---------------------------------------------


def pick_example(creq: Columns,
                 lru_usable: Dict[str, int],
                 workload: str,
                 w0: float,
                 stop: float,
                 max_blocks: int = 64) -> Optional[str]:
    """A request the ceiling mostly reuses and LRU mostly loses: the
    largest ceiling prefix among `workload`'s windowed requests of at most
    `max_blocks` blocks whose LRU prefix is at most a quarter of it."""
    m = (window(creq["planned_s"], w0, stop) & (creq["workload"] == workload)
         & (creq["prompt_blocks"] <= max_blocks)
         & (creq["usable_hits"] >= 16))
    best = None
    for r, uc in zip(creq["req"][m].tolist(), creq["usable_hits"][m].tolist()):
        if lru_usable.get(r, 0) <= uc / 4:
            key = (-uc, r)
            if best is None or key < best[0]:
                best = (key, r)
    return None if best is None else best[1]


def block_states(lk: Columns, req: str) -> List[str]:
    """Each prompt block of `req`'s first prefill: "hit" (usable), "cut"
    (cached behind a miss) or "miss"."""
    m = (lk["req"] == req) & (lk["prefill"] == 0)
    order = np.argsort(lk["index"][m])
    hit, usable = lk["hit"][m][order], lk["usable"][m][order]
    return [
        "hit" if u else "cut" if h else "miss"
        for h, u in zip(hit.tolist(), usable.tolist())
    ]


def distance_summary(parts: List[np.ndarray], capacity_blocks: int,
                     points: int) -> Row:
    """A workload's reuse distances: their CDF and the share at or beyond
    the cache's capacity in blocks."""
    d = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    out = cdf_points(d, points)
    out["beyond_capacity"] = (float(np.mean(
        d >= capacity_blocks)) if len(d) else None)
    return out


# Assembly -----------------------------------------------------------------


def _usable_map(req: Columns) -> Dict[str, int]:
    return dict(zip(req["req"].tolist(), req["usable_hits"].tolist()))


def load_requests(run_dir: Path) -> Columns:
    return read(run_dir, "requests", [
        "req", "workload", "session", "turn", "planned_s", "status",
        "prompt_tokens", "prompt_blocks", "hits", "usable_hits",
        "hits_after_miss", "cached_tokens", "preempted"
    ])


LOOKUP_COLUMNS = [
    "req", "workload", "turn", "planned_s", "prefill", "index", "hit",
    "usable", "in_use", "source", "interval_s", "requests_between",
    "evicted_before"
]
RESIDENCY_COLUMNS = [
    "owner_workload", "owner_planned_s", "end", "idle_s", "hits_before",
    "evictor_workload", "next_lookup_t", "index"
]


def per_run(run_dir: Path, cfg: Row, w0: float) -> Row:
    """Everything one policy run contributes (headline, lost reuse needs the
    ceiling and is done in `study`)."""
    stop, bs, wl = cfg["stop_s"], cfg["block_size"], cfg["workloads"]
    req = load_requests(run_dir)
    lk = read(run_dir, "lookups", LOOKUP_COLUMNS)
    rs = read(run_dir, "residencies", RESIDENCY_COLUMNS)
    return {
        "usable":
        usable_by_workload(req, wl, w0, stop, bs),
        "hit_by_interval":
        hit_by_interval(lk, wl, cfg["interval_bins_s"], w0, stop),
        "eviction_ages":
        eviction_ages(rs, wl, w0, stop),
        "frequency":
        frequency_outcomes(rs, wl, w0, stop),
        "hit_density":
        hit_density(rs, wl, w0, stop),
        "evictions":
        evictions_matrix(rs, wl, w0, stop),
        "requests_with_hits_after_miss":
        int((window(req["planned_s"], w0, stop)
             & (req["hits_after_miss"] > 0)).sum()),
        "preempted":
        int(req["preempted"].sum()),
        "_req":
        req,
        "_lk":
        lk,
    }


def study(runs: Path, cfg: Row, sweep: Optional[Path], reports: Optional[Path],
          plans_root: Path) -> Row:
    w0, stop, bs = cfg["w0_s"], cfg["stop_s"], cfg["block_size"]
    wl, cap = cfg["workloads"], cfg["capacity_tokens"]
    mix, pacing = cfg["mix"], cfg["pacing"]
    seeds = cfg["reported_seeds"]
    policies = cfg["policies"]
    summary: Row = {"config": cfg}

    # Headline, Q2 portion and the policy views, per reported seed.
    headline: Row = {p: {} for p in policies + ["ceiling"]}
    lost: Row = {p: {} for p in policies}
    position_inputs: Dict[str, List] = defaultdict(list)
    q1_hits: Row = {p: [] for p in policies}
    views: Row = {p: defaultdict(list) for p in policies}
    interval_pool: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
    sources: List[Row] = []
    come_backs: List[Row] = []
    distances: Dict[str, List[np.ndarray]] = defaultdict(list)
    example: Row = {}
    for seed in seeds:
        cdir = runs / run_name(mix, pacing, seed, "ceiling", cap)
        creq = load_requests(cdir)
        clk = read(cdir, "lookups", LOOKUP_COLUMNS)
        crs = read(cdir, "residencies", RESIDENCY_COLUMNS)
        headline["ceiling"][seed] = usable_by_workload(creq, wl, w0, stop, bs)
        for key, vals in intervals(clk, w0, stop).items():
            interval_pool[key].append(vals)
        sources.append(source_shares(clk, wl, w0, stop))
        come_backs.append(come_back(crs, wl, cfg["km_grid_s"], w0, stop))
        planned = dict(zip(creq["req"].tolist(), creq["planned_s"].tolist()))
        for w, d in reuse_distance(cdir, planned, wl, cap // bs, w0,
                                   stop).items():
            distances[w].append(d)
        cusable = _usable_map(creq)
        mwin = window(creq["planned_s"], w0, stop)
        position_inputs["blocks"].append(creq["prompt_blocks"][mwin])
        position_inputs["ceiling"].append(creq["usable_hits"][mwin])
        position_inputs["workload"].append(creq["workload"][mwin])
        for policy in policies:
            pdir = runs / run_name(mix, pacing, seed, policy, cap)
            r = per_run(pdir, cfg, w0)
            headline[policy][seed] = r["usable"]
            pus = _usable_map(r["_req"])
            lost[policy][seed] = lost_reuse(r["_lk"], cusable, pus, wl, w0,
                                            stop)
            position_inputs[policy].append(
                np.array([pus.get(x, 0) for x in creq["req"][mwin]]))
            q1_hits[policy].append(r["hit_by_interval"])
            if seed == seeds[0]:
                if policy == policies[0]:
                    pick = pick_example(creq, pus, "sharegpt", w0, stop)
                    if pick is not None:
                        k = list(creq["req"]).index(pick)
                        # Block 0 is the system prompt every
                        # conversation shares; block 1's previous access
                        # is this conversation's previous turn.
                        prev = clk["interval_s"][(clk["req"] == pick)
                                                 & (clk["index"] == 1)]
                        example = {
                            "seed":
                            seed,
                            "req":
                            pick,
                            "turn":
                            int(creq["turn"][k]),
                            "prompt_tokens":
                            int(creq["prompt_tokens"][k]),
                            "since_previous_s":
                            (float(prev[0]) if len(prev) else None),
                            "ceiling":
                            block_states(clk, pick),
                        }
                if example:
                    example[policy] = block_states(r["_lk"], example["req"])
            for k in ("eviction_ages", "frequency", "hit_density",
                      "evictions"):
                views[policy][k].append(r[k])
            views[policy]["hits_after_miss"].append(
                r["requests_with_hits_after_miss"])
            views[policy]["preempted"].append(r["preempted"])
    summary["headline"] = headline_summary(headline, wl, policies, seeds)
    summary["lost"] = lost
    summary["example"] = example
    summary["q1"] = {
        "intervals": {
            w: {
                s: cdf_points(
                    np.concatenate(interval_pool[(w, s)]) if interval_pool.get(
                        (w, s)) else np.array([]), cfg["cdf_points"])
                for s in SOURCES
            }
            for w in wl
        },
        "intervals_all": {
            w: cdf_points(
                np.concatenate([
                    np.concatenate(interval_pool[(w, s)]) for s in SOURCES
                    if interval_pool.get((w, s))
                ]) if any(interval_pool.get((w, s))
                          for s in SOURCES) else np.array([]),
                cfg["cdf_points"])
            for w in wl
        },
        "come_back": merge_seeds(come_backs),
        "hit_by_interval": {p: merge_seeds(q1_hits[p])
                            for p in policies},
        "reuse_distance": {
            w: distance_summary(distances[w], cap // bs, cfg["cdf_points"])
            for w in wl
        },
    }
    summary["q2"] = {
        "sources":
        merge_seeds(sources),
        "position":
        position_by_workload(position_inputs, wl, policies, cfg["index_bins"]),
    }
    summary["policy_views"] = {
        p: {
            k:
            merge_seeds(v) if k not in ("hits_after_miss", "preempted") else v
            for k, v in views[p].items()
        }
        for p in policies
    }
    summary["sensitivity"] = sensitivity(runs, cfg)
    summary["mix"] = mix_summary(plans_root, cfg)
    if sweep is not None and sweep.exists():
        summary["sim"] = sim_summary(sweep, cfg)
        summary["validation"] = validation(runs, sweep, cfg)
    if reports is not None:
        for name in ("identify", "signals", "paper"):
            path = reports / f"{name}.json"
            if path.exists():
                summary[name] = json.loads(path.read_text())
    return summary


def merge_seeds(rows: Sequence[Row]) -> Row:
    """Average per-seed dicts leaf by leaf: numbers become the seed mean
    (with the per-seed values kept as `_seeds` for scalars), lists of
    numbers are averaged elementwise, other values come from seed 1."""
    if not rows:
        return {}
    first = rows[0]
    if isinstance(first, dict):
        keys = first.keys()
        return {k: merge_seeds([r[k] for r in rows if k in r]) for k in keys}
    if isinstance(first, list):
        if all(isinstance(r, list) and len(r) == len(first) for r in rows):
            if all(x is None or isinstance(x, (int, float)) for r in rows
                   for x in r):
                cols = list(zip(*rows))
                return [
                    None if any(x is None for x in c) else float(np.mean(c))
                    for c in cols
                ]
            if all(isinstance(x, list) for x in first):
                return [
                    merge_seeds([r[i] for r in rows])
                    for i in range(len(first))
                ]
        return first
    if isinstance(first, bool) or first is None:
        return first
    if isinstance(first, (int, float)):
        vals = [r for r in rows if isinstance(r, (int, float))]
        if len(vals) != len(rows):
            return None
        return float(np.mean(vals))
    return first


def headline_summary(headline: Row, workloads: Sequence[str],
                     policies: Sequence[str], seeds: Sequence[int]) -> Row:
    """Per policy and workload: the usable fraction's seed mean and spread,
    efficiency against the ceiling, and the paired change against LRU."""
    out: Row = {}
    for p in list(policies) + ["ceiling"]:
        out[p] = {}
        for w in list(workloads) + ["all", "macro"]:
            vals = [headline[p][s][w]["fraction"] for s in seeds]
            row: Row = {"usable": spread(vals)}
            if p != "ceiling":
                ceil = [headline["ceiling"][s][w]["fraction"] for s in seeds]
                row["efficiency"] = spread([
                    None if (v is None or c in (None, 0)) else v / c
                    for v, c in zip(vals, ceil)
                ])
                lru = [headline["lru"][s][w]["fraction"] for s in seeds]
                row["vs_lru"] = spread([
                    None if (v is None or b is None) else v - b
                    for v, b in zip(vals, lru)
                ])
            if w not in ("macro", ):
                row["prompt_tokens"] = int(
                    np.mean(
                        [headline[p][s][w]["prompt_tokens"] for s in seeds]))
            out[p][w] = row
    return out


def position_by_workload(inputs: Dict[str, List], workloads: Sequence[str],
                         policies: Sequence[str],
                         edges: Sequence[float]) -> Row:
    blocks = np.concatenate(inputs["blocks"])
    ceil = np.concatenate(inputs["ceiling"])
    work = np.concatenate(inputs["workload"])
    pol = {p: np.concatenate(inputs[p]) for p in policies}
    out: Row = {}
    for w in list(workloads) + ["all"]:
        k = np.ones(len(work), bool) if w == "all" else work == w
        out[w] = position_profile(blocks[k], ceil[k],
                                  {p: v[k]
                                   for p, v in pol.items()}, edges)
    return out


def sensitivity(runs: Path, cfg: Row) -> Row:
    """Fast pacing against human pacing (seed 2), other warm-ups, and the
    LRU replicate (run-to-run noise)."""
    stop, bs, wl, cap = (cfg["stop_s"], cfg["block_size"], cfg["workloads"],
                         cfg["capacity_tokens"])
    out: Row = {}
    seed = cfg["noise_replicate"]["seed"]

    def usable(name: str, w0: float = cfg["w0_s"]) -> Optional[Row]:
        path = runs / name
        if not (path / "tables" / "requests.parquet").exists():
            return None
        return usable_by_workload(load_requests(path), wl, w0, stop, bs)

    pacing = {}
    for pace in (cfg["pacing"], cfg["sensitivity_pacing"]):
        pacing[pace] = {
            p: usable(run_name(cfg["mix"], pace, seed, p, cap))
            for p in cfg["policies"] + ["ceiling"]
        }
    out["pacing"] = {"seed": seed, **pacing}
    out["warmup"] = {}
    for w0 in [cfg["w0_s"]] + cfg["w0_sensitivity_s"]:
        out["warmup"][str(int(w0))] = {
            p:
            spread([(usable(run_name(cfg["mix"], cfg["pacing"], s, p, cap), w0)
                     or {}).get("all", {}).get("fraction")
                    for s in cfg["reported_seeds"]])
            for p in cfg["policies"] + ["ceiling"]
        }
    base = usable(run_name(cfg["mix"], cfg["pacing"], seed, "lru", cap))
    rep = usable(run_name(cfg["mix"], cfg["pacing"], seed, "lru", cap, 1))
    if base and rep:
        out["noise"] = {
            w: {
                "run":
                base[w]["fraction"],
                "replicate":
                rep[w]["fraction"],
                "diff":
                (None if None in (base[w]["fraction"],
                                  rep[w]["fraction"]) else rep[w]["fraction"] -
                 base[w]["fraction"])
            }
            for w in list(wl) + ["all"]
        }
    return out


def mix_summary(plans_root: Path, cfg: Row) -> Row:
    """The mix as built: per workload, request and token shares, sessions,
    turns per session, pacing source and pauses, and Bailian's thinning,
    for every reported and training seed."""
    out: Row = {}
    for seed in cfg["train_seeds"] + cfg["reported_seeds"]:
        path = plans_root / f"{cfg['mix']}-{cfg['pacing']}-s{seed}" / \
            "manifest.json"
        m = json.loads(path.read_text())
        out[str(seed)] = {
            "requests": m["requests"],
            "sessions": m["sessions"],
            "prompt_tokens": m["prompt_tokens"],
            "workloads": {
                w: {
                    k: s.get(k)
                    for k in ("requests", "sessions", "request_share",
                              "prompt_token_share", "turns_per_session_mean",
                              "prompt_tokens_p50", "pacing", "pause_s",
                              "kept_session_fraction", "kept_request_fraction",
                              "turns_after_stop", "window_turns_after_stop",
                              "skipped_turns_context", "window_index")
                }
                for w, s in m["workloads"].items()
            },
        }
    return out


def read_sweep(path: Path) -> List[Row]:
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def sim_summary(path: Path, cfg: Row) -> Row:
    """Simulator guide: usable reuse against capacity per policy (mean and
    spread over the plans of each group), and Belady's split per workload
    at the fixed capacity."""
    rows = [r for r in read_sweep(path) if "summary" in r]
    groups = {
        "balanced-human": [f"balanced-human-s{s}" for s in (1, 2, 3, 4)],
        "balanced-fast": [f"balanced-fast-s{s}" for s in (1, 2, 3, 4)],
        "chat-heavy-human": ["chat-heavy-human-s1", "chat-heavy-human-s2"],
        "agent-heavy-human": ["agent-heavy-human-s1", "agent-heavy-human-s2"],
    }
    out: Row = {"note": "cache simulator, model timing; a design guide"}
    for group, plans in groups.items():
        curves: Row = {}
        for policy in cfg["sim_policies"] + ["ceiling"]:
            curve = {}
            for r in rows:
                plan = Path(r["plan"]).name
                if plan not in plans:
                    continue
                is_ceiling = str(r.get("capacity")) == "ceiling"
                if (policy == "ceiling") != is_ceiling:
                    continue
                if policy != "ceiling" and r["policy"] != policy:
                    continue
                key = "ceiling" if is_ceiling else str(r["capacity_tokens"])
                s = r["summary"]
                per = s["workloads"]
                curve.setdefault(key, []).append({
                    "all":
                    s["usable_reused_fraction"],
                    **{
                        w: per.get(w, {}).get("usable_reused_fraction")
                        for w in cfg["workloads"]
                    }
                })
            curves[policy] = {
                k: {
                    w: spread([x[w] for x in v])
                    for w in ["all"] + cfg["workloads"]
                }
                for k, v in curve.items()
            }
        out[group] = curves
    return out


def validation(runs: Path, sweep: Path, cfg: Row) -> Row:
    """Engine against simulator (model timing) on the same plans, whole
    plan, overall and per workload."""
    sims = {(Path(r["plan"]).name, r["policy"], str(r.get("capacity"))): r
            for r in read_sweep(sweep) if "summary" in r}
    out: List[Row] = []
    cap = cfg["capacity_tokens"]
    for seed in cfg["train_seeds"] + cfg["reported_seeds"]:
        plan = f"{cfg['mix']}-{cfg['pacing']}-s{seed}"
        for policy in cfg["policies"] + ["ceiling"]:
            path = runs / run_name(cfg["mix"], cfg["pacing"], seed, policy,
                                   cap)
            if not (path / "tables" / "requests.parquet").exists():
                continue
            key = ((plan, "lru", "ceiling") if policy == "ceiling" else
                   (plan, policy, str(cap)))
            sim = sims.get(key)
            if sim is None:
                continue
            eng = usable_by_workload(load_requests(path), cfg["workloads"],
                                     -math.inf, math.inf, cfg["block_size"])
            out.append({
                "plan":
                plan,
                "policy":
                policy,
                "engine":
                eng["all"]["fraction"],
                "sim":
                sim["summary"]["usable_reused_fraction"],
                "max_workload_diff":
                max(
                    abs((eng[w]["fraction"] or 0) -
                        (sim["summary"]["workloads"].get(w, {}).get(
                            "usable_reused_fraction") or 0))
                    for w in cfg["workloads"]),
            })
    return {
        "runs": out,
        "max_diff": max((abs(r["engine"] - r["sim"]) for r in out),
                        default=None)
    }


def _round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, digits)
    if isinstance(value, dict):
        return {
            str(k): _round(v, digits)
            for k, v in value.items() if not str(k).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, np.generic):
        return _round(value.item(), digits)
    return value


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/m3-task2"))
    parser.add_argument("--plans",
                        type=Path,
                        default=Path("data/processed/m3"))
    parser.add_argument("--sweep",
                        type=Path,
                        default=Path("results/m3-sim/sweep.jsonl"))
    parser.add_argument("--reports",
                        type=Path,
                        default=Path("results/m3-study"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out",
                        type=Path,
                        default=Path("results/m3-study/summary.json"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    summary = study(args.runs, cfg, args.sweep, args.reports, args.plans)
    from analysis.run_m2_task2 import git_state
    summary["provenance"] = git_state()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_round(summary), indent=1))
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
