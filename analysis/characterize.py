"""Per-workload reuse profile of a mixed run (spec 007 User Story 2).

Answers the course's questions for each workload in a mix, from one run's
cache event log (engine or ``analysis/cachesim.py``) and client results::

    python -m analysis.characterize results/m3-task2/balanced-s1-lru

- **Reuse interval**: for every revive (a freed block read again), the time
  it spent in the free pool and the number of requests started meanwhile;
  for every access after the first, the gap since the block's previous
  access (``004``'s reuse interval, `metrics.reuse_intervals`; it spans
  evictions). A free block that waits longer than the policy keeps it is
  lost, so the free-pool time is what an eviction policy has to cover.
- **Reuse portion**: usable reused fraction (``004``), hits per block,
  share of blocks ever hit, and how many hits fall on blocks written while
  decoding.
- **Hit sources**: the share of hits on blocks inserted (computed) by the
  same session, another session of the same workload, or another workload.
- **Cache pressure**: evictions and misses on blocks evicted earlier. An
  eviction counts against the workload of the block's last holder (the
  request that freed it), not the request whose allocation caused it.
- **Identification**: a request's *prefix family* at depth ``k`` is the
  content hash of its ``k``-th block (or of its last full block if it has
  fewer). vLLM's hashes are chained, so equal hashes mean equal ``k``-block
  prefixes: the family is a fingerprint of the template, system prompt or
  task a server can compute without any label. Purity is the share of
  requests whose family's majority workload is their own; a family seen in
  one request only is trivially pure, so purity is also given without
  those singletons.

A block's workload is that of the request that inserted it, updated to
the request that last freed it. Every request id carries its workload as
its prefix (``006`` FR-002).
"""
import argparse
import bisect
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from analysis import events as ev
from analysis import metrics as m

FAMILY_DEPTHS = (1, 2, 4, 8)


def quantiles(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)

    def q(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "p10": q(0.1),
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
    }


def _session(req: str) -> str:
    parsed = ev.parse_request_id(req)
    return parsed[0] if parsed else ev.client_request_id(req)


def reuse_profile(events: Iterable[ev.Event]) -> Dict[str, Any]:
    """Free-pool times, access gaps, hit sources and eviction costs per
    workload. An access gap is ``004``'s reuse interval
    (`metrics.reuse_intervals`), counted for the accessing request's
    workload."""
    events = list(events)
    starts = sorted(m.request_starts(events).values())

    def started_between(a: float, b: float) -> int:
        return bisect.bisect_left(starts, b) - bisect.bisect_right(starts, a)

    holder: Dict[int, str] = {}  # hash -> request that last held it
    inserted_by: Dict[int, str] = {}
    decode_block: Dict[int, bool] = {}
    freed_at: Dict[int, float] = {}
    evicted: set = set()
    hits_per_block: Dict[int, int] = Counter()
    block_workload: Dict[int, str] = {}
    pool_s: Dict[str, List[float]] = defaultdict(list)
    pool_requests: Dict[str, List[int]] = defaultdict(list)
    gap_s: Dict[str, List[float]] = defaultdict(list)
    for row in m.reuse_intervals(events):
        if row["req"] is not None:
            gap_s[ev.workload_of(row["req"])].append(row["interval_s"])
    counts: Dict[str, Counter] = defaultdict(Counter)
    for e in events:
        w = ev.workload_of(e.req) if e.req is not None else None
        h = e.hash
        if e.ev == "insert":
            inserted_by[h] = e.req
            holder[h] = e.req
            decode_block[h] = e.op == "append_slots"
            block_workload.setdefault(h, w)
            counts[w]["inserts"] += 1
            counts[w][
                "decode_inserts" if decode_block[h] else "prompt_inserts"] += 1
        elif e.ev == "hit":
            counts[w]["hits"] += 1
            hits_per_block[h] += 1
            if decode_block.get(h):
                counts[w]["hits_on_decode_blocks"] += 1
            # Who computed the block (its inserter), not who freed it last.
            owner = inserted_by.get(h)
            if owner is None:
                counts[w]["hits_unknown_owner"] += 1
            elif ev.workload_of(owner) != w:
                counts[w]["hits_cross_workload"] += 1
            elif _session(owner) == _session(e.req):
                counts[w]["hits_same_session"] += 1
            else:
                counts[w]["hits_other_session"] += 1
        elif e.ev == "miss":
            counts[w]["misses"] += 1
            if h in evicted:
                counts[w]["misses_on_evicted"] += 1
        elif e.ev == "free":
            holder[h] = e.req
            freed_at[h] = e.t
            counts[w]["frees"] += 1
        elif e.ev == "revive":
            if h in freed_at:
                t0 = freed_at.pop(h)
                pool_s[w].append(e.t - t0)
                pool_requests[w].append(started_between(t0, e.t))
            counts[w]["revives"] += 1
        elif e.ev == "evict":
            owner = holder.get(h)
            ow = ev.workload_of(owner) if owner else "unknown"
            counts[ow]["evicted"] += 1
            if h in freed_at:
                pool_s[ow + "#evicted"].append(e.t - freed_at.pop(h))
            evicted.add(h)
            for table in (holder, inserted_by, decode_block):
                table.pop(h, None)
        elif e.ev == "dedupe":
            counts[w]["dedupes"] += 1
    blocks_by_workload: Dict[str, List[int]] = defaultdict(list)
    for h, w in block_workload.items():
        blocks_by_workload[w].append(hits_per_block.get(h, 0))
    out: Dict[str, Any] = {}
    for w in sorted(k for k in counts if k is not None):
        c = counts[w]
        hits = c["hits"]
        blocks = blocks_by_workload.get(w, [])
        out[w] = {
            **dict(c),
            "block_hit_rate":
            hits / (hits + c["misses"]) if hits + c["misses"] else None,
            "hit_share_same_session":
            c["hits_same_session"] / hits if hits else None,
            "hit_share_other_session":
            c["hits_other_session"] / hits if hits else None,
            "hit_share_cross_workload":
            c["hits_cross_workload"] / hits if hits else None,
            "hit_share_on_decode_blocks":
            c["hits_on_decode_blocks"] / hits if hits else None,
            "blocks":
            len(blocks),
            "blocks_ever_hit_share":
            (sum(n > 0 for n in blocks) / len(blocks) if blocks else None),
            "hits_per_block":
            quantiles(blocks),
            "free_pool_s_until_revive":
            quantiles(pool_s.get(w, [])),
            "requests_between_free_and_revive":
            quantiles(pool_requests.get(w, [])),
            "free_pool_s_until_evict":
            quantiles(pool_s.get(w + "#evicted", [])),
            "access_gap_s":
            quantiles(gap_s.get(w, [])),
            "revive_share_of_frees":
            c["revives"] / c["frees"] if c["frees"] else None,
        }
    return out


def occupancy(events: Iterable[ev.Event],
              window_s: float = 30.0) -> List[Dict[str, Any]]:
    """Cached blocks per workload over time (spec 007 FR-011, figure 3):
    the count at the end of each window, a block counting for the workload
    of the request that inserted it, then of the request that last freed
    it."""
    label: Dict[int, str] = {}
    count: Counter = Counter()
    rows: List[Dict[str, Any]] = []
    start = None
    next_edge = None
    for e in events:
        if start is None:
            start = e.t
            next_edge = start + window_s
        while e.t >= next_edge:
            rows.append({"t_s": next_edge - start, **dict(count)})
            next_edge += window_s
        w = ev.workload_of(e.req) if e.req is not None else "unknown"
        if e.ev == "insert":
            label[e.hash] = w
            count[w] += 1
        elif e.ev == "free" and e.hash in label:
            count[label[e.hash]] -= 1
            label[e.hash] = w
            count[w] += 1
        elif e.ev == "evict" and e.hash in label:
            count[label.pop(e.hash)] -= 1
    if start is not None:
        rows.append({"t_s": next_edge - start, **dict(count)})
    return rows


def usable_by_workload(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Usable and vLLM-reported reuse per workload, from `analyze_run`'s
    request rows (which carry ``usable_cached_tokens``)."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[ev.workload_of(r["request_id"])].append(r)
    out = {}
    for w, rs in sorted(groups.items()):
        out[w] = {
            "requests":
            len(rs),
            "prompt_tokens":
            sum(r["prompt_tokens"] for r in rs),
            "usable_reused_fraction":
            m.token_reused_fraction(rs, "usable_cached_tokens"),
            "vllm_cached_share":
            m.token_reused_fraction(rs),
            "mean_usable_cached_fraction":
            statistics.fmean(r["usable_cached_fraction"] for r in rs),
            "requests_with_hits_after_miss":
            sum(r["hits_after_miss"] > 0 for r in rs),
        }
    return out


def prefix_families(events: Iterable[ev.Event],
                    depths: Sequence[int] = FAMILY_DEPTHS) -> Dict[str, Any]:
    """Each request's family at each depth: the hash of its k-th prompt
    block (its last if it has fewer), from its first-prefill lookups
    (`metrics.Prefills`)."""
    lookups: Dict[str, List[int]] = defaultdict(list)
    prefills = m.Prefills()
    for e in events:
        lookup = prefills.step(e.ev, e.hash, e.req, e.op)
        if lookup is not None and lookup.prefill == 0:
            lookups[e.req].append(e.hash)
    out: Dict[str, Any] = {}
    for k in depths:
        families: Dict[str, int] = {}
        for req, hashes in lookups.items():
            families[ev.client_request_id(req)] = hashes[min(k, len(hashes)) -
                                                         1]
        out[str(k)] = families
    return out


def family_purity(families: Dict[str, int]) -> Dict[str, Any]:
    """How well a family names its workload within one run.

    A family seen in one request only (a singleton) is pure by
    construction, and a classifier trained on another run has not seen it,
    so purity is also given over the other families' requests, with the
    singletons' number and share of families and of requests.
    """
    by_family: Dict[int, Counter] = defaultdict(Counter)
    for rid, family in families.items():
        by_family[family][ev.workload_of(rid)] += 1
    majority = {f: c.most_common(1)[0][0] for f, c in by_family.items()}
    correct = sum(c[majority[f]] for f, c in by_family.items())
    total = sum(sum(c.values()) for c in by_family.values())
    singletons = sum(1 for c in by_family.values() if sum(c.values()) == 1)
    shared = total - singletons  # requests in families of two or more
    # Every singleton's request counts as correct in `purity`.
    without = (correct - singletons) / shared if shared else None
    return {
        "requests": total,
        "families": len(by_family),
        "singleton_families": singletons,
        "singleton_family_share": _share(singletons, len(by_family)),
        "singleton_request_share": _share(singletons, total),
        "purity": _share(correct, total),
        "purity_without_singletons": without,
    }


def _share(part: int, whole: int) -> Optional[float]:
    return part / whole if whole else None


def family_classifier(train: Dict[str, int]) -> Dict[int, str]:
    """Family -> majority workload, learned on one run (spec 007 FR-006)."""
    by_family: Dict[int, Counter] = defaultdict(Counter)
    for rid, family in train.items():
        by_family[family][ev.workload_of(rid)] += 1
    return {f: c.most_common(1)[0][0] for f, c in by_family.items()}


def classify(table: Dict[int, str],
             test: Dict[str, int],
             default: str = "unknown") -> Dict[str, Any]:
    """Accuracy and confusion matrix of `table` on another run's requests;
    an unseen family is labelled `default`."""
    confusion: Dict[str, Counter] = defaultdict(Counter)
    for rid, family in test.items():
        confusion[ev.workload_of(rid)][table.get(family, default)] += 1
    total = sum(sum(c.values()) for c in confusion.values())
    correct = sum(c[w] for w, c in confusion.items())
    unseen = sum(c[default] for c in confusion.values())
    return {
        "requests": total,
        "accuracy": correct / total if total else None,
        "unseen_share": unseen / total if total else None,
        "confusion": {w: dict(c)
                      for w, c in sorted(confusion.items())},
    }


def trace_reuse(records: Iterable[Dict[str, Any]],
                block_size: int = 16) -> Dict[str, Any]:
    """Reuse in a raw Bailian trace at its full rate, before any sampling:
    with a cache that never evicts, the share of each request's prompt that
    repeats an earlier request's prefix (whole blocks, prefix-closed), and
    for each such block the time since its prefix was last used.

    A block's key is its whole prefix of hash ids, as vLLM's chained hash
    is, so a repeated block counts only after a repeated prefix.
    """
    last_seen: Dict[int, float] = {}
    gaps: List[float] = []
    reused_tokens = prompt_tokens = 0
    fractions: List[float] = []
    for r in sorted(records, key=lambda r: (r["timestamp"], r["chat_id"])):
        t = r["timestamp"]
        full = r["input_length"] // block_size
        key = 0
        keys = []
        for h in r["hash_ids"][:full]:
            key = hash((key, h))
            keys.append(key)
        reused = 0
        for key in keys:
            if key not in last_seen:
                break
            gaps.append(t - last_seen[key])
            reused += 1
        for key in keys:
            last_seen[key] = t
        tokens = min(reused * block_size, r["input_length"])
        reused_tokens += tokens
        prompt_tokens += r["input_length"]
        fractions.append(tokens /
                         r["input_length"] if r["input_length"] else 0)
    return {
        "requests":
        len(fractions),
        "reusable_token_fraction":
        reused_tokens / prompt_tokens if prompt_tokens else None,
        "mean_reusable_fraction":
        statistics.fmean(fractions) if fractions else None,
        "reuse_gap_s":
        quantiles(gaps),
    }


def characterize(run_dir: Path) -> Dict[str, Any]:
    from analysis import run_m2_task2 as task2
    log = ev.load_events(ev.log_path(run_dir))
    summary = task2.analyze_run(run_dir, 10.0)
    rows, _ = m.cached_fractions(ev.load_results(run_dir / "client.jsonl"))
    m.add_usable_fractions(rows, m.prefix_hits(log.events),
                           log.header["block_size"])
    families = prefix_families(log.events)
    return {
        "run": str(run_dir),
        "overall": {
            k: summary.get(k)
            for k in ("block_hit_rate_log", "usable_reused_fraction",
                      "token_reused_fraction", "evictions",
                      "requests_with_hits_after_miss", "sc003_pass")
        },
        "usable": usable_by_workload(rows),
        "reuse": reuse_profile(log.events),
        "families": {k: family_purity(f)
                     for k, f in families.items()},
        "occupancy": occupancy(log.events),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="JSON file to write")
    args = parser.parse_args(argv)
    result = characterize(args.run_dir)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
