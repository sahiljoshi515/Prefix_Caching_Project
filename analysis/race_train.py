"""Train RACE-RP's sketches on the seed-1 trace (spec 007, Milestone 3).

RACE-RP (``vllm/core/reuse_evictors.py``) scores an idle block by the
discounted reuse of training blocks with similar access features. This
module builds those training observations and the two RACE sketches:

1. **Trace.** The seed-1 plan is replayed through vLLM's own allocator
   (``analysis/cachesim.py``) with room for every block, so nothing is
   evicted: each block's accesses (lookup hits, insertions,
   deduplications, the evictors' ``record_access``) and the times it
   became idle are those of the requests alone, not of a policy. Timing is
   that of the seed-1 no-eviction engine run (``engine:<run dir>``) by
   default.
2. **Observations.** For each time a block became idle, at idle ages
   0, 5, 30 and 120 s: an observation if the block is still idle (no
   access since) and at least H seconds before the trace's last event.
   Features are computed from accesses up to that time only
   (``observation_features``, the evictor's ``AccessHistory.features``
   offline). The target is ``exp(-delta / tau)`` when the next access comes
   delta <= H seconds later, else 0 (H = 180 s, tau = 60 s).
3. **Sketches.** Feature means and standard deviations come from the
   observations; then each observation adds 1 to N and its target to W at
   its bucket in each of L tables of P random-hyperplane bits
   (``RaceSketch``). The file written is what ``race_rp`` loads through
   ``VLLM_EVICTION_PARAMS=sketch=<file>``.

Usage::

    python -m analysis.race_train --plan data/processed/m3/balanced-human-s1 \\
        --sketch 16x6 --out results/m3-race

writes ``results/m3-race/sketch-L16-P6.json``; several ``--sketch LxP``
values share one trace (seed-1 tuning, ``analysis/m3t2_policies.py``).
"""
import argparse
import json
import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vllm.core.block import cache_events
from vllm.core.evictor import LRUEvictor
from vllm.core.reuse_evictors import LONG_S, NUM_FEATURES, SHORT_S, RaceSketch

AGES = (0.0, 5.0, 30.0, 120.0)
HORIZON_S = 180.0
TAU_S = 60.0
SEED = 0
DEFAULT_PLAN = Path("data/processed/m3/balanced-human-s1")
DEFAULT_TIMING = "engine:results/m3-task2/balanced-human-s1-ceiling"


@dataclass
class Trace:
    """Every block's access times and idle starts, in the order they
    happened. An idle start keeps the number of the block's accesses before
    it, so an access at the same instant but after the free (the replay
    frees before it allocates) counts as the next access."""
    accesses: Dict[int, List[float]] = field(
        default_factory=lambda: defaultdict(list))
    idle_starts: Dict[int, List[Tuple[float, int]]] = field(
        default_factory=lambda: defaultdict(list))
    depth: Dict[int, int] = field(default_factory=dict)
    parent: Dict[int, Optional[int]] = field(default_factory=dict)
    end: float = 0.0

    def access(self, content_hash: int, depth: int, prev_hash: Optional[int],
               now: float) -> None:
        self.accesses[content_hash].append(now)
        self.depth.setdefault(content_hash, depth)
        self.parent.setdefault(content_hash, prev_hash)
        self.end = max(self.end, now)

    def idle(self, content_hash: int, now: float) -> None:
        self.idle_starts[content_hash].append(
            (now, len(self.accesses[content_hash])))
        self.end = max(self.end, now)

    def family(self, content_hash: int) -> int:
        """The block among its first two whose requests are its family's:
        the block itself at depth 1 or 2, else its depth-2 ancestor."""
        while self.depth[content_hash] > 2:
            parent = self.parent[content_hash]
            assert parent is not None, "a deep block without a parent"
            content_hash = parent
        return content_hash


class TraceEvictor(LRUEvictor):
    """LRU that records accesses and idle starts into a `Trace`. It runs
    with room for every block, so it must never evict."""

    def __init__(self,
                 trace: Trace,
                 clock: Callable[[], float] = cache_events.event_time):
        super().__init__()
        self.trace = trace
        self.clock = clock

    def record_access(self, block_id: int, content_hash: int, depth: int,
                      prev_hash: Optional[int]) -> None:
        self.trace.access(content_hash, depth, prev_hash, self.clock())

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        super().add(block_id, content_hash, num_hashed_tokens, last_accessed)
        self.trace.idle(content_hash, self.clock())

    def evict(self) -> Tuple[int, int]:
        raise RuntimeError("The training trace's cache must hold every block")


def record_trace(plan: Path,
                 timing: str = DEFAULT_TIMING,
                 block_size: int = 16) -> Trace:
    """Replay `plan` with room for every block (the ceiling's capacity) and
    return its trace."""
    from analysis import cachesim
    from analysis.run_m3_task2 import ceiling_capacity
    from client_simulator.plan_replay import group_sessions
    from client_simulator.token_plan import read_plan
    requests = list(read_plan(plan / "plan.jsonl.gz"))
    if timing == "model":
        times = cachesim.model_timing(group_sessions(requests))
    elif timing.startswith("engine:"):
        times = cachesim.engine_timing(Path(timing[len("engine:"):]), requests)
    else:
        raise ValueError(f"Unknown timing {timing!r}")
    sim = cachesim.CacheSim(
        ceiling_capacity(plan, block_size) // block_size, block_size, "lru")
    trace = Trace()
    sim.gpu.evictor = TraceEvictor(trace)
    sim.run(requests, times)
    waits = {
        k: sim.counts[k]
        for k in ("deferred", "decode_out_of_blocks") if sim.counts[k]
    }
    if waits:
        raise RuntimeError(
            f"The training trace's cache was too small: {waits}")
    return trace


def observation_features(trace: Trace, content_hash: int, family: int,
                         now: float) -> Tuple[float, ...]:
    """x_b(now) from accesses at or before `now` only, as
    ``AccessHistory.features`` computes it online."""

    def count(h: int, span: float) -> int:
        times = trace.accesses[h]
        return bisect_right(times, now) - bisect_right(times, now - span)

    times = trace.accesses[content_hash]
    last = times[bisect_right(times, now) - 1]
    return (math.log1p(count(content_hash,
                             SHORT_S)), math.log1p(count(content_hash,
                                                         LONG_S)),
            math.log1p(max(now - last,
                           0.0)), math.log1p(trace.depth[content_hash]),
            math.log1p(count(family, LONG_S)))


def target(delta: float,
           horizon: float = HORIZON_S,
           tau: float = TAU_S) -> float:
    """r = exp(-delta / tau) for a reuse within the horizon, else 0."""
    return math.exp(-delta / tau) if delta <= horizon else 0.0


def observations(
        trace: Trace,
        ages: Sequence[float] = AGES,
        horizon: float = HORIZON_S,
        tau: float = TAU_S) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, r, age) of every observation of `trace` (module docstring)."""
    last_t = trace.end - horizon
    ages = sorted(ages)
    rows: List[Tuple[float, ...]] = []
    rewards: List[float] = []
    obs_ages: List[float] = []
    for content_hash, starts in trace.idle_starts.items():
        times = trace.accesses[content_hash]
        family = trace.family(content_hash)
        for start, i in starts:
            following = times[i] if i < len(times) else math.inf
            for age in ages:
                now = start + age
                if now > last_t or now >= following:
                    break  # later ages fail too
                rows.append(
                    observation_features(trace, content_hash, family, now))
                rewards.append(target(following - now, horizon, tau))
                obs_ages.append(age)
    x = np.array(rows, dtype=np.float64).reshape(-1, NUM_FEATURES)
    return x, np.array(rewards), np.array(obs_ages)


def build_sketch(x: np.ndarray,
                 r: np.ndarray,
                 L: int,
                 P: int,
                 seed: int = SEED,
                 alpha: float = 5.0,
                 meta: Optional[Dict] = None) -> RaceSketch:
    """The two RACE sketches of observations (x, r), with standardization
    from the same observations."""
    if not len(x):
        raise ValueError("No training observations")
    sketch = RaceSketch.random(L,
                               P,
                               seed,
                               x.mean(axis=0),
                               x.std(axis=0),
                               r_bar=float(r.mean()),
                               alpha=alpha,
                               meta=meta)
    sketch.add(x, r)
    return sketch


def sketch_path(out: Path, L: int, P: int) -> Path:
    return out / f"sketch-L{L}-P{P}.json"


def parse_size(text: str) -> Tuple[int, int]:
    try:
        L, P = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not LxP (tables x bits), e.g. 16x6") from None
    if L < 1 or P < 1:
        raise argparse.ArgumentTypeError("L and P must be positive")
    return L, P


def train(plan: Path,
          sizes: Sequence[Tuple[int, int]],
          out: Path,
          timing: str = DEFAULT_TIMING,
          seed: int = SEED,
          alpha: float = 5.0) -> List[Path]:
    """Record `plan`'s trace once and write one sketch per (L, P)."""
    from analysis.run_m2_task2 import git_state
    trace = record_trace(plan, timing)
    x, r, age = observations(trace)
    meta = {
        "plan": str(plan),
        "timing": timing,
        "observations": int(len(x)),
        "observations_by_age": {str(a): int((age == a).sum())
                                for a in AGES},
        "reused_within_horizon": float((r > 0).mean()) if len(r) else None,
        "ages_s": list(AGES),
        "horizon_s": HORIZON_S,
        "tau_s": TAU_S,
        "seed": seed,
        "trace_end_s": trace.end,
        **git_state(),
    }
    sketches = [(L, P, build_sketch(x, r, L, P, seed, alpha, meta))
                for L, P in sizes]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "observations.npz", x=x, r=r, age=age)
    paths = []
    for L, P, sketch in sketches:
        path = sketch_path(out, L, P)
        sketch.save(path)
        paths.append(path)
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--timing", default=DEFAULT_TIMING)
    parser.add_argument("--sketch",
                        type=parse_size,
                        nargs="+",
                        default=[(16, 6)],
                        help="LxP: hash tables x hyperplane bits")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=Path("results/m3-race"))
    args = parser.parse_args(argv)
    for path in train(args.plan, args.sketch, args.out, args.timing, args.seed,
                      args.alpha):
        meta = json.loads(path.read_text())["meta"]
        print(f"{path}: {meta['observations']} observations")


if __name__ == "__main__":
    main()
