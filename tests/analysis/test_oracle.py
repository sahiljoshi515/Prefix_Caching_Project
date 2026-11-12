"""Belady's offline optimum in the cache simulator (spec 007 headroom).

Toy plans with 4-token blocks, written with `write_plan` and replayed by
`cachesim.simulate` (model timing), so victims are derived by hand. Seeded
random plans check what MIN must do: keep prefix closure, reuse at least as
much as every vLLM policy, and match LRU when nothing is evicted.
"""
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict

import pytest

from analysis import cachesim
from analysis.cachesim import CacheSim, Timing
from analysis.events import client_request_id
from analysis.oracle import NEVER, BeladyEvictor, FutureLookups, future_lookups
from client_simulator.plan_replay import group_sessions
from client_simulator.schema import TokenRequest
from client_simulator.token_plan import write_plan

pytestmark = pytest.mark.m3

BS = 4
SEEDS = range(20)


def req(rid, prompt, output=(), arrival=0.0, after=None):
    return TokenRequest(request_id=rid,
                        session_id=rid.rsplit(":", 1)[0],
                        turn_index=int(rid.rsplit(":", 1)[1]),
                        prompt_token_ids=list(prompt),
                        output_token_ids=list(output),
                        arrival_s=None if after is not None else arrival,
                        after_s=after)


def chain(tokens):
    return cachesim.block_hashes(list(tokens), BS)


def write(tmp_path, requests):
    path = tmp_path / "plan"
    write_plan(requests, path / "plan.jsonl.gz", path / "responses.jsonl")
    return path


def events(run_dir):
    with gzip.open(run_dir / "events.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f]


def evicted(run_dir):
    return [e["hash"] for e in events(run_dir) if e["ev"] == "evict"]


def two_families(*later):
    """R1 caches A = [1-4][5-8], R2 caches B = [101-104][105-108], R3 needs
    one block for C = [201-204], then `later` requests read again."""
    return [
        req("w:r1:0", range(1, 9), arrival=0.0),
        req("w:r2:0", range(101, 109), arrival=1.0),
        req("w:r3:0", range(201, 205), arrival=2.0),
        *later,
    ]


def test_belady_evicts_the_block_never_read_again(tmp_path):
    """Four blocks. R4 reads A again and nothing reads B, so R3 evicts B,
    its deeper block first, and R4 hits both of A's blocks. LRU evicts
    A2 (A is older), and R4's miss on it then evicts B2."""
    r4 = req("w:r4:0", range(1, 9), arrival=3.0)
    plan = write(tmp_path, two_families(r4))
    a, b = chain(range(1, 9)), chain(range(101, 109))
    expected = {"belady": ([b[1]], 2), "lru": ([a[1], b[1]], 1)}
    for policy, (victims, r4_hits) in expected.items():
        out = tmp_path / policy
        result = cachesim.simulate(plan, policy, 4 * BS, BS, out=out)
        rows = {row["request_id"]: row for row in result.rows}
        assert evicted(out) == victims
        assert rows["w:r4:0"]["hits"] == r4_hits
        assert rows["w:r4:0"]["usable_cached_tokens"] == r4_hits * BS


def test_belady_evicts_the_block_read_again_last(tmp_path):
    """R4 reads A and R5 reads B, later: R3 evicts B2. R5 then misses B2
    and evicts A2, the deepest of the blocks never read again (A1, A2, C1).
    Belady hits 3 blocks; LRU (A2, B2, C1) hits 2."""
    r4 = req("w:r4:0", range(1, 9), arrival=3.0)
    r5 = req("w:r5:0", range(101, 109), arrival=4.0)
    plan = write(tmp_path, two_families(r4, r5))
    a, b = chain(range(1, 9)), chain(range(101, 109))
    c = chain(range(201, 205))
    expected = {"belady": ([b[1], a[1]], 3), "lru": ([a[1], b[1], c[0]], 2)}
    for policy, (victims, hits) in expected.items():
        out = tmp_path / policy
        result = cachesim.simulate(plan, policy, 4 * BS, BS, out=out)
        assert evicted(out) == victims
        assert result.summary["hits"] == hits


def test_waiting_prompts_keep_their_place_in_the_oracle(tmp_path):
    """Seven blocks. R0 holds C (three blocks) until 1.8 s and R1 holds D
    (four) until 2.0 s. R2 (four new blocks, at 1.0 s) waits for space and
    R3 (at 1.5 s, reads C) waits behind it: both start at 2.0 s. C is freed
    after R3's planned start but before R3 is allocated, so keyed on
    planned times R3's read would look past and C3 would go with D4
    (deepest first). Waiting keeps R3's place in the allocation order, which
    the oracle keys on: R2 evicts only D (never read again) and R3 hits all
    of C. LRU evicts C first (older), and R3's misses evict the rest of D."""
    requests = [
        req("w:r0:0", range(201, 213)),
        req("w:r1:0", range(1, 17)),
        req("w:r2:0", range(51, 67)),
        req("w:r3:0", range(201, 213)),
    ]
    times = {
        "w:r0:0": (0.0, 1.8),
        "w:r1:0": (0.0, 2.0),
        "w:r2:0": (1.0, 1.2),
        "w:r3:0": (1.5, 1.6)
    }
    c, d = chain(range(201, 213)), chain(range(1, 17))
    for policy, victims, r3_hits in (("belady", d[::-1], 3),
                                     ("lru", c[::-1] + d[::-1], 0)):
        log = tmp_path / f"{policy}.jsonl.gz"
        sim = CacheSim(7, BS, policy, event_log=log)
        timing = {r: Timing(s, f, la=f) for r, (s, f) in times.items()}
        result = sim.run(requests, timing)
        rows = {row["request_id"]: row for row in result.rows}
        assert result.summary["deferred"] == 2
        assert rows["w:r2:0"]["start_s"] == rows["w:r3:0"]["start_s"] == 2.0
        with gzip.open(log, "rt") as f:
            logged = [json.loads(line) for line in f]
        assert [e["hash"] for e in logged if e["ev"] == "evict"] == victims
        assert rows["w:r3:0"]["hits"] == r3_hits


def test_future_lookups_number_allocations_as_the_simulator_does():
    """By start time, ties in plan order; a request with no timing is never
    sent. Each hash maps to the allocations that read it."""
    requests = [
        req("w:a:0", range(1, 9)),  # A1 A2
        req("w:b:0", range(1, 7)),  # A1 and a partial block
        req("w:c:0", range(1, 13)),  # A1 A2 A3
        req("w:d:0", range(1, 5)),  # never sent
    ]
    timing = {
        "w:a:0": Timing(2.0, 2.1, 2.1),
        "w:b:0": Timing(1.0, 1.1, 1.1),
        "w:c:0": Timing(2.0, 2.1, 2.1)
    }
    lookups = future_lookups(requests, timing, BS)
    a = chain(range(1, 13))
    assert lookups.order == ["w:b:0", "w:a:0", "w:c:0"]
    assert lookups.times == [1.0, 2.0, 2.0]
    assert lookups.lookups == {a[0]: [0, 1, 2], a[1]: [1, 2], a[2]: [2]}
    assert lookups.next_lookup(a[1]) == 1
    assert lookups.next_lookup(a[1], 2) == 2
    assert lookups.next_lookup(a[1], 3) == NEVER
    assert lookups.next_lookup(12345) == NEVER
    assert lookups.time(1) == 2.0 and lookups.time(NEVER) == math.inf
    lookups.allocated("w:b:0")
    assert lookups.next_lookup(a[0]) == 1
    with pytest.raises(RuntimeError, match="expected 'w:a:0'"):
        lookups.allocated("w:c:0")


def test_evictor_order_and_lazy_heap_bookkeeping():
    """Allocation 0 reads hash 10, allocation 1 hash 11, allocation 2 hashes
    10 and 12; 97 to 99 are never read."""
    lookups = FutureLookups(["a", "b", "c"], [0.0, 1.0, 2.0], {
        10: [0, 2],
        11: [1],
        12: [2]
    })
    evictor = BeladyEvictor(lookups)
    for block_id, content_hash, tokens in ((0, 10, 4), (1, 11, 4), (2, 12, 8),
                                           (3, 98, 4), (4, 99, 8), (5, 97, 8)):
        evictor.add(block_id, content_hash, tokens, 0.0)
    assert evictor.num_blocks == 6 and 3 in evictor and 6 not in evictor
    # A new last access does not move a Belady key: no new heap entry.
    entries = len(evictor._heap)
    evictor.update(1, 9.0)
    assert len(evictor._heap) == entries
    # Allocation "a" hits block 0, taking it out; its old entry goes stale.
    evictor.remove(0)
    assert 0 not in evictor and evictor.num_blocks == 5
    with pytest.raises(ValueError):
        evictor.remove(0)
    lookups.allocated("a")
    evictor.add(0, 10, 4, 1.0)  # freed again: next read is allocation 2
    # Never read first (deeper, then larger id), then the latest read.
    assert [evictor.evict() for _ in range(6)] == [(5, 97), (4, 99), (3, 98),
                                                   (2, 12), (0, 10), (1, 11)]
    with pytest.raises(ValueError):
        evictor.evict()  # only the stale entry is left


def test_evictor_heap_stays_bounded():
    """Stale entries are dropped once they outnumber live ones."""
    evictor = BeladyEvictor(FutureLookups([], [], {}))
    for content_hash in range(1000):
        evictor.add(0, content_hash, 4, 0.0)
        evictor.remove(0)
        assert len(evictor._heap) <= evictor.REBUILD_MIN + 1


def tokens(rng, count):
    """Fresh tokens: no two requests write the same decode block."""
    return [rng.randrange(1000, 10**6) for _ in range(count)]


def random_plan(seed):
    """Six to ten sessions of one to four turns over three workloads. Each
    session opens with one of two shared system prompts, and each turn's
    prompt is the session's history (earlier prompts and outputs) plus new
    tokens, so chains share roots and grow turn by turn. Sessions start at
    0, 1 or 2 s within 4 ms, and turns follow within 10 ms, so requests
    overlap."""
    rng = random.Random(seed)
    systems = [[rng.randrange(1, 1000) for _ in range(rng.choice((4, 8, 12)))]
               for _ in range(2)]
    requests = []
    for s in range(rng.randint(6, 10)):
        session = f"w{s % 3}:s{s}"
        history = list(rng.choice(systems))
        start = rng.choice((0.0, 1.0, 2.0)) + rng.uniform(0.0, 0.004)
        for turn in range(rng.randint(1, 4)):
            history += tokens(rng, rng.randint(1, 10))
            output = tokens(rng, rng.randint(0, 10))
            requests.append(
                req(f"{session}:{turn}",
                    history,
                    output,
                    arrival=start,
                    after=rng.uniform(0.0, 0.01) if turn else None))
            history = history + output
    return requests


def small_capacities(requests):
    """In blocks: the largest prompt alone, a little more, and twice it."""
    need = max(-(-len(r.prompt_token_ids) // BS) for r in requests)
    return need, need + 4, 2 * need


def closure_breaks(requests, run_dir):
    """Evictions in the run's log of a block with a descendant cached."""
    parent: Dict[Any, Any] = {}
    for r in requests:
        hashes = chain(r.prompt_token_ids + r.output_token_ids)
        parent.update(zip(hashes[1:], hashes))
    children: Counter = Counter()
    breaks = 0
    for e in events(run_dir):
        if e["ev"] == "insert" and e["hash"] in parent:
            children[parent[e["hash"]]] += 1
        elif e["ev"] == "evict":
            breaks += children[e["hash"]] > 0
            if e["hash"] in parent:
                children[parent[e["hash"]]] -= 1
    return breaks


def readers(run_dir):
    """The requests that looked up each hash while allocating a prompt."""
    found = defaultdict(set)
    for e in events(run_dir):
        if e.get("op") == "allocate" and e["ev"] in ("hit", "miss"):
            found[e["hash"]].add(client_request_id(e["req"]))
    return dict(found)


def oracle_readers(lookups):
    """The same, as the oracle predicts it."""
    return {
        content_hash: {lookups.order[p]
                       for p in positions}
        for content_hash, positions in lookups.lookups.items()
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_belady_keeps_prefix_closure_and_beats_every_policy(tmp_path, seed):
    """At small capacities Belady never evicts an ancestor of a cached block
    (so no request hits after a miss) and reuses at least as many tokens as
    LRU, LFU and FIFO. Its lookups are the ones the allocator makes."""
    requests = random_plan(seed)
    plan = write(tmp_path, requests)
    timing = cachesim.model_timing(group_sessions(requests))
    predicted = oracle_readers(future_lookups(requests, timing, BS))
    for blocks in small_capacities(requests):
        out = tmp_path / f"belady-{blocks}"
        belady = cachesim.simulate(plan, "belady", blocks * BS, BS,
                                   out=out).summary
        assert readers(out) == predicted  # the allocator's hashes
        assert belady["requests_with_hits_after_miss"] == 0
        assert closure_breaks(requests, out) == 0
        for policy in ("lru", "lfu", "fifo"):
            other = cachesim.simulate(plan, policy, blocks * BS, BS).summary
            assert belady["usable_tokens"] >= other["usable_tokens"], policy
            # Waits depend only on blocks in use, the same under any policy.
            for count in ("deferred", "never_allocated",
                          "decode_out_of_blocks"):
                assert belady.get(count, 0) == other.get(count, 0), count


def test_the_closure_check_catches_fifo(tmp_path):
    """FIFO evicts by first free, ancestors included, so the check above
    can fail."""
    requests = random_plan(0)
    plan = write(tmp_path, requests)
    blocks = small_capacities(requests)[0]
    cachesim.simulate(plan, "fifo", blocks * BS, BS, out=tmp_path / "fifo")
    assert closure_breaks(requests, tmp_path / "fifo") > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_belady_is_lru_when_nothing_is_evicted(tmp_path, seed):
    """With room for every block the plan writes, no evictor is asked for a
    victim, so Belady's run is LRU's."""
    requests = random_plan(seed)
    plan = write(tmp_path, requests)
    blocks = sum(-(-len(r.prompt_token_ids + r.output_token_ids) // BS)
                 for r in requests)
    belady = cachesim.simulate(plan, "belady", blocks * BS, BS)
    lru = cachesim.simulate(plan, "lru", blocks * BS, BS)
    assert belady.summary.pop("offline_oracle") is True
    assert belady.summary.get("evictions", 0) == 0
    assert belady.summary == lru.summary
    assert belady.rows == lru.rows


def test_belady_run_says_it_is_an_offline_oracle(tmp_path, capsys):
    plan = write(tmp_path, [req("w:a:0", range(1, 9))])
    cachesim.simulate(plan, "belady", 16 * BS, BS, out=tmp_path / "run")
    sim = json.loads((tmp_path / "run" / "sim.json").read_text())
    assert sim["policy"] == "belady"
    assert sim["summary"]["offline_oracle"] is True
    options = f"--policy belady --capacity {16 * BS} --block-size {BS}"
    cachesim.main(["--plan", str(plan), *options.split()])
    assert '"offline_oracle": true' in capsys.readouterr().out
    lru = cachesim.simulate(plan, "lru", 16 * BS, BS).summary
    assert "offline_oracle" not in lru
