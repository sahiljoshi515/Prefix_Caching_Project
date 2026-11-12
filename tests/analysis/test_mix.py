"""Mixed-workload plans (specs/007-m3-mixed-workload-policy, FR-001, US1).

Uses handcrafted conversations, a toy Bailian trace and a fake tokenizer,
so the expected counts and times can be worked out by hand.
"""
import json
from dataclasses import asdict
from typing import Any, Dict

import pytest

from analysis import mix
from client_simulator.schema import Conversation, Message, workload_of
from client_simulator.token_plan import read_plan, tokenize_conversation

pytestmark = pytest.mark.m3


class FakeTokenizer:
    """One token per character; a role marker per message; 7 = generation
    prompt."""
    ROLE = {"system": 1, "user": 2, "assistant": 3}

    def encode(self, text, add_special_tokens=False):
        return [100 + ord(c) % 50 for c in text]

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert tokenize
        out = []
        for m in messages:
            out += [self.ROLE[m["role"]]] + self.encode(m["content"])
        return out + ([7] if add_generation_prompt else [])


def conversation(cid, turns, length=3):
    messages = []
    for t in range(turns):
        messages += [
            Message("user", f"q{t}" + "x" * length),
            Message("assistant", f"a{t}" + "y" * length)
        ]
    return Conversation(cid, messages)


def bailian_sessions(count, start=0.0, gap=10.0):
    """`count` two-turn sessions, one every `gap` seconds."""
    sessions = []
    for i in range(count):
        t = start + i * gap
        root = {
            "chat_id": 2 * i,
            "parent_chat_id": -1,
            "timestamp": t,
            "input_length": 20,
            "output_length": 4,
            "type": "text",
            "turn": 1,
            "hash_ids": [1, 1000 + i]
        }
        child = dict(root,
                     chat_id=2 * i + 1,
                     parent_chat_id=2 * i,
                     timestamp=t + 3,
                     input_length=40,
                     hash_ids=[1, 2000 + i, 3000 + i])
        sessions.append([root, child])
    return sessions


def build(shares,
          seed=1,
          rate=0.2,
          duration=100.0,
          max_model_len=10_000,
          conversations=None,
          traces=None,
          tail=20.0,
          pacer=None):
    """A plan with Phase 0's constant pauses and, by default, a tail long
    enough that no session is cut by the stop."""
    spec = mix.MixSpec(name="test",
                       shares=shares,
                       duration_s=duration,
                       rate=rate,
                       max_model_len=max_model_len,
                       tail_s=tail)
    conversations = conversations or {
        "sharegpt": [conversation(f"sharegpt:c{i}", 2) for i in range(40)],
        "agentbank": [conversation(f"agentbank:t-{i}", 1) for i in range(40)],
    }
    traces = traces or {"bailian-chat": bailian_sessions(100)}
    return mix.build(spec,
                     seed,
                     conversations,
                     traces,
                     FakeTokenizer(),
                     pacer=pacer or mix.Pacer("fast"))


def test_tokenize_conversation_matches_the_chat_client():
    conv = conversation("sharegpt:c", 2, length=1)
    requests, skipped = tokenize_conversation(conv,
                                              FakeTokenizer(),
                                              10_000,
                                              after_s=5.0)
    tok = FakeTokenizer()
    assert skipped == 0
    assert [r.request_id for r in requests] == ["sharegpt:c:0", "sharegpt:c:1"]
    # Turn 1's prompt is the history through turn 1's question.
    history = [{"role": m.role, "content": m.content} for m in conv.messages]
    assert requests[1].prompt_token_ids == tok.apply_chat_template(
        history[:3], True, True)
    assert requests[1].output_token_ids == tok.encode("a1y")
    assert (requests[0].after_s, requests[1].after_s) == (None, 5.0)


def test_turns_past_the_context_are_dropped_with_the_rest():
    conv = conversation("sharegpt:c", 3, length=1)
    # Turn 0: prompt 1+3+1 = 5, output 3 (+1 EOS) = 9 tokens. Turn 1's
    # prompt is 5 + 4 + 4 = 13 tokens, so a 12-token context keeps one turn.
    requests, skipped = tokenize_conversation(conv, FakeTokenizer(), 12)
    assert [r.turn_index for r in requests] == [0]
    assert skipped == 2


def test_fifty_fifty_mix_hits_its_targets_and_keeps_sessions_whole():
    # 0.2 req/s x 100 s = 20 requests, 10 per workload.
    requests, manifest = build({"sharegpt": 0.5, "bailian-chat": 0.5})
    counts = {
        w: manifest["workloads"][w]["requests"]
        for w in ("sharegpt", "bailian-chat")
    }
    assert counts == {"sharegpt": 10, "bailian-chat": 10}
    assert manifest["requests"] == 20
    by_session: Dict[Any, Any] = {}
    for r in requests:
        by_session.setdefault(r.session_id, []).append(r)
    for session in by_session.values():
        # Whole sessions, turns in order, one workload each.
        assert [r.turn_index for r in session] == list(range(len(session)))
        assert len({workload_of(r.request_id) for r in session}) == 1
    sharegpt = [s for s in by_session.values() if s[0].workload == "sharegpt"]
    assert len(sharegpt) == 5 and all(len(s) == 2 for s in sharegpt)


def test_text_turns_are_closed_loop_and_sessions_start_in_the_window():
    requests, manifest = build({"sharegpt": 1.0})
    for r in requests:
        if r.turn_index == 0:
            assert 0 <= r.arrival_s < 100 and r.after_s is None
        else:
            assert r.arrival_s is None and r.after_s == mix.THINK_S["sharegpt"]
    firsts = [r.arrival_s for r in requests if r.turn_index == 0]
    assert firsts == sorted(firsts)
    assert manifest["workloads"]["sharegpt"]["think_s"] == 5.0


def test_bailian_keeps_recorded_times_relative_to_its_window():
    requests, manifest = build({"bailian-chat": 1.0}, seed=3)
    info = manifest["workloads"]["bailian-chat"]
    start = info["window_start_s"]
    # The trace spans 0 to 993 s: nine 100 s windows; seed 3 takes the third.
    assert (info["windows"], info["window_index"], start) == (9, 2, 200.0)
    for r in requests:
        assert r.after_s is None
        original = r.meta["chat_id"] // 2 * 10.0 + 3 * (r.meta["chat_id"] % 2)
        assert r.arrival_s == pytest.approx(original - start)
        assert r.request_id.startswith("bailian-chat:")
    roots = [r for r in requests if r.turn_index == 0]
    assert all(0 <= r.arrival_s < 100 for r in roots)
    assert info["requests"] == 20 and info["sessions"] == 10
    assert info["window_sessions"] == 10
    assert info["kept_session_fraction"] == 1.0


def test_same_seed_same_plan_other_seed_other_plan():
    a, _ = build({"sharegpt": 0.5, "bailian-chat": 0.5}, seed=3)
    b, _ = build({"sharegpt": 0.5, "bailian-chat": 0.5}, seed=3)
    c, _ = build({"sharegpt": 0.5, "bailian-chat": 0.5}, seed=4)
    assert [asdict(r) for r in a] == [asdict(r) for r in b]
    assert [asdict(r) for r in a] != [asdict(r) for r in c]


def test_ccbench_takes_at_most_one_run_per_task():
    runs = [
        conversation(f"ccbench:{task}-{model}", 1) for task in range(4)
        for model in ("glm-4-5", "kimi-k2")
    ]
    requests, _ = build({"ccbench": 1.0},
                        rate=0.08,
                        conversations={"ccbench": runs})
    tasks = [mix.ccbench_task(r.session_id) for r in requests]
    assert sorted(tasks) == ["0", "1", "2", "3"]


def test_unknown_workload_is_refused():
    with pytest.raises(ValueError, match="Unknown workload 'chat'"):
        build({"chat": 1.0})


def test_main_writes_plan_responses_and_manifest(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "processed").mkdir(parents=True)
    with (data / "processed" / "sharegpt.conversations.jsonl").open("w") as f:
        for i in range(10):
            c = conversation(f"sharegpt:c{i}", 2)
            f.write(
                json.dumps({
                    "conversation_id": c.conversation_id,
                    "messages": [asdict(m) for m in c.messages]
                }) + "\n")
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda name: FakeTokenizer())
    out = tmp_path / "plan"
    mix.main([
        "--mix", "only-sharegpt", "--seed", "1", "--out",
        str(out), "--data",
        str(data), "--duration", "20", "--rate", "0.4", "--pacing", "fast",
        "--tail", "10"
    ])
    plan = list(read_plan(out / "plan.jsonl.gz"))
    assert len(plan) == 8
    answers = [json.loads(x) for x in (out / "responses.jsonl").open()]
    assert [a["request_id"] for a in answers] == [r.request_id for r in plan]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["mix"] == "only-sharegpt" and manifest["seed"] == 1
    assert manifest["pacing"] == {
        "kind": "fast",
        "think_scale": 1.0,
        "think_s": mix.THINK_S
    }
    assert manifest["stop_s"] == 30.0
    source = data / "processed" / "sharegpt.conversations.jsonl"
    assert manifest["inputs"] == {str(source): mix.sha256(source)}


def test_bailian_turns_after_the_stop_are_dropped():
    # Roots every 10 s from 0 s, each with a child 8 s later. Seed 1 takes
    # the first window. With a 100 s window and no tail, the roots at 0..90 s
    # and all their children (8..98 s) are inside. With a 95 s window, the
    # root at 90 s is still inside but its child at 98 s is not.
    sessions = bailian_sessions(30, start=0.0)
    for s in sessions:
        s[1]["timestamp"] += 5
    spec = mix.MixSpec(name="t",
                       shares={"bailian-chat": 1.0},
                       duration_s=100.0,
                       rate=1.0,
                       tail_s=0.0)

    def plan():
        return mix.build(spec,
                         1, {}, {"bailian-chat": sessions},
                         FakeTokenizer(),
                         pacer=mix.Pacer("fast"))

    requests, manifest = plan()
    info = manifest["workloads"]["bailian-chat"]
    assert info["stop_s"] == 100.0
    assert info["window_turns_after_stop"] == 0
    assert info["window_requests"] == len(requests) == 20
    spec.duration_s = 95.0
    requests, manifest = plan()
    info = manifest["workloads"]["bailian-chat"]
    assert info["window_turns_after_stop"] == 1
    assert info["window_requests"] == len(requests) == 19
    assert all(r.meta["planned_s"] == r.arrival_s < 95.0 for r in requests)


# Pacing, the plan's stop and Bailian windows (spec 007 Phase 1) ----------

CHAT_GAPS = {2: [10.0], 3: [20.0], 4: [30.0], 5: [40.0], 6: [60.0]}


def test_human_pacing_borrows_chat_gaps_by_turn():
    # Turn index k is the trace's turn k + 1; turns 6 and later share a pool.
    pacer = mix.Pacer("human", chat_gaps=CHAT_GAPS)
    pauses = pacer.pauses("sharegpt", conversation("sharegpt:c", 7),
                          mix.random.Random(0))
    assert pauses == [None, 10.0, 20.0, 30.0, 40.0, 60.0, 60.0]
    scaled = mix.Pacer("human", think_scale=0.5, chat_gaps=CHAT_GAPS)
    assert scaled.pauses("sharegpt", conversation("sharegpt:c", 3),
                         mix.random.Random(0)) == [None, 5.0, 10.0]


def test_agents_use_recorded_tool_latencies():
    pacer = mix.Pacer("human", tool_gaps=[0.5])
    ccbench = conversation("ccbench:1-glm", 3)
    ccbench.turn_gaps_s = [None, 2.5, None]  # turn 2 has no timestamp
    rng = mix.random.Random(0)
    assert pacer.pauses("ccbench", ccbench, rng) == [None, 2.5, 0.5]
    assert pacer.pauses("agentbank", conversation("agentbank:t-1", 3),
                        rng) == [None, 0.5, 0.5]
    with pytest.raises(ValueError, match="CC-Bench's recorded tool"):
        mix.Pacer("human").pauses("agentbank",
                                  conversation("agentbank:t-1", 2), rng)


def test_fast_pacing_is_phase_0s_constants():
    pauses = mix.Pacer("fast").pauses("ccbench",
                                      conversation("ccbench:1-glm", 3),
                                      mix.random.Random(0))
    assert pauses == [None, 2.0, 2.0]
    with pytest.raises(ValueError, match="Unknown pacing"):
        mix.Pacer("slow")


def test_chat_gap_pool_removes_the_answer_time_and_floors():

    def r(chat_id, parent, t, out=100, turn=2, kind="text"):
        return {
            "chat_id": chat_id,
            "parent_chat_id": parent,
            "timestamp": t,
            "output_length": out,
            "turn": turn,
            "type": kind
        }

    records = [
        r(1, -1, 0.0, turn=1),
        r(2, 1, 10.0),  # 10 - 0.03 x 100 = 7 s
        r(3, 2, 12.0, turn=3),  # 2 - 3 < 1: the floor, 1 s
        r(4, 2, 50.0, turn=3, kind="file"),  # not a text request
        r(5, 99, 70.0),  # parent outside the trace
        r(6, 3, 200.0, turn=9),  # 200 - 12 - 3 = 185 s; turn 9 pools at 6
    ]
    assert mix.chat_gap_pool(records) == {2: [7.0], 3: [1.0], 6: [185.0]}


def test_tool_gap_pool_skips_unknown_gaps():
    a = conversation("ccbench:1-x", 3)
    a.turn_gaps_s = [None, 1.0, None]
    b = conversation("ccbench:2-x", 2)
    b.turn_gaps_s = [None, 0.25]
    assert mix.tool_gap_pool([a, b, conversation("ccbench:3-x",
                                                 2)]) == [1.0, 0.25]


def test_quantiles_are_nearest_rank():
    q = mix.quantiles([float(v) for v in range(1, 11)])
    assert q == {
        "n": 10,
        "mean": 5.5,
        "p10": 1.0,
        "p50": 5.0,
        "p90": 9.0,
        "p99": 10.0
    }
    assert mix.quantiles([]) == {}


def test_text_turns_planned_at_or_after_the_stop_are_dropped():
    # Three-turn sessions with 5 s pauses in a 100 s plan with no tail: a
    # session starting at s keeps the turns planned at s, s + 5, s + 10 that
    # are before 100 s, and only kept turns count toward the target.
    convs = [conversation(f"sharegpt:c{i}", 3) for i in range(60)]
    requests, manifest = build({"sharegpt": 1.0},
                               rate=0.3,
                               conversations={"sharegpt": convs},
                               tail=0.0)
    info = manifest["workloads"]["sharegpt"]
    sessions: Dict[Any, Any] = {}
    for r in requests:
        sessions.setdefault(r.session_id, []).append(r)
    cut = 0
    for session in sessions.values():
        start = session[0].arrival_s
        expected = [start + 5.0 * k for k in range(3) if start + 5 * k < 100]
        assert [r.meta["planned_s"]
                for r in session] == pytest.approx(expected)
        cut += 3 - len(session)
    assert info["turns_after_stop"] == cut
    assert info["requests"] == len(requests) == 30
    assert info["pause_s"]["p50"] == 5.0


def test_partition_windows_give_seeds_disjoint_trace_quarters():
    # Roots every 10 s from 0 to 400 s: four 100 s windows; seed k takes
    # window k - 1, and seed 5 wraps around to seed 1's window.
    traces = {"bailian-chat": bailian_sessions(41)}
    chats = {}
    for seed in (1, 2, 3, 4, 5):
        requests, manifest = build({"bailian-chat": 1.0},
                                   seed=seed,
                                   rate=1.0,
                                   traces=traces,
                                   tail=0.0)
        info = manifest["workloads"]["bailian-chat"]
        assert info["windows"] == 4
        assert info["window_start_s"] == 100.0 * ((seed - 1) % 4)
        chats[seed] = {r.meta["chat_id"] for r in requests}
    for a in (1, 2, 3, 4):
        for b in (1, 2, 3, 4):
            if a < b:
                assert not chats[a] & chats[b]
    assert chats[5] == chats[1]


def test_pacings_of_a_seed_share_session_starts():
    convs = [conversation(f"sharegpt:c{i}", 4) for i in range(80)]
    fast, _ = build({"sharegpt": 1.0},
                    conversations={"sharegpt": convs},
                    tail=0.0)
    human, manifest = build({"sharegpt": 1.0},
                            conversations={"sharegpt": convs},
                            tail=0.0,
                            pacer=mix.Pacer("human", chat_gaps=CHAT_GAPS))
    starts_fast = {
        r.session_id: r.arrival_s
        for r in fast if r.turn_index == 0
    }
    starts_human = {
        r.session_id: r.arrival_s
        for r in human if r.turn_index == 0
    }
    shared = starts_fast.keys() & starts_human.keys()
    assert shared
    assert all(starts_fast[s] == starts_human[s] for s in shared)
    pacing = manifest["pacing"]
    assert pacing["kind"] == "human"
    assert pacing["chat_gap_pool"] == {str(k): 1 for k in CHAT_GAPS}
    assert manifest["workloads"]["sharegpt"]["think_s"] is None


def test_main_with_human_pacing_reads_the_chat_trace(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "processed").mkdir(parents=True)
    with (data / "processed" / "sharegpt.conversations.jsonl").open("w") as f:
        for i in range(10):
            c = conversation(f"sharegpt:c{i}", 2)
            f.write(
                json.dumps({
                    "conversation_id": c.conversation_id,
                    "messages": [asdict(m) for m in c.messages]
                }) + "\n")
    trace = data / "raw" / "bailian" / "qwen_traceA_blksz_16.jsonl"
    trace.parent.mkdir(parents=True)
    with trace.open("w") as f:
        for root, child in bailian_sessions(3):
            child["turn"] = 2
            f.write(json.dumps(root) + "\n" + json.dumps(child) + "\n")
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda name: FakeTokenizer())
    out = tmp_path / "plan"
    mix.main([
        "--mix", "only-sharegpt", "--seed", "1", "--out",
        str(out), "--data",
        str(data), "--duration", "200", "--rate", "0.04"
    ])
    manifest = json.loads((out / "manifest.json").read_text())
    # Each child arrives 3 s after its root, whose answer is 4 tokens:
    # 3 - 0.12 = 2.88 s.
    assert manifest["pacing"]["chat_gap_pool"] == {"2": 3}
    assert set(manifest["inputs"]) == {
        str(data / "processed" / "sharegpt.conversations.jsonl"),
        str(trace)
    }
    plan = list(read_plan(out / "plan.jsonl.gz"))
    assert {r.after_s for r in plan if r.turn_index == 1} == {2.88}
    # bailian-chat only paced ShareGPT; it is not a workload of this mix.
    assert set(manifest["workloads"]) == {"sharegpt"}
