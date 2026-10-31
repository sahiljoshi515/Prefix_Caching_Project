"""Mixed-workload plans (specs/007-m3-mixed-workload-policy, FR-001, US1).

Uses handcrafted conversations, a toy Bailian trace and a fake tokenizer,
so the expected counts and times can be worked out by hand.
"""
import json
from dataclasses import asdict

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
          traces=None):
    spec = mix.MixSpec(name="test",
                       shares=shares,
                       duration_s=duration,
                       rate=rate,
                       max_model_len=max_model_len)
    conversations = conversations or {
        "sharegpt": [conversation(f"sharegpt:c{i}", 2) for i in range(40)],
        "agentbank": [conversation(f"agentbank:t-{i}", 1) for i in range(40)],
    }
    traces = traces or {"bailian-chat": bailian_sessions(100)}
    return mix.build(spec, seed, conversations, traces, FakeTokenizer())


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
    by_session = {}
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
    requests, manifest = build({"bailian-chat": 1.0})
    info = manifest["workloads"]["bailian-chat"]
    start = info["window_start_s"]
    assert 0 <= start <= 993 - 100  # the trace ends at 993 s
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
        str(data), "--duration", "20", "--rate", "0.4"
    ])
    plan = list(read_plan(out / "plan.jsonl.gz"))
    assert len(plan) == 8
    answers = [json.loads(x) for x in (out / "responses.jsonl").open()]
    assert [a["request_id"] for a in answers] == [r.request_id for r in plan]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["mix"] == "only-sharegpt" and manifest["seed"] == 1
