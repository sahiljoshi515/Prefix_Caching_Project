"""Milestone 3 workload adapters (specs/006-m3-workload-eviction FR-001..005).

Each fixture is a handcrafted record in the dataset's raw format, and every
expected value is worked out by hand from it.
"""
import gzip
import json
import sys
from collections import Counter

import pytest

from client_simulator import process_raw_traces
from client_simulator.adapters import agentbank, bailian, ccbench
from client_simulator.schema import Message, workload_of
from client_simulator.token_plan import read_plan

pytestmark = pytest.mark.m3

# AgentBank ---------------------------------------------------------------


def test_agentbank_maps_roles_and_merges_repeats():
    record = {
        "id":
        "alfworld_7",
        "conversations": [
            {
                "from": "human",
                "value": "Task: put a mug in the sink."
            },
            {
                "from": "human",
                "value": "You see a mug."
            },
            {
                "from": "gpt",
                "value": "Action: take mug"
            },
            {
                "from": "human",
                "value": "Observation: you hold the mug."
            },
            {
                "from": "gpt",
                "value": "Action: put mug in sink"
            },
            {
                "from": "human",
                "value": "Observation: done."
            },
        ]
    }
    counts = Counter()
    conversation = agentbank.to_conversation(record, counts, "alfworld")
    assert conversation.conversation_id == "agentbank:alfworld-alfworld_7"
    assert conversation.messages == [
        Message("user", "Task: put a mug in the sink.\n\nYou see a mug."),
        Message("assistant", "Action: take mug"),
        Message("user", "Observation: you hold the mug."),
        Message("assistant", "Action: put mug in sink"),
    ]
    assert counts == {
        "merged_messages": 1,
        "unanswered_final_users_removed": 1
    }


@pytest.mark.parametrize("messages,reason", [
    ([{
        "from": "gpt",
        "value": "hi"
    }], "skipped_assistant_first"),
    ([{
        "from": "human",
        "value": "hi"
    }], "skipped_no_exchange"),
    ([{
        "from": "tool",
        "value": "x"
    }], "skipped_unsupported_role"),
])
def test_agentbank_skips_unusable_trajectories(messages, reason):
    counts = Counter()
    record = {"id": "x_0", "conversations": messages}
    assert agentbank.to_conversation(record, counts, "x") is None
    assert counts[reason] == 1


# CC-Bench ----------------------------------------------------------------


def ccbench_row(events, task_id=12, model="GLM-4.5"):
    return {
        "task_id": task_id,
        "model_name": model,
        "trajectory": json.dumps(events)
    }


def event(kind, content, sidechain=False):
    return {
        "type": kind,
        "isSidechain": sidechain,
        "message": {
            "role": kind,
            "content": content
        }
    }


def test_ccbench_serializes_tools_and_drops_side_events():
    events = [
        {
            "type": "summary",
            "summary": "s"
        },
        event("user", "Build a todo app"),
        event("assistant", [{
            "type": "text",
            "text": "Planning."
        }]),
        event("assistant", [{
            "type": "tool_use",
            "name": "Write",
            "input": {
                "path": "a.py"
            }
        }]),
        event("assistant", [{
            "type": "text",
            "text": "sub-agent"
        }],
              sidechain=True),
        event("user", [{
            "type": "tool_result",
            "content": "ok"
        }]),
        {
            "type": "system",
            "content": "compacted"
        },
        event("assistant", [{
            "type": "thinking",
            "thinking": "hmm"
        }, {
            "type": "text",
            "text": "Done."
        }]),
        event("user", "thanks"),
    ]
    counts = Counter()
    conversation = ccbench.to_conversation(ccbench_row(events), counts)
    assert conversation.conversation_id == "ccbench:12-glm-4-5"
    assert conversation.messages == [
        Message("user", "Build a todo app"),
        Message(
            "assistant",
            'Planning.\n\n<tool_use name="Write">{"path": "a.py"}</tool_use>'),
        Message("user", "<tool_result>\nok\n</tool_result>"),
        Message("assistant", "Done."),
    ]
    assert counts == {
        "dropped_other_events": 2,
        "dropped_sidechain_events": 1,
        "merged_messages": 1,
        "unanswered_final_users_removed": 1,
    }


def test_ccbench_skips_a_trajectory_that_starts_with_the_assistant():
    counts = Counter()
    row = ccbench_row([event("assistant", "hi"), event("user", "x")])
    assert ccbench.to_conversation(row, counts) is None
    assert counts["skipped_assistant_first"] == 1


# Bailian -----------------------------------------------------------------


def rec(chat_id, parent, t, input_length, output_length, hash_ids):
    return {
        "chat_id": chat_id,
        "parent_chat_id": parent,
        "timestamp": t,
        "input_length": input_length,
        "output_length": output_length,
        "type": "text",
        "turn": 1,
        "hash_ids": hash_ids,
    }


W = "bailian-chat"


def blk(hash_id, workload=W):
    return list(bailian.block_tokens(workload, hash_id))


def test_block_tokens_are_fixed_ordinary_tokens_namespaced_by_trace():
    a = bailian.block_tokens("bailian-chat", 10)
    assert a == bailian.block_tokens("bailian-chat", 10)
    assert len(a) == 16
    assert all(bailian.TOKEN_LOW <= t < bailian.TOKEN_HIGH for t in a)
    # The traces were anonymized separately: the same id is other content.
    assert a != bailian.block_tokens("bailian-api", 10)
    assert a != bailian.block_tokens("bailian-chat", 11)


def test_root_prompt_is_its_blocks_cut_to_the_input_length():
    [request] = bailian.synthesize(W, [rec(1, -1, 3.5, 20, 5, [10, 11])])
    assert request.request_id == "bailian-chat:1:0"
    assert request.session_id == "bailian-chat:1"
    assert request.prompt_token_ids == blk(10) + blk(11)[:4]
    assert len(request.output_token_ids) == 5
    assert request.arrival_s == 3.5
    assert request.meta["output_from_child"] is False


def test_child_extends_parent_and_parent_output_is_the_continuation():
    # Parent: 20 tokens (one full block + 4); output 8. The child repeats
    # the parent's full block (10) and is long enough (40 >= 28) to include
    # the parent's whole output.
    parent = rec(1, -1, 0.0, 20, 8, [10, 11])
    child = rec(2, 1, 5.0, 40, 3, [10, 12, 13])
    p, c = bailian.synthesize(W, [parent, child])
    assert p.prompt_token_ids == blk(10) + blk(11)[:4]
    # The child keeps the parent's exact prompt, then its own blocks.
    assert c.prompt_token_ids == p.prompt_token_ids + blk(12)[4:] + blk(13)[:8]
    assert p.output_token_ids == c.prompt_token_ids[20:28] == blk(12)[4:12]
    assert p.meta["output_from_child"] is True
    assert c.request_id == "bailian-chat:1:1"


def test_child_that_drops_part_of_the_output_leaves_filler():
    # 24 < 20 + 8: like a thinking trace, the next turn omits part of the
    # previous output, so the parent's output cannot be the continuation.
    parent = rec(1, -1, 0.0, 20, 8, [10, 11])
    child = rec(2, 1, 5.0, 24, 3, [10, 12])
    p, c = bailian.synthesize(W, [parent, child])
    assert c.prompt_token_ids == p.prompt_token_ids + blk(12)[4:8]
    assert p.output_token_ids == bailian.filler_tokens(W, 1, 8)
    assert p.meta["output_from_child"] is False


def test_child_with_other_full_blocks_does_not_extend_its_parent():
    parent = rec(1, -1, 0.0, 20, 8, [10, 11])
    child = rec(2, 1, 5.0, 40, 3, [99, 12, 13])
    p, c = bailian.synthesize(W, [parent, child])
    assert c.prompt_token_ids == blk(99) + blk(12) + blk(13)[:8]
    assert p.meta["output_from_child"] is False


def test_first_qualifying_child_sets_the_output():
    parent = rec(1, -1, 0.0, 20, 8, [10, 11])
    short = rec(2, 1, 3.0, 24, 1, [10, 12])
    long_ = rec(3, 1, 6.0, 40, 1, [10, 14, 15])
    p, _, c = bailian.synthesize(W, [parent, short, long_])
    assert p.output_token_ids == c.prompt_token_ids[20:28] == blk(14)[4:12]


def test_sessions_group_trees_and_orphans_in_time_order():
    records = [
        rec(3, 1, 9.0, 16, 1, [1]),
        rec(1, -1, 1.0, 16, 1, [1]),
        rec(2, 1, 4.0, 16, 1, [1]),
        rec(5, 77, 2.0, 16, 1, [2]),  # parent not in the trace
        rec(4, -1, 0.5, 16, 1, [3]),
    ]
    groups = [[r["chat_id"] for r in s] for s in bailian.sessions(records)]
    assert groups == [[4], [1, 2, 3], [5]]


def test_synthesize_refuses_other_block_sizes_and_bad_lengths():
    with pytest.raises(ValueError, match="block size 32"):
        bailian.synthesize(W, [rec(1, -1, 0, 16, 1, [1])], block_size=32)
    with pytest.raises(ValueError, match="3 hash ids for 20"):
        bailian.synthesize(W, [rec(1, -1, 0, 20, 1, [1, 2, 3])])


def test_bailian_is_a_token_level_dataset_of_process_raw_traces(
        tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in bailian.TRACES.values():
        (raw / name).write_text("")
    records = [
        rec(1, -1, 0.0, 20, 8, [10, 11]),
        rec(2, 1, 5.0, 40, 3, [10, 12, 13])
    ]
    (raw / bailian.TRACES[W]).write_text("\n".join(map(json.dumps, records)))
    out = tmp_path / "out" / "chat.responses.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "process_raw_traces", "--dataset", W, "--raw",
        str(raw), "--output",
        str(out)
    ])
    process_raw_traces.main()
    plan = list(read_plan(out.with_name("chat.responses.plan.jsonl")))
    assert [r.request_id
            for r in plan] == ["bailian-chat:1:0", "bailian-chat:1:1"]
    answers = [json.loads(line) for line in out.read_text().splitlines()]
    assert answers[0] == {
        "request_id": "bailian-chat:1:0",
        "response_token_ids": plan[0].output_token_ids
    }


def test_workload_tag_is_the_text_before_the_first_colon():
    assert workload_of("bailian-chat:12:3") == "bailian-chat"
    assert workload_of("agentbank:alfworld-alfworld_0:2") == "agentbank"


def test_plan_round_trips_through_gzip(tmp_path):
    requests = bailian.synthesize(W, [rec(1, -1, 0.0, 20, 8, [10, 11])])
    from client_simulator.token_plan import write_plan
    assert write_plan(requests, tmp_path / "p.jsonl.gz",
                      tmp_path / "r.jsonl") == 1
    with gzip.open(tmp_path / "p.jsonl.gz", "rt") as f:
        assert json.loads(f.readline())["request_id"] == "bailian-chat:1:0"
    assert list(read_plan(tmp_path / "p.jsonl.gz")) == requests
