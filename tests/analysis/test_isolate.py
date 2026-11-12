"""Isolated plans cut from a mixed plan (spec 007, mixed vs isolated)."""
import json

import pytest

from analysis import isolate
from client_simulator.schema import TokenRequest
from client_simulator.token_plan import read_plan, write_plan

pytestmark = pytest.mark.m3


def req(rid, prompt, arrival=None, after=None, planned=None):
    session, turn = rid.rsplit(":", 1)
    return TokenRequest(request_id=rid,
                        session_id=session,
                        turn_index=int(turn),
                        prompt_token_ids=list(prompt),
                        output_token_ids=[7, 8],
                        arrival_s=arrival,
                        after_s=after,
                        meta={} if planned is None else {"planned_s": planned})


# A mix in plan order: ShareGPT closed loop, two Bailian traces open loop.
MIX = [
    req("bailian-chat:1:0", range(10), arrival=0.5),
    req("sharegpt:s1:0", range(4), arrival=1.0),
    req("bailian-api:2:0", range(6), arrival=2.0),
    req("sharegpt:s1:1", range(8), after=3.0, planned=4.0),
    req("bailian-chat:1:1", range(12), arrival=9.0),
]


def write_mix(path):
    write_plan(MIX, path / "plan.jsonl.gz", path / "responses.jsonl")
    stats = {}
    for w in ("sharegpt", "bailian-chat", "bailian-api"):
        mine = [r for r in MIX if r.workload == w]
        stats[w] = {
            "requests": len(mine),
            "prompt_tokens": sum(len(r.prompt_token_ids) for r in mine),
            "request_share": 0.0,
        }
    manifest = {
        "mix": "balanced",
        "seed": 2,
        "pacing": {
            "kind": "human"
        },
        "stop_s": 1800.0,
        "shares": {w: 1 / 3
                   for w in stats},
        "requests": len(MIX),
        "sessions": 3,
        "prompt_tokens": 40,
        "workloads": stats,
        "commit": "src123",
        "dirty": False,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_groups_are_the_four_course_workloads():
    assert list(
        isolate.GROUPS) == ["sharegpt", "agentbank", "ccbench", "bailian"]
    assert isolate.GROUPS["bailian"] == ("bailian-api", "bailian-chat",
                                         "bailian-coder", "bailian-thinking")
    assert isolate.group_of("bailian-coder") == "bailian"
    assert isolate.group_of("ccbench") == "ccbench"
    with pytest.raises(ValueError, match="no group"):
        isolate.group_of("nope")
    assert isolate.isolated_mix("bailian") == "alone-bailian"
    assert isolate.group_of_mix("alone-sharegpt") == "sharegpt"
    assert isolate.group_of_mix("balanced") is None
    with pytest.raises(ValueError, match="Unknown group"):
        isolate.group_of_mix("alone-nope")


def test_isolated_plan_keeps_the_groups_requests_and_times(
        tmp_path, monkeypatch):
    monkeypatch.setattr("analysis.run_m2_task2.git_state", lambda: {
        "commit": "new456",
        "dirty": False
    })
    source = tmp_path / "balanced-human-s2"
    write_mix(source)
    out = tmp_path / "alone-bailian-human-s2"
    manifest = isolate.isolate(source, "bailian", out)
    kept = list(read_plan(out / "plan.jsonl.gz"))
    # Same requests, same order, same times and pauses: only the others go.
    assert kept == [MIX[0], MIX[2], MIX[4]]
    answers = [
        json.loads(line)
        for line in (out / "responses.jsonl").read_text().splitlines()
    ]
    assert [a["request_id"] for a in answers] == [r.request_id for r in kept]
    assert json.loads((out / "manifest.json").read_text()) == manifest
    assert manifest["mix"] == "alone-bailian"
    assert manifest["seed"] == 2 and manifest["pacing"] == {"kind": "human"}
    assert manifest["requests"] == 3 and manifest["sessions"] == 2
    assert manifest["prompt_tokens"] == 28
    assert set(manifest["workloads"]) == {"bailian-chat", "bailian-api"}
    chat = manifest["workloads"]["bailian-chat"]
    assert chat["request_share"] == pytest.approx(2 / 3)
    assert chat["prompt_token_share"] == pytest.approx(22 / 28)
    assert manifest["shares"] == {"bailian-api": 1 / 3, "bailian-chat": 1 / 3}
    src = manifest["isolated_from"]
    assert src["plan"] == "balanced-human-s2" and src["group"] == "bailian"
    assert src["commit"] == "src123" and len(src["plan_sha256"]) == 64
    assert manifest["commit"] == "new456" and manifest["dirty"] is False
    # ShareGPT alone keeps its closed-loop turn's pause and planned time.
    isolate.isolate(source, "sharegpt", tmp_path / "alone-sharegpt-human-s2")
    alone = list(
        read_plan(tmp_path / "alone-sharegpt-human-s2" / "plan.jsonl.gz"))
    assert alone == [MIX[1], MIX[3]]
    assert alone[1].after_s == 3.0 and alone[1].meta["planned_s"] == 4.0


def test_a_group_missing_from_the_plan_or_a_stale_manifest_is_refused(
        tmp_path):
    source = tmp_path / "balanced-human-s2"
    manifest = write_mix(source)
    with pytest.raises(ValueError, match="no ccbench requests"):
        isolate.isolate(source, "ccbench", tmp_path / "out")
    manifest["workloads"]["sharegpt"]["prompt_tokens"] += 1
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sharegpt: the source plan holds"):
        isolate.isolate(source, "sharegpt", tmp_path / "out2")
    assert not (tmp_path / "out2" / "manifest.json").exists()
