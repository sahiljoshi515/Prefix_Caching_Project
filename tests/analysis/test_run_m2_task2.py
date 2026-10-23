"""Offline parts of the Task 2 experiment command (no server)."""
import csv
import json

import pytest

from analysis import run_m2_task2 as run

from .test_metrics import LOG, RESULTS, write_log

pytestmark = pytest.mark.m2


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def conversation(cid, turns):
    messages = []
    for i in range(turns):
        messages += [{
            "role": "user",
            "content": f"q{i}"
        }, {
            "role": "assistant",
            "content": f"a{i}"
        }]
    return {"conversation_id": cid, "messages": messages}


def test_write_sample_keeps_first_conversations_and_their_responses(tmp_path):
    conversations = write_jsonl(tmp_path / "c.jsonl", [
        conversation("s:A", 2),
        conversation("s:B", 1),
        conversation("s:C", 3)
    ])
    responses = write_jsonl(tmp_path / "r.jsonl", [{
        "request_id": f"{cid}:{t}",
        "response": "x"
    } for cid, n in (("s:A", 2), ("s:B", 1), ("s:C", 3)) for t in range(n)])
    out = tmp_path / "out"
    out.mkdir()
    sample = run.write_sample(conversations, responses, 2, out)
    assert (sample["conversations"], sample["turns"]) == (2, 3)
    kept = [
        json.loads(line)["request_id"]
        for line in (out / "sample.responses.jsonl").open()
    ]
    assert kept == ["s:A:0", "s:A:1", "s:B:0"]


def test_write_sample_rejects_missing_responses(tmp_path):
    conversations = write_jsonl(tmp_path / "c.jsonl", [conversation("s:A", 2)])
    responses = write_jsonl(tmp_path / "r.jsonl", [{
        "request_id": "s:A:0",
        "response": "x"
    }])
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="1 responses for 2"):
        run.write_sample(conversations, responses, 1, out)


def make_run(run_dir, hit_rate=0.5):
    run_dir.mkdir(parents=True)
    write_log(run_dir / "events.jsonl", LOG)
    write_jsonl(run_dir / "client.jsonl", RESULTS)
    (run_dir / "client.jsonl.metrics.after.txt").write_text(
        f'vllm:gpu_prefix_cache_hit_rate{{model_name="m"}} {hit_rate}\n')
    return run_dir


def test_analyze_run_writes_csvs_and_self_checks(tmp_path):
    run_dir = make_run(tmp_path / "multi")
    summary = run.analyze_run(run_dir, window_s=5)
    assert summary["requests_completed"] == 3
    assert summary["requests_excluded"] == {"skipped_context": 1}
    assert summary["log_complete"]
    assert summary["sc003_pass"]
    assert summary["sc004_gap_signs"] == {"zero": 2, "positive": 1}
    assert summary["blocks"] == 4
    assert summary["blocks_with_hits_share"] == 0.75
    assert summary["hit_sources"] == {
        "cross/allocate": 0.5,
        "intra/allocate": 0.25,
        "intra/append_slots": 0.25,
    }
    with (run_dir / "requests.csv").open() as f:
        requests = list(csv.DictReader(f))
    assert [r["gap_tokens"] for r in requests] == ["0", "0", "16"]
    assert json.loads((run_dir / "summary.json").read_text()) == json.loads(
        json.dumps(summary))
    for name in ("blocks.csv", "reuse_intervals.csv", "hit_rate_windows.csv"):
        assert (run_dir / name).stat().st_size > 0


def test_analyze_run_fails_sc003_when_the_metric_disagrees(tmp_path):
    summary = run.analyze_run(make_run(tmp_path / "single", hit_rate=0.6),
                              window_s=5)
    assert not summary["sc003_pass"]


def test_make_figures_draws_every_figure(tmp_path):
    pytest.importorskip("matplotlib")
    for setting in run.SETTINGS:
        run.analyze_run(make_run(tmp_path / setting), window_s=5)
    written = run.make_figures(tmp_path,
                               tmp_path / "figures",
                               run.SETTINGS,
                               window_s=5)
    names = sorted(p.name for p in written)
    assert len(names) == 14  # seven figures, PDF and PNG
    assert "fig4_cached_by_turn.pdf" in names
    assert all(p.stat().st_size > 0 for p in written)


def test_existing_output_folder_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="exists"):
        run.main(["--out", str(tmp_path)])
