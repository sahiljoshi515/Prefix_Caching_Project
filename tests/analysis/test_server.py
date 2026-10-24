"""Server launcher checks that need no vLLM process."""
import socket

import pytest

from analysis import server

pytestmark = pytest.mark.m2


def test_busy_port_is_refused_before_starting(tmp_path):
    with socket.socket() as busy:
        busy.bind((server.HOST, 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert server.port_in_use(port)
        launcher = server.SimulatorServer("m",
                                          tmp_path / "trace.jsonl",
                                          log_path=tmp_path / "server.log",
                                          port=port)
        with pytest.raises(RuntimeError, match="already in use"):
            launcher.__enter__()
        assert launcher.process is None


def test_command_binds_the_probed_host(tmp_path):
    launcher = server.SimulatorServer("m",
                                      tmp_path / "trace.jsonl",
                                      log_path=tmp_path / "server.log",
                                      event_log=tmp_path / "events.jsonl",
                                      args=["--enable-prefix-caching"],
                                      port=8123)
    command = launcher.command
    assert command[command.index("--host") + 1] == server.HOST
    assert launcher.base_url == f"http://{server.HOST}:8123/v1"
    assert launcher.env["VLLM_CACHE_EVENT_LOG"] == str(tmp_path /
                                                       "events.jsonl")
    assert launcher.env["VLLM_USE_SIMULATOR"] == "1"


def test_kv_cache_blocks_are_read_from_the_log(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("INFO simulator.py:294] Simulated KV cache: 21845 blocks "
                   "of 16 tokens (+ 21845 swap blocks)\n")
    launcher = server.SimulatorServer("m", tmp_path / "t", log_path=log)
    assert launcher.kv_cache_blocks() == {
        "num_gpu_blocks": 21845,
        "block_size": 16
    }


def test_shell_cache_settings_are_not_inherited(tmp_path, monkeypatch):
    """A leftover VLLM_EVICTION_POLICY would run a server under a policy
    its manifest does not record."""
    monkeypatch.setenv("VLLM_EVICTION_POLICY", "lfu")
    monkeypatch.setenv("VLLM_CACHE_EVENT_LOG", str(tmp_path / "stale.jsonl"))
    plain = server.SimulatorServer("m",
                                   tmp_path / "t",
                                   log_path=tmp_path / "l")
    assert "VLLM_EVICTION_POLICY" not in plain.env
    assert "VLLM_CACHE_EVENT_LOG" not in plain.env
    chosen = server.SimulatorServer("m",
                                    tmp_path / "t",
                                    log_path=tmp_path / "l",
                                    env={"VLLM_EVICTION_POLICY": "fifo"})
    assert chosen.env["VLLM_EVICTION_POLICY"] == "fifo"
