"""Start and stop `vllm serve` on the simulator for one experiment run.

Shared by the Milestone 2 and 3 experiments (specs 004 to 007). The server
is stopped with SIGINT so the engine closes the cache event log cleanly.
"""
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Sequence

# Bind and probe 127.0.0.1 explicitly. With vLLM's default 0.0.0.0, macOS
# lets the server start beside another program on 127.0.0.1:<port>, which
# then receives our requests.
HOST = "127.0.0.1"
KV_CACHE_LINE = re.compile(r"Simulated KV cache: (\d+) blocks of (\d+) tokens")


class SimulatorServer:
    """Context manager around one `vllm serve` process.

    Args:
        model: served model (only its config and tokenizer are read).
        trace: VLLM_SIMULATOR_TRACE, the responses JSONL.
        log_path: where the server's stdout and stderr go.
        event_log: VLLM_CACHE_EVENT_LOG, or None for no event log.
        args: extra `vllm serve` arguments.
        env: extra environment variables (e.g. VLLM_CPU_KVCACHE_SPACE,
            VLLM_EVICTION_POLICY).

    The shell's VLLM_CACHE_EVENT_LOG and VLLM_EVICTION_POLICY are never
    inherited: an experiment sets them through `event_log` and `env`, so its
    manifest describes the server it ran.
    """

    def __init__(self,
                 model: str,
                 trace: Path,
                 log_path: Path,
                 event_log: Optional[Path] = None,
                 args: Sequence[str] = (),
                 env: Optional[Dict[str, str]] = None,
                 port: int = 8000,
                 startup_timeout_s: float = 300.0):
        self.model = model
        self.port = port
        self.log_path = Path(log_path)
        self.startup_timeout_s = startup_timeout_s
        self.command = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--host", HOST, "--port",
            str(port), *args
        ]
        self.env = dict(os.environ)
        self.env.update(VLLM_USE_SIMULATOR="1",
                        VLLM_NO_USAGE_STATS="1",
                        VLLM_SIMULATOR_TRACE=str(trace))
        for name in ("VLLM_CACHE_EVENT_LOG", "VLLM_EVICTION_POLICY"):
            self.env.pop(name, None)
        if event_log is not None:
            self.env["VLLM_CACHE_EVENT_LOG"] = str(event_log)
        self.env.update(env or {})
        self.process: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{HOST}:{self.port}/v1"

    def __enter__(self) -> "SimulatorServer":
        if port_in_use(self.port):
            raise RuntimeError(f"Port {self.port} is already in use; stop "
                               "that program or choose another port")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w") as log:
            self.process = subprocess.Popen(self.command,
                                            env=self.env,
                                            stdout=log,
                                            stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited with {self.process.returncode}; "
                    f"see {self.log_path}:\n{self.log_tail()}")
            if self._healthy():
                return self
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"vllm serve not healthy after "
                           f"{self.startup_timeout_s} s; see {self.log_path}")

    def __exit__(self, *exc) -> None:
        self.stop()

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{self.port}/health",
                                        timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return False

    def stop(self, timeout_s: float = 60.0) -> None:
        """SIGINT, then SIGKILL if the server does not exit in time."""
        if self.process is None or self.process.poll() is not None:
            return
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout_s)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def log_tail(self, lines: int = 20) -> str:
        text = self.log_path.read_text(errors="replace").splitlines()
        return "\n".join(text[-lines:])

    def kv_cache_blocks(self) -> Optional[Dict[str, int]]:
        """GPU blocks and block size from the simulator's startup line."""
        match = KV_CACHE_LINE.search(self.log_path.read_text(errors="replace"))
        if match is None:
            return None
        return {
            "num_gpu_blocks": int(match.group(1)),
            "block_size": int(match.group(2))
        }


def port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex((HOST, port)) == 0
