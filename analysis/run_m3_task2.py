"""Milestone 3 Task 2: build plans, replay the engine grid, sweep the
simulator (spec 007).

::

    # 1. Build the plans a grid needs (analysis/mix.py), once per machine:
    python -m analysis.run_m3_task2 plans --grid phase1
    # 2. Replay the grid on fresh simulator servers, 6 at a time:
    python -m analysis.run_m3_task2 grid --grid phase1 --out results/m3-task2
    #    Interrupted? Run the same command again: finished runs are kept.
    #    Two laptops: add --shard 1/2 on one and --shard 2/2 on the other
    #    (to both commands); each plan's runs stay on one machine.
    # Sweep the fast cache simulator (a design tool; reported numbers come
    # from the grid), timed by each plan's own no-eviction engine run:
    python -m analysis.run_m3_task2 sim \\
        --plans data/processed/m3/balanced-human-s1 \\
        --policies lru lfu fifo --capacities 65536 131072 \\
        --timing 'engine:results/m3-task2/{plan}-ceiling' \\
        --out results/m3-sim/train.jsonl
    # Replay one plan once:
    python -m analysis.run_m3_task2 replay \\
        --plan data/processed/m3/balanced-human-s1 --policy lru \\
        --capacity 131072 --out results/m3-task2/one-off

A run replays a plan on a fresh server (empty cache) with the cache event
log on. Its folder is laid out like Task 2's (``client.jsonl``, its metric
snapshots, ``events.jsonl.gz``, ``server.log``, ``run.json``), so
``analysis/run_m2_task2.analyze_run`` reads it. ``run.json`` records what
constitution V asks for: the plan's manifest (datasets, sample, pacing,
seed), the model, the capacity in blocks, the block size, the policy and the
commit.

A grid run is named ``<mix>-<pacing>-s<seed>-<policy>-cap<tokens>``, or
``<mix>-<pacing>-s<seed>-ceiling``, plus ``-r<n>`` for a replicate, and
lives in that folder under ``--out``. The grid:

- skips runs whose ``run.json`` has ``finished_at_utc``, once it has
  checked that each holds the configuration its name says (`run_config`;
  runs made with ``replay`` qualify) and that a ceiling never evicted;
- deletes and replays an unfinished run's folder once it has stopped: the
  previous grid saw it fail, or nothing in it changed for ``QUIET_S``. If
  one changed more recently, something may still be replaying it (another
  checkout's ``replay``), and the grid refuses to start;
- refuses to start on a dirty working tree (``--allow-dirty`` overrides),
  since each run records the commit it ran;
- gives each of its ``--jobs`` workers its own port (``--base-port`` + i);
- has the client stop sending ``--stop-slack`` seconds after the plan's
  stop (``plan_replay --stop-at``), a guard against a replay that falls far
  behind;
- replays each ceiling on a cache of ``CEILING_MARGIN`` times the distinct
  full blocks its plan can ever cache (or ``--ceiling-capacity``, if that is
  enough), and fails a ceiling run that evicted;
- keeps a Mac awake while it runs, and writes each run's status to
  ``<out>/grid.json``.

Ctrl-C reaches the servers and clients too, so the runs in progress stop
and are replayed next time. ``kill -INT`` of this process alone starts no
new run and lets the running ones finish.
"""
import argparse
import contextlib
import dataclasses
import fcntl
import functools
import gzip
import json
import math
import multiprocessing
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (IO, Any, Callable, Dict, Iterator, List, Optional,
                    Sequence, Set, Tuple, Union)

from analysis import events as ev
from analysis.cachesim import block_hashes  # noqa: F401 (re-exported)
from analysis.run_m2_task2 import git_state
from analysis.server import SimulatorServer, port_in_use
from analysis.sweep import is_finished
from client_simulator.token_plan import read_plan

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# The plans drop turns longer than this (analysis/mix.py --max-model-len).
MAX_MODEL_LEN = 32768
GRIDS = ("phase1", "smoke")
BASELINES = ("lru", "lfu", "fifo")
# Phase 0's capacity (plan.md): there LRU's usable reuse fell 12.9 points
# short of the no-eviction ceiling, so the mix is under pressure (FR-007).
DEFAULT_CAPACITY = 131072
# A ceiling's cache holds this many times the distinct full blocks its plan
# can ever cache. The extra room holds the partial last blocks of running
# requests, which are never cached, so the ceiling never has to evict.
CEILING_MARGIN = 1.25
# Plans whose name is not the mix they are built from, with their extra
# analysis.mix arguments: the smoke plan is the balanced mix over 180 s.
PLAN_RECIPES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "smoke": ("balanced", ("--duration", "180")),
}

CapacitySpec = Union[int, str]  # tokens, or "ceiling"


class RunnerError(RuntimeError):
    """A run or grid that cannot go ahead; the message says what to fix."""


def _say(message: str) -> None:
    # Flushed, so progress shows up when the output goes to a file (nohup).
    print(message, flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: Any) -> None:
    """Write through a rename, so an interruption never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def replay_plan(plan_dir: Path,
                run_dir: Path,
                *,
                policy: str,
                capacity_tokens: int,
                block_size: int = 16,
                model: str = DEFAULT_MODEL,
                max_model_len: int = MAX_MODEL_LEN,
                port: int = 8000,
                labels: Optional[str] = None,
                event_log: bool = True,
                max_concurrency: int = 64,
                stop_at: Optional[float] = None,
                manifest_updates: Optional[Dict[str, Any]] = None,
                extra_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Replay `plan_dir`'s plan into `run_dir` (which must not exist).

    `stop_at` is passed to the client as ``--stop-at``: requests not ready
    that many seconds after the replay starts are not sent. The plan already
    ends at its stop, so this only bounds a replay that falls far behind.
    `manifest_updates` are added to ``run.json`` (the grid records its
    configuration there).
    """
    if capacity_tokens % block_size:
        raise ValueError("The capacity must be a whole number of blocks")
    plan = json.loads((plan_dir / "manifest.json").read_text())
    run_dir.mkdir(parents=True)
    num_blocks = capacity_tokens // block_size
    # --disable-log-requests: vLLM would log every token-id prompt (about
    # 100 MB per run); it does not change what the cache does.
    server_args = [
        "--enable-prefix-caching", "--enable-prompt-tokens-details",
        "--disable-log-requests", "--max-model-len",
        str(max_model_len), "--block-size",
        str(block_size), "--num-gpu-blocks-override",
        str(num_blocks)
    ]
    env = {"VLLM_EVICTION_POLICY": policy, **(extra_env or {})}
    if labels is not None:
        env["VLLM_WA_LABELS"] = labels
    client_output = run_dir / "client.jsonl"
    manifest: Dict[str, Any] = {
        "experiment": "m3-task2",
        **git_state(),
        "started_at_utc": _now(),
        "plan_dir": str(plan_dir),
        "plan": plan,
        "model": model,
        "server_args": server_args,
        "eviction_policy": policy,
        "wa_labels": labels,
        "block_size": block_size,
        "capacity_tokens": capacity_tokens,
        "num_gpu_blocks": num_blocks,
        "env": env,
        "max_concurrency": max_concurrency,
        "stop_at": stop_at,
        **(manifest_updates or {}),
    }
    with SimulatorServer(model,
                         plan_dir / "responses.jsonl",
                         log_path=run_dir / "server.log",
                         event_log=run_dir /
                         "events.jsonl.gz" if event_log else None,
                         args=server_args,
                         env=env,
                         port=port) as server:
        manifest["kv_cache"] = server.kv_cache_blocks()
        command = [
            sys.executable, "-m", "client_simulator.plan_replay", "--plan",
            str(plan_dir / "plan.jsonl.gz"), "--model", model, "--base-url",
            server.base_url, "--max-concurrency",
            str(max_concurrency), "--output",
            str(client_output), "--capture-metrics"
        ]
        if stop_at is not None:
            command += ["--stop-at", str(stop_at)]
        client = subprocess.run(command,
                                capture_output=True,
                                text=True,
                                check=False)
    (run_dir / "client.out").write_text(client.stdout + client.stderr)
    client_run = json.loads(Path(str(client_output) + ".run.json").read_text())
    # Exit code 1 only means some requests were skipped or failed; those are
    # counted in the summary. Anything else is a broken run.
    if client.returncode not in (0, 1) or client_run["status"] != "finished":
        raise RuntimeError(f"client failed ({client.returncode}); see "
                           f"{run_dir / 'client.out'}")
    manifest["client_summary"] = client_run.get("summary")
    manifest["finished_at_utc"] = _now()
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --- Configurations and grids -------------------------------------------


@dataclass(frozen=True)
class MixConfig:
    """One engine run of a grid: a plan (mix, pacing, seed) replayed under
    one eviction policy at one capacity.

    A ceiling run replays its plan under LRU on a cache that never evicts
    (`ceiling_capacity`), so `capacity_tokens` is ignored and left out of
    its name: one ceiling serves every capacity. `replicate` > 0 repeats an
    otherwise identical run, to measure run-to-run noise.
    """
    mix: str
    pacing: str
    seed: int
    policy: str
    capacity_tokens: int
    block_size: int = 16
    ceiling: bool = False
    replicate: int = 0

    def __post_init__(self) -> None:
        if self.ceiling and self.policy != "lru":
            raise ValueError("A ceiling run uses LRU: without evictions the "
                             "policy has nothing to decide")
        if self.block_size <= 0 or self.replicate < 0:
            raise ValueError("The block size must be positive and the "
                             "replicate not negative")
        if not self.ceiling and (self.capacity_tokens <= 0
                                 or self.capacity_tokens % self.block_size):
            raise ValueError(f"Capacity {self.capacity_tokens} tokens is not "
                             "a positive multiple of block size "
                             f"{self.block_size}")

    @property
    def plan_name(self) -> str:
        return f"{self.mix}-{self.pacing}-s{self.seed}"

    @property
    def name(self) -> str:
        if self.ceiling:
            name = f"{self.plan_name}-ceiling"
        else:
            name = f"{self.plan_name}-{self.policy}-cap{self.capacity_tokens}"
        return name + (f"-r{self.replicate}" if self.replicate else "")

    def record(self) -> Dict[str, Any]:
        """What ``run.json`` records about the configuration."""
        return {
            "config": self.name,
            "mix": self.mix,
            "pacing": self.pacing,
            "seed": self.seed,
            "replicate": self.replicate,
            "ceiling": self.ceiling,
        }


def grid_configs(name: str,
                 capacity_tokens: int = DEFAULT_CAPACITY) -> List[MixConfig]:
    """The configurations of grid `name`, in run order.

    ``phase1`` (21 runs) replays balanced-mix plans, each under LRU, LFU and
    FIFO at `capacity_tokens` plus its ceiling:

    - reported: human pacing, seeds 2 to 4 (12 runs), the comparison;
    - train: human pacing, seed 1 (4), for tuning only (FR-006);
    - fast: Phase 0's constant pauses, seed 2 (4), pacing sensitivity;
    - noise: seed 2's LRU run once more (1), the run-to-run spread.

    ``smoke`` is one LRU run of a 180 s plan, to check the setup.
    """

    def plan(mix: str, pacing: str, seed: int) -> List[MixConfig]:
        return [
            MixConfig(mix, pacing, seed, policy, capacity_tokens)
            for policy in BASELINES
        ] + [
            MixConfig(mix, pacing, seed, "lru", capacity_tokens, ceiling=True)
        ]

    if name == "phase1":
        configs = [
            c for seed in (2, 3, 4) for c in plan("balanced", "human", seed)
        ]
        configs += plan("balanced", "human", 1)
        configs += plan("balanced", "fast", 2)
        lru = MixConfig("balanced", "human", 2, "lru", capacity_tokens)
        configs.append(dataclasses.replace(lru, replicate=1))
    elif name == "smoke":
        configs = [MixConfig("smoke", "human", 101, "lru", capacity_tokens)]
    else:
        raise ValueError(f"Unknown grid {name!r}; use one of "
                         f"{', '.join(GRIDS)}")
    names = [c.name for c in configs]
    assert len(set(names)) == len(names), f"repeated run names in {name}"
    return configs


def shard_configs(configs: Sequence[MixConfig], index: int,
                  count: int) -> List[MixConfig]:
    """Part `index` of `count` (1-based) of a grid, in grid order.

    A plan's runs stay in one part, so every policy of a seed, its ceiling
    and any replicate are replayed on the same machine: a machine's speed
    sets the response latency, which sets when closed-loop turns are sent,
    so runs on two laptops would differ by more than their policy. Plans go
    to the part with the fewest runs so far, larger plans first (ties: the
    earlier plan, the lower part). The parts are disjoint, cover the grid,
    and are the same on every machine.
    """
    if count < 1 or not 1 <= index <= count:
        raise ValueError(f"Shard {index}/{count} is not one of 1/{count} "
                         f"to {count}/{count}")
    groups: Dict[str, List[MixConfig]] = {}
    for config in configs:
        groups.setdefault(config.plan_name, []).append(config)
    if count > len(groups):
        raise ValueError(f"The grid has {len(groups)} plans, so it splits "
                         f"into at most {len(groups)} parts")
    loads = [0] * count
    part_of: Dict[str, int] = {}
    # sorted() is stable, so plans of equal size keep their grid order.
    for plan in sorted(groups, key=lambda name: -len(groups[name])):
        part = min(range(count), key=lambda i: loads[i])
        part_of[plan] = part
        loads[part] += len(groups[plan])
    return [c for c in configs if part_of[c.plan_name] == index - 1]


def parse_shard(text: str) -> Tuple[int, int]:
    """``"i/k"`` -> (i, k), with 1 <= i <= k."""
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"a shard is i/k, such as 1/2, not {text!r}")
    index, count = int(match.group(1)), int(match.group(2))
    if count < 1 or not 1 <= index <= count:
        raise argparse.ArgumentTypeError(
            f"shard {text} is not one of 1/{count} to {count}/{count}")
    return index, count


# --- Plans -----------------------------------------------------------------


def plan_dir(data: Path, config: MixConfig) -> Path:
    """Where `build_plans` puts a configuration's plan."""
    return data / "processed" / "m3" / config.plan_name


def plan_command(config: MixConfig, data: Path) -> List[str]:
    """The ``analysis.mix`` command that builds a configuration's plan."""
    mix, extra = PLAN_RECIPES.get(config.mix, (config.mix, ()))
    return [
        sys.executable, "-m", "analysis.mix", "--mix", mix, "--pacing",
        config.pacing, "--seed",
        str(config.seed), "--data",
        str(data), "--out",
        str(plan_dir(data, config)), *extra
    ]


def build_plans(configs: Sequence[MixConfig],
                data: Path,
                force: bool = False,
                run: Callable[..., Any] = subprocess.run,
                log: Callable[[str], None] = _say) -> List[Path]:
    """Build each plan `configs` need that is not built yet; return the
    plans built.

    ``analysis.mix`` writes ``manifest.json`` last, so a plan without one
    was interrupted and is built again. With `force`, the manifest is
    deleted first, so an interrupted rebuild is not taken for a plan.
    """
    built = []
    seen: Set[str] = set()
    for config in configs:
        if config.plan_name in seen:
            continue
        seen.add(config.plan_name)
        out = plan_dir(data, config)
        manifest = out / "manifest.json"
        if manifest.exists() and not force:
            log(f"{config.plan_name}: built earlier, kept")
            continue
        manifest.unlink(missing_ok=True)
        command = plan_command(config, data)
        log(f"{config.plan_name}: building ({' '.join(command[1:])})")
        result = run(command, check=False)
        if result.returncode != 0:
            raise RunnerError(f"analysis.mix exited with {result.returncode} "
                              f"while building {config.plan_name}")
        built.append(out)
    return built


def read_plan_manifest(path: Path) -> Dict[str, Any]:
    """A plan's manifest; the grid needs its ``stop_s``."""
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise RunnerError(f"No plan in {path}; build it first with "
                          "`python -m analysis.run_m3_task2 plans --grid "
                          "<grid>`")
    manifest = json.loads(manifest_path.read_text())
    if "stop_s" not in manifest:
        raise RunnerError(f"{manifest_path} has no stop_s (an older "
                          "analysis.mix built it); rebuild it with "
                          "`plans --force`")
    return manifest


# --- The no-eviction ceiling -----------------------------------------------


def distinct_full_blocks(plan_dir: Path, block_size: int = 16) -> int:
    """How many distinct full blocks the plan's requests can ever cache.

    A request's sequence is its prompt followed by its output (the
    simulator replays the recorded output; its final EOS is never stored),
    and only full blocks are cached, so this is every block the cache could
    hold if it never evicted.
    """
    seen: Set[int] = set()
    for request in read_plan(plan_dir / "plan.jsonl.gz"):
        seen.update(
            block_hashes(request.prompt_token_ids + request.output_token_ids,
                         block_size))
    return len(seen)


def _ceiling_tokens(distinct_blocks: int, block_size: int) -> int:
    return math.ceil(CEILING_MARGIN * distinct_blocks) * block_size


def ceiling_capacity(plan_dir: Path, block_size: int = 16) -> int:
    """A capacity in tokens at which replaying the plan never evicts:
    ``ceil(CEILING_MARGIN x distinct full blocks)`` blocks."""
    return _ceiling_tokens(distinct_full_blocks(plan_dir, block_size),
                           block_size)


def _open_text(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def count_evictions(log: Path) -> Tuple[int, bool]:
    """The evictions in a cache event log, and whether it is complete (ends
    with its ``end`` line).

    Streamed, because a ceiling run's log has millions of lines.
    """
    evictions = 0
    last = ""
    with _open_text(log) as lines:
        for line in lines:
            # The substring test skips json for almost every line.
            if '"evict"' in line and json.loads(line)["ev"] == "evict":
                evictions += 1
            if line.strip():
                last = line
    complete = bool(last) and json.loads(last).get("ev") == "end"
    return evictions, complete


def check_ceiling(run_dir: Path,
                  log: Optional[Callable[[str], None]] = None) -> None:
    """Record a ceiling run's eviction count in its ``run.json`` (once) and
    raise unless it is 0.

    A ceiling that evicted would understate what a policy can reach. The
    count is also taken for a ceiling finished without it (one made with
    ``replay``, or a grid interrupted between the two), so it is never
    missing.
    """
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text())
    if "ceiling_evictions" not in manifest:
        if log is not None:
            log(f"{run_dir.name}: counting the ceiling's evictions")
        evictions, complete = count_evictions(ev.log_path(run_dir))
        manifest.update(ceiling_evictions=evictions,
                        ceiling_log_complete=complete)
        _write_json(manifest_path, manifest)
    if not manifest.get("ceiling_log_complete", True):
        raise RunnerError(f"{run_dir}: the event log has no end line, so "
                          "the ceiling cannot be shown to never evict; "
                          "delete the folder to replay it")
    if manifest["ceiling_evictions"]:
        raise RunnerError(f"{run_dir}: the ceiling run evicted "
                          f"{manifest['ceiling_evictions']} blocks; raise "
                          "CEILING_MARGIN and delete the folder to replay it")


# --- Run folders -----------------------------------------------------------

# A grid run's folder name (`MixConfig.name`): <plan>-<policy>-cap<tokens>
# or <plan>-ceiling, then -r<n> for a replicate.
RUN_NAME = re.compile(r"(?P<plan>.+?)-(?:(?P<policy>[a-z0-9_]+)-cap"
                      r"(?P<capacity>\d+)|(?P<ceiling>ceiling))"
                      r"(?:-r(?P<replicate>\d+))?")
# What `MixConfig.record` adds to run.json, besides the run's name.
RECORD_FIELDS = ("mix", "pacing", "seed", "replicate", "ceiling")
# A replay changes its folder every few seconds (the client records each
# request, the server logs its throughput), so an unfinished folder that
# changed more recently than this may still be replaying: another
# checkout's `replay`, or a grid killed without cleanup whose servers and
# clients live on.
QUIET_S = 900.0
# A run the previous grid saw fail had stopped by then. This short wait
# only covers a folder that something else has made again since.
STOPPED_QUIET_S = 60.0


def run_config(run_dir: Path) -> MixConfig:
    """The configuration of the run in `run_dir`, read from its
    ``run.json``.

    A grid run records it. A run made with ``replay`` does not: its pacing
    (``pacing.kind``) and seed then come from the plan manifest that
    ``run.json`` holds, its mix from the plan folder's name (a recipe plan's
    manifest names the mix it was built from: smoke's says balanced), and
    whether it is a ceiling and its replicate from the run folder's name.
    """
    run = json.loads((run_dir / "run.json").read_text())
    if all(field in run for field in RECORD_FIELDS):
        mix, pacing, seed = run["mix"], run["pacing"], run["seed"]
        ceiling, replicate = run["ceiling"], run["replicate"]
    else:
        plan = run["plan"]
        seed = plan["seed"]
        pacing = plan.get("pacing")
        if isinstance(pacing, dict):
            pacing = pacing.get("kind")
        if not pacing:
            raise ValueError(f"{run_dir}: its plan manifest has no pacing")
        plan_name, suffix = Path(run["plan_dir"]).name, f"-{pacing}-s{seed}"
        mix = (plan_name[:-len(suffix)]
               if plan_name.endswith(suffix) else plan["mix"])
        match = RUN_NAME.fullmatch(run_dir.name)
        if match is None:
            raise ValueError(f"{run_dir.name} is not a grid run's name")
        ceiling = match["ceiling"] is not None
        replicate = int(match["replicate"] or 0)
    return MixConfig(mix,
                     pacing,
                     int(seed),
                     run["eviction_policy"],
                     int(run["capacity_tokens"]),
                     block_size=int(run["block_size"]),
                     ceiling=bool(ceiling),
                     replicate=int(replicate))


def check_finished(run_dir: Path,
                   config: MixConfig,
                   log: Optional[Callable[[str], None]] = None) -> None:
    """Raise unless the finished run in `run_dir` is `config`'s, and, for a
    ceiling, one that never evicted. A ceiling's capacity is not checked:
    any capacity that never evicts gives the same run."""
    held = run_config(run_dir)
    if held.name != config.name or held.block_size != config.block_size:
        raise RunnerError(f"{run_dir} holds {held.name} at block size "
                          f"{held.block_size}, not {config.name}")
    if config.ceiling:
        check_ceiling(run_dir, log)


def last_change(run_dir: Path) -> float:
    """The newest modification time of a run folder and everything in it
    (the folder's own time only changes when files are added)."""
    newest = run_dir.stat().st_mtime
    for root, dirs, files in os.walk(run_dir):
        for name in dirs + files:
            with contextlib.suppress(OSError):  # removed meanwhile
                path = os.path.join(root, name)
                newest = max(newest,
                             os.stat(path, follow_symlinks=False).st_mtime)
    return newest


def _failed_in_previous_grid(out: Path) -> Set[str]:
    """The runs the previous grid in `out` replayed and saw fail.

    Their servers and clients had stopped when it recorded that, and that
    grid is gone (this one holds the lock), so they are not replaying.
    """
    try:
        runs = json.loads((out / "grid.json").read_text())["runs"]
        return {
            run["name"]
            for run in runs
            if run.get("replayed") and run.get("status") == "failed"
        }
    except (OSError, ValueError, KeyError, TypeError):
        return set()


# --- The engine grid -------------------------------------------------------


@contextlib.contextmanager
def keep_awake() -> Iterator[None]:
    """Keep macOS from sleeping while this process runs.

    A grid takes hours, and an idle laptop that sleeps suspends every
    server and client mid-run. ``caffeinate -w`` also exits by itself if
    this process dies. Does nothing elsewhere or without caffeinate.
    """
    process = None
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        with contextlib.suppress(OSError):
            process = subprocess.Popen(
                ["caffeinate", "-i", "-w",
                 str(os.getpid())])
    try:
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()


@contextlib.contextmanager
def grid_lock(out: Path) -> Iterator[None]:
    """Allow one grid per results folder at a time.

    A second grid in the same folder would take the first one's runs in
    progress for interrupted runs and delete them.
    """
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".grid.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunnerError(f"Another grid is running in {out}; wait for "
                              "it or choose another --out") from None
        yield


class _Grid:
    """The state of one `run_grid` call, shared by its worker threads."""

    def __init__(self, configs: Sequence[MixConfig], out: Path, data: Path,
                 plans: Dict[str, Dict[str, Any]], meta: Dict[str, Any],
                 replay: Callable[..., Any], log: Callable[[str], None], *,
                 stop_slack: float, ceiling_tokens: Optional[int]):
        self.configs = list(configs)
        self.out, self.data, self.plans = out, data, plans
        self.meta, self.replay, self.log = meta, replay, log
        self.stop_slack, self.ceiling_tokens = stop_slack, ceiling_tokens
        # Read before this grid writes its own grid.json over it.
        self.failed_earlier = _failed_in_previous_grid(out)
        self.lock = threading.Lock()
        self.ports: queue.Queue[int] = queue.Queue()
        self.counts: Counter = Counter()
        self.todo = 0
        self.entries: Dict[str, Dict[str, Any]] = {
            c.name: {
                "name": c.name,
                "status": "pending",
                "run_dir": str(out / c.name),
                "wall_s": None,
                "error": None,
            }
            for c in self.configs
        }

    def write(self) -> None:
        """Refresh ``grid.json``; the caller holds the lock (or is alone)."""
        _write_json(
            self.out / "grid.json", {
                **self.meta, "updated_at_utc": _now(),
                "runs": [self.entries[c.name] for c in self.configs]
            })

    def say(self, index: int, config: MixConfig, message: str) -> None:
        with self.lock:
            self.log(f"[{index}/{len(self.configs)}] {config.name}: "
                     f"{message}")

    def locked_log(self, message: str) -> None:
        with self.lock:
            self.log(message)

    def classify(self, config: MixConfig) -> Tuple[str, Optional[str]]:
        """What to do with a configuration's folder now, and why.

        "skipped": finished earlier, and checked; "failed": finished, but
        it holds another run, or a ceiling that evicted or whose log was
        cut short; "busy": unfinished and changed recently, so it may still
        be replaying; "replay": absent, or unfinished and stopped (to be
        deleted first).
        """
        run_dir = self.out / config.name
        if is_finished(run_dir):
            try:
                check_finished(run_dir, config, self.locked_log)
            except Exception as exc:  # a bad old run must not stop the grid
                return "failed", f"{type(exc).__name__}: {exc}"
            return "skipped", None
        if not run_dir.exists():
            return "replay", None
        quiet = time.time() - last_change(run_dir)
        failed_earlier = config.name in self.failed_earlier
        if quiet >= (STOPPED_QUIET_S if failed_earlier else QUIET_S):
            return "replay", ("the previous grid saw it fail"
                              if failed_earlier else
                              f"unfinished, unchanged for {quiet / 60:.0f} "
                              "min")
        return "busy", f"unfinished, changed {quiet:.0f} s ago"

    def run(self, jobs: int, base_port: int) -> List[Dict[str, Any]]:
        todo: List[Tuple[int, MixConfig]] = []
        busy: List[str] = []
        for index, config in enumerate(self.configs, 1):
            status, note = self.classify(config)
            if status == "replay":
                todo.append((index, config))
            elif status == "busy":
                busy.append(f"{config.name} ({note})")
            else:
                self.entries[config.name].update(status=status, error=note)
                message = ("finished earlier, skipped" if status == "skipped"
                           else f"finished earlier, but {note}")
                self.say(index, config, message)
        # Both refusals come before this grid writes grid.json, which still
        # tells the next grid which runs the previous one saw fail.
        if busy:
            raise RunnerError(f"{len(busy)} unfinished runs changed in the "
                              f"last {QUIET_S / 60:.0f} minutes, so "
                              "something may still be replaying them (a "
                              "`replay` from another checkout, or a grid "
                              "killed without cleanup): " + "; ".join(busy) +
                              ". Wait for it to finish, or delete those "
                              "folders if nothing is running.")
        ports = [base_port + i for i in range(min(jobs, len(todo)))]
        taken = [port for port in ports if port_in_use(port)]
        if taken:
            raise RunnerError(f"Ports {taken} are in use; stop those programs "
                              "or choose another --base-port")
        self.todo = len(todo)
        self.write()
        if not todo:
            self.log("Nothing to replay")
            return [self.entries[c.name] for c in self.configs]
        for port in ports:
            self.ports.put(port)
        try:
            with ThreadPoolExecutor(max_workers=len(ports)) as pool:
                futures = [pool.submit(self.run_one, i, c) for i, c in todo]
                try:
                    for future in as_completed(futures):
                        future.result()
                except KeyboardInterrupt:
                    cancelled = sum(future.cancel() for future in futures)
                    with self.lock:
                        self.log(f"Interrupted: {cancelled} runs will not "
                                 "start; waiting for the running ones")
                    raise
        finally:
            with self.lock:
                self.write()
        return [self.entries[c.name] for c in self.configs]

    def run_one(self, index: int, config: MixConfig) -> None:
        # Again: the folder may have changed while this run waited.
        status, note = self.classify(config)
        if status != "replay":
            self.done(index, config, status, note)
            return
        port = self.ports.get()
        with self.lock:
            self.entries[config.name].update(status="running",
                                             port=port,
                                             started_at_utc=_now())
            self.write()
        self.say(index, config,
                 f"started on port {port}" + (f" ({note})" if note else ""))
        start = time.monotonic()
        status, error = "failed", None
        try:
            self.replay_one(index, config, port)
            status = "finished"
        except Exception as exc:  # one failed run must not stop the others
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self.ports.put(port)
        self.done(index, config, status, error, time.monotonic() - start)

    def done(self,
             index: int,
             config: MixConfig,
             status: str,
             error: Optional[str],
             wall: Optional[float] = None) -> None:
        with self.lock:
            entry = self.entries[config.name]
            entry.update(status=status, error=error)
            if wall is not None:
                entry.update(wall_s=round(wall, 1), finished_at_utc=_now())
            self.counts[status] += 1
            self.write()
            left = self.todo - sum(self.counts.values())
            took = "" if wall is None else f" in {wall:.0f} s"
            self.log(f"[{index}/{len(self.configs)}] {config.name}: {status}"
                     f"{took}{': ' + error if error else ''} "
                     f"({self.counts['finished']} finished, "
                     f"{self.counts['failed']} failed, {left} to go)")

    def ceiling_capacity(self, config: MixConfig, path: Path,
                         max_model_len: int, updates: Dict[str, Any]) -> int:
        distinct = distinct_full_blocks(path, config.block_size)
        # vLLM refuses a cache that cannot hold one longest request.
        whole_blocks = -(-max_model_len // config.block_size)
        needed = max(_ceiling_tokens(distinct, config.block_size),
                     whole_blocks * config.block_size)
        updates.update(ceiling_distinct_blocks=distinct,
                       ceiling_margin=CEILING_MARGIN)
        if self.ceiling_tokens is None:
            return needed
        if self.ceiling_tokens < needed:
            raise RunnerError(f"A ceiling of {self.ceiling_tokens} tokens is "
                              f"below the {needed} that {config.plan_name} "
                              f"needs ({CEILING_MARGIN} x its {distinct} "
                              "distinct blocks)")
        return self.ceiling_tokens

    def replay_one(self, index: int, config: MixConfig, port: int) -> None:
        run_dir = self.out / config.name
        path = plan_dir(self.data, config)
        plan = self.plans[config.plan_name]
        max_model_len = plan.get("max_model_len", MAX_MODEL_LEN)
        if port_in_use(port):
            raise RunnerError(f"Port {port} is in use")
        if run_dir.exists():
            self.say(index, config, "removing its unfinished folder")
            shutil.rmtree(run_dir)
        with self.lock:
            # From here the folder is this grid's: if it fails, the next
            # grid knows nothing else is replaying it.
            self.entries[config.name]["replayed"] = True
            self.write()
        updates = {"grid": self.meta.get("grid"), **config.record()}
        capacity = config.capacity_tokens
        if config.ceiling:
            capacity = self.ceiling_capacity(config, path, max_model_len,
                                             updates)
        self.replay(path,
                    run_dir,
                    policy=config.policy,
                    capacity_tokens=capacity,
                    block_size=config.block_size,
                    max_model_len=max_model_len,
                    port=port,
                    stop_at=plan["stop_s"] + self.stop_slack,
                    manifest_updates=updates)
        if config.ceiling:
            check_ceiling(run_dir)


def run_grid(configs: Sequence[MixConfig],
             out: Path,
             *,
             data: Path = Path("data"),
             jobs: int = 6,
             base_port: int = 8100,
             stop_slack: float = 600.0,
             allow_dirty: bool = False,
             ceiling_tokens: Optional[int] = None,
             grid: Optional[str] = None,
             shard: Optional[Tuple[int, int]] = None,
             replay: Callable[..., Any] = replay_plan,
             log: Callable[[str], None] = _say) -> List[Dict[str, Any]]:
    """Replay every configuration not finished yet, `jobs` at a time.

    Returns one status entry per configuration (name, status, run_dir,
    wall_s, error), also written to ``<out>/grid.json``: "skipped"
    (finished earlier and checked, by `check_finished`), "finished",
    "failed", "busy" (its folder came alive while it waited), or "pending"
    if interrupted before it started. A failed run does not stop the
    others. Refuses to start while an unfinished folder of `configs` may
    still be replaying (see `QUIET_S`). Ceilings take `ceiling_tokens` if
    given (refused if a plan needs more), else what their plan needs.
    `replay` takes `replay_plan`'s arguments (tests pass a fake).
    """
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if stop_slack < 0:
        raise ValueError("stop_slack must not be negative")
    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError("Two configurations have the same name")
    if ceiling_tokens is not None:
        sizes = {c.block_size for c in configs if c.ceiling}
        if ceiling_tokens <= 0 or any(ceiling_tokens % size for size in sizes):
            raise ValueError(f"A ceiling of {ceiling_tokens} tokens is not a "
                             "positive whole number of blocks")
    state = git_state()
    if state["dirty"] and not allow_dirty:
        raise RunnerError("The working tree has uncommitted changes, which "
                          "the commit each run records would not show; "
                          "commit them or pass --allow-dirty")
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        raise RunnerError("Unset PROMETHEUS_MULTIPROC_DIR: servers that "
                          "share it report each other's metrics")
    plans: Dict[str, Dict[str, Any]] = {}
    for config in configs:
        if config.plan_name not in plans:
            path = plan_dir(data, config)
            plans[config.plan_name] = read_plan_manifest(path)
        max_model_len = plans[config.plan_name].get("max_model_len",
                                                    MAX_MODEL_LEN)
        if not config.ceiling and config.capacity_tokens < max_model_len:
            raise RunnerError(f"{config.name}: vLLM does not start with a "
                              f"cache smaller than max_model_len "
                              f"({max_model_len} tokens)")
    meta = {
        "grid": grid,
        "shard": f"{shard[0]}/{shard[1]}" if shard else None,
        "jobs": jobs,
        "base_port": base_port,
        "stop_slack_s": stop_slack,
        "ceiling_tokens": ceiling_tokens,
        "data": str(data),
        **state,
        "started_at_utc": _now(),
    }
    with grid_lock(out), keep_awake():
        return _Grid(configs,
                     out,
                     data,
                     plans,
                     meta,
                     replay,
                     log,
                     stop_slack=stop_slack,
                     ceiling_tokens=ceiling_tokens).run(jobs, base_port)


# --- The cache-simulator sweep ---------------------------------------------


def sim_key(row: Dict[str, Any]) -> Tuple[str, str, str, int, str]:
    """What identifies a simulator run in the output file."""
    return (str(row["plan"]), str(row["policy"]), str(row["capacity"]),
            int(row["block_size"]), str(row["timing"]))


def _finished_sims(
        out: Path) -> Tuple[Set[Tuple[str, str, str, int, str]], bool]:
    """The keys of `out`'s successful runs, and whether its last line was
    cut off (an interrupted write), so the next line must start afresh.

    Runs recorded with an error are tried again (a policy added since).
    """
    if not out.exists():
        return set(), False
    text = out.read_text(encoding="utf-8")
    done = set()
    for line in text.splitlines():
        try:
            row = json.loads(line)
            if "error" not in row and "summary" in row:
                done.add(sim_key(row))
        except (ValueError, KeyError, TypeError):
            continue  # the cut-off line, or not a run
    return done, bool(text) and not text.endswith("\n")


def _sim_task(task: Dict[str, Any],
              simulate: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """One simulator run (in a pool worker). A failure becomes the row's
    "error", so one bad combination does not stop the sweep."""
    run = simulate
    if run is None:
        from analysis import cachesim
        run = cachesim.simulate
    row = dict(task)
    start = time.monotonic()
    try:
        plan = Path(task["plan"])
        capacity = task["capacity"]
        tokens = (ceiling_capacity(plan, task["block_size"])
                  if capacity == "ceiling" else int(capacity))
        row["capacity_tokens"] = tokens
        row["summary"] = run(plan, task["policy"], tokens, task["block_size"],
                             task["timing"]).summary
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = round(time.monotonic() - start, 3)
    return row


def plan_timing(timing: str, plan: Path) -> str:
    """`timing` for one plan: ``{plan}`` becomes the plan folder's name, so
    each plan takes its own engine run's timing
    (``engine:results/m3-task2/{plan}-ceiling``)."""
    return timing.replace("{plan}", Path(plan).name)


def run_sims(plans: Sequence[Path],
             policies: Sequence[str],
             capacities: Sequence[CapacitySpec],
             out: Path,
             *,
             block_size: int = 16,
             timing: str = "model",
             jobs: int = 8,
             nice: int = 0,
             simulate: Optional[Callable[..., Any]] = None,
             log: Callable[[str], None] = _say) -> List[Dict[str, Any]]:
    """Simulate every (plan, capacity, policy) not yet in `out`.

    Appends one JSON line per run, as it finishes: plan, policy, capacity
    (as asked: tokens or "ceiling"), capacity_tokens, block_size, timing
    (`plan_timing`), summary (``cachesim.simulate``'s), seconds, commit,
    dirty, and error for a run that failed. Policies are passed through
    unchanged. Engine timing must come from finished engine runs. With
    `jobs` > 1 the runs go to a process pool whose workers lower their CPU
    priority by `nice`; `simulate` replaces ``cachesim.simulate`` (tests).
    Returns the new rows.
    """
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    for capacity in capacities:
        if capacity != "ceiling" and (not isinstance(capacity, int) or
                                      capacity <= 0 or capacity % block_size):
            raise ValueError(f"Capacity {capacity!r} is neither 'ceiling' nor "
                             f"a positive multiple of block size {block_size}")
    # A running engine run's log is cut short: its timing would be too.
    unfinished = []
    for plan in plans:
        resolved = plan_timing(timing, plan)
        if resolved.startswith("engine:") and not is_finished(
                Path(resolved[len("engine:"):])):
            unfinished.append(resolved)
    if unfinished:
        raise ValueError("Engine timing needs finished engine runs "
                         "(run.json with finished_at_utc), not "
                         f"{', '.join(unfinished)}")
    done, cut = _finished_sims(out)
    tasks = []
    keys = set()
    for plan in plans:
        for capacity in capacities:
            for policy in policies:
                task = {
                    "plan": str(plan),
                    "policy": policy,
                    "capacity": capacity,
                    "block_size": block_size,
                    "timing": plan_timing(timing, plan),
                }
                key = sim_key(task)
                if key not in keys:  # a value given twice runs once
                    keys.add(key)
                    if key not in done:
                        tasks.append(task)
    log(f"{len(tasks)} simulator runs to do, {len(keys) - len(tasks)} done "
        "earlier")
    if not tasks:
        return []
    state = git_state()
    worker = functools.partial(_sim_task, simulate=simulate)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with contextlib.ExitStack() as stack:
        results: Iterator[Dict[str, Any]]
        if jobs == 1:
            if nice:
                os.nice(nice)
            results = map(worker, tasks)
        else:
            # spawn: a forked copy of a parent that loaded torch can hang.
            context = multiprocessing.get_context("spawn")
            pool = stack.enter_context(
                context.Pool(min(jobs, len(tasks)),
                             initializer=os.nice if nice else None,
                             initargs=(nice, ) if nice else ()))
            results = pool.imap_unordered(worker, tasks)
        output = stack.enter_context(out.open("a", encoding="utf-8"))
        if cut:
            output.write("\n")
        for count, row in enumerate(results, 1):
            row.update(state)
            output.write(json.dumps(row) + "\n")
            output.flush()
            rows.append(row)
            if "error" in row:
                outcome = row["error"]
            else:
                usable = row["summary"].get("usable_reused_fraction")
                outcome = ("usable reuse " +
                           ("n/a" if usable is None else f"{usable:.4f}"))
            log(f"[{count}/{len(tasks)}] {Path(row['plan']).name} "
                f"{row['policy']} {row['capacity']}: {outcome} "
                f"({row['seconds']:.0f} s)")
    return rows


# --- Command line ------------------------------------------------------------


def capacity_arg(text: str) -> CapacitySpec:
    if text == "ceiling":
        return text
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"a capacity is a positive number of tokens or 'ceiling', not "
            f"{text!r}")
    return value


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="Replay one plan once")
    replay.add_argument("--plan", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--policy", default="lru")
    replay.add_argument("--labels")
    replay.add_argument("--capacity", type=int, required=True)
    replay.add_argument("--block-size", type=int, default=16)
    replay.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    replay.add_argument("--port", type=int, default=8000)
    replay.add_argument("--model", default=DEFAULT_MODEL)
    replay.add_argument("--no-event-log", action="store_true")
    replay.add_argument("--stop-at",
                        type=float,
                        help="Do not send requests that are not ready this "
                        "many seconds after the start (plan_replay guard)")

    plans = sub.add_parser("plans", help="Build the plans a grid needs")
    plans.add_argument("--grid", choices=GRIDS, required=True)
    plans.add_argument("--data", type=Path, default=Path("data"))
    plans.add_argument("--shard",
                       type=parse_shard,
                       help="i/k: only the plans of grid part i of k")
    plans.add_argument("--force",
                       action="store_true",
                       help="Rebuild plans that were built earlier")

    grid = sub.add_parser("grid", help="Replay a named grid, resumably")
    grid.add_argument("--grid", choices=GRIDS, required=True)
    grid.add_argument("--out", type=Path, default=Path("results/m3-task2"))
    grid.add_argument("--data",
                      type=Path,
                      default=Path("data"),
                      help="Plans are read from <data>/processed/m3")
    grid.add_argument("--jobs",
                      type=int,
                      default=6,
                      help="Runs at once, each on its own server")
    grid.add_argument("--base-port",
                      type=int,
                      default=8100,
                      help="Worker i serves on this port + i")
    grid.add_argument("--shard",
                      type=parse_shard,
                      help="i/k: run only part i of k (1-based) of the grid; "
                      "a plan's runs stay in one part")
    grid.add_argument("--capacity",
                      type=int,
                      default=DEFAULT_CAPACITY,
                      help="Cache size in tokens of the non-ceiling runs")
    grid.add_argument("--allow-dirty",
                      action="store_true",
                      help="Run with uncommitted changes (recorded)")
    grid.add_argument("--stop-slack",
                      type=float,
                      default=600.0,
                      help="Stop sending this many seconds after the plan's "
                      "stop_s")
    grid.add_argument("--ceiling-capacity",
                      type=int,
                      help="Cache size in tokens of every ceiling run "
                      "(default: what its plan needs; refused if smaller)")

    sim = sub.add_parser("sim",
                         help="Sweep plans x policies x capacities in the "
                         "cache simulator")
    sim.add_argument("--plans", type=Path, nargs="+", required=True)
    sim.add_argument("--policies", nargs="+", default=list(BASELINES))
    sim.add_argument("--capacities",
                     type=capacity_arg,
                     nargs="+",
                     required=True,
                     help="Tokens, or 'ceiling' (never evicts)")
    sim.add_argument("--block-size", type=int, default=16)
    sim.add_argument("--timing",
                     default="model",
                     help="model, or engine:<finished run dir>; {plan} in "
                     "it becomes each plan's name, as in "
                     "engine:results/m3-task2/{plan}-ceiling")
    sim.add_argument("--jobs", type=int, default=8)
    sim.add_argument("--nice",
                     type=int,
                     default=10,
                     help="Lower the workers' CPU priority by this much, so "
                     "engine runs on this machine keep theirs (0: unchanged)")
    sim.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if getattr(args, "jobs", 1) < 1:
        parser.error("--jobs must be at least 1")
    if getattr(args, "stop_slack", 0) < 0:
        parser.error("--stop-slack must not be negative")
    if not 0 <= getattr(args, "nice", 0) <= 19:
        parser.error("--nice must be between 0 and 19")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        if args.command == "replay":
            manifest = replay_plan(args.plan,
                                   args.out,
                                   policy=args.policy,
                                   labels=args.labels,
                                   capacity_tokens=args.capacity,
                                   block_size=args.block_size,
                                   model=args.model,
                                   max_model_len=args.max_model_len,
                                   port=args.port,
                                   event_log=not args.no_event_log,
                                   stop_at=args.stop_at)
            print(json.dumps(manifest["client_summary"], indent=2))
        elif args.command == "plans":
            configs = grid_configs(args.grid)
            if args.shard:
                configs = shard_configs(configs, *args.shard)
            build_plans(configs, args.data, force=args.force)
        elif args.command == "grid":
            configs = grid_configs(args.grid, args.capacity)
            if args.shard:
                configs = shard_configs(configs, *args.shard)
            entries = run_grid(configs,
                               args.out,
                               data=args.data,
                               jobs=args.jobs,
                               base_port=args.base_port,
                               stop_slack=args.stop_slack,
                               allow_dirty=args.allow_dirty,
                               ceiling_tokens=args.ceiling_capacity,
                               grid=args.grid,
                               shard=args.shard)
            counts = Counter(e["status"] for e in entries)
            _say(f"Done: {counts['finished']} finished, {counts['skipped']} "
                 f"finished earlier, {counts['busy']} left to whatever is "
                 f"replaying them, {counts['failed']} failed; see "
                 f"{args.out / 'grid.json'}")
            if counts["failed"]:
                raise SystemExit(1)
        elif args.command == "sim":
            with keep_awake():
                rows = run_sims(args.plans,
                                args.policies,
                                args.capacities,
                                args.out,
                                block_size=args.block_size,
                                timing=args.timing,
                                jobs=args.jobs,
                                nice=args.nice)
            failed = sum("error" in row for row in rows)
            _say(f"Wrote {len(rows)} rows ({failed} with an error) to "
                 f"{args.out}")
    except RunnerError as exc:
        raise SystemExit(f"error: {exc}") from None
    except KeyboardInterrupt:
        if args.command == "replay":  # its folder must be new: no resuming
            raise
        # 130: the shell's code for a command stopped by Ctrl-C.
        _say("Interrupted; run the same command again to resume")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
