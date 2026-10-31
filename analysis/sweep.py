"""Resumable sweep over KV-cache capacity, block size and eviction policy.

Spec ``005-m2-cache-tuning``, User Story 2. A configuration is a policy, a
block size and a capacity in **tokens**; its block count is
``capacity / block_size``, so changing the block size never changes how much
the cache holds. Each configuration replays the same sample into its own run
folder, named after it, through a caller-supplied ``replay`` function (one
fresh server per configuration), and is then analyzed with Task 2's metrics.

Resuming: a run folder whose ``run.json`` has ``finished_at_utc`` is
finished and is not replayed again. A folder without it was interrupted; it
is deleted and replayed. Configurations whose cache cannot hold one
``max_model_len`` request are infeasible: vLLM refuses to start, so they are
reported as such and never run.

No vLLM import, so the grid logic is testable without a server.
"""
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from analysis import events as ev

POLICIES = ("lru", "fifo", "lfu")
# What `vllm serve --block-size` accepts in this vLLM version.
BLOCK_SIZES = (8, 16, 32, 64, 128)
PREEMPTIONS_METRIC = "vllm:num_preemptions_total"
# Fields of a run's summary.json that analyses before the usable reused
# fraction did not write; a cached summary without them is analyzed again.
CURRENT_SUMMARY_FIELDS = ("usable_reused_fraction",
                          "mean_usable_cached_fraction",
                          "requests_with_hits_after_miss")


@dataclass(frozen=True)
class Config:
    policy: str
    block_size: int
    capacity_tokens: int

    @property
    def num_blocks(self) -> int:
        return self.capacity_tokens // self.block_size

    @property
    def name(self) -> str:
        return f"{self.policy}-bs{self.block_size}-cap{self.capacity_tokens}"

    def server_args(self) -> List[str]:
        return [
            "--block-size",
            str(self.block_size), "--num-gpu-blocks-override",
            str(self.num_blocks)
        ]

    def env(self) -> Dict[str, str]:
        return {"VLLM_EVICTION_POLICY": self.policy}


def expand_grid(policies: Iterable[str], block_sizes: Iterable[int],
                capacities_tokens: Iterable[int]) -> List[Config]:
    """Every combination, ordered by capacity, then block size, then policy.

    Raises ValueError for an unknown policy or block size, or a capacity that
    is not a whole number of blocks for every block size.
    """
    policies, block_sizes = list(policies), list(block_sizes)
    capacities_tokens = list(capacities_tokens)
    for name, values in (("policy", policies), ("block size", block_sizes),
                         ("capacity", capacities_tokens)):
        if not values:
            raise ValueError(f"the grid needs at least one {name}")
        if len(set(values)) != len(values):
            raise ValueError(f"repeated {name} in {values}")
    for policy in policies:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; use one of "
                             f"{', '.join(POLICIES)}")
    for block_size in block_sizes:
        if block_size not in BLOCK_SIZES:
            raise ValueError(f"block size {block_size} is not one of "
                             f"{BLOCK_SIZES}")
    for capacity in capacities_tokens:
        for block_size in block_sizes:
            if capacity <= 0 or capacity % block_size:
                raise ValueError(
                    f"capacity {capacity} tokens is not a positive multiple "
                    f"of block size {block_size}")
    return [
        Config(policy, block_size, capacity)
        for capacity in sorted(capacities_tokens)
        for block_size in sorted(block_sizes) for policy in policies
    ]


def grid_order(configs: Iterable[Config]) -> List[Config]:
    """Unique configurations in `expand_grid`'s order: capacity, then
    block size, then policy in `POLICIES` order."""
    return sorted(set(configs),
                  key=lambda c:
                  (c.capacity_tokens, c.block_size, POLICIES.index(c.policy)))


def feasible(config: Config, max_model_len: int) -> bool:
    """vLLM only starts if the cache holds one max_model_len request."""
    return config.capacity_tokens >= max_model_len


def is_finished(run_dir: Path) -> bool:
    try:
        manifest = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(manifest.get("finished_at_utc"))


def summary_row(config: Config, run_dir: Optional[Path],
                summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One row of summary.csv; metric columns are empty when not run."""
    row: Dict[str, Any] = {
        "name": config.name,
        "policy": config.policy,
        "block_size": config.block_size,
        "capacity_tokens": config.capacity_tokens,
        "num_blocks": config.num_blocks,
        "status": "infeasible" if summary is None else "finished",
    }
    summary = summary or {}
    manifest: Dict[str, Any] = {}
    preemptions = None
    if run_dir is not None:
        manifest = json.loads((run_dir / "run.json").read_text())
        preemptions = ev.read_metric(
            run_dir / "client.jsonl.metrics.after.txt", PREEMPTIONS_METRIC)
    kv_cache = manifest.get("kv_cache") or {}
    client = manifest.get("client_summary") or {}
    residency = summary.get("hits_per_residency") or {}
    interval = summary.get("reuse_interval_s") or {}
    distance = summary.get("reuse_requests_between") or {}
    row.update({
        "kv_blocks_reported":
        kv_cache.get("num_gpu_blocks"),
        "requests_completed":
        summary.get("requests_completed"),
        "requests_skipped_context":
        client.get("skipped_context"),
        "requests_failed":
        client.get("failed"),
        "block_hit_rate":
        summary.get("block_hit_rate_log"),
        "mean_cached_fraction":
        summary.get("mean_cached_fraction"),
        "usable_reused_fraction":
        summary.get("usable_reused_fraction"),
        "mean_usable_cached_fraction":
        summary.get("mean_usable_cached_fraction"),
        "token_reused_fraction":
        summary.get("token_reused_fraction"),
        "requests_with_hits_after_miss":
        summary.get("requests_with_hits_after_miss"),
        "evictions":
        summary.get("evictions"),
        "eviction_rate_per_1k_lookups":
        summary.get("eviction_rate_per_1k_lookups"),
        "preemptions":
        preemptions,
        "blocks":
        summary.get("blocks"),
        "blocks_with_hits_share":
        summary.get("blocks_with_hits_share"),
        "hits_per_residency_mean":
        residency.get("mean"),
        "reuse_interval_s_p50":
        interval.get("p50"),
        "reuse_requests_between_p50":
        distance.get("p50"),
        "sc003_pass":
        summary.get("sc003_pass"),
        "commit":
        manifest.get("commit"),
    })
    return row


def run_sweep(configs: Sequence[Config],
              out: Path,
              replay: Callable[[Config, Path], Any],
              analyze: Callable[[Path], Dict[str, Any]],
              max_model_len: int,
              reanalyze: bool = False,
              replay_if: Callable[[Config], bool] = lambda config: True,
              log: Callable[[str], None] = print) -> List[Dict[str, Any]]:
    """Run every configuration not already finished; return summary rows.

    `replay(config, run_dir)` must create `run_dir` and write its manifest
    with `finished_at_utc` last; `analyze(run_dir)` returns its summary,
    which is cached as `summary.json` unless `reanalyze` is set or the
    cached one lacks a `CURRENT_SUMMARY_FIELDS` field.
    Unfinished configurations for which `replay_if(config)` is False are
    reported as "missing" instead of being replayed.
    """
    rows = []
    for index, config in enumerate(configs, 1):
        prefix = f"[{index}/{len(configs)}] {config.name}"
        run_dir = out / config.name
        if not feasible(config, max_model_len):
            log(f"{prefix}: infeasible, {config.capacity_tokens} tokens < "
                f"max_model_len {max_model_len}")
            rows.append(summary_row(config, None, None))
            continue
        if is_finished(run_dir):
            log(f"{prefix}: finished earlier, analyzing")
        elif not replay_if(config):
            log(f"{prefix}: missing, not replayed")
            row = summary_row(config, None, None)
            row["status"] = "missing"
            rows.append(row)
            continue
        else:
            if run_dir.exists():
                log(f"{prefix}: removing an interrupted run")
                shutil.rmtree(run_dir)
            log(f"{prefix}: replaying")
            replay(config, run_dir)
        summary_path = run_dir / "summary.json"
        summary = None
        if summary_path.exists() and not reanalyze:
            summary = json.loads(summary_path.read_text())
            if any(f not in summary for f in CURRENT_SUMMARY_FIELDS):
                log(f"{prefix}: summary.json is from an older analysis, "
                    "analyzing again")
                summary = None
        if summary is None:
            summary = analyze(run_dir)
        rows.append(summary_row(config, run_dir, summary))
    return rows
