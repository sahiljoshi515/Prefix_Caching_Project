"""Milestone 3 Task 2: one workload's requests from a mixed plan (spec 007).

The mixed-versus-isolated comparison runs each workload on its own, with
exactly the requests it sends in the mix: the same sessions, prompts,
answers, recorded arrival times and closed-loop pauses. Only the other
workloads are gone. So each workload's traffic intensity (its requests and
tokens per second) is the same alone as in the mix, and any difference in
its reuse comes from sharing the cache::

    python -m analysis.isolate --group bailian \\
        --source data/processed/m3/balanced-human-s2 \\
        --out data/processed/m3/alone-bailian-human-s2

writes ``plan.jsonl.gz``, ``responses.jsonl`` and ``manifest.json`` into
``--out``, as ``analysis/mix.py`` does. The manifest is the source plan's,
restricted to the group, with ``isolated_from`` naming the source plan, its
sha256 and the commit that built it.

The course counts four workloads: ShareGPT, AgentBank, CC-Bench and
Bailian. Bailian's four traces (chat, API, thinking, coder) are one group
here, so "Bailian alone" replays all four together.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from client_simulator.adapters import bailian
from client_simulator.schema import TokenRequest
from client_simulator.token_plan import read_plan, write_plan

# The four workloads the course names, as groups of the mix's workloads.
GROUPS: Dict[str, Tuple[str, ...]] = {
    "sharegpt": ("sharegpt", ),
    "agentbank": ("agentbank", ),
    "ccbench": ("ccbench", ),
    "bailian": tuple(sorted(bailian.TRACES)),
}
PREFIX = "alone-"


def group_of(workload: str) -> str:
    """The group a mix workload belongs to."""
    for group, members in GROUPS.items():
        if workload in members:
            return group
    raise ValueError(f"Workload {workload!r} is in no group; groups: "
                     f"{', '.join(GROUPS)}")


def isolated_mix(group: str) -> str:
    """The mix name of a group's isolated plans (``alone-<group>``)."""
    if group not in GROUPS:
        raise ValueError(f"Unknown group {group!r}; use one of "
                         f"{', '.join(GROUPS)}")
    return PREFIX + group


def group_of_mix(mix: str) -> Optional[str]:
    """The group an ``alone-<group>`` mix isolates, else None."""
    if not mix.startswith(PREFIX):
        return None
    group = mix[len(PREFIX):]
    isolated_mix(group)  # raises for an unknown group
    return group


def select(requests: List[TokenRequest], group: str) -> List[TokenRequest]:
    """The requests of `group`'s workloads, in plan order."""
    members = set(GROUPS[group])
    return [r for r in requests if r.workload in members]


def isolated_manifest(source: Dict[str, Any], group: str,
                      requests: List[TokenRequest],
                      source_info: Dict[str, Any]) -> Dict[str, Any]:
    """The source manifest restricted to `group`, with shares recomputed
    within the isolated plan."""
    members = GROUPS[group]
    workloads = {
        w: dict(stats)
        for w, stats in source["workloads"].items() if w in members
    }
    total_requests = len(requests)
    total_tokens = sum(len(r.prompt_token_ids) for r in requests)
    for w, stats in workloads.items():
        mine = [r for r in requests if r.workload == w]
        tokens = sum(len(r.prompt_token_ids) for r in mine)
        if len(mine) != stats["requests"] or tokens != stats["prompt_tokens"]:
            raise ValueError(f"{w}: the source plan holds {len(mine)} "
                             f"requests and {tokens} prompt tokens, its "
                             f"manifest {stats['requests']} and "
                             f"{stats['prompt_tokens']}")
        stats["request_share"] = (len(mine) /
                                  total_requests if total_requests else 0)
        stats["prompt_token_share"] = (tokens /
                                       total_tokens if total_tokens else 0)
    manifest = {
        k: v
        for k, v in source.items() if k not in ("commit", "dirty")
    }
    manifest.update(
        mix=isolated_mix(group),
        shares={
            w: source["shares"][w]
            for w in members if w in source["shares"]
        },
        requests=total_requests,
        sessions=len({r.session_id
                      for r in requests}),
        prompt_tokens=total_tokens,
        workloads=workloads,
        isolated_from=source_info,
    )
    return manifest


def isolate(source: Path, group: str, out: Path) -> Dict[str, Any]:
    """Write `group`'s requests of the plan in `source` to `out`; return
    the manifest. The manifest is written last, as ``analysis.mix`` does,
    so a folder without one was interrupted."""
    from analysis.mix import sha256
    from analysis.run_m2_task2 import git_state
    source_manifest = json.loads((source / "manifest.json").read_text())
    requests = select(list(read_plan(source / "plan.jsonl.gz")), group)
    if not requests:
        raise ValueError(f"{source} has no {group} requests")
    info = {
        "plan": source.name,
        "mix": source_manifest.get("mix"),
        "group": group,
        "workloads": list(GROUPS[group]),
        "plan_sha256": sha256(source / "plan.jsonl.gz"),
        "commit": source_manifest.get("commit"),
        "dirty": source_manifest.get("dirty"),
    }
    manifest = isolated_manifest(source_manifest, group, requests, info)
    manifest.update(git_state())
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    write_plan(requests, out / "plan.jsonl.gz", out / "responses.jsonl")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=list(GROUPS), required=True)
    parser.add_argument("--source",
                        type=Path,
                        required=True,
                        help="A mixed plan folder (analysis/mix.py's --out)")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest = isolate(args.source, args.group, args.out)
    print(f"Wrote {manifest['requests']} {args.group} requests in "
          f"{manifest['sessions']} sessions from {args.source.name} to "
          f"{args.out}")


if __name__ == "__main__":
    main()
