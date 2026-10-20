"""Check a pull request against the team's requirements.

Run on every pull request by ``.github/workflows/pr-requirements.yml``
(``specs/001-ci-cd``, User Story 7). The rules come from ``CONTRIBUTING.md``
and ``.specify/memory/constitution.md``. Errors fail the check; warnings are
reported but do not.

In GitHub Actions the pull request is read from ``GITHUB_EVENT_PATH``. To run
it locally on your branch before opening a PR::

    python tools/pr_check/check_pr.py --base origin/main --head HEAD \\
        --body-file my_pr_description.md

Pass ``--no-description`` instead of ``--body-file`` to check only the
changed files and commits.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

# vLLM commit the project forks (CONTRIBUTING.md section 1). A file under
# vllm/ that exists there is an upstream file.
UPSTREAM_BASE = "94d545a1a18e20ea8763a6760194589b8a3c9065"
TEMPLATE_PATH = ".github/PULL_REQUEST_TEMPLATE.md"

# CONTRIBUTING.md section 4: the PR description items.
REQUIRED_SECTIONS = (
    "Summary",
    "Milestone and task",
    "Upstream changes",
    "Testing",
    "Effect on results",
    "AI usage",
)
# CONTRIBUTING.md section 7: no datasets, bulk results or environments, and
# committed fixtures stay small.
FORBIDDEN_PREFIXES = ("data/", "results/", ".venv/")
MAX_FILE_BYTES = 1_000_000
# Constitution I: the v1 engine is never modified.
V1_PREFIX = "vllm/v1/"
# CONTRIBUTING.md section 6.3: new functionality comes with tests.
CODE_PREFIXES = ("vllm/", "client_simulator/", "analysis/")
TEST_PREFIXES = ("tests/", "tools/pr_check/test_")
# CONTRIBUTING.md section 5: commit subject lines.
MAX_SUBJECT_CHARS = 72
# CONTRIBUTING.md section 4: <type>-<name>-<description>.
BRANCH_PATTERN = re.compile(
    r"^(milestone[1-3]|fix|docs|ci|report)-[a-z0-9]+-[a-z0-9]+(-[a-z0-9]+)*$")

_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True)
class ChangedFile:
    status: str  # "A" added, "M" modified, "D" deleted
    path: str
    size: int  # bytes at the PR head; 0 when deleted


@dataclass(frozen=True)
class Commit:
    sha: str
    subject: str
    body: str
    is_merge: bool


@dataclass(frozen=True)
class Finding:
    level: str  # "error" or "warning"
    rule: str
    message: str
    path: Optional[str] = None


def _normalize(text: str) -> str:
    return " ".join(text.split())


def parse_sections(markdown: str) -> Dict[str, str]:
    """Map each lower-cased heading to the text under it.

    HTML comments are dropped, and lines inside fenced code blocks are never
    read as headings.
    """
    sections: Dict[str, str] = {}
    current: Optional[str] = None
    lines: List[str] = []
    in_fence = False
    for line in _COMMENT.sub("", markdown).splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = match.group(1).strip().lower()
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    return sections


def check_description(body: str, template: str) -> List[Finding]:
    """Every required section is present and filled in."""
    findings = []
    sections = parse_sections(body)
    defaults = parse_sections(template)
    for name in REQUIRED_SECTIONS:
        key = name.lower()
        text = sections.get(key)
        if text is None:
            findings.append(
                Finding(
                    "error", "description",
                    f'Missing section "## {name}" in the PR description '
                    "(CONTRIBUTING.md section 4)."))
        elif not text or _normalize(text) == _normalize(defaults.get(key, "")):
            findings.append(
                Finding(
                    "error", "description",
                    f'Section "## {name}" is empty or still the template '
                    "text."))
    milestone = sections.get("milestone and task", "")
    if milestone and "specs/" not in milestone:
        findings.append(
            Finding(
                "warning", "spec-link",
                'Link the spec under "## Milestone and task" '
                "(specs/<NNN>-<name>/spec.md)."))
    return findings


def check_upstream(files: Iterable[ChangedFile], body: str,
                   is_upstream: Callable[[str], bool]) -> List[Finding]:
    """No v1 changes; every changed upstream file is listed with a reason."""
    findings = []
    listed = parse_sections(body).get("upstream changes", "")
    for f in files:
        if f.path.startswith(V1_PREFIX):
            findings.append(
                Finding(
                    "error", "v1-engine",
                    f"{f.path} is in the v1 engine, which the project never "
                    "modifies (constitution I).", f.path))
        elif (f.path.startswith("vllm/") and f.status in ("M", "D")
              and is_upstream(f.path) and f.path not in listed):
            findings.append(
                Finding(
                    "error", "upstream",
                    f"{f.path} is an upstream vLLM file. List it with its "
                    'reason under "## Upstream changes" (constitution II).',
                    f.path))
    return findings


def check_files(files: Iterable[ChangedFile]) -> List[Finding]:
    """No datasets, results, environments or oversized files."""
    findings = []
    for f in files:
        if f.status == "D":
            continue
        if f.path.startswith(FORBIDDEN_PREFIXES):
            findings.append(
                Finding(
                    "error", "data",
                    f"{f.path}: datasets, results and environments are not "
                    "committed; commit the script that produces them "
                    "(CONTRIBUTING.md section 7).", f.path))
        elif f.size > MAX_FILE_BYTES:
            findings.append(
                Finding(
                    "error", "size",
                    f"{f.path} is {f.size / 1e6:.1f} MB; files over 1 MB are "
                    "not committed (CONTRIBUTING.md section 7).", f.path))
    return findings


def check_tests(files: Sequence[ChangedFile]) -> List[Finding]:
    """Code changes come with test changes."""
    code = [
        f for f in files if f.status != "D" and f.path.endswith(".py")
        and f.path.startswith(CODE_PREFIXES)
    ]
    tests = [f for f in files if f.path.startswith(TEST_PREFIXES)]
    if code and not tests:
        return [
            Finding(
                "warning", "tests",
                f"{len(code)} Python file(s) changed under "
                f"{', '.join(CODE_PREFIXES)} but no test did. New "
                "functionality needs tests (CONTRIBUTING.md section 6).")
        ]
    return []


def check_commits(commits: Iterable[Commit]) -> List[Finding]:
    """Commit messages follow CONTRIBUTING.md section 5."""
    findings = []
    for c in commits:
        if c.is_merge:
            continue
        problems = []
        if len(c.subject) > MAX_SUBJECT_CHARS:
            problems.append(f"subject is {len(c.subject)} characters "
                            f"(max {MAX_SUBJECT_CHARS})")
        if c.subject[:1].islower():
            problems.append("subject starts with a lowercase letter")
        if c.subject.endswith("."):
            problems.append("subject ends with a period")
        if not c.body:
            problems.append("no body explaining what changed and why")
        if problems:
            findings.append(
                Finding(
                    "warning", "commit-message",
                    f'{c.sha[:8]} "{c.subject}": ' + "; ".join(problems) +
                    " (CONTRIBUTING.md section 5)."))
    return findings


def check_branch(branch: Optional[str]) -> List[Finding]:
    """The branch name follows CONTRIBUTING.md section 4."""
    if branch is None or BRANCH_PATTERN.match(branch):
        return []
    return [
        Finding(
            "warning", "branch-name",
            f'Branch "{branch}" does not follow <type>-<name>-<description>, '
            'for example "milestone2-triet-client-simulator" '
            "(CONTRIBUTING.md section 4).")
    ]


def run_checks(body: Optional[str],
               template: str,
               files: Sequence[ChangedFile],
               commits: Sequence[Commit],
               is_upstream: Callable[[str], bool],
               branch: Optional[str] = None) -> List[Finding]:
    """All checks; ``body=None`` skips the description-based ones."""
    findings = []
    if body is not None:
        findings += check_description(body, template)
        findings += check_upstream(files, body, is_upstream)
    findings += check_files(files)
    findings += check_tests(files)
    findings += check_commits(commits)
    findings += check_branch(branch)
    return findings


# ---------------------------------------------------------------------------
# Reporting


def _escape_data(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape_data(text).replace(":", "%3A").replace(",", "%2C")


def annotation(finding: Finding) -> str:
    """A GitHub Actions workflow command that annotates the PR."""
    props = f"title={_escape_property(finding.rule)}"
    if finding.path:
        props = f"file={_escape_property(finding.path)}," + props
    return f"::{finding.level} {props}::{_escape_data(finding.message)}"


def summary_markdown(findings: Sequence[Finding]) -> str:
    """The job summary shown on the workflow run page."""
    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    if not findings:
        return "## PR requirements\n\nAll checks passed.\n"
    lines = [
        "## PR requirements", "",
        f"{len(errors)} error(s), {len(warnings)} warning(s). Errors fail "
        "this check; see `CONTRIBUTING.md`.", "", "| Level | Rule | Detail |",
        "|---|---|---|"
    ]
    for f in errors + warnings:
        detail = f.message.replace("|", "\\|")
        lines.append(f"| {f.level} | {f.rule} | {detail} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Collecting data from git and the GitHub event


def _git(*args: str) -> str:
    return subprocess.run(["git", *args],
                          check=True,
                          capture_output=True,
                          text=True).stdout


def changed_files(base: str, head: str) -> List[ChangedFile]:
    """Files changed by the PR since it branched from ``base``."""
    files = []
    out = _git("diff", "--name-status", "--no-renames", f"{base}...{head}")
    for line in out.splitlines():
        status, path = line.split("\t", 1)
        status = "M" if status[0] == "T" else status[0]
        size = 0 if status == "D" else int(
            _git("cat-file", "-s", f"{head}:{path}"))
        files.append(ChangedFile(status, path, size))
    return files


def pr_commits(base: str, head: str) -> List[Commit]:
    """Commits on the PR branch that are not on ``base``."""
    commits = []
    out = _git("log", "--format=%H%x00%P%x00%s%x00%b%x1e", f"{base}..{head}")
    for record in out.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        sha, parents, subject, body = record.split("\x00", 3)
        commits.append(
            Commit(sha, subject, body.strip(),
                   len(parents.split()) > 1))
    return commits


def is_upstream_file(path: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{UPSTREAM_BASE}:{path}"],
        capture_output=True)
    return result.returncode == 0


def _require_upstream_base() -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{UPSTREAM_BASE}^{{commit}}"],
        capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"Commit {UPSTREAM_BASE[:8]} is missing; check out "
                         "the full history (actions/checkout fetch-depth: 0).")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a pull request against CONTRIBUTING.md.")
    parser.add_argument("--base", help="Base commit (default: from the event)")
    parser.add_argument("--head", help="Head commit (default: from the event)")
    parser.add_argument("--branch",
                        help="PR branch name (default: from the event, or "
                        "the current branch)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--body-file", type=Path, help="PR description")
    group.add_argument("--no-description",
                       action="store_true",
                       help="Skip the checks that need a PR description")
    parser.add_argument("--template", type=Path, default=Path(TEMPLATE_PATH))
    args = parser.parse_args(argv)

    event_path = os.environ.get("GITHUB_EVENT_PATH")
    pr = {}
    if event_path:
        with open(event_path, encoding="utf-8") as f:
            pr = json.load(f).get("pull_request") or {}
    base = args.base or pr.get("base", {}).get("sha")
    head = args.head or pr.get("head", {}).get("sha")
    if not base or not head:
        parser.error("--base and --head are required outside a PR event")
    body: Optional[str]
    if args.no_description:
        body = None
    elif args.body_file:
        body = args.body_file.read_text(encoding="utf-8")
    elif pr:
        body = pr.get("body") or ""
    else:
        parser.error("pass --body-file or --no-description outside a PR event")
    template = (args.template.read_text(
        encoding="utf-8") if args.template.is_file() else "")

    branch = args.branch or pr.get("head", {}).get("ref")
    if branch is None and not event_path:
        current = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
        branch = None if current == "HEAD" else current  # detached: skip

    _require_upstream_base()
    findings = run_checks(body, template, changed_files(base, head),
                          pr_commits(base, head), is_upstream_file, branch)

    for finding in findings:
        print(
            annotation(finding) if event_path else
            f"{finding.level.upper()} [{finding.rule}] {finding.message}")
    summary = summary_markdown(findings)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(summary)
    else:
        print(summary)
    return 1 if any(f.level == "error" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
