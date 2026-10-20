"""Tests for check_pr.py: ``python -m pytest tools/pr_check``.

They live outside ``tests/`` because ``tests/conftest.py`` imports vLLM and
PyTorch, which this check deliberately does not install.
"""
import subprocess
from pathlib import Path

import pytest
from check_pr import (REQUIRED_SECTIONS, ChangedFile, Commit, Finding,
                      annotation, changed_files, check_branch, check_commits,
                      check_description, check_files, check_tests,
                      check_upstream, parse_sections, pr_commits, run_checks,
                      summary_markdown)

TEMPLATE = (Path(__file__).resolve().parents[2] / ".github" /
            "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")


def filled_body(**overrides: str) -> str:
    """A complete PR description; ``overrides`` replace section texts."""
    texts = {
        "Summary": "Add the LFU evictor.",
        "Milestone and task": "Milestone 2, Task 3 "
        "(specs/005-m2-cache-tuning/spec.md)",
        "Upstream changes": "none",
        "Testing": "pytest tests/core: 950 passed",
        "Effect on results": "None yet.",
        "AI usage": "None.",
    }
    texts.update(overrides)
    return "\n\n".join(f"## {name}\n\n{text}" for name, text in texts.items())


def rules(findings, level=None):
    return [f.rule for f in findings if level is None or f.level == level]


def never_upstream(path: str) -> bool:
    return False


# --- description -----------------------------------------------------------


def test_parse_sections_ignores_comments_and_code_fences():
    body = ("## Summary\n<!-- hint -->\nReal text\n"
            "## Testing\n```\n## not a heading\n```\n")
    sections = parse_sections(body)
    assert sections["summary"] == "Real text"
    assert "## not a heading" in sections["testing"]
    assert "not a heading" not in sections


def test_complete_description_passes():
    assert check_description(filled_body(), TEMPLATE) == []


def test_unfilled_template_fails_every_section():
    findings = check_description(TEMPLATE, TEMPLATE)
    assert rules(findings, "error") == ["description"] * len(REQUIRED_SECTIONS)


def test_missing_section_is_an_error():
    body = filled_body().replace("## AI usage", "## Notes")
    findings = check_description(body, TEMPLATE)
    assert len(findings) == 1
    assert 'Missing section "## AI usage"' in findings[0].message


def test_empty_section_is_an_error():
    findings = check_description(filled_body(Testing=""), TEMPLATE)
    assert rules(findings) == ["description"]
    assert "## Testing" in findings[0].message


def test_headings_match_case_insensitively_at_any_level():
    body = filled_body().replace("## Summary", "### SUMMARY")
    assert check_description(body, TEMPLATE) == []


def test_missing_spec_link_is_a_warning():
    findings = check_description(
        filled_body(**{"Milestone and task": "Milestone 2, Task 3"}), TEMPLATE)
    assert rules(findings, "warning") == ["spec-link"]
    assert rules(findings, "error") == []


# --- upstream files and the v1 engine --------------------------------------


def test_v1_engine_change_is_an_error():
    files = [ChangedFile("M", "vllm/v1/engine/core.py", 10)]
    assert rules(check_upstream(files, filled_body(),
                                never_upstream)) == ["v1-engine"]


def test_unlisted_upstream_file_is_an_error():
    files = [ChangedFile("M", "vllm/core/evictor.py", 10)]
    findings = check_upstream(files, filled_body(), lambda path: True)
    assert rules(findings) == ["upstream"]
    assert findings[0].path == "vllm/core/evictor.py"


def test_listed_upstream_file_passes():
    body = filled_body(
        **{
            "Upstream changes":
            "- `vllm/core/evictor.py`: register LFU in make_evictor"
        })
    files = [ChangedFile("M", "vllm/core/evictor.py", 10)]
    assert check_upstream(files, body, lambda path: True) == []


def test_new_project_module_under_vllm_is_not_upstream():
    files = [ChangedFile("A", "vllm/core/eviction_policies.py", 10)]
    assert check_upstream(files, filled_body(), lambda path: True) == []


# --- files -----------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["data/raw/sharegpt.json", "results/multi.jsonl", ".venv/bin/python"])
def test_data_results_and_envs_are_errors(path):
    assert rules(check_files([ChangedFile("A", path, 10)])) == ["data"]


def test_large_file_is_an_error():
    findings = check_files([ChangedFile("A", "tests/fixture.json", 2_000_000)])
    assert rules(findings) == ["size"]
    assert "2.0 MB" in findings[0].message


def test_deleting_results_is_allowed():
    assert check_files([ChangedFile("D", "results/multi.jsonl", 0)]) == []


# --- tests -----------------------------------------------------------------


def test_code_without_tests_is_a_warning():
    files = [ChangedFile("M", "client_simulator/client_sim.py", 10)]
    assert rules(check_tests(files), "warning") == ["tests"]


def test_code_with_tests_passes():
    files = [
        ChangedFile("M", "client_simulator/client_sim.py", 10),
        ChangedFile("A", "tests/client_simulator/test_client_sim.py", 10),
    ]
    assert check_tests(files) == []


def test_docs_only_change_needs_no_tests():
    assert check_tests([ChangedFile("M", "README.md", 10)]) == []


# --- commits ---------------------------------------------------------------


def test_good_commit_passes():
    commit = Commit("a" * 40, "Add LFU evictor", "Why and what.", False)
    assert check_commits([commit]) == []


def test_bad_commit_lists_every_problem():
    commit = Commit("b" * 40, "implement client sim.", "", False)
    findings = check_commits([commit])
    assert rules(findings, "warning") == ["commit-message"]
    message = findings[0].message
    assert "lowercase" in message
    assert "period" in message
    assert "no body" in message


def test_long_subject_and_merge_commits():
    long = Commit("c" * 40, "A" * 73, "body", False)
    merge = Commit("d" * 40, "merge branch main", "", True)
    findings = check_commits([long, merge])
    assert len(findings) == 1
    assert "73 characters" in findings[0].message


# --- branch name -----------------------------------------------------------


@pytest.mark.parametrize("branch", [
    "milestone2-marcus-speckit", "milestone3-jasmine-lfu-eviction",
    "fix-xinliang-reuse-gap-units", "docs-marcus-contributing-guide",
    "ci-marcus-simulator-tests", "report-sahil-milestone2-results"
])
def test_branch_names_that_follow_the_pattern(branch):
    assert check_branch(branch) == []


@pytest.mark.parametrize("branch", [
    "client-simulator", "m2/client-simulator", "milestone4-marcus-x",
    "milestone2-Marcus-speckit", "milestone2-marcus"
])
def test_other_branch_names_are_a_warning(branch):
    assert rules(check_branch(branch), "warning") == ["branch-name"]


def test_unknown_branch_is_not_checked():
    assert check_branch(None) == []


# --- combined and reporting ------------------------------------------------


def test_run_checks_without_description_skips_description_rules():
    files = [ChangedFile("M", "vllm/core/evictor.py", 10)]
    findings = run_checks(None, TEMPLATE, files, [], lambda path: True)
    assert "description" not in rules(findings)
    assert "upstream" not in rules(findings)


def test_annotation_escapes_workflow_command_syntax():
    finding = Finding("error", "data", "50% done\nnext line", "a,b:c.py")
    assert annotation(finding) == (
        "::error file=a%2Cb%3Ac.py,title=data::50%25 done%0Anext line")


def test_summary_counts_and_escapes_pipes():
    findings = [
        Finding("warning", "tests", "x | y"),
        Finding("error", "data", "bad"),
    ]
    summary = summary_markdown(findings)
    assert "1 error(s), 1 warning(s)" in summary
    assert "x \\| y" in summary
    assert summary.index("| error |") < summary.index("| warning |")


def test_summary_when_clean():
    assert "All checks passed." in summary_markdown([])


# --- git integration -------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          check=True,
                          capture_output=True,
                          text=True).stdout.strip()


def test_changed_files_and_commits_from_a_real_repository(
        tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "keep.txt").write_text("keep\n")
    (repo / "gone.txt").write_text("gone\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "Initial commit", "-m", "Body.")
    base = _git(repo, "rev-parse", "HEAD")

    _git(repo, "switch", "-q", "-c", "feature")
    (repo / "keep.txt").write_text("changed\n")
    (repo / "gone.txt").unlink()
    (repo / "new.txt").write_text("12345")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change files")
    head = _git(repo, "rev-parse", "HEAD")

    monkeypatch.chdir(repo)
    files = {f.path: f for f in changed_files(base, head)}
    commits = pr_commits(base, head)

    assert files["keep.txt"].status == "M"
    assert files["gone.txt"] == ChangedFile("D", "gone.txt", 0)
    assert files["new.txt"] == ChangedFile("A", "new.txt", 5)
    assert [(c.subject, c.body, c.is_merge)
            for c in commits] == [("change files", "", False)]
