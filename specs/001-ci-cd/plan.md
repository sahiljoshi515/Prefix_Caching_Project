# Implementation Plan: Project CI/CD

**Spec**: [spec.md](spec.md) | **Branch**: `milestone2-marcus-speckit` |
**Date**: 2026-10-18

## Summary

Add one workflow, `project-ci`, that installs the project the way the README
does (CPU-only, pinned) and runs the simulator suite and vLLM's `tests/core`
subset on every PR and push to `main`. Remove or scope the inherited vLLM
workflows that fail for reasons unrelated to our code, fix the `mypy` job
that already fails on `main`, and replace vLLM's PR template and Dependabot
settings. The end-to-end replay smoke test (US4) and the report/submission
delivery (US5) are later phases, gated on `003` and on `report/` existing.

## Technical Context

**Language/Version**: GitHub Actions YAML; Python 3.12 on `ubuntu-latest`
**Primary Dependencies**: `uv` 0.12.24 (the version on the team's laptops),
PyTorch 2.5.1 CPU wheels from `https://download.pytorch.org/whl/cpu`
**Testing**: `tests/simulator` (67 tests, about 21 s locally),
`tests/core` subset (944 tests, about 13 s locally), measured 2026-10-18
**Target Platform**: GitHub-hosted Linux runners
**Performance Goals**: whole job under 15 minutes (SC-003)
**Constraints**: no secrets (fork PRs), public downloads only

## Constitution Check

- [x] I. Tests run the simulator on CPU without weights; only `config.json`
      and tokenizer files are downloaded
- [x] II. No vLLM source file changes; only `.github/` files
- [x] III. The workflow runs the existing deterministic suites
- [x] IV. n/a (no metrics)
- [x] V. n/a (no experiments)
- [x] VI. Delivered through a PR; the workflow makes the review signal real
- [x] VII. AI usage recorded in the PR description

## Files

### New

| Path | Purpose |
|---|---|
| `.github/workflows/project-ci.yml` | Install the project, run `tests/simulator`, `tests/client_simulator` and `tests/analysis` when present, then the `tests/core` subset |
| `.github/workflows/pr-requirements.yml` | Run the PR checker's tests, then the checker on the PR (US7) |
| `tools/run_project_tests.sh` | Run only the project's tests, all or one milestone (`m1`/`m2`/`m3`), plus `check-markers`; used by CI and on laptops |
| `Makefile` | One set of commands for everyone and for CI: `setup` (macOS or Linux, CPU only), `test`, `test-m1/2/3`, `test-core`, `check-markers`, `ci`, `lint`, `check-pr`, `help`. GNU Make 3.81 compatible (macOS default) |
| `tools/pr_check/check_pr.py` | Standard-library checker: rules as pure functions, plus git and event-file collection and GitHub annotations |
| `tools/pr_check/test_check_pr.py` | 39 tests, one or more per rule, plus one against a real temporary git repository |

### Modified repository files (not vLLM source)

| Path | Change | Reason |
|---|---|---|
| `pyproject.toml` | Register pytest markers `m1`, `m2`, `m3` | Per-milestone test selection without unknown-marker warnings |
| `tests/simulator/test_{simulator,engine,online}.py` | `pytestmark = pytest.mark.m1` (one line each) | Tag the Milestone 1 tests |
| `.github/workflows/mypy.yaml` | Also install `setuptools<76` | On Python 3.12 the runner has no `setuptools`, so `pkg_resources` (imported by `ray_gpu_executor.py` and `ray_hpu_executor.py`) cannot be found; 3.9 to 3.11 runners ship it. Reproduced locally with `mypy==1.11.1`, which passes once setuptools is present |
| `.github/workflows/lint-and-deploy.yaml` | Add a `paths` filter for `examples/chart-helm/**` | Helm chart test unrelated to the project; it always fails at the minio image pull |
| `.github/PULL_REQUEST_TEMPLATE.md` | Replace with the six items of `CONTRIBUTING.md` §4 | vLLM's template points contributors to vLLM's process |
| `.github/dependabot.yml` | Drop the `pip` ecosystem and the vLLM reviewers | Dependency pins are deliberate (`constraints-simulator.txt`); a bump breaks import |

### Deleted

| Path | Reason |
|---|---|
| `.github/workflows/reminder_comment.yml` | Posts vLLM's Buildkite instructions; fails on fork PRs |
| `.github/workflows/publish.yml` | Builds vLLM release wheels on `v*` tags; must never run here |

`cleanup_pr_body.yml` stays: it only strips vLLM template strings and is a
no-op with the new template. `stale.yml` (90 + 30 days) cannot affect a
one-semester project. Both can be removed later if they cause noise.

## Design

### Installing on a CPU-only Linux runner

The README's Linux line is
`VLLM_TARGET_DEVICE=empty uv pip install --editable . -c constraints-simulator.txt`.
With `empty`, `setup.py` reads `requirements-cuda.txt`, so PyPI would
deliver the CUDA build of `torch==2.5.1` plus NVIDIA libraries (several GB),
both at runtime and in the isolated build environment (`pyproject.toml`
lists `torch == 2.5.1` as a build requirement). The workflow therefore adds
the PyTorch CPU index:

```
--extra-index-url https://download.pytorch.org/whl/cpu
--index-strategy unsafe-best-match
```

`torch==2.5.1` matches the local version `2.5.1+cpu` (PEP 440 ignores local
labels when the specifier has none), and `unsafe-best-match` lets uv choose
it over PyPI's `2.5.1`. uv uses the same index settings for build isolation.
`VLLM_TARGET_DEVICE=empty` is set only on the install step; at runtime
`VLLM_USE_SIMULATOR=1` pins the CPU platform.

Options considered:

- **Plain pip.** Works the same way, but the README and the course spec use
  `uv`, and uv resolves this tree much faster.
- **`VLLM_TARGET_DEVICE=cpu`.** Would compile vLLM's CPU kernels with CMake,
  which takes many minutes and is unnecessary because the simulator never
  runs a kernel.

### PR requirements checker (US7)

- Rules are pure functions over plain data (description text, changed files
  with status and size, commits), so each is unit-tested without git or
  GitHub. A thin layer collects that data with `git diff --name-status
  base...head`, `git log base..head` and `git cat-file`.
- "Upstream file" means the path exists at the fork's base commit
  `94d545a1`, so new project modules under `vllm/` (for example
  `vllm/simulator.py`) are not flagged, while files the project already
  edited (for example `vllm/envs.py`) are.
- The tests live in `tools/pr_check/`, not `tests/`, because
  `tests/conftest.py` imports PyTorch and vLLM. The check therefore installs
  only pytest and runs in seconds.
- Security: the workflow uses `pull_request` (read-only token, safe for
  forks), and the description reaches the script through
  `GITHUB_EVENT_PATH`, never through `${{ }}` in a shell line.

### One job, separate steps

The simulator, client-simulator and analysis suites and the `tests/core`
subset run as separate steps of one job, which pays for the install once
(the expensive part). The step that fails says which suite broke. Because
the job always runs, it can be a required check, which a path-filtered
workflow cannot.

### Later phases

- **US4, e2e smoke** (after `003` merges): a step that starts
  `vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching
  --enable-prompt-tokens-details` with a committed 20-conversation fixture
  (`tests/fixtures/sharegpt_small.json`), runs `process_raw_traces` and
  `client_sim` in both modes, and asserts that all requests complete and that
  the multi-turn mean cached fraction is above 0.
- **US5, report and submission** (after `report/` exists): `report.yml` builds
  the MLSys LaTeX source with a pinned TeX Live action and checks the page
  count with `pdfinfo`. `submission.yml` runs on `submission-*` tags with
  `contents: write`, creates a `git bundle --all`, and attaches it and the PDF
  to a release.

## Verification

1. `python -c "import yaml; [yaml.safe_load(open(f)) for f in ...]"` on every
   changed workflow (syntax).
2. Run the job's commands locally on macOS (same pins), already done for the
   test commands: 67 and 944 passed.
3. Push the branch and open the PR: `project-ci / tests` must be green, and
   `lint-and-deploy` and `pr_reminder` must not appear (SC-001).
4. On a scratch commit, break one simulator assertion and confirm exactly one
   failing step (SC-002), then drop the commit.
5. Read the job duration from the run (SC-003).

## Risks

| Risk | Mitigation |
|---|---|
| uv resolves a non-CPU torch or a different version | The constraint `torch==2.5.1` plus the CPU index; the job prints `torch.__version__` and fails unless it ends in `+cpu` |
| `ray[default]` and `xformers` from `requirements-cuda.txt` slow the install | Accept for now (they are pure wheels); cache uv's directory later if the install exceeds 5 minutes |
| Hugging Face download flakes | Tests already prefer the local cache; add an `actions/cache` step for `~/.cache/huggingface` if flakes appear |
| `test_online.py` port clashes on a shared runner | Each job has its own VM; tests pick free ports via `RemoteOpenAIServer` |
