# Contributing Guidelines

This document defines how team members contribute to this repository. It
applies to all code, tests, experiment scripts and documentation.

The keywords **shall**, **should** and **may** are used as defined in
[RFC 2119](https://www.rfc-editor.org/rfc/rfc2119): *shall* denotes a
requirement, *should* a recommendation that may be departed from only for a
stated reason, and *may* an option.

## 1. Purpose and Scope

This repository is the team's fork of [vLLM](https://github.com/vllm-project/vllm)
at commit `94d545a1a18e20ea8763a6760194589b8a3c9065`, extended for the
COMP 536 project *A Study of Prefix Sharing in LLM Serving*. Contributions
shall serve the objectives of that project:

- **Milestone 1:** a simulator that bypasses model execution and replays
  responses from traces;
- **Milestone 2:** prefix sharing for a homogeneous chatbot workload
  (ShareGPT), including the client simulator, the prefix-sharing metrics and
  the cache-parameter study;
- **Milestone 3:** cache eviction for heterogeneous and mixed workloads.

Changes intended for vLLM itself are out of scope (see Section 11).

## 2. Project Constraints

The following constraints derive from the project specification. Every
contribution shall respect them.

- **Engine version.** All modifications shall target the vLLM v0 engine
  under `vllm/`. The v1 engine (`vllm/v1/`) is out of scope and shall not be
  modified.
- **Serving mode.** Reported results shall be produced in online serving
  mode (`vllm serve`), not with the offline `LLM` entry point.
- **Hardware.** The code shall run on a CPU-only machine. No contribution
  may require a GPU or model weights.
- **Base commit.** The upstream base commit shall not be changed.
- **Isolation of changes.** Simulator logic should reside in
  `vllm/simulator.py` or in new modules. Changes to upstream vLLM modules
  should be minimal, and each one shall be justified in the pull request
  description.

## 3. Development Environment

The environment shall be set up as described in the *Setup* section of
[`README.md`](README.md); `make setup` does the same in one command on macOS
and Linux, and `make help` lists the other shortcuts. In particular:

- Dependencies shall be installed with the constraints file
  `constraints-simulator.txt`. The pinned versions are required for this
  vLLM commit to import correctly.
- `VLLM_USE_SIMULATOR=1` shall be exported before Python starts, because
  vLLM selects its platform at import time.
- A pull request that introduces a new dependency shall pin its version in
  `constraints-simulator.txt` and add the installation step to the README.

## 4. Branching and Review

All changes shall be developed on a dedicated feature branch created from
`main`. Direct commits to `main` are not permitted.

Branch names shall take the form `<type>-<name>-<description>`: the type
below, the author's first name, and a short description, all lowercase with
words separated by hyphens:

| Type | Purpose | Example |
|---|---|---|
| `milestone1`, `milestone2`, `milestone3` | Work on a project milestone | `milestone2-triet-client-simulator`, `milestone3-jasmine-lfu-eviction` |
| `fix` | Correction of a defect | `fix-xinliang-reuse-gap-units` |
| `docs` | Documentation only | `docs-marcus-contributing-guide` |
| `ci` | Continuous-integration configuration | `ci-marcus-simulator-tests` |
| `report` | The technical report | `report-sahil-milestone2-results` |

Each feature should be specified first under `specs/` (see
[`specs/README.md`](specs/README.md)). Once it has been implemented and
tested, its author shall open a pull request (PR) against `main`. Each PR
shall:

- address a single feature or fix and contain no unrelated changes;
- pass the tests required by Section 6 and include tests for new
  functionality;
- describe what was changed, why it was changed and how it was tested.

The PR description shall contain the following items:

1. **Summary** of the change.
2. **Milestone and task** addressed (e.g. *Milestone 2, Task 1*).
3. **Upstream changes:** each modified vLLM file and the reason for the
   modification, or "none".
4. **Testing:** the commands run and their outcome.
5. **Effect on results:** whether any reported number or figure changes,
   and why.
6. **AI usage**, as required by Section 10.

The PR template provides these headings. The `pr-requirements` check fails a
PR whose description lacks one of them, which changes an upstream vLLM file
without listing it, which touches `vllm/v1/`, or which commits data, results
or a file over 1 MB; it warns about missing tests, commit messages that break
Section 5 and branch names that break the pattern above. Run it before
opening the PR:

```bash
python tools/pr_check/check_pr.py --base origin/main --head HEAD \
    --body-file my_pr_description.md
```

At least one other team member shall review and approve the PR before it is
merged. Review feedback should be addressed in follow-up commits rather than
by rewriting branch history. PRs shall be merged with a merge commit, not
squashed, so that the complete commit history required for submission is
preserved. The history of `main` shall never be rewritten or force-pushed.

## 5. Commit Conventions

- Each commit shall contain one logical change, and the test suite should
  pass at every commit.
- The subject line shall use the imperative mood ("Add LFU evictor", not
  "Added LFU evictor"), begin with a capital letter, omit the trailing
  period and not exceed 72 characters.
- Every commit except a trivial one shall include a body, separated from the
  subject by a blank line and wrapped at 72 characters, that explains what
  changed and why.
- Commits shall not contain datasets, model files, bulk experiment output,
  credentials or local environment files such as `.venv/`.

Example:

```
Add LFU evictor for the prefix-caching allocator

LRU is the only eviction policy in vLLM v0. LFUEvictor evicts the free
block with the fewest reuses and breaks ties by least recent access, so
that Milestone 2, Task 3 can compare it against LRU and FIFO.
```

## 6. Testing Requirements

All commands run from the repository root with the virtual environment
active.

1. Every PR shall pass the project's tests, which these commands run with
   the required environment variables, for all milestones or for one:

   ```bash
   make test                          # all milestones
   make test-m2                       # one milestone: test-m1, -m2, -m3
   make ci                            # everything CI runs
   ```

   `make` calls `tools/run_project_tests.sh`, which can also be run directly.

   Each test module shall declare its milestone with
   `pytestmark = pytest.mark.m1` (or `m2`, `m3`); CI fails a project test
   without one.

2. A PR that modifies `vllm/core/` (scheduler, block manager or evictors)
   shall also pass vLLM's own scheduler and block-manager tests:

   ```bash
   python -m pytest tests/core \
       --ignore=tests/core/block/e2e \
       --ignore=tests/core/test_num_computed_tokens_update.py
   ```

   The two excluded paths exercise real model execution and multi-step
   scheduling respectively, neither of which the simulator supports.

3. New functionality shall be accompanied by tests, and a bug fix should
   include a test that fails without the fix.
4. Tests shall be deterministic. They should use fixed random seeds, a fake
   clock instead of wall-clock time, and small handcrafted traces whose
   expected results (for example, the number of cached blocks or the block
   a policy evicts) can be derived by hand.
5. The author shall run the applicable tests locally before requesting
   review. All required checks (`project-ci`, `pr-requirements`, `ruff`,
   `yapf`, `mypy`, `codespell`) shall pass before a PR is merged.

## 7. Experiments and Data

- Every figure and table in the technical report shall be produced by a
  committed script, and it shall be possible to regenerate it from a clean
  checkout by following documented steps.
- Each experiment shall record its full configuration together with its
  output: dataset and sample, tokenizer, prefix-cache size, block size,
  eviction policy, request rate, random seed and commit hash.
- Datasets (ShareGPT, AgentBank, CC-Bench-trajectories,
  qwen-bailian-usagetraces-anon and others) shall not be committed. The
  scripts that download and preprocess them shall be committed instead.
  Small samples used as test fixtures may be committed and should be kept
  under 1 MB in total.
- Raw logs and bulk results should not be committed. Final figures that
  appear in the report may be, under `report/figures/<task>/`.

## 8. Documentation

Documentation lives in this repository as Markdown and is reviewed in the
same PR as the change it describes. Each kind of information has one home;
other documents link to it rather than repeat it:

| Question | Home |
|---|---|
| How to install, run and test the project | [`README.md`](README.md) |
| How the team works | `CONTRIBUTING.md` (this document) |
| What a task builds and why, how, and what is left | `specs/<NNN>-<name>/spec.md`, `plan.md`, `tasks.md` |
| What a task found: numbers, figures, interpretation | `report/notes/<task>.md` |
| The figures the technical report uses | `report/figures/<task>/` |
| How AI tools were used | `report/notes/ai-usage.md` (Section 10) |
| File formats, classes and functions | Docstrings in the code |

- A PR that changes behaviour, commands or results shall update the
  affected documents in the same PR: the README for commands, the spec's
  `plan.md` and `tasks.md` for design decisions and progress, and the
  task's note for results.
- A fact should be written once and linked from elsewhere. For example,
  metric definitions live in the spec, and the README and the notes link to
  them.
- A task's note shall be written when its results are final, in the shape
  [`report/README.md`](report/README.md) describes, and shall give the
  command and the commit behind every number.

## 9. Code Style

- Code should follow the conventions of the surrounding vLLM code,
  including type hints and docstrings for public classes and functions.
- Changed Python files should pass the repository's pinned linters, which
  are configured in `pyproject.toml`:

  ```bash
  uv pip install ruff==0.6.5 isort==5.13.2 yapf==0.32.0 toml==0.10.2
  ruff check <files>
  isort --check-only <files>
  yapf --diff <files>
  ```

## 10. AI-Assisted Contributions

The project specification permits AI tools subject to the following rules:

- The author shall fully understand all code they submit, including code
  generated by AI tools, and shall be able to explain it during review.
- The PR description shall state whether AI tools were used and for which
  parts of the change.
- Prompts that produced a significant portion of a change shall be
  recorded, so that they can be collected for the AI-usage section and the
  appendix of the technical report. A PR that used AI tools shall add an
  entry with them to `report/notes/ai-usage.md`, because PR descriptions are
  not part of the submitted git history; the description may point to that
  entry instead of repeating the prompts.
- Commits written largely by an AI tool may carry a `Co-Authored-By:`
  trailer that names the tool.

## 11. Relationship to Upstream vLLM

This repository is a course project and is not affiliated with the vLLM
project. Contributions to vLLM itself shall follow the upstream
[contributing guide](https://docs.vllm.ai/en/latest/contributing/overview.html).
