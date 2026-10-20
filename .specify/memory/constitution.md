# Prefix-Sharing Study Constitution

The non-negotiable rules for every spec, plan and change in this repository.
`CONTRIBUTING.md` holds the day-to-day procedures (branch names, commit
format, PR description, lint commands); this document holds the principles
those procedures serve. When the two disagree, this document wins and
`CONTRIBUTING.md` is amended.

Source documents: the course specification *Project: A study of prefix
sharing in LLM serving* (COMP 536) and the team's *COMP 536 Project Charter*.

## Core Principles

### I. Faithful simulation, nothing else changes

The simulator replaces the model forward pass and nothing more. Scheduling,
continuous batching, request handling, the KV-cache block manager and prefix
caching shall behave exactly as in unmodified vLLM, so that every cache
statistic we report is one real vLLM would produce.

- All engine work targets the **v0 engine** under `vllm/`. `vllm/v1/` is
  never modified.
- All reported results come from **online serving** (`vllm serve` plus the
  client simulator), never from the offline `LLM` entry point.
- Everything runs on a **CPU-only machine without model weights**.
- The upstream base commit `94d545a1a18e20ea8763a6760194589b8a3c9065` is
  never changed.

### II. Small, justified upstream changes

New behaviour lives in new modules (`vllm/simulator.py`, new evictors, new
instrumentation modules, `client_simulator/`, `analysis/`). Edits to existing
vLLM files are kept to the minimum hook needed. Every such edit is listed in
the PR description with its reason. A change that alters vLLM's default
behaviour when the project's flags are off is a defect.

### III. Tested and deterministic

Every feature ships with tests, and a bug fix ships with a test that fails
without it. Tests use fixed seeds, fake clocks and small handcrafted traces
whose expected results can be derived by hand (cached blocks, the block a
policy evicts, a reuse gap). Every project test module declares its
milestone (`pytestmark = pytest.mark.m1`, `m2` or `m3`), and
`tools/run_project_tests.sh` runs them all, or one milestone, on every PR.
vLLM's own `tests/core` subset also passes, because later milestones change
`vllm/core/`. CI enforces both (spec `001-ci-cd`).

### IV. One definition per metric

Each prefix-sharing metric has exactly one written definition (in the spec of
the task that introduces it) and exactly one implementation, which every later
task reuses. A metric definition states its unit, its denominator and how
edge cases count (a request with zero cache hits counts as `0`, never as
missing). The shared definitions are:

| Metric | Definition owner |
|---|---|
| Per-request cached fraction | `004-m2-prefix-sharing-analysis` |
| Hits per cache block | `004-m2-prefix-sharing-analysis` |
| Reuse interval of a block | `004-m2-prefix-sharing-analysis` |
| Eviction count and rate | `005-m2-cache-tuning` |

### V. Reproducible experiments

Every figure and table in the report is produced by a committed script from a
clean checkout by following documented steps. Every run records its full
configuration next to its output: dataset and sample size, model/tokenizer,
KV-cache capacity (blocks), block size, eviction policy, conversation rate,
think time, seed and commit hash. Datasets and bulk results are never
committed. Download and preprocessing scripts are committed instead. Small
fixtures under 1 MB in total may be committed.

### VI. Reviewed, preserved history

The submission is the `.git` folder itself, so history is part of the
deliverable. Work happens on feature branches created from the current
milestone branch (the default branch, e.g. `milestone-2`), reaches it only
through a reviewed PR (at least one other member approves), and is merged
with a merge commit; milestone branches reach `main` the same way. `main`
and the milestone branches are never rewritten or force-pushed.

### VII. Understood and disclosed AI use

Authors fully understand every line they submit, including AI-generated
code, and can explain it in review. PR descriptions state where AI tools were
used and include prompts that produced a significant portion of the change,
so the report's AI-usage section and appendix can be assembled from them.

## Deliverable Constraints

- **Due:** 2026-11-30 23:59 (the Canvas deadline wins if it differs).
- **Code:** the `.git` folder with full history; a root `README.md` with
  setup and run instructions for every milestone.
- **Report:** a PDF of at most 6 pages excluding references and appendix, in
  the MLSys 2025 LaTeX style, covering design and results of all milestones,
  with a dedicated AI-usage section.

## Development Workflow

1. **Specify:** write `specs/<NNN>-<name>/spec.md` from
   `.specify/templates/spec-template.md`. It describes *what* and *why*, with
   testable requirements and measurable success criteria.
2. **Plan:** write `plan.md` from the plan template. It describes *how*,
   names the files to touch, and passes the Constitution Check below.
3. **Tasks:** write `tasks.md` from the tasks template: ordered, small,
   each naming its file and its test.
4. **Implement:** follow `tasks.md` on a feature branch, ticking tasks off
   in the same PR.
5. **Review:** open a PR against the milestone branch with the description
   required by `CONTRIBUTING.md` §4. The reviewer checks the change against
   the spec's requirements and success criteria, not only the code.

A spec may be amended during implementation; the amendment is part of the
same PR and called out in its description.

### Constitution Check (every plan answers these)

- [ ] v0 engine only, online serving only, CPU only, no weights (I)
- [ ] Every edited upstream file listed with its reason; defaults unchanged
      when project flags are off (II)
- [ ] Tests named for every requirement; deterministic (III)
- [ ] Metrics reuse the single shared definition and implementation (IV)
- [ ] Each experiment records its full configuration; no data committed (V)
- [ ] Delivered through a reviewed PR with merge commit (VI)
- [ ] AI usage recorded for the PR description (VII)

## Governance

This constitution supersedes other practices in the repository. Amendments
are made by PR, need approval from at least two members, and bump the version
below: major for removing or redefining a principle, minor for adding one,
patch for wording.

**Version**: 1.0.1 | **Status**: Ratified 2026-10-20 (PR #7) |
**Last Amended**: 2026-10-20 (PRs target the milestone branch)
