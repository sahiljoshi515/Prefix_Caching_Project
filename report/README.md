# Technical report

The course asks for one PDF on the design and results of all milestones: at
most 6 pages excluding references and appendix, in the
[MLSys 2025 LaTeX format](https://media.mlsys.org/Conferences/MLSYS2025/mlsys2025style.zip),
with a section on how we used AI and the most helpful prompts in an
appendix. This folder is where that PDF is built from.

The report is written in two stages. While the milestones are in progress,
each task writes a **note**: a Markdown page with its findings, committed in
the task's PR. At the end, the paper condenses the notes. Writing the notes
as results come in means the paper is assembly, not a reconstruction from
memory a week before the deadline.

## Layout

| Path | Holds | Written |
|---|---|---|
| `notes/<task>.md` | One task's findings: setup, numbers, figures, interpretation | In the task's PR, once its results are final |
| `notes/ai-usage.md` | How AI tools were used, and the prompts worth quoting | See [AI usage](#ai-usage) |
| `figures/<task>/` | The figures the paper uses (PDF for LaTeX, PNG for the notes) | With the note |
| LaTeX source (MLSys 2025 style) | The paper | Once Milestone 3's notes exist; `specs/001-ci-cd` (User Story 5) then builds the PDF in CI |

Bulk output (event logs, CSVs, every figure a run draws) stays in
`results/`, which git ignores. Only the figures the paper uses are copied
here.

## Notes

| Note | Task | Spec |
|---|---|---|
| [`m1-simulator.md`](notes/m1-simulator.md) | Milestone 1: the simulator | [`002`](../specs/002-m1-simulator-verification/spec.md) |
| `m2-task1.md` | Milestone 2, Task 1: the client simulator (design only) | [`003`](../specs/003-m2-client-simulator/spec.md) |
| `m2-task2.md` | Milestone 2, Task 2: ShareGPT single- vs multi-turn sharing | [`004`](../specs/004-m2-prefix-sharing-analysis/spec.md) |
| `m2-task3.md` | Milestone 2, Task 3: cache size, block size, eviction policy | [`005`](../specs/005-m2-cache-tuning/spec.md) |
| `m3-task1.md` | Milestone 3, Task 1: the best policy per workload | [`006`](../specs/006-m3-workload-eviction/spec.md) |
| `m3-task2.md` | Milestone 3, Task 2: a policy for the mixed workload | [`007`](../specs/007-m3-mixed-workload-policy/spec.md) |

Each note follows the shape of a paper section, so that condensing it is
mostly cutting:

1. **Question.** What the task asks, in one or two sentences, with a link to
   the spec. Definitions stay in the spec; the note links to them.
2. **Setup.** Sample, settings and cache configuration, the exact command,
   and the commit it ran at (the run's `run.json` records it).
3. **Results.** The figures from `figures/<task>/` and the key numbers in a
   table.
4. **Interpretation.** What the numbers mean, what was unexpected, and how it
   connects to the other tasks.
5. **Limitations.** What the simulator or the sample cannot show.

One to two pages of Markdown is enough; the paper gives each task about a
page.

## Paper outline (draft)

A starting point for the team to change. Page counts are rough and add up to
the 6-page limit.

| Section | Content | From | Pages |
|---|---|---|---|
| 1. Introduction | Prefix sharing in LLM serving, what we measured, the main findings | All notes | 0.5 |
| 2. Simulator | Replacing model execution while keeping vLLM's scheduler and cache | `m1-simulator.md` | 0.75 |
| 3. Workload replay and metrics | Client simulator, cache event log, metric definitions | `m2-task1.md`, `m2-task2.md` | 0.5 |
| 4. Homogeneous chatbot workload | ShareGPT single- vs multi-turn; cache parameters and policies | `m2-task2.md`, `m2-task3.md` | 1.75 |
| 5. Heterogeneous and mixed workloads | Best policy per workload; the workload-aware policy against the baselines | `m3-task1.md`, `m3-task2.md` | 1.75 |
| 6. AI usage | Required by the course | `ai-usage.md` | 0.25 |
| 7. Conclusion | | | 0.5 |
| Appendix | The most helpful prompts (required) | `ai-usage.md` | not counted |

## AI usage

The course requires a section on how we used AI and, in an appendix, the
prompts that produced a significant part of the code. `CONTRIBUTING.md`
(Section 10) asks every PR to record this. PR descriptions live on GitHub,
not in the git history we submit, so each PR also adds an entry to
[`notes/ai-usage.md`](notes/ai-usage.md), which the paper draws from.

## Rules

These come from `CONTRIBUTING.md` (Sections 7 and 8):

- Every figure here is produced by a committed script, and its note gives
  the command and the commit that produced it.
- A PR that changes a reported number or figure updates the note in the
  same PR and says so under "Effect on results".
- Numbers in the paper come from the notes, and notes link to the specs for
  definitions instead of repeating them.
