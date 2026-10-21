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

One folder per milestone, so each milestone can be reviewed on its own:

```
report/
├── README.md              this page
├── ai-usage.md            AI usage of every PR, for the paper's AI section
├── milestone1/
│   ├── changelog.md       what changed, who changed it, in which commit or PR
│   └── simulator.md       how the simulator works and how it was tested
├── milestone2/
│   ├── changelog.md
│   ├── task1-client-simulator.md
│   ├── task2-prefix-sharing-analysis.md
│   ├── task3-cache-tuning.md
│   └── figures/task<K>/   the figures the paper uses
└── milestone3/
    ├── changelog.md
    ├── task1-workload-eviction.md
    └── task2-mixed-workload-policy.md
```

A task's note is named `task<K>-<name>.md`, with the name of its spec
(`specs/005-m2-cache-tuning` gives `task3-cache-tuning.md`). A task that has
not started has a placeholder with the sections to fill.

| File | Holds | Written |
|---|---|---|
| `milestone<N>/changelog.md` | What the group changed in the milestone, task by task and newest first, with who and which commit or PR | An entry in each PR that adds to the milestone |
| `milestone<N>/task<K>-<name>.md` | One task's findings: setup, numbers, figures, interpretation | A placeholder until the task starts; filled in the task's PR once its results are final |
| `milestone<N>/figures/task<K>/` | The figures the paper uses (PDF for LaTeX, PNG for the notes) | With the note |
| `ai-usage.md` | How AI tools were used, and the prompts worth quoting | See [AI usage](#ai-usage) |
| LaTeX source (MLSys 2025 style) | The paper | Once Milestone 3's notes exist; `specs/001-ci-cd` (User Story 5) then builds the PDF in CI |

Bulk output (event logs, CSVs, every figure a run draws) stays in
`results/`, which git ignores. Only the figures the paper uses are copied
here.

## Notes

| Milestone | Note | Covers | Spec | State |
|---|---|---|---|---|
| 1 | [`simulator.md`](milestone1/simulator.md) | The simulator: design, changes to vLLM, tests | [`002`](../specs/002-m1-simulator-verification/spec.md) | Written |
| 2 | [`task1-client-simulator.md`](milestone2/task1-client-simulator.md) | Task 1: the client simulator and the ShareGPT data | [`003`](../specs/003-m2-client-simulator/spec.md) | Written |
| 2 | [`task2-prefix-sharing-analysis.md`](milestone2/task2-prefix-sharing-analysis.md) | Task 2: ShareGPT single- vs multi-turn sharing | [`004`](../specs/004-m2-prefix-sharing-analysis/spec.md) | Written, with the 1,000-conversation results |
| 2 | [`task3-cache-tuning.md`](milestone2/task3-cache-tuning.md) | Task 3: cache size, block size, eviction policy | [`005`](../specs/005-m2-cache-tuning/spec.md) | Placeholder |
| 3 | [`task1-workload-eviction.md`](milestone3/task1-workload-eviction.md) | Task 1: the best policy per workload | [`006`](../specs/006-m3-workload-eviction/spec.md) | Placeholder |
| 3 | [`task2-mixed-workload-policy.md`](milestone3/task2-mixed-workload-policy.md) | Task 2: a policy for the mixed workload | [`007`](../specs/007-m3-mixed-workload-policy/spec.md) | Placeholder |

The changelogs: [Milestone 1](milestone1/changelog.md),
[Milestone 2](milestone2/changelog.md), [Milestone 3](milestone3/changelog.md).

Each note follows the shape of a paper section, so that condensing it is
mostly cutting:

1. **Question.** What the task asks, in one or two sentences, with a link to
   the spec. Definitions stay in the spec; the note links to them.
2. **Setup.** Sample, settings and cache configuration, the exact command,
   and the commit it ran at (the run's `run.json` records it).
3. **Results.** The figures from `figures/task<K>/` and the key numbers in
   a table.
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
| 2. Simulator | Replacing model execution while keeping vLLM's scheduler and cache | `milestone1/simulator.md` | 0.75 |
| 3. Workload replay and metrics | Client simulator, cache event log, metric definitions | Milestone 2, tasks 1 and 2 | 0.5 |
| 4. Homogeneous chatbot workload | ShareGPT single- vs multi-turn; cache parameters and policies | Milestone 2, tasks 2 and 3 | 1.75 |
| 5. Heterogeneous and mixed workloads | Best policy per workload; the workload-aware policy against the baselines | Milestone 3, tasks 1 and 2 | 1.75 |
| 6. AI usage | Required by the course | `ai-usage.md` | 0.25 |
| 7. Conclusion | | | 0.5 |
| Appendix | The most helpful prompts (required) | `ai-usage.md` | not counted |

## AI usage

The course requires a section on how we used AI and, in an appendix, the
prompts that produced a significant part of the code. `CONTRIBUTING.md`
(Section 10) asks every PR to record this. PR descriptions live on GitHub,
not in the git history we submit, so each PR also adds an entry to
[`ai-usage.md`](ai-usage.md), which the paper draws from.

## Rules

These come from `CONTRIBUTING.md` (Sections 7 and 8):

- Every figure here is produced by a committed script, and its note gives
  the command and the commit that produced it.
- A PR that changes a reported number or figure updates the note in the
  same PR and says so under "Effect on results".
- Numbers in the paper come from the notes, and notes link to the specs for
  definitions instead of repeating them.
