# AI usage

Source material for the paper's AI-usage section and its prompt appendix,
which the course requires.

Every PR records its AI usage (`CONTRIBUTING.md`, Section 10). PR
descriptions live on GitHub, not in the git history we submit, so each PR
also adds its entry here; its description may then point to the entry
instead of repeating it. A PR merged without an entry gets one added later
(for example PR #6, see `specs/003-m2-client-simulator/tasks.md`, T012).

Entry format:

```markdown
### PR #<number>: <title>

- **Tools:** which AI tools were used.
- **Used for:** which parts of the change they wrote or helped with, and
  what the author checked or rewrote.
- **Prompts:** the prompts that produced a significant part of the change,
  quoted, or "none".
```

## Entries

### Pending PR: Bailian adapter and shared experiment runner

- **Tools:** OpenAI Codex.
- **Used for:** adapting milestone 2 runners into a common runner;
  reusing the shared analyzer; document changes.
- **Prompts:** "adapt milestone 2 runners in the analysis folder into
  a common runner capable of repeating the experiment on any workload
  for milestone 3 such as qwen balian or any future workload such as
  mixed workload. Make sure this runner reuse any analyzer that
  milestone 2 already implemented such as LFU, MRU, etc. For this runner,
  have one of its argument be a config file that will specify invariants
  such as cache capacity, block size, etc (that will be kept the same
  across all replays), eviction policy analysis workloads (such as sharegpt,
  qwen balian, or multiple) and put an template of these files in example
  folder. After that, document my adapter implementation and the shared runner."

### PR #17: Milestone 3 Task 1: AgentBank and CC-Bench

- **Tools:** Claude Code (Claude Opus 5.5).
- **Used for:** the AgentBank and CC-Bench adapters, fixtures and tests,
  the experiment configs, the full runs and their checks, the client's
  keep-alive retry (spec 003, T015) after diagnosing the failed requests,
  the AgentBank and CC-Bench sections of the Task 1 note and the document
  updates. Jasmine chose the shared 32,768-token context and 65,536-token
  cache, ShareGPT's pacing, a seeded 1,000-trajectory AgentBank sample, no
  stand-in for Claude Code's system prompt, and reporting CC-Bench at the
  shared rate rather than adding a lower-rate run.
- **Prompts:** "ok now we are moving on to milestone 3 task 1. [...] im
  responsible for AgentBank, CC-Bench trajectories while triet is
  responsible for Qwen Bailian usage traces. he already did his part
  [...] my [understanding] is that you just have to do the same but for
  AgentBank, CC-Bench trajectories. follow his conventions and ask me if
  anything's unclear"; "if we are comparing to other workloads, we should
  just keep the arrive rate the same right? so if cc-bench failed under
  that arrive rate, we should just report that?"; "yes do that" (the
  retry fix and the CC-Bench rerun).
- **Review fix:** Claude Code checked the reviewer's points against the
  CC-Bench logs and the event logs, fixed the adapter (session trees,
  resumed sessions, dropped events) with tests, reran CC-Bench, and
  rewrote the CC-Bench conclusion and the LFU explanation for AgentBank.
  It also ran a simulated review of the PR before the second review. The
  first fix treated resumed sessions' copies as re-runs and replayed them;
  the second review caught it. Prompts: "got some feedback on the PR, is
  this related to CC-bench crashing bc arrive rate?"; "sounds good, go with
  the plan"; "could you simulate my teammate reviewing my code changes in
  the PR"; "yea do fix it, and check for other bugs. how did this one get
  past last time?"

### PR #7: Add contributing guide, spec-kit specs and project CI

- **Tools:** Claude Code (Claude Opus 5.5).
- **Used for:** the review of PR #6, the specs, the workflows,
  `tools/pr_check/`, `tools/run_project_tests.sh`, the Makefile and the
  README update. Marcus chose the scope, the branch-name rule and the
  one-PR packaging.
- **Prompts:** "can you help me to create a speckit that related to this
  task: <the charter>"; "What I would like to do is if someone make pr, then
  the test will be run on it to detect is there fits the requirement for
  us"; "I mean we can do make file for everyone for testing?"

### PR #8: Run CI checks on PRs into the milestone branches

- **Tools:** Claude Code (Claude Opus 5.5).
- **Used for:** finding that PRs into `milestone-2` ran no checks, and the
  workflow filter, Makefile and document changes. Marcus asked for the
  `milestone-2` branch and gave the go-ahead for the fix.
- **Prompts:** none beyond the request for the branch.

### PR #9: Milestone 2 Task 2: Prefix-sharing analysis

- **Tools:** Claude Code (Claude Opus 5.5).
- **Used for:** reading vLLM's v0 allocator, the cache event log and its
  hooks, `analysis/`, the tests, the experiment runs and their analysis,
  the notes, the changelogs and the documentation rules. Marcus chose one
  PR for all of Task 2, the report layout (a folder per milestone,
  changelogs, notes named after their specs, placeholders), port 8000 in
  the docs, and keeping PR #8 separate.
- **Prompts:** "right now I am going to continue the task 2 in milestone2,
  please create a branch and plan for it to implement"; "can you test it
  first the task?"; "Since we are a group, so what method do you suggest
  that we should do for our project? a way to document all things";
  "please run the full run testing for me and record it".

### PR #10: Milestone 2 Task 3, Phase 1: FIFO and LFU evictors

- **Tools:** Claude Code (Claude Opus 5.5).
- **Used for:** reading spec 005 and Task 2's allocator hooks, the
  evictors, the `VLLM_EVICTION_POLICY` wiring, both test modules, the
  check that an unset variable replays identically to `milestone-2`, and
  the document updates. Jasmine chose the full 1,000-conversation sample
  for the sweep and to keep spec 005's capacity range as written.
- **Prompts:** "now milestone 2 task 2 is finished. start working on task 3. read the specs/ and report/ folders first. ask me if anything's
  unclear, don't just assume."

### PR #11: Milestone 2 Task 3, Phase 2: cache sweep runner

- **Tools:** Claude Code (Claude Opus 5.5).
- **Used for:** the sweep runner (`analysis/sweep.py`,
  `analysis/run_m2_task3.py`), the eviction metrics, the small extensions of
  Task 2's `run_setting` and `analyze_run`, the tests, the smoke run and
  the memory check, and the document updates. Jasmine asked for one PR per
  phase.
- **Prompts:** "now work on the next phase".

### PR #12: Milestone 2 Task 3, Phase 3: cache-tuning results

- **Tools:** Claude Code (Claude Opus 5.5).
- **Used for:** running the 45-run grid and checking every run, the
  figures and the best-configuration table, the analyses behind the note's
  interpretation (what LFU evicts and when, LRU against FIFO by turn), the
  Task 3 note and the document updates. Jasmine chose block sizes 8, 16
  and 32.
- **Prompts:** "next work on phase 3"; "what's the status now".
- **Review fix:** Claude Code checked the reviewer's report that vLLM's
  `cached_tokens` is wrong under FIFO against vLLM's source and the 45
  event logs, added the usable reused fraction, re-analyzed the runs and
  rewrote the affected findings. Prompt: "ok so after applying the fix,
  our result and his should match? if so, yea make the changes and draft
  a reply response".
- **Second review:** Claude Code ran the nine 131,072-token runs, updated
  the note, qualified spec 004's metric definition and fixed the sweep
  tooling issues the reviewer listed, in this PR as Jasmine chose.
  Prompts: "think i got some comments on pr 12, can you address the";
  "keep them in PR 12".

### PR #13: Skip PDF and PNG figures in codespell

- **Tools:** Claude Code (Claude Fable 5.1).
- **Used for:** reproducing the `codespell` failure that PR #9's figures
  cause on `milestone-2`, the one-line skip in `pyproject.toml`, the PR
  description and this entry. Sahil asked for the fix and reviewed it.
- **Prompts:** "Can you fix the CI for codespell/codecheck and make a
  commit, and then tell me if we can merge it?"

### PR pending (branch `milestone3_task2`): Milestone 3 Task 2, Phase 1

- **Tools:** Claude Code (Claude Opus 5.5), with parallel subagents for the
  runner, the tables, the Belady oracle, identification and signals, the
  paper checks, and an independent code review.
- **Used for:** designing the study (the design was stress-tested by a
  review agent before any run), porting the earlier local work, the
  pacing and window changes, the grid runner, every analysis module and
  its tests, the 21 engine runs and the simulator sweep, the results page
  and figures, and these documents. Sahil chose the workloads, the model,
  the branch and the plan's direction.
- **Prompts:** "I need your help with Milestone 3 Task 2 [...] We need a
  systematic study using the Qwen2.5-0.5B-Instruct Model that
  specifically plots these 3 things for LRU, LFU, and FIFO [...] create
  this mixed workload generator, and then run the baselines, see how these
  baselines and metrics that are asked look like in plots, and then based
  on that we can get some signals to focus and design the smart eviction
  policy [...] read the following paper [...] write neat code on a new
  branch called milestone3_task2 [...] commit and push code to this new
  branch. Create a good, interpretable, coherent artifact".


### PR #16: Milestone 3 Task 2, Phase 2: spec revision and first wa experiments

- **Tools:** Claude Code (Claude Opus 5.5), with one multi-agent literature
  search that informed the spec revision.
- **Used for:** the spec 007 revision, the leaf-LRU controls, the label
  probe, the `wa` prototype and their tests, the table script, rebuilding
  the data and Phase 1's runs on a second laptop, the cache-replay
  experiments and their record in `plan.md`, and this PR's documents.
  Marcus chose to revise the spec on a branch off `milestone3_task2`, to
  run experiments before planning, and to test `wa` before its review.
- **Prompts:** "can you help me to write the spec of milestone 3 task 2?";
  "Can you do some experiment first? and then we need to plan how to do
  it."; "Can you test the wa first? and get some result"; "write a pr?".
- **Review fix:** Claude Code checked the review's findings against the
  code, wrote regression tests that fail on the first version, fixed the
  evictor, the tests and `sim_table`, reran the seed-1 tuning and the
  held-out replays, compared four bin rules on seed 1, and rewrote the
  results in `plan.md` and the changelog. Marcus passed on the review.
  Prompt: "This is his feedback" (with the review pasted).
- **Engine runs and scope:** Claude Code found that `wa` had run only in
  the replay, ran it on the engine, added the runner's `wa` grid, removed
  the probe and the second control, added S3-FIFO (and removed it again
  at Marcus's request) and narrowed spec 007 to the basic study Marcus
  chose. Prompts (Marcus's, translated from
  Chinese): "Can you check whether our repo has really run it?"; "OK, can
  you just run it? Has Sahil run it?"; "Can you implement it? I think it
  should be fine."; "We can do S3-FIFO, but delete the other unnecessary
  ones"; "I think don't do too much, just the basics"; "remove the
  S3-FIFO".

### PR #18: Milestone 3: merge Task 2 into milestone-3

- **Tools:** Claude Code (Claude Opus 5.5).
- **Used for:** the trial merge, resolving the ten conflicts, keeping
  Task 2's AgentBank and CC-Bench processing as the `-mix` datasets,
  finding and fixing the positional `Conversation` field clash, the
  reproducibility checks (input hashes and a rebuilt plan), the documents
  and this entry. Marcus chose to keep each task's processing (option A)
  over moving Task 2 to Task 1's adapters.
- **Prompts:** "should we merge the milestone3-task2 into milestone3?";
  "Should we follow Jasmine? I mean task 2 is after task 1"; "Okay, then
  just use option A"; "just implement all".

### PR pending (branch `docs-sahil-milestone-run-guide`): README on running each milestone

- **Tools:** Claude Code (Claude Opus 5.5), with independent agents: one
  checked every command, flag, path and number against the code, and a
  "cold reader" followed the README from a fresh clone.
- **Used for:**
  - reading the course handout to learn what the README is graded on;
  - rewriting the README around running each milestone and task;
  - two smoke configurations, `examples/agentbank_smoke.json` and
    `examples/ccbench_smoke.json`;
  - running the README's commands from a fresh clone with real downloads,
    with each full grid only started;
  - this entry and the changelog line.

  Sahil chose to base the branch on PR #18's branch, and to verify with the
  quick runs plus every analysis step rather than every full grid.
- **Prompts:** "we need to modify the main README file by telling the user
  how to run the results for Milestone1, Milestone2, Milestone3 clearly.
  Can you create a PR by branching off the latest working branch? Make
  sure you run the code yourself to see every command you write works, and
  spawn multiple agents to track behavioral changes. [...] read this to get
  what we are going to be graded on, and write the README".

