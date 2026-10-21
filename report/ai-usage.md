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
