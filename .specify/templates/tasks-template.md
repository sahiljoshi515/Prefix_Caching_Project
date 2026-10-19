# Tasks: [FEATURE NAME]

**Input**: `spec.md` and `plan.md` in this folder.

**Format**: `- [ ] T### [P?] [US#] Description (path)`

- **[P]**: can run in parallel with other [P] tasks (different files, no
  dependency on an unfinished task)
- **[US#]**: the user story the task serves (omitted for setup and polish)
- Each implementation task is preceded by, or includes, its test.
- Tick a box in the same commit that finishes the task.

## Phase 1: Setup

- [ ] T001 [description] ([path])

## Phase 2: Foundation (blocks all user stories)

- [ ] T002 [description] ([path])

**Checkpoint**: [what must work before user stories start]

## Phase 3: User Story 1 (P1)

**Goal**: [...]
**Independent test**: [...]

- [ ] T003 [P] [US1] Test: [...] ([path])
- [ ] T004 [US1] [...] ([path])

**Checkpoint**: [...]

## Phase 4: User Story 2 (P2)

- [ ] T005 [US2] [...]

## Phase N: Polish

- [ ] T0NN Update README / report notes
- [ ] T0NN Run the full verification from plan.md and paste results into
  the PR description

## Dependencies

- [which phases and tasks block which]
