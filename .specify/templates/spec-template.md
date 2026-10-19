# Feature Specification: [FEATURE NAME]

**Spec**: `[NNN-short-name]` | **Milestone/Task**: [e.g. Milestone 2, Task 2]
**Owner**: [name] | **Reviewers**: [names]
**Status**: Draft | In progress | In review (PR #N) | Done
**Created**: [YYYY-MM-DD] | **Depends on**: [spec ids or "none"]

**Input**: [the charter or course-spec text this feature comes from, quoted]

## User Scenarios & Testing *(mandatory)*

<!--
  User stories are ordered by priority. Each one is independently testable:
  implementing only that story still gives a demonstrable result.
  "User" here is usually a team member running an experiment or a reviewer.
-->

### User Story 1 - [Brief title] (Priority: P1)

[What the user needs to do, in plain language]

**Why this priority**: [the value, and what depends on it]

**Independent Test**: [how to show this story works on its own]

**Acceptance Scenarios**:

1. **Given** [initial state], **When** [action], **Then** [observable result]
2. **Given** [initial state], **When** [action], **Then** [observable result]

---

### User Story 2 - [Brief title] (Priority: P2)

[...]

### Edge Cases

- What happens when [boundary condition]?
- How does the system handle [error scenario]?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST [specific, testable capability]
- **FR-002**: The system MUST [...]
- **FR-003**: [NEEDS CLARIFICATION: what is unclear and the options]

### Metric Definitions *(if the feature produces or consumes a metric)*

| Metric | Unit | Definition | Edge cases |
|---|---|---|---|
| [name] | [unit] | [formula, numerator and denominator] | [how zero, missing and partial cases count] |

### Key Entities *(if the feature involves data)*

- **[Entity]**: [what it represents, key attributes, no implementation]

## Success Criteria *(mandatory)*

- **SC-001**: [measurable outcome, e.g. "a 1,000-conversation replay
  finishes on a laptop in under 10 minutes"]
- **SC-002**: [...]

## Assumptions

- [defaults chosen where the source text is silent]

## Out of Scope

- [what this feature deliberately does not do]
