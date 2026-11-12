from dataclasses import dataclass, field
from typing import Optional


def request_id(conversation_id, turn_index):
    return f"{conversation_id}:{turn_index}"


def workload_of(conversation_or_request_id):
    """The workload tag: the id's text before its first colon (006 FR-002)."""
    return conversation_or_request_id.split(":", 1)[0]


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Conversation:
    conversation_id: str
    messages: list[Message]
    # Turns before this one are history only: they were already replayed
    # as part of another conversation (a re-run branch of a CC-Bench log).
    first_turn: int = 0
    # Recorded pause before each turn, in seconds: from the end of the
    # previous answer to this turn's message, for sources with timestamps
    # (CC-Bench's tool runs). Index = turn; turn 0 and unknown gaps are None.
    # None when the source has no timestamps (ShareGPT, AgentBank).
    turn_gaps_s: Optional[list] = None


@dataclass
class ReplayRequest:
    request_id: str
    conversation_id: str
    turn_index: int
    messages: list[Message]
    response: str


@dataclass
class ScheduledConversation:
    conversation: Conversation
    arrival_s: float


@dataclass
class TokenRequest:
    """One request of a token-level replay plan (specs 006 and 007).

    A session's first request has `arrival_s`, its send time relative to the
    plan's start. A later request has either `arrival_s` (open loop: recorded
    timestamps) or `after_s` (closed loop: sent that long after the session's
    previous request completes).
    """
    request_id: str
    session_id: str
    turn_index: int
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    arrival_s: Optional[float] = None
    after_s: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def workload(self) -> str:
        return workload_of(self.request_id)
