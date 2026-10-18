from dataclasses import dataclass

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Conversation:
    conversation_id: str
    messages: list[Message]

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