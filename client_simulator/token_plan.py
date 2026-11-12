"""Token-level replay plans (specs 006 and 007).

A plan is a JSONL file of `TokenRequest`s, optionally gzipped, plus the
simulator trace that answers them (``{"request_id", "response_token_ids"}``
per request, for ``VLLM_SIMULATOR_TRACE``). Text conversations become plans
with `tokenize_conversation`; the Bailian adapter produces plans directly.
"""
import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from client_simulator.schema import Conversation, TokenRequest, request_id


def _open(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def write_plan(requests: Iterable[TokenRequest], plan_path: Path,
               responses_path: Path) -> int:
    """Write the plan and its simulator trace; return the request count."""
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen = set()
    with _open(plan_path, "w") as plan, _open(responses_path, "w") as answers:
        for request in requests:
            if request.request_id in seen:
                raise ValueError(f"Duplicate request id {request.request_id}")
            seen.add(request.request_id)
            plan.write(json.dumps(asdict(request)) + "\n")
            answers.write(
                json.dumps({
                    "request_id": request.request_id,
                    "response_token_ids": request.output_token_ids,
                }) + "\n")
            count += 1
    return count


def read_plan(path: Path) -> Iterator[TokenRequest]:
    with _open(path, "r") as source:
        for line in source:
            if line.strip():
                yield TokenRequest(**json.loads(line))


def tokenize_conversation(
    conversation: Conversation,
    tokenizer,
    max_model_len: int,
    after_s: Union[None, float, Sequence[Optional[float]]] = None
) -> Tuple[List[TokenRequest], int]:
    """One request per assistant turn, as the chat client would send it.

    `after_s` is the closed-loop pause before every later turn: one number
    for all of them, or one entry per turn (index = turn; entry 0 unused).

    The prompt is the chat template over the history with the generation
    prompt added (what the server's ``/tokenize`` returns for the chat
    client); the output is the recorded answer tokenized on its own (what the
    simulator replays). A turn whose prompt and output (plus the simulator's
    final EOS) exceed `max_model_len` is skipped with every later turn, as
    the client does. Returns the requests and the number of skipped turns.
    """
    messages = [{
        "role": m.role,
        "content": m.content
    } for m in conversation.messages]
    requests: List[TokenRequest] = []
    turns = len(messages) // 2
    for turn in range(turns):
        answer = messages[2 * turn + 1]
        prompt = tokenizer.apply_chat_template(messages[:2 * turn + 1],
                                               add_generation_prompt=True,
                                               tokenize=True)
        output = tokenizer.encode(answer["content"], add_special_tokens=False)
        if len(prompt) + len(output) + 1 > max_model_len:
            return requests, turns - turn
        requests.append(
            TokenRequest(request_id=request_id(conversation.conversation_id,
                                               turn),
                         session_id=conversation.conversation_id,
                         turn_index=turn,
                         prompt_token_ids=list(prompt),
                         output_token_ids=list(output),
                         after_s=None if turn == 0 else _pause(after_s, turn)))
    return requests, 0


def _pause(after_s: Union[None, float, Sequence[Optional[float]]],
           turn: int) -> Optional[float]:
    if after_s is None or isinstance(after_s, (int, float)):
        return after_s
    return after_s[turn]
