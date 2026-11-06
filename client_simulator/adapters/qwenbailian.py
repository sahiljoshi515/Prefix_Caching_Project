import json
import math
import random
import shutil
import urllib.request
from itertools import islice
from pathlib import Path

TRACE_FILES = ("qwen_traceA_blksz_16.jsonl", "qwen_traceB_blksz_16.jsonl",
               "qwen_thinking_blksz_16.jsonl", "qwen_coder_blksz_16.jsonl")


def check_download(raw_path):
    if raw_path.is_file():
        return
    name = raw_path.name if raw_path.name in TRACE_FILES else TRACE_FILES[0]
    url = ("https://media.githubusercontent.com/media/alibaba-edu/"
           f"qwen-bailian-usagetraces-anon/main/{name}")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(".download")
    try:
        with urllib.request.urlopen(
                url, timeout=60) as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        temporary.replace(raw_path)
    finally:
        temporary.unlink(missing_ok=True)


def load_token_requests(raw_path,
                        tokenizer,
                        seed=0,
                        limit=None,
                        block_size=16):
    if block_size != 16:
        raise ValueError("Bailian files require block size 16")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token")
    special = set(tokenizer.all_special_ids)
    vocab = [
        i for i in sorted(set(tokenizer.get_vocab().values()))
        if i not in special and tokenizer.decode([i]).strip()
    ]
    if not vocab:
        raise ValueError("Tokenizer has no ordinary tokens")
    namespace = Path(raw_path).stem

    def tokens(key, count):
        return random.Random(f"{seed}:{namespace}:{key}").choices(vocab,
                                                                  k=count)

    # -1 is the marker for a request with no parent
    roots = {-1: -1}
    turns = {-1: 0}
    previous_time = -1.0
    with raw_path.open() as source:
        for line_number, line in enumerate(islice(source, limit), 1):
            row = json.loads(line)
            cid, parent = row["chat_id"], row["parent_chat_id"]
            length, output = row["input_length"], row["output_length"]
            timestamp, hashes = row["timestamp"], row["hash_ids"]
            turn = row["turn"]
            integers = [cid, parent, length, output, turn, *hashes]
            if (any(type(i) is not int for i in integers) or cid in roots
                    or length <= 0 or output < 0 or turn < 1 or len(hashes) !=
                (length + 15) // 16 or timestamp < 0
                    or not math.isfinite(timestamp)
                    or timestamp < previous_time):
                raise ValueError(f"{raw_path}:{line_number}: invalid record")
            if parent not in roots:
                raise ValueError(f"{raw_path}:{line_number}: missing parent")
            if turn != turns[parent] + 1:
                raise ValueError(
                    f"{raw_path}:{line_number}: inconsistent turn")
            root = cid
            if parent != -1:
                root = roots[parent]
            roots[cid] = root
            turns[cid] = turn
            previous_time = timestamp
            full, tail = divmod(length, 16)
            prompt = []
            for h in hashes[:full]:
                prompt.extend(tokens(f"block:{h}", 16))
            prompt.extend(tokens(f"tail:{cid}", tail))
            yield {
                "request_id": f"bailian:{namespace}:{cid}:{turn - 1}",
                "conversation_id": f"bailian:{namespace}:{root}",
                "turn_index": turn - 1,
                "timestamp_s": timestamp,
                "prompt_token_ids": prompt,
                "response_token_ids": tokens(f"output:{cid}", output),
                "eos_token_id": tokenizer.eos_token_id,
                "block_size": 16,
                "request_type": row.get("type", "text")
            }
