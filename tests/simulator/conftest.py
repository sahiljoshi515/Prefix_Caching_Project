"""Shared fixtures for the prefix-sharing simulator tests.

``VLLM_USE_SIMULATOR=1`` must already be exported when pytest starts: the
root ``tests/conftest.py`` imports vllm first, and the platform is chosen at
import time (``vllm/platforms/__init__.py``).
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Mapping, Union

import pytest

if os.environ.get("VLLM_USE_SIMULATOR") != "1":
    raise RuntimeError(
        "tests/simulator needs `export VLLM_USE_SIMULATOR=1` before pytest "
        "starts (the platform is selected when vllm is imported).")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# `vllm serve` (test_online.py) must resolve even if the venv is not active.
os.environ["PATH"] = (os.path.dirname(sys.executable) + os.pathsep +
                      os.environ.get("PATH", ""))

from vllm.platforms import current_platform  # noqa: E402

assert current_platform.is_cpu(), current_platform

# Config and tokenizer files only: the simulator never loads weights.
MODEL_FILES = [
    "config.json", "generation_config.json", "tokenizer*", "vocab.json",
    "merges.txt", "special_tokens_map.json"
]


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    """Skip the root conftest's distributed-env cleanup (no GPU here)."""
    return False


def write_trace(path: Union[str, Path],
                entries: Mapping[str, Union[str, List[int]]]) -> str:
    """Write ``{request_id: response text | token ids}`` as a JSONL trace."""
    with open(path, "w") as f:
        for request_id, response in entries.items():
            key = ("response_token_ids"
                   if isinstance(response, list) else "response")
            f.write(json.dumps({"request_id": request_id, key: response}))
            f.write("\n")
    return str(path)


@pytest.fixture(scope="session")
def gpt2_dir() -> str:
    """Local directory holding gpt2's config and tokenizer (about 2 MB)."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download("gpt2",
                                 allow_patterns=MODEL_FILES,
                                 local_files_only=True)
    except Exception:
        return snapshot_download("gpt2", allow_patterns=MODEL_FILES)
