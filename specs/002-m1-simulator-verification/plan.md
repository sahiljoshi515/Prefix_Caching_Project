# Implementation Plan: Independent verification of the Milestone 1 simulator

**Spec**: [spec.md](spec.md) | **Branch**: none (verification only; fixes get
their own `fix/` branches) | **Date**: 2026-10-18

## Summary

Two teammates each reproduce Milestone 1 from a fresh clone and report in
one tracking issue. No code is written unless a deviation is found.

## Constitution Check

- [x] I. Verifies CPU-only, weight-free, online mode (`test_online.py`)
- [x] II. No code changes
- [x] III. Uses the existing deterministic suite
- [x] IV, V. n/a
- [x] VI. Fixes, if any, go through PRs
- [x] VII. n/a

## Procedure (each verifier)

```bash
git clone https://github.com/sahiljoshi515/Prefix_Caching_Project.git m1-verify
cd m1-verify
git rev-parse HEAD
uv --version && python3 --version && uname -a
# README "Setup" section, verbatim, for your OS
uv venv --python 3.12 .venv && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -c constraints-simulator.txt   # macOS
uv pip install pytest==8.3.3 pytest-asyncio==0.24.0
export VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1
python -c "from vllm.platforms import current_platform; print(current_platform)"
python -c "from vllm.engine.llm_engine import LLMEngine; print(LLMEngine._get_executor_cls(None))"
python -m pytest tests/simulator -v -o asyncio_default_fixture_loop_scope=function 2>&1 | tee m1-tests.log
# Terminal 1
export VLLM_SIMULATOR_TRACE=$PWD/examples/simulator/trace_example.jsonl
vllm serve gpt2 --enable-prefix-caching --max-model-len 1024 --port 8000
# Terminal 2
python examples/simulator/send_requests.py
```

## Report template (issue comment)

```
Verifier: <name>
Machine: <OS + version>, <arch>, Python <x.y.z>, uv <x.y.z>
Commit: <hash>
Install: OK / deviations: <...>
Tests: <N passed, M failed> (log attached)
Example replay: <output>
Deviations filed: #<issue>, ...
```

## Risks

| Risk | Mitigation |
|---|---|
| A verifier reuses an old `.venv` and hides an install problem | FR-001: fresh clone and fresh venv |
| Both verifiers use the same OS | Prefer one macOS and one Linux/WSL2 verification; CI adds Linux |
