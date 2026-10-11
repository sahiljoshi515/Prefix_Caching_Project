"""Prefix-sharing simulator: an executor that never runs the model.

COMP 536 project, Milestone 1.  With ``VLLM_USE_SIMULATOR=1`` vLLM keeps all
of its serving machinery (request handling, continuous batching, the
scheduler, the KV-cache block manager and prefix caching) but replaces the
model forward pass with this module: every scheduled sequence receives its
next token from a pre-collected trace instead of a GPU.

Trace format (JSONL, one object per line, selected with
``VLLM_SIMULATOR_TRACE``)::

    {"request_id": "conv0-turn0", "response": "The answer is 42."}
    {"request_id": "conv0-turn1", "response_token_ids": [464, 3280, 318]}

A request is matched to a trace entry by its request id (clients set it with
the ``X-Request-Id`` header or the chat ``request_id`` field).  Requests with
no trace entry receive a filler token until ``max_tokens`` is reached.
"""
import json
import re
from typing import Dict, List, Optional, Set, Tuple

import vllm.envs as envs
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (CompletionSequenceGroupOutput, ExecuteModelRequest,
                           Logprob, SequenceGroupMetadata, SequenceOutput)
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer
from vllm.transformers_utils.tokenizers import MistralTokenizer
from vllm.worker.cache_engine import CacheEngine

logger = init_logger(__name__)

# Decorations the engine adds to client request ids: the OpenAI front-end
# prefixes (serving_chat.py / serving_completion.py), the per-prompt index of
# the completions endpoint and the suffix ParallelSampleSequenceGroup appends
# for n > 1 (vllm/sequence.py).
_CHAT_PREFIX = "chatcmpl-"
_COMPLETION_PREFIX = "cmpl-"
_PROMPT_INDEX_RE = re.compile(r"-\d+$")
_PARALLEL_SAMPLE_RE = re.compile(r"_parallel_sample_\d+$")


def candidate_ids(request_id: str) -> List[str]:
    """Trace keys to try, most specific first, for an engine request id.

    ``cmpl-abc-0_parallel_sample_1`` yields ``[itself, "cmpl-abc-0",
    "abc-0", "abc"]`` so a trace may be keyed by the client id or by any
    decorated form.
    """
    candidates = [request_id]
    base = _PARALLEL_SAMPLE_RE.sub("", request_id)
    candidates.append(base)
    if base.startswith(_CHAT_PREFIX):
        candidates.append(base[len(_CHAT_PREFIX):])
    elif base.startswith(_COMPLETION_PREFIX):
        stripped = base[len(_COMPLETION_PREFIX):]
        candidates.append(stripped)
        candidates.append(_PROMPT_INDEX_RE.sub("", stripped))
    return list(dict.fromkeys(candidates))  # Unique, order preserved.


def _encode(tokenizer: AnyTokenizer, text: str) -> List[int]:
    """Tokenize text without adding BOS/EOS tokens."""
    if isinstance(tokenizer, MistralTokenizer):
        raise ValueError("The simulator does not support "
                         "--tokenizer-mode mistral.")
    return tokenizer.encode(text, add_special_tokens=False)


def load_trace(path: str, tokenizer: AnyTokenizer) -> Dict[str, List[int]]:
    """Load a JSONL trace into ``{request_id: response_token_ids}``.

    ``response`` text is tokenized with the served model's tokenizer (no
    special tokens); ``response_token_ids`` are used verbatim.  A trailing EOS
    token is dropped because the simulator emits its own.
    """
    traces: Dict[str, List[int]] = {}
    vocab_size = len(tokenizer)
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError as e:
        raise ValueError(
            f"Cannot read VLLM_SIMULATOR_TRACE={path}: {e}") from e
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path}:{line_no}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{where}: invalid JSON ({e})") from e
        if "request_id" not in record:
            raise ValueError(f"{where}: missing 'request_id'")
        request_id = str(record["request_id"])
        token_ids = record.get("response_token_ids")
        if token_ids is None:
            if "response" not in record:
                raise ValueError(f"{where}: needs 'response' or "
                                 "'response_token_ids'")
            token_ids = _encode(tokenizer, record["response"])
        elif not (isinstance(token_ids, list)
                  and all(isinstance(t, int) for t in token_ids)):
            raise ValueError(f"{where}: 'response_token_ids' must be a "
                             "list of integers")
        token_ids = list(token_ids)
        if token_ids and token_ids[-1] == tokenizer.eos_token_id:
            token_ids.pop()
        if any(t < 0 or t >= vocab_size for t in token_ids):
            raise ValueError(f"{where}: token id outside the vocabulary "
                             f"(size {vocab_size})")
        if request_id in traces:
            logger.warning(
                "%s: duplicate request_id %r, keeping the "
                "last one", where, request_id)
        traces[request_id] = token_ids
    return traces


def _certain(token_id: int) -> Dict[int, Logprob]:
    # The engine requires the emitted token to be a key of this dict.  0.0
    # (a "certain" token) stays JSON-serialisable if a client asks for
    # logprobs, unlike the sampler's inf placeholder.
    return {token_id: Logprob(logprob=0.0, rank=1)}


class Simulator:
    """Decides which token each scheduled sequence emits next.

    Pure Python with no engine state, so it can be unit-tested on its own.
    The executor below calls :meth:`make_sampler_output` once per step.
    """

    def __init__(self,
                 tokenizer: AnyTokenizer,
                 trace_path: Optional[str] = None,
                 block_size: int = 16) -> None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The simulator needs a tokenizer with an EOS "
                             "token to know when a response ends.")
        self.eos_token_id: int = tokenizer.eos_token_id
        # Ordinary word token emitted for requests without a trace entry.
        self.filler_token_id: int = _encode(tokenizer, "the")[-1]
        self.block_size = block_size
        self.traces: Dict[str, List[int]] = {}
        if trace_path:
            self.traces = load_trace(trace_path, tokenizer)
        else:
            logger.warning("VLLM_SIMULATOR_TRACE is not set: every request "
                           "gets filler tokens until max_tokens.")

    def lookup(self, request_id: str) -> Optional[List[int]]:
        """Trace response for an engine request id, or None."""
        for candidate in candidate_ids(request_id):
            if candidate in self.traces:
                return self.traces[candidate]
        return None

    def next_token(self,
                   request_id: str,
                   num_generated: int,
                   min_tokens: int = 0) -> int:
        """Token number ``num_generated`` of the response for a request."""
        trace = self.lookup(request_id)
        if trace is None:
            return self.filler_token_id
        if num_generated < len(trace):
            return trace[num_generated]
        if num_generated < min_tokens:
            # A real model has EOS masked out below min_tokens; keep emitting
            # visible tokens rather than EOS tokens the stop checker ignores.
            return self.filler_token_id
        return self.eos_token_id  # Response exhausted: finish the request.

    def make_sampler_output(
            self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> SamplerOutput:
        """Build what the real sampler would return for one step.

        One entry per scheduled sequence group, in order.  A group whose
        prefill is split across steps (chunked prefill) is not sampled until
        its last chunk arrives, exactly like the real model runner.
        """
        outputs: List[CompletionSequenceGroupOutput] = []
        for meta in seq_group_metadata_list:
            params = meta.sampling_params
            min_tokens = params.min_tokens if params is not None else 0
            samples: List[SequenceOutput] = []
            # Groups hold one sequence at this commit (n > 1 is split into
            # sub-requests); the loop keeps the sampler's general shape.
            for seq_id, seq_data in meta.seq_data.items():
                if meta.is_prompt and meta.computed_block_nums:
                    # Record the prefix-cache hit like the GPU runner does;
                    # it is reported as RequestOutput.num_cached_tokens.
                    seq_data.update_num_cached_tokens(
                        len(meta.computed_block_nums) * self.block_size)
                if not meta.do_sample:
                    continue
                # Tokens generated so far (kept when a preempted request is
                # recomputed) give the position in the trace.
                token_id = self.next_token(meta.request_id,
                                           seq_data.get_output_len(),
                                           min_tokens)
                samples.append(
                    SequenceOutput(parent_seq_id=seq_id,
                                   output_token=token_id,
                                   logprobs=_certain(token_id)))
            outputs.append(
                CompletionSequenceGroupOutput(samples=samples,
                                              prompt_logprobs=None))
        return SamplerOutput(outputs=outputs)


class SimulatorExecutor(ExecutorAsyncBase):
    """vLLM executor that replays traces instead of running a model.

    Chosen by ``LLMEngine._get_executor_cls`` and
    ``AsyncLLMEngine._get_executor_cls`` when ``VLLM_USE_SIMULATOR=1``.  It
    never loads model weights; only the model config and tokenizer are read.
    """

    uses_ray: bool = False

    def _init_executor(self) -> None:
        self._check_config()
        # Same tokenizer construction as the engine's TokenizerGroup, so the
        # EOS token id matches the one the engine's stop checker uses.
        tokenizer = get_tokenizer(
            self.model_config.tokenizer,
            tokenizer_mode=self.model_config.tokenizer_mode,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.tokenizer_revision)
        self.simulator = Simulator(tokenizer, envs.VLLM_SIMULATOR_TRACE,
                                   self.cache_config.block_size)
        logger.info(
            "Simulator enabled: %d trace entries from %s, eos_token_id=%d, "
            "filler_token_id=%d", len(self.simulator.traces),
            envs.VLLM_SIMULATOR_TRACE, self.simulator.eos_token_id,
            self.simulator.filler_token_id)

    def _check_config(self) -> None:
        """Fail early, with a clear message, on features not simulated."""
        if not current_platform.is_cpu():
            raise ValueError("VLLM_USE_SIMULATOR must be set before vllm is "
                             "imported so that the CPU platform is selected.")
        unsupported = {
            "runner types other than generate":
            self.model_config.runner_type != "generate",
            "tensor or pipeline parallelism":
            self.parallel_config.world_size > 1,
            "multi-step scheduling": self.scheduler_config.is_multi_step,
            "speculative decoding": self.speculative_config is not None,
            "LoRA adapters": self.lora_config is not None,
            "prompt adapters": self.prompt_adapter_config is not None,
        }
        for feature, used in unsupported.items():
            if used:
                raise ValueError(f"The simulator does not support {feature}.")

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        # Same accounting as the CPU worker: the KV-cache budget from
        # VLLM_CPU_KVCACHE_SPACE (GiB) is divided into blocks of
        # `block_size` tokens.  Swap space provides "CPU" blocks so that
        # --preemption-mode swap works too.  --num-gpu-blocks-override is
        # applied afterwards by LLMEngine._initialize_kv_caches.
        block_bytes = CacheEngine.get_cache_block_size(self.cache_config,
                                                       self.model_config,
                                                       self.parallel_config)
        if block_bytes <= 0:
            raise ValueError("The simulator needs a model with attention "
                             "layers (its KV-cache block size is 0 bytes).")
        cache_config = self.cache_config
        kv_cache_bytes = cache_config.cpu_kvcache_space_bytes  # type: ignore
        num_gpu_blocks = int(kv_cache_bytes // block_bytes)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes // block_bytes)
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        # Nothing to allocate (blocks are bookkeeping), but keep the sanity
        # check the real workers apply in raise_if_cache_size_invalid.
        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        if num_gpu_blocks <= 0 or self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The KV cache holds {max_seq_len} tokens but max_model_len "
                f"is {self.model_config.max_model_len}: raise "
                "VLLM_CPU_KVCACHE_SPACE / --num-gpu-blocks-override or "
                "lower --max-model-len.")
        logger.info(
            "Simulated KV cache: %d blocks of %d tokens (+ %d swap blocks)",
            num_gpu_blocks, self.cache_config.block_size, num_cpu_blocks)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        # Real model runners invoke this callback (asynchronous processing of
        # the previous step's outputs) while the forward pass runs.  The CPU
        # platform disables async output processing, so it is None here.
        if execute_model_req.async_callback is not None:
            execute_model_req.async_callback()
        # Delta metadata only exists with Ray SPMD workers, never here.
        metas = [
            m for m in execute_model_req.seq_group_metadata_list
            if isinstance(m, SequenceGroupMetadata)
        ]
        if len(metas) != len(execute_model_req.seq_group_metadata_list):
            raise ValueError("SequenceGroupMetadataDelta is not supported.")
        # blocks_to_swap_in/out/copy are ignored on purpose: there is no
        # physical cache to move, the block manager's bookkeeping is enough.
        return [self.simulator.make_sampler_output(metas)]

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        return self.execute_model(execute_model_req)

    def check_health(self) -> None:
        return  # No workers and no devices, so nothing can be unhealthy.

    def _run_workers(self, method: str, *args, **kwargs) -> List:
        return []  # Only reached by the profiling endpoints.

    # LoRA adapters and prompt adapters are not supported by the simulator.
    def add_lora(self, lora_request: LoRARequest) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> Set[int]:
        return set()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return False

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return False

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return False

    def list_prompt_adapters(self) -> Set[int]:
        return set()
