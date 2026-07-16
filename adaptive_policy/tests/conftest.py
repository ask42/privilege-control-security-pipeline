from __future__ import annotations

import pytest
from vllm import LLM

VLLM_MODEL = "Qwen/Qwen3-8B"


@pytest.fixture(scope="session")
def shared_llm():
    return LLM(
        model=VLLM_MODEL,
        dtype="half",
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        block_size=16,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )
