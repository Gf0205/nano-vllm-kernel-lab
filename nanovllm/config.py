import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig


def resolve_torch_dtype(hf_config: AutoConfig) -> torch.dtype:
    # Transformers versions differ: some expose `dtype`, older ones use `torch_dtype`.
    dtype = getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        if dtype == "auto":
            return torch.float16
        name = dtype.removeprefix("torch.")
        resolved = getattr(torch, name, None)
        if isinstance(resolved, torch.dtype):
            return resolved
    return torch.float16


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    decode_aware_prefill_interleave: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config.dtype = resolve_torch_dtype(self.hf_config)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
