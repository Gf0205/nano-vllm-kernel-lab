# RTX 3090 Phase 3 Prefix Cache Results

Environment matches the Phase 2 smoke setup:

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.5.1+cu124
- CUDA runtime: 12.4
- Triton: 3.1.0
- Transformers: 4.51.0
- FlashAttention: 2.7.4.post1
- Model: Qwen3-0.6B

## Setup

The prefix-cache benchmark warms one request with a shared 1024-token prefix, then compares:

- `shared_prefix_after_warmup`: 32 requests sharing the same 1024-token prefix plus unique 32-token suffixes.
- `unique_prefix_baseline`: 32 synthetic requests with unique 1056-token prompts.

The KV cache block size is 256 tokens, so the 1024-token shared prefix spans exactly 4 reusable full blocks.

## Key Results

| case | prefix_cache_hit_rate | prefix_cache_hit_blocks | physical_block_allocations | output_tokens_per_s | peak_memory_gb |
| --- | --- | --- | --- | --- | --- |
| shared_prefix_after_warmup | 1.0 | 128 | 32 | 2167.79 | 20.397 |
| unique_prefix_baseline | 0.0 | 0 | 160 | 1709.11 | 20.793 |

## Interpretation

- Shared-prefix requests hit all reusable full prefix blocks: `32 requests * 4 blocks = 128 hit blocks`.
- Physical block allocations drop from 160 to 32 because shared prefix blocks are reused after warmup.
- Output throughput improves from 1709.11 tok/s to 2167.79 tok/s in this benchmark.
- Peak memory remains around 20 GB because the engine pre-allocates KV cache according to `gpu_memory_utilization`; the useful signal here is block allocation/reuse, not incremental CUDA memory.

## Stop Condition

Phase 3 prefix-cache validation is complete:

- BlockManager metrics are exposed.
- Prefix-cache smoke and standard RTX 3090 benchmarks run successfully.
- Shared-prefix cache hit rate reaches 100% for full prefix blocks.
- Physical block allocation reduction is visible and explainable.

Next phase: scheduler/chunked-prefill metrics or the first safe Triton kernel integration, depending on project priority.
