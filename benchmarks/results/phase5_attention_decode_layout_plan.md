# Phase 5 Attention Decode Layout / Correctness Study

Phase 5 starts with an attention decode contract study, not a custom kernel.

## 1. Goal

Confirm the exact data contract of the current FlashAttention decode path:

- KV cache layout;
- block table layout;
- context length semantics;
- GQA head mapping;
- q/k/v tensor shapes passed to FlashAttention.

## 2. Current Contract

From `ModelRunner.allocate_kv_cache` and `Attention.forward`:

```text
global kv_cache:
  [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]

per-layer k_cache/v_cache:
  [num_blocks, block_size, num_kv_heads, head_dim]

block_tables:
  [batch, blocks_per_sequence]

context_lens:
  [batch]

decode q:
  [batch, num_heads, head_dim]

FlashAttention q input:
  q.unsqueeze(1) -> [batch, 1, num_heads, head_dim]
```

The physical token offset is:

```text
physical_slot = block_table[token_index // block_size] * block_size
              + token_index % block_size
```

GQA mapping repeats each KV head across:

```text
num_heads / num_kv_heads
```

query heads.

## 3. Validation Script

Run:

```bash
python benchmarks/bench_attention_decode_contract.py \
  --model /root/huggingface/Qwen3-0.6B \
  --batch-size 4 \
  --context-len 513 \
  --block-size 256 \
  --num-blocks 16 \
  --no-write
```

The script compares `flash_attn_with_kvcache` against a direct PyTorch
reference implementation using the same block table and GQA mapping.

## 4. Go / No-Go Criteria

Go only if:

- the contract check passes against the PyTorch reference;
- the layout and strides are fully documented;
- there is a concrete hypothesis for why a custom path can beat
  FlashAttention.

No-go if:

- correctness is fragile;
- the only idea is "rewrite FlashAttention in Triton";
- no measurable overhead outside FlashAttention itself is identified.

## 5. Current Bias

Attention decode remains the most relevant candidate because it is the largest
single hotspot and aligns with the scheduler/KV-cache/decode story. But
FlashAttention is a strong baseline, so the first deliverable is correctness
and layout evidence, not speedup.

## 6. AutoDL RTX 3090 Contract Result

The first Phase 5 contract validation passed on the AutoDL RTX 3090
environment with Qwen3-0.6B:

```text
model: Qwen3-0.6B
batch_size: 4
context_len: 513
block_size: 256
blocks_per_seq: 3
num_heads: 16
num_kv_heads: 8
gqa_group_size: 2
head_dim: 128
dtype: torch.bfloat16
k_cache_shape: (16, 256, 8, 128)
k_cache_stride: (262144, 1024, 128, 1)
block_tables_shape: (4, 3)
block_tables_stride: (3, 1)
q_shape_for_flash: (4, 1, 16, 128)
max_abs_err: 0.000977
mean_abs_err: 8.1e-05
passed: True
```

Interpretation:

- the synthetic paged KV-cache layout matches `flash_attn_with_kvcache`;
- block table indexing and stride semantics are understood for this shape;
- `context_lens` semantics match the PyTorch reference path;
- GQA mapping is confirmed as two query heads per KV head;
- BF16 numerical error is small and acceptable for this contract check.

This closes the first correctness gate:

```text
Current FlashAttention decode contract is understood for the tested
Qwen3-0.6B / BF16 / paged-KV decode shape.
```

This does not yet justify replacing FlashAttention. The next decision still
requires a concrete optimization hypothesis and a standalone benchmark.

## 7. Recommended Next Step

Do not integrate a custom attention kernel yet.

The next Phase 5 artifact should be a standalone attention-decode microbenchmark
that compares the existing FlashAttention decode path against a minimal
candidate or diagnostic baseline under controlled decode shapes:

- same KV-cache layout: `[num_blocks, block_size, num_kv_heads, head_dim]`;
- same block table layout: `[batch, blocks_per_sequence]`;
- Qwen3-0.6B decode shapes first: `num_heads=16`, `num_kv_heads=8`,
  `head_dim=128`;
- representative batch sizes and context lengths;
- correctness check before timing;
- timing split for FlashAttention call overhead vs candidate path.

Go forward only if this benchmark exposes a measurable and explainable
opportunity. If the standalone evidence is weak, switch Phase 5 toward the
GEMM/MLP candidate instead of forcing an attention replacement.

The baseline script is now:

```bash
python benchmarks/bench_attention_decode_microbench.py \
  --model /root/huggingface/Qwen3-0.6B \
  --cases 1x128,4x513,8x512,16x512,32x512,32x1024,64x512 \
  --warmup-iters 20 \
  --timing-iters 100 \
  --no-write
```
