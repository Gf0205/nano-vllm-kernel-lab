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
