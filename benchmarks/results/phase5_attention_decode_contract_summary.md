# Phase 5 Attention Decode Contract Summary

Phase 5 started with a contract validation before any custom kernel work. The
goal was to prove that the current FlashAttention decode path is understood
well enough to compare or replace it later.

## 1. Validation Result

The AutoDL RTX 3090 validation passed:

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

## 2. Supported Conclusion

The tested decode contract is correct:

- KV cache uses `[num_blocks, block_size, num_kv_heads, head_dim]` per layer;
- block tables use `[batch, blocks_per_sequence]` with stride `(3, 1)` in this
  test;
- decode query reaches FlashAttention as `[batch, 1, num_heads, head_dim]`;
- GQA repeats each KV head across two query heads for Qwen3-0.6B;
- `flash_attn_with_kvcache` matches the PyTorch reference within BF16-tolerable
  error.

This passes the Phase 5 correctness gate for the tested shape.

## 3. What This Does Not Prove

This result does not prove that a custom Triton decode-attention kernel will be
faster than FlashAttention.

It only proves that the project can now build a standalone comparison against
the correct layout and semantics. FlashAttention remains a strong baseline.

## 4. Recommended Next Step

Build a standalone decode-attention microbenchmark before engine integration.
The benchmark should first compare the current FlashAttention path across
representative batch sizes and context lengths. Then, only if there is a clear
hypothesis, add a minimal candidate path and compare correctness plus timing.

If no measurable attention opportunity appears, Phase 5 should switch to the
GEMM/MLP candidate described in `phase5_candidate_optimization_plan.md`.

## 5. Standalone Baseline Command

The next AutoDL RTX 3090 run should use:

```bash
python benchmarks/bench_attention_decode_microbench.py \
  --model /root/huggingface/Qwen3-0.6B \
  --cases 1x128,4x513,8x512,16x512,32x512,32x1024,64x512 \
  --warmup-iters 20 \
  --timing-iters 100 \
  --no-write
```

Paste back each printed row. The important fields are:

- `batch_size`;
- `context_len`;
- `gqa_group_size`;
- `q_shape_for_flash`;
- `k_cache_shape`;
- `block_tables_shape`;
- `passed`;
- `max_abs_err`;
- `mean_abs_err`;
- `latency_ms_avg`;
- `latency_ms_p50`;
- `latency_ms_p95`;
- `latency_ms_min`;
- `latency_ms_max`.
- `latency_us_per_token_avg`;
- `tokens_per_s`.

This run is still evidence gathering. It should produce a FlashAttention decode
baseline and a go/no-go discussion, not an immediate Triton kernel integration.
