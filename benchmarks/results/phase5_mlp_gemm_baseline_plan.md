# Phase 5 MLP / GEMM Baseline Plan

Attention decode replacement is temporarily no-go after the standalone
FlashAttention baseline. Phase 5 now pivots to the Qwen3 MLP/GEMM candidate,
but still stays in evidence-gathering mode.

## 1. Goal

Measure the BF16 standalone MLP baseline for Qwen3-0.6B before considering any
custom GEMM, W4A16, or fused MLP implementation.

The first question is:

```text
Which MLP subcomponent dominates for real Qwen3 shapes: gate_up projection,
SiluAndMul activation, or down projection?
```

## 2. Scope

This phase measures:

- `gate_up_proj`: one BF16 linear from hidden size to `2 * intermediate_size`;
- `SiluAndMul`: split, SiLU, and elementwise multiply;
- `down_proj`: one BF16 linear from intermediate size back to hidden size;
- full standalone MLP: `gate_up -> SiluAndMul -> down`.

It records:

- real model hidden/intermediate sizes from `Config`;
- projection weight and activation shapes;
- CUDA-event latency for each segment;
- percent of full standalone MLP time;
- tokens/s for gate_up, down, and full MLP.

## 3. Non-Goals

Do not implement these yet:

- W4A16 full-model quantization;
- fused MLP kernel;
- custom GEMM kernel;
- engine integration;
- accuracy or perplexity evaluation.

## 4. AutoDL Command

Run on the RTX 3090 environment:

```bash
python benchmarks/bench_mlp_gemm_microbench.py \
  --model /root/huggingface/Qwen3-0.6B \
  --token-cases 1,8,16,32,64,128,256,512,1024 \
  --warmup-iters 20 \
  --timing-iters 100 \
  --no-write
```

Paste back every printed row. Important fields:

- `num_tokens`;
- `hidden_size`;
- `intermediate_size`;
- `gate_up_weight_shape`;
- `down_weight_shape`;
- `gate_up_ms_avg`;
- `silu_mul_ms_avg`;
- `down_ms_avg`;
- `full_mlp_ms_avg`;
- `gate_up_pct_of_full`;
- `silu_mul_pct_of_full`;
- `down_pct_of_full`;
- `gate_up_tokens_per_s`;
- `down_tokens_per_s`;
- `full_mlp_tokens_per_s`.

## 5. Go / No-Go Criteria

Go deeper into MLP/GEMM only if the standalone baseline shows a clear and
explainable hotspot that aligns with Phase 4 profiler evidence.

No-go if:

- BF16 GEMM is already too strong to justify a custom path;
- activation or launch overhead dominates instead of GEMM;
- the only idea is to jump directly to W4A16 without a controlled baseline.

If go conditions are met, the next step should still be a narrow candidate
study, not immediate engine replacement.
