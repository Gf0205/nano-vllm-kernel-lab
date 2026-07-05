# Phase 5 Candidate Optimization Plan

Phase 5 should be profile-driven. Based on Phase 4, the two realistic
optimization directions are attention decode and GEMM.

## 1. Candidate A: Attention Decode

Why it is attractive:

- it is the largest single hotspot in graph-mode profiling;
- it connects directly to the project's scheduler, KV cache, prefix cache, and
  decode-interleaving work;
- improving decode attention would strengthen the project's core inference
  systems story.

Risks:

- FlashAttention is already highly optimized;
- paged KV layout and block-table indexing are easy to get wrong;
- custom Triton decode attention must match FlashAttention numerically;
- CUDA Graph compatibility must be preserved;
- one-off kernels may lose to the existing library.

Minimal viable route:

1. Write a standalone single-batch/single-head or small-shape attention decode
   checker.
2. Match FlashAttention output with the current KV cache layout.
3. Add detailed pointer/stride assertions before optimizing.
4. Benchmark standalone against FlashAttention.
5. Only if standalone is promising, test eager engine integration.
6. Only after eager correctness, test CUDA Graph capture.

Success criteria:

- correctness vs FlashAttention;
- measurable improvement or clearer explanation of why FlashAttention is hard
  to beat;
- no scheduler/KV-cache regression.

## 2. Candidate B: GEMM / MLP

Why it is attractive:

- BF16 GEMM is the second major hotspot;
- MLP GEMM is easier to isolate than attention decode;
- W4A16 or fused MLP experiments can produce clear benchmark tables.

Risks:

- cuBLAS/CUTLASS kernels are already strong;
- W4A16 changes model accuracy/weight-loading assumptions;
- tensor-parallel and packing details can expand scope quickly;
- optimizing GEMM may become a quantization project rather than serving
  systems project.

Minimal viable route:

1. Start with MLP `gate_up_proj` only.
2. Keep QKV and output projection unchanged.
3. Benchmark BF16 baseline vs candidate in standalone form.
4. Add accuracy/error checks before engine integration.

Success criteria:

- clear standalone speedup;
- controlled numerical error;
- no broad architecture rewrite.

## 3. Recommendation

Start with attention decode analysis, not implementation.

Reason:

```text
Attention decode is the largest single hotspot and is more aligned with the
project's KV-cache/scheduler/decode story.
```

However, do not immediately replace FlashAttention. The next step should be a
minimal correctness and layout study for decode attention. If that study shows
the existing FlashAttention path is already near-optimal or too risky, switch
to GEMM/MLP as the safer optimization candidate.

The concrete next artifact is `phase5_attention_decode_layout_plan.md` and
`bench_attention_decode_contract.py`.

## 4. Explicit Non-Goals

Do not start these yet:

- adaptive scheduler policy;
- speculative decoding;
- W4A16 full-model quantization;
- KV compression;
- Triton RoPE/RMSNorm replacement.
