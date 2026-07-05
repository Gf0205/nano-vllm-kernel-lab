# Phase 5 Attention Decode Microbenchmark Summary

This summary records the first standalone FlashAttention decode baseline after
the Phase 5 contract check passed.

## 1. Question

Does the current FlashAttention paged decode path expose a clear standalone
optimization window for a custom attention kernel?

This benchmark does not test a replacement kernel. It measures the existing
`flash_attn_with_kvcache` baseline with the same paged KV-cache, block table,
GQA, and BF16 contract used by the engine decode path.

## 2. AutoDL RTX 3090 Result

Command:

```bash
python benchmarks/bench_attention_decode_microbench.py \
  --model /root/huggingface/Qwen3-0.6B \
  --cases 1x128,4x513,8x512,16x512,32x512,32x1024,64x512 \
  --warmup-iters 20 \
  --timing-iters 100 \
  --no-write
```

All cases passed the PyTorch reference correctness check.

| batch | context | blocks/seq | max abs err | mean abs err | avg ms | p50 ms | p95 ms | us/token avg | tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1 | 0.001953 | 0.000155 | 0.0585 | 0.0573 | 0.0645 | 58.5 | 17094.02 |
| 4 | 513 | 3 | 0.000977 | 0.000079 | 0.0624 | 0.0614 | 0.0676 | 15.6 | 64102.56 |
| 8 | 512 | 2 | 0.000977 | 0.000082 | 0.0698 | 0.0676 | 0.0707 | 8.725 | 114613.18 |
| 16 | 512 | 2 | 0.001953 | 0.000084 | 0.0820 | 0.0819 | 0.0850 | 5.125 | 195121.95 |
| 32 | 512 | 2 | 0.001953 | 0.000083 | 0.1304 | 0.1300 | 0.1331 | 4.075 | 245398.77 |
| 32 | 1024 | 4 | 0.000977 | 0.000061 | 0.2265 | 0.2263 | 0.2304 | 7.0781 | 141280.35 |
| 64 | 512 | 2 | 0.001953 | 0.000083 | 0.2095 | 0.2079 | 0.2171 | 3.2734 | 305489.26 |

## 3. Interpretation

Correctness is stable:

- every tested shape reported `passed=True`;
- BF16 maximum absolute error stayed at or below `0.001953`;
- mean absolute error stayed around `6.1e-05` to `1.55e-04`.

The baseline is strong:

- small-batch launch/dispatch overhead is visible at batch 1, about `58.5 us`
  per decoded token;
- batching amortizes the cost quickly, reaching about `4.075 us/token` at
  batch 32, context 512;
- batch 64, context 512 reaches about `3.2734 us/token`;
- doubling context from 512 to 1024 at batch 32 increases average latency from
  `0.1304 ms` to `0.2265 ms`, which is expected for decode attention reading
  more KV history.

One case had a single high max value:

```text
batch=8, context=512, avg=0.0698 ms, p95=0.0707 ms, max=0.2427 ms
```

Because p95 stayed close to p50, this looks like an isolated timing outlier,
not a stable performance issue.

## 4. Go / No-Go

Attention decode replacement is no-go for immediate implementation.

Reason:

```text
The standalone FlashAttention decode baseline is correct and already very fast
for the tested Qwen3-0.6B shapes. The current evidence does not show a clear,
measurable optimization window that justifies writing and integrating a custom
Triton attention-decode kernel.
```

This does not mean attention decode is unimportant. It remains a major
end-to-end hotspot from Phase 4. But the standalone evidence says the existing
library call is a strong baseline, so replacing it now would likely be a
high-risk kernel project without enough supporting evidence.

## 5. Recommendation

Do not continue directly into a custom attention decode kernel.

Recommended next step:

1. Report this no-go recommendation to GPT.
2. Ask GPT whether to accept the pivot toward the GEMM/MLP candidate from
   `phase5_candidate_optimization_plan.md`.
3. If GPT agrees, start a GEMM/MLP standalone baseline focused on Qwen3 MLP
   shapes, beginning with BF16 `gate_up_proj`.

Do not start W4A16 full-model quantization yet. Keep the next step as a
standalone BF16 GEMM/MLP measurement unless GPT explicitly recommends a
different path.
