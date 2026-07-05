# Phase 5 MLP / GEMM Baseline Summary

This summary records the first standalone Qwen3-0.6B BF16 MLP/GEMM baseline
after attention decode replacement was marked no-go.

## 1. Question

Does the Qwen3 MLP path expose a clearer optimization target than attention
decode?

This run measures standalone BF16 shapes from the model config. It does not
implement W4A16, a fused MLP kernel, a custom GEMM kernel, or engine
integration.

## 2. AutoDL RTX 3090 Result

Command:

```bash
python benchmarks/bench_mlp_gemm_microbench.py \
  --model /root/huggingface/Qwen3-0.6B \
  --token-cases 1,8,16,32,64,128,256,512,1024 \
  --warmup-iters 20 \
  --timing-iters 100 \
  --no-write
```

Model shape:

```text
hidden_size: 1024
intermediate_size: 3072
gate_up_weight_shape: (6144, 1024)
down_weight_shape: (1024, 3072)
dtype: torch.bfloat16
```

| tokens | gate_up avg ms | silu_mul avg ms | down avg ms | full MLP avg ms | gate_up % full | silu_mul % full | down % full | full MLP tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0388 | 0.0423 | 0.0335 | 0.0913 | 42.4973 | 46.3308 | 36.6922 | 10952.90 |
| 8 | 0.0390 | 0.0438 | 0.0404 | 0.1009 | 38.6521 | 43.4093 | 40.0396 | 79286.42 |
| 16 | 0.0393 | 0.0440 | 0.0400 | 0.1013 | 38.7957 | 43.4353 | 39.4867 | 157946.69 |
| 32 | 0.0488 | 0.0441 | 0.0521 | 0.1063 | 45.9078 | 41.4864 | 49.0122 | 301034.81 |
| 64 | 0.0435 | 0.0450 | 0.0456 | 0.1067 | 40.7685 | 42.1743 | 42.7366 | 599812.56 |
| 128 | 0.0635 | 0.0440 | 0.0454 | 0.1017 | 62.4385 | 43.2645 | 44.6411 | 1258603.74 |
| 256 | 0.0868 | 0.0468 | 0.0539 | 0.1393 | 62.3116 | 33.5966 | 38.6935 | 1837760.23 |
| 512 | 0.1586 | 0.0458 | 0.0899 | 0.2535 | 62.5641 | 18.0671 | 35.4635 | 2019723.87 |
| 1024 | 0.2468 | 0.0639 | 0.1367 | 0.4023 | 61.3473 | 15.8837 | 33.9796 | 2545364.16 |

## 3. Interpretation

For very small token counts, the benchmark is launch/overhead dominated:

- `num_tokens <= 64` keeps most segment averages around `0.04 ms`;
- `SiluAndMul` looks large in percentage terms for tiny inputs because the GEMM
  work is too small to dominate yet.

For larger token counts, the MLP path becomes GEMM dominated:

- from `num_tokens=128` onward, `gate_up` is consistently about `61-63%` of
  full standalone MLP time;
- `down` remains the second important GEMM segment, about `34-39%` for
  `num_tokens >= 256`;
- `SiluAndMul` falls to about `15.9%` at `num_tokens=1024`;
- full MLP latency grows from `0.1017 ms` at 128 tokens to `0.4023 ms` at
  1024 tokens.

The percent columns are diagnostic, not additive. Each segment is timed
separately, while `full_mlp` is timed as one combined call, so separate segment
percentages can sum above 100%.

## 4. Go / No-Go

MLP/GEMM candidate is go for deeper standalone study.

Supported conclusion:

```text
For Qwen3-0.6B BF16 standalone MLP shapes, larger token cases are dominated by
the gate_up projection, with down projection as the second GEMM hotspot.
This provides a clearer optimization target than immediate attention decode
replacement.
```

This is not yet a go for W4A16 or a fused kernel implementation. It is a go for
the next measurement step.

## 5. Recommended Next Step

Add a focused GEMM baseline that compares:

- `torch.nn.functional.linear` / cuBLAS BF16 baseline;
- candidate standalone GEMM route only if it can be isolated cleanly;
- `gate_up` shape first: `[tokens, 1024] x [6144, 1024]^T`;
- then `down` shape: `[tokens, 3072] x [1024, 3072]^T`;
- token cases centered on `128, 256, 512, 1024`.

Do not start full-model W4A16 quantization yet. If a custom path is explored,
it should begin as a standalone GEMM comparison with correctness and timing,
not as an engine replacement.
