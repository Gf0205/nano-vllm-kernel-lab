# Phase 3.6 Benchmark Audit Summary

Environment: AutoDL RTX 3090, Qwen3-0.6B, PyTorch 2.5.1 + CUDA 12.4,
FlashAttention 2.7.4.post1. The audit ran the same six cases under
natural/reverse/shuffled order with two repeats.

## Key finding

The original Phase 2 anomaly where `batch=1,input=512` looked much faster than
`batch=1,input=128` is not stable after audit controls. With CUDA Graph enabled,
both shapes converge to roughly the same output-token throughput:

| mode | case | avg tok/s | min tok/s | max tok/s |
| --- | --- | --- | --- | --- |
| graph | bs1_in128_out128 | 313.29 | 308.74 | 314.80 |
| graph | bs1_in512_out128 | 312.70 | 310.90 | 314.02 |
| eager | bs1_in128_out128 | 41.23 | 40.35 | 42.09 |
| eager | bs1_in512_out128 | 41.14 | 40.07 | 41.88 |

Interpretation: the Phase 2 single-run smoke result should be described as a
warmup/order artifact, not as a real claim that longer prompts improve
single-request decode throughput.

## Stable post-audit shape trends

| mode | case | avg tok/s | note |
| --- | --- | --- | --- |
| graph | bs8_in128_out128 | 2147.00 | one first natural-order run was low at 1329.20 tok/s |
| graph | bs8_in512_out128 | 1971.75 | mostly stable around 2.0k tok/s |
| graph | bs32_in128_out128 | 5757.82 | stable best-throughput case in this audit |
| graph | bs32_in512_out128 | 3832.16 | lower than input 128 because prefill cost dominates more |
| eager | bs8_in128_out128 | 299.79 | stable after repeat/order controls |
| eager | bs8_in512_out128 | 299.41 | close to input 128 |
| eager | bs32_in128_out128 | 1217.81 | stable |
| eager | bs32_in512_out128 | 1179.46 | slightly lower than input 128 |

## CUDA Graph vs eager

The eager/graph gap remains large after the audit, but the explanation should be
phrased carefully. The timed region excludes LLM initialization and CUDA Graph
capture, so the measured gap reflects the steady-state generation path,
including Python scheduling overhead avoided by graph replay. It should not be
claimed as pure kernel speedup.

Approximate audited speedups:

| case | graph avg tok/s | eager avg tok/s | speedup |
| --- | --- | --- | --- |
| bs1_in128_out128 | 313.29 | 41.23 | 7.60x |
| bs1_in512_out128 | 312.70 | 41.14 | 7.60x |
| bs8_in128_out128 | 2147.00 | 299.79 | 7.16x |
| bs8_in512_out128 | 1971.75 | 299.41 | 6.59x |
| bs32_in128_out128 | 5757.82 | 1217.81 | 4.73x |
| bs32_in512_out128 | 3832.16 | 1179.46 | 3.25x |

## Metric notes for reporting

- `output_tokens_per_s` is output-token-only throughput:
  `total_output_tokens / wall_time_s`.
- `ignore_eos=True` and `output_complete=True` confirm each case generated the
  expected number of output tokens.
- The audit script synchronizes CUDA before and after each timed `llm.generate`.
- `llm_init_time_s` is recorded but excluded from throughput timing.
- Latency `tpot_s` is kept for compatibility, but the clearer field is
  `decode_tpot_s_aggregate`: aggregate decode cadence after average TTFT, not
  the reciprocal of system throughput.

## Workflow update

On AutoDL, use `--no-write` for audit runs and paste the console output back into
the local development thread. The local repo will store curated summaries like
this file, which avoids untracked result files blocking later `git pull`.
