# Canonical Workloads Before Profiling

Before Phase 4 bottleneck profiling, freeze three workloads so every later
optimization is evaluated against the same regimes.

## 1. Latency regime

Purpose: understand single-request and small-batch latency.

Suggested command:

```bash
python benchmarks/bench_latency.py \
  --model /root/huggingface/Qwen3-0.6B \
  --num-seqs 1 \
  --input-len 512 \
  --output-len 128
```

Primary metrics:

- TTFT
- aggregate decode TPOT
- output tok/s
- peak memory

## 2. Throughput regime

Purpose: measure steady-state batched generation.

Suggested command:

```bash
python benchmarks/bench_throughput.py \
  --model /root/huggingface/Qwen3-0.6B \
  --batch-sizes 8,32 \
  --input-lens 128,512 \
  --output-lens 128 \
  --warmup
```

Primary metrics:

- output tok/s
- wall time
- CUDA Graph vs eager gap
- output completeness

## 3. Long-context interference regime

Purpose: test whether long prefill disrupts active decode.

Suggested command:

```bash
python benchmarks/bench_chunked_prefill_interference.py \
  --model /root/huggingface/Qwen3-0.6B \
  --active-decode-seqs 8 \
  --active-input-len 128 \
  --active-output-len 128 \
  --long-input-len 3072 \
  --long-output-len 32 \
  --inject-after-decode-steps 8 \
  --normal-budget 8192 \
  --chunked-budget 512 \
  --long-decode-reserve-blocks 1 \
  --no-write \
  --output-prefix chunked_prefill_interference_3090
```

Primary metrics:

- active decode gap average / P95 / max
- long request TTFT
- number of chunked prefill steps
- system output tok/s
- waiting/running queue peaks

Note: keep `long_input_len + long_output_len` within the model's effective
context length. For the current Qwen3-0.6B setup, `3072 + 32` is a safer
long-context stress case than `4096 + 32`, which can exceed the effective
context limit after configuration clipping. The benchmark also records
`effective_long_input_len`; if available KV blocks are insufficient while active
decode requests are running, it trims the injected prompt instead of crashing
the scheduler on an allocation-capacity edge case.

## Roadmap after these workloads

1. Finish Scheduler/Chunked Prefill interference analysis.
2. Treat Phase 2 as smoke only; use audited baseline for claims.
3. Enter Phase 4 bottleneck profiling.
4. Use profiler evidence to choose Phase 5 optimization targets.
5. Keep advanced features like W4A16, speculative decoding, paged decode
   attention, and KV compression as optional later phases.
