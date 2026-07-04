# Phase 3.6 Benchmark Audit Plan

Phase 3.6 is a measurement-audit phase, not a new optimization phase. It exists
to make the Phase 2 throughput and latency numbers explainable before moving to
kernel work.

## Questions to answer

1. Why did the original smoke run show `batch=1,input=512` faster than
   `batch=1,input=128`?
2. What exactly does TPOT mean in the latency script?
3. How much of the eager-vs-CUDA-Graph gap is stable after warmup and case-order
   changes?

## Scope

- Add an audit benchmark with repeat and case-order controls.
- Keep timing boundaries explicit: `llm.generate` only, with CUDA synchronize
  before and after each timed case.
- Verify output-token counts, because `output_tokens_per_s` is output-token only.
- Clarify latency TPOT as an aggregate decode cadence after average TTFT.

## Non-goals

- No Triton or CUDA kernel integration.
- No scheduler/model-runner refactor.
- No Nsight or PyTorch profiler unless the audit still leaves unexplained
  anomalies.
- No quantization, speculative decoding, or KV-cache compression in this phase.

## AutoDL RTX 3090 command

```bash
python benchmarks/bench_benchmark_audit.py \
  --model /root/huggingface/Qwen3-0.6B \
  --batch-sizes 1,8,32 \
  --input-lens 128,512 \
  --output-lens 128 \
  --orders natural,reverse,shuffle \
  --repeats 2 \
  --no-write \
  --output-prefix benchmark_audit_3090
```

Optional eager comparison:

```bash
python benchmarks/bench_benchmark_audit.py \
  --model /root/huggingface/Qwen3-0.6B \
  --batch-sizes 1,8,32 \
  --input-lens 128,512 \
  --output-lens 128 \
  --orders natural,reverse,shuffle \
  --repeats 2 \
  --enforce-eager \
  --no-write \
  --output-prefix benchmark_audit_3090_eager
```

## Stop condition

If the `batch=1,input=512` advantage disappears or changes sign under reverse
or shuffled order, treat the original Phase 2 smoke result as an order/warmup
artifact and stop the audit. If it stays stable across orders and repeats, keep
Phase 3.6 open and inspect decode graph selection and scheduler step behavior
next.
