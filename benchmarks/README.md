# Phase 2 Benchmark Guide

This directory contains the Phase 2 benchmark entrypoints. Run these scripts on
the AutoDL RTX 3090 machine, not on a CPU-only laptop.

## Throughput Baseline

```bash
python benchmarks/bench_throughput.py \
  --model ~/huggingface/Qwen3-0.6B/ \
  --batch-sizes 1,8,32,128,256 \
  --input-lens 128,512,1024 \
  --output-lens 128,512 \
  --max-model-len 4096
```

Compare CUDA Graph on/off:

```bash
python benchmarks/bench_throughput.py --model ~/huggingface/Qwen3-0.6B/ --enforce-eager
python benchmarks/bench_throughput.py --model ~/huggingface/Qwen3-0.6B/
```

## Latency Baseline

```bash
python benchmarks/bench_latency.py \
  --model ~/huggingface/Qwen3-0.6B/ \
  --num-seqs 32 \
  --input-len 512 \
  --output-len 128
```

The latency script uses the current `LLMEngine.step()` loop directly so it can
observe when each sequence receives its first generated token.

## Prefix Cache Baseline

```bash
python benchmarks/bench_prefix_cache.py \
  --model ~/huggingface/Qwen3-0.6B/ \
  --num-seqs 32 \
  --prefix-len 1024 \
  --suffix-len 32 \
  --output-len 64
```

This benchmark first warms one shared-prefix request so `BlockManager` can hash
full prefix blocks. It then compares later requests with the same prefix against
synthetic requests with unique prefixes.

## Scheduler / Chunked Prefill Metrics

```bash
python benchmarks/bench_scheduler_metrics.py \
  --model ~/huggingface/Qwen3-0.6B/ \
  --num-seqs 16 \
  --input-len 2048 \
  --output-len 64 \
  --normal-budget 16384 \
  --chunked-budget 512
```

This benchmark runs the same prompts twice and only changes
`max_num_batched_tokens`. The smaller budget should force chunked prefill and
make `num_chunked_prefill_steps` visible in the scheduler metrics.

For the interference workload, keep active decode requests running and inject a
long prompt:

```bash
python benchmarks/bench_chunked_prefill_interference.py \
  --model /root/huggingface/Qwen3-0.6B \
  --active-decode-seqs 8 \
  --active-input-len 128 \
  --active-output-len 128 \
  --long-input-len 4096 \
  --long-output-len 32 \
  --inject-after-decode-steps 8 \
  --normal-budget 8192 \
  --chunked-budget 512 \
  --no-write \
  --output-prefix chunked_prefill_interference_3090
```

## Benchmark Audit

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

Run this before making performance claims from the smoke benchmark. It times
`llm.generate` only, synchronizes CUDA before and after each measured case,
checks generated token counts, and changes case order to expose warmup or
ordering artifacts. Use `--no-write` on AutoDL when you only want to paste the
console output back into the local development thread; omit it when you want
jsonl/md artifacts.

## Output

All benchmark scripts write:

- `benchmarks/results/*.jsonl` for machine-readable records.
- `benchmarks/results/*.md` for README-friendly tables.

Record the GPU, CUDA, PyTorch, Triton, Transformers, and FlashAttention
versions together with the result files before moving to kernel optimization.
