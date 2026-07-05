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
  --long-input-len 3072 \
  --long-output-len 32 \
  --inject-after-decode-steps 8 \
  --normal-budget 8192 \
  --chunked-budget 512 \
  --long-decode-reserve-blocks 0 \
  --timeline-limit 48 \
  --no-write \
  --output-prefix chunked_prefill_interference_3090
```

This is also the Phase 3.7 attribution entrypoint. Besides the top-level
scheduler metrics, it reports decode batch histograms, total prefill/decode wall
time, decode step latency, CUDA Graph decode replay counts, and a bounded
post-injection timeline.

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
## Phase 4: Bottleneck profiling

Start with internal timing before choosing an optimization target:

```bash
python benchmarks/bench_internal_timing.py \
  --model /root/huggingface/Qwen3-0.6B \
  --num-seqs 32 \
  --input-len 512 \
  --output-len 128 \
  --no-write \
  --output-prefix internal_timing_3090
```

This reports prefill/decode time split into scheduler, model runner, and
postprocess segments.

If `model_runner.call` dominates, split it further:

```bash
python benchmarks/bench_model_runner_timing.py \
  --model /root/huggingface/Qwen3-0.6B \
  --num-seqs 32 \
  --input-len 512 \
  --output-len 128 \
  --no-write \
  --output-prefix model_runner_timing_3090
```

Then profile steady-state decode:

```bash
python benchmarks/bench_decode_profiler.py \
  --model /root/huggingface/Qwen3-0.6B \
  --num-seqs 32 \
  --input-len 512 \
  --output-len 128 \
  --warmup-decode-steps 4 \
  --profile-decode-steps 16 \
  --no-write \
  --output-prefix decode_profiler_3090
```

For Nsight Systems on AutoDL, first check:

```bash
which nsys
nsys --version
```

Then capture steady-state decode only:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o /root/autodl-tmp/nsys_decode_graph \
  python benchmarks/bench_decode_nsys.py \
    --model /root/huggingface/Qwen3-0.6B \
    --num-seqs 32 \
    --input-len 512 \
    --output-len 128 \
    --warmup-decode-steps 8 \
    --profile-decode-steps 32
```

## Phase 5: Attention decode contract

Before writing a custom attention kernel, validate the current decode contract:

```bash
python benchmarks/bench_attention_decode_contract.py \
  --model /root/huggingface/Qwen3-0.6B \
  --batch-size 4 \
  --context-len 513 \
  --block-size 256 \
  --num-blocks 16 \
  --no-write
```

After the contract passes, measure the current FlashAttention decode baseline
standalone before attempting a replacement:

```bash
python benchmarks/bench_attention_decode_microbench.py \
  --model /root/huggingface/Qwen3-0.6B \
  --cases 1x128,4x513,8x512,16x512,32x512,32x1024,64x512 \
  --warmup-iters 20 \
  --timing-iters 100 \
  --no-write
```

This script uses the same paged KV-cache and GQA contract as the engine decode
path. It checks FlashAttention output against a PyTorch reference for each case
before reporting CUDA-event latency, average microseconds per decoded token, and
tokens/s for the standalone decode call.

After attention decode replacement is no-go, measure the Qwen3 MLP/GEMM BF16
baseline before considering any MLP kernel work:

```bash
python benchmarks/bench_mlp_gemm_microbench.py \
  --model /root/huggingface/Qwen3-0.6B \
  --token-cases 128,256,512,1024 \
  --warmup-iters 20 \
  --timing-iters 100 \
  --no-write
```

This script reports `gate_up_proj`, `SiluAndMul`, `down_proj`, and full MLP
latency for standalone BF16 shapes derived from the model config. It is a
baseline and attribution step only; it does not implement W4A16, fused MLP, or
engine integration. Prefer the `*_boundary_*` fields for percentage
attribution because they come from one consistently timed full MLP call.
