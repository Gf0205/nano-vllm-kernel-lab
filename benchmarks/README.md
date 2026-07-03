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

## Output

Both scripts write:

- `benchmarks/results/*.jsonl` for machine-readable records.
- `benchmarks/results/*.md` for README-friendly tables.

Record the GPU, CUDA, PyTorch, Triton, Transformers, and FlashAttention
versions together with the result files before moving to kernel optimization.
