# Phase 4 Bottleneck Profiling Plan

Phase 4 starts profiling before adding new kernels or optimization features.

## 1. Goal

Find where time is spent in the current Qwen3-0.6B / RTX 3090 serving path.

## 2. First Step: Internal Timing

Use `benchmarks/bench_internal_timing.py` to split each engine step into:

| Segment | Meaning |
| --- | --- |
| schedule | scheduler queue/block decisions |
| model | `ModelRunner.run`, including prepare, model execution, logits, sampling |
| postprocess | token append, cache bookkeeping, finished cleanup |

The script reports prefill and decode separately.

Suggested command:

```bash
python benchmarks/bench_internal_timing.py \
  --model /root/huggingface/Qwen3-0.6B \
  --num-seqs 32 \
  --input-len 512 \
  --output-len 128 \
  --no-write \
  --output-prefix internal_timing_3090
```

## 3. Interpretation

If `decode_model_s_*` dominates, move next to PyTorch profiler/Nsight to break
down attention, MLP, RMSNorm, RoPE, logits, and sampling.

If `schedule` or `postprocess` is unexpectedly large, inspect Python-side
scheduler/block-manager overhead before writing kernels.

If eager and CUDA Graph differ strongly, profile both:

```bash
python benchmarks/bench_internal_timing.py ... --enforce-eager
```

## 4. Stop Rule

Do not choose Triton, quantization, speculative decoding, or KV compression
until this timing identifies a concrete hotspot.
