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

## 5. Second Step: ModelRunner Breakdown

If internal timing shows `model_runner.call` dominates, split
`ModelRunner.run` itself:

```bash
python benchmarks/bench_model_runner_timing.py \
  --model /root/huggingface/Qwen3-0.6B \
  --num-seqs 32 \
  --input-len 512 \
  --output-len 128 \
  --no-write \
  --output-prefix model_runner_timing_3090
```

Run the eager comparison as well:

```bash
python benchmarks/bench_model_runner_timing.py \
  --model /root/huggingface/Qwen3-0.6B \
  --num-seqs 32 \
  --input-len 512 \
  --output-len 128 \
  --enforce-eager \
  --no-write \
  --output-prefix model_runner_timing_3090_eager
```

This checks whether the decode outlier is in input preparation, graph replay /
forward, logits projection, or sampler.

## 6. Third Step: Steady Decode Profiler

After confirming decode `forward_or_graph` dominates, profile steady-state
decode only. Skip prefill and the first decode steps.

Graph mode:

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

Eager mode, with module labels for attention/MLP/RMSNorm/RoPE:

```bash
python benchmarks/bench_decode_profiler.py \
  --model /root/huggingface/Qwen3-0.6B \
  --num-seqs 32 \
  --input-len 512 \
  --output-len 128 \
  --enforce-eager \
  --warmup-decode-steps 4 \
  --profile-decode-steps 16 \
  --no-write \
  --output-prefix decode_profiler_3090_eager
```

Use graph mode to inspect CUDA Graph replay and kernel launch pattern. Use
eager mode to attribute time to module-level regions before choosing a kernel
optimization target.

## 7. Fourth Step: Nsight Systems

Use Nsight Systems only after PyTorch profiler confirms steady decode hotspots.
On AutoDL, use the CLI `nsys` command. The GUI is not required in the cloud.

First check availability:

```bash
which nsys
nsys --version
```

Capture only steady-state decode with CUDA profiler API gating:

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

Inspect the report on AutoDL:

```bash
nsys stats /root/autodl-tmp/nsys_decode_graph.nsys-rep
```

What to check:

- whether `flash_fwd_splitkv_kernel` dominates the Graph replay window;
- how GEMM kernels interleave with attention kernels;
- whether there are visible idle gaps between kernels;
- whether synchronizations or memcpys appear inside the profiled decode range;
- whether many small Triton kernels are meaningful in total time or just noise.

If one FlashAttention decode kernel dominates, move to Nsight Compute for that
kernel. If the timeline shows many fragmented small kernels or gaps, focus on
launch pattern and graph replay behavior instead.

Current AutoDL image does not include `nsys`, so Phase 4 is closed using
PyTorch Profiler evidence. See `phase4_profiling_summary.md`.
