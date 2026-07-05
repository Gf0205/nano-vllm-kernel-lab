# nano-vLLM Kernel Lab

This repository is a profile-driven LLM inference systems project built on top
of the upstream `nano-vLLM` codebase. The goal is not to turn nano-vLLM into a
full production serving framework. The goal is to make a small inference engine
measurable, explainable, and extensible on a single RTX 3090.

The project focuses on three themes:

1. benchmark credibility;
2. scheduler optimization for long-prefill interference;
3. profile-driven optimization decisions before writing custom kernels.

## What Changed From Upstream

The upstream project provides a compact vLLM-like inference engine with prefix
caching, paged KV cache, CUDA Graph decode, tensor parallel hooks, and a small
Python codebase.

This fork adds the systems work around it:

- benchmark scripts for throughput, latency, prefix cache, scheduler
  interference, internal timing, ModelRunner timing, and steady decode
  profiling;
- BlockManager and scheduler metrics for prefix-cache reuse, physical block
  allocation, chunked prefill behavior, and decode-aware interleaving;
- a decode-aware chunked-prefill scheduler policy with static cadence
  ablation;
- Phase 4 profiling summaries showing where decode time is actually spent;
- Phase 5 go/no-go studies for attention decode and MLP/GEMM candidates.

## Current Status

Main development is frozen after the first profile-driven optimization cycle.
The repository is now in maintenance / interview-asset mode.

This does not mean there are no possible future optimizations. It means the
current project has reached a useful stopping point:

- benchmark anomalies were audited instead of blindly accepted;
- a real scheduler policy was implemented and measured;
- profiling identified attention decode and BF16 GEMM as the main steady-state
  decode hotspots;
- attention replacement was rejected for immediate implementation after a
  standalone FlashAttention decode microbenchmark;
- MLP/GEMM was kept as a future standalone-study candidate, not rushed into
  engine integration.

## Key Results

All measurements below were collected on AutoDL with an RTX 3090, Qwen3-0.6B,
PyTorch 2.5.1 + CUDA 12.4, Triton 3.1.0, and FlashAttention 2.7.4.post1.

### Scheduler Optimization

The long-prefill interference workload uses 8 active decode requests and injects
one 3072-token long prompt after decode has started.

| Policy | Active decode max gap | Long request TTFT | Notes |
| --- | ---: | ---: | --- |
| upstream chunked prefill | about 182 ms | about 177 ms | no explicit decode opportunity between chunks |
| decode-aware N=1 | about 35 ms | about 195 ms | best continuity, highest TTFT cost |
| decode-aware N=2 | about 64 ms | about 184 ms | balanced candidate for this workload |
| decode-aware N=4 | about 124 ms | about 185 ms | too sparse; trends back toward upstream |

Supported conclusion:

```text
Explicit decode-aware interleaving reduces the worst active-decode service gap.
The cadence controls a continuity/progress trade-off. N=2 is a balanced
candidate for this specific workload, not a global default.
```

See:

- `benchmarks/results/phase3_8_policy_a_summary.md`
- `benchmarks/results/phase3_9_cadence_ablation_summary.md`

### Profiling

Phase 4 used internal timing, ModelRunner timing, and PyTorch Profiler.

| Layer | Supported finding |
| --- | --- |
| Engine timing | scheduler and postprocess are not first-order bottlenecks |
| ModelRunner timing | steady decode is dominated by forward / CUDA Graph replay |
| PyTorch Profiler | FlashAttention decode and BF16 GEMM dominate steady decode |

Graph-mode steady decode over 16 profiled decode steps:

| Event | CUDA total |
| --- | ---: |
| FlashAttention split-kv decode kernel | 52.2 ms |
| BF16 GEMM 64x64 kernel | 32.1 ms |
| small Triton kernels combined | 8.0 ms |
| lm_head/logits GEMM | 6.1 ms |
| sampler fused softmax/argmax | 1.9 ms |

Supported conclusion:

```text
RoPE, RMSNorm, sampler, logits, and small Triton kernels are not the first
optimization targets for this workload. Attention decode and GEMM are the real
candidate areas.
```

See `benchmarks/results/phase4_profiling_summary.md`.

### Phase 5 Candidate Decisions

| Candidate | Current decision | Reason |
| --- | --- | --- |
| attention decode replacement | immediate no-go | FlashAttention standalone baseline is correct and already fast |
| MLP/GEMM | go for deeper standalone study only | gate_up dominates larger-token standalone MLP cases |
| W4A16 | no-go for now | would expand scope into quantization and accuracy validation |
| fused MLP / custom GEMM | no-go for immediate engine integration | needs stronger standalone evidence first |

See:

- `benchmarks/results/phase5_attention_decode_contract_summary.md`
- `benchmarks/results/phase5_attention_decode_microbench_summary.md`
- `benchmarks/results/phase5_mlp_gemm_baseline_summary.md`

## Reproducing Core Runs

Install dependencies in an environment with CUDA 12.x, PyTorch, Triton, and
FlashAttention. The project was validated with Qwen3-0.6B.

Download the model:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B \
  --local-dir-use-symlinks False
```

Run a basic generation check:

```bash
python example.py --model /root/huggingface/Qwen3-0.6B
```

Run the scheduler cadence workload:

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
  --include-decode-aware \
  --decode-aware-cadences 1,2,4 \
  --repeats 3 \
  --no-write
```

Run steady decode profiling:

```bash
python benchmarks/bench_decode_profiler.py \
  --model /root/huggingface/Qwen3-0.6B \
  --num-seqs 32 \
  --input-len 512 \
  --output-len 128 \
  --warmup-decode-steps 4 \
  --profile-decode-steps 16 \
  --no-write
```

## Important Caveats

- Phase 2 smoke benchmark numbers are preserved as smoke evidence, not final
  performance conclusions.
- Nsight Systems was not available in the AutoDL image used for this project;
  PyTorch Profiler provides the current event-level evidence.
- The scheduler cadence result is workload-specific. N=2 is a balanced
  candidate for the tested setup, not a universal serving policy.
- Phase 5 did not implement a custom kernel. It deliberately stopped at
  correctness contracts, standalone baselines, and go/no-go decisions.

## Original Project

This repository builds on the upstream nano-vLLM project:

https://github.com/GeeeekExplorer/nano-vllm
