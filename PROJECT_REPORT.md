# Project Report: nano-vLLM Kernel Lab

## 1. Project Positioning

`nano-vLLM Kernel Lab` is a controlled LLM inference systems project based on
the upstream nano-vLLM engine. It targets one practical environment: Qwen3-0.6B
on a single RTX 3090.

The project is intentionally narrow. It is not a generic serving framework and
not a collection of unrelated kernel demos. The core goal is to show a complete
optimization method:

```text
Observation -> Hypothesis -> Instrumentation -> Experiment -> Conclusion -> Caveat
```

## 2. Main Contributions

### Benchmark Credibility

The early throughput smoke results contained suspicious behavior, such as
longer prompts appearing faster in some cases. Instead of treating those numbers
as final performance claims, the project added a benchmark audit phase.

The audit controlled:

- case order;
- repeats;
- warmup;
- CUDA synchronization;
- output completeness;
- graph/eager mode separation.

Supported conclusion:

```text
Phase 2 smoke results are useful for environment validation, but not strong
enough as final performance conclusions.
```

### Prefix Cache Observability

The project added metrics around logical blocks, physical block allocations,
prefix-cache eligible blocks, prefix-cache hits, and block reuse.

The important lesson was not just that prefix cache can improve throughput. The
project separated:

- allocator pressure;
- KV block reuse;
- prefill compute avoidance;
- TTFT / throughput effect.

### Scheduler Optimization

The strongest original project contribution is decode-aware scheduling for
chunked prefill.

Problem:

```text
A long prefill request can create large service gaps for active decode
requests, even when prefill is chunked.
```

Implemented policy:

```text
After N chunked-prefill steps, force one decode opportunity if active decode
requests are running.
```

Measured result in the canonical interference workload:

| Policy | Max active decode gap | Long request TTFT | Interpretation |
| --- | ---: | ---: | --- |
| upstream chunked | about 182 ms | about 177 ms | baseline |
| N=1 | about 35 ms | about 195 ms | strongest continuity |
| N=2 | about 64 ms | about 184 ms | balanced candidate |
| N=4 | about 124 ms | about 185 ms | too sparse |

Supported conclusion:

```text
The optimization does not make a single kernel faster. It improves service
continuity by changing scheduling policy.
```

### Profile-Driven Optimization

Phase 4 profiled before optimizing.

Evidence chain:

```text
internal timing
-> ModelRunner.run breakdown
-> steady decode PyTorch profiler
```

Supported finding:

```text
The decode path is dominated by FlashAttention decode and BF16 GEMM.
Scheduler/postprocess, sampler, logits, RoPE, RMSNorm, and small Triton kernels
are not first-order bottlenecks for the tested workload.
```

### Phase 5 Go / No-Go

The project then evaluated optimization candidates before writing kernels.

Attention decode:

- contract validated against PyTorch reference;
- FlashAttention standalone baseline was correct and fast;
- immediate custom attention replacement was marked no-go.

MLP/GEMM:

- standalone BF16 MLP baseline showed `gate_up` dominates larger-token cases;
- this is a go for deeper standalone study;
- it is not yet a go for W4A16, fused MLP, or engine integration.

## 3. What This Project Is Not Claiming

This project does not claim:

- a production-grade serving stack;
- universal scheduler defaults;
- a custom kernel faster than FlashAttention;
- full-model W4A16 quantization;
- Nsight Systems timeline validation.

The value is the controlled engineering path and the evidence-backed stopping
points.

## 4. Final Status

Main development is frozen.

The project is ready to be used as:

- a public GitHub portfolio project;
- an interview discussion project;
- a foundation for a later standalone GEMM/kernel study if needed.

Recommended next personal step:

```text
Stop adding nano-vLLM features for now.
Use the project for interview preparation.
Start the next project only after README, result table, and Q&A are stable.
```
