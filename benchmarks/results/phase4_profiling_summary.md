# Phase 4 Profiling Summary

Phase 4 profiles the current Qwen3-0.6B / RTX 3090 decode path before choosing
an optimization target.

## 1. Internal Timing

Internal timing split engine steps into scheduler, model runner, and
postprocess.

Key result:

| Mode | Wall time | Prefill model | Decode model sum | Decode model avg |
| --- | --- | --- | --- | --- |
| CUDA Graph | 2.23 s | 1.34 s | 0.89 s | 7.0 ms |
| Eager | 5.28 s | 1.18 s | 4.09 s | 32.2 ms |

Supported conclusion:

- scheduler and postprocess are not first-order bottlenecks;
- the main bottleneck is inside `ModelRunner.run`, especially decode;
- CUDA Graph mainly improves decode execution.

## 2. ModelRunner Breakdown

`ModelRunner.run` was split into input preparation, graph setup, forward/graph,
logits, and sampler.

Key result:

| Mode | Decode total avg | Decode forward/graph avg | Logits avg | Sampler avg |
| --- | --- | --- | --- | --- |
| CUDA Graph | 7.41 ms | 6.05 ms | 0.45 ms | 0.40 ms |
| Eager | 33.9 ms | 32.7 ms | 0.43 ms | 0.43 ms |

Eager had a cold-start outlier:

```text
decode_run_total_s_max: 826 ms
decode_run_total_s_max_step: 1
decode_forward_or_graph_s_first: 824 ms
```

Supported conclusion:

- prepare, logits, and sampler are not the primary bottleneck;
- eager's large outlier is a first-decode cold-start effect;
- steady decode bottleneck is model forward / CUDA Graph replay.

## 3. PyTorch Profiler

Steady-state decode profiling skipped prefill and early decode steps.

Graph-mode top CUDA events:

| Event | CUDA total over 16 decode steps |
| --- | --- |
| FlashAttention split-kv decode kernel | 52.2 ms |
| BF16 GEMM 64x64 kernel | 32.1 ms |
| small Triton kernels combined | 8.0 ms |
| lm_head/logits GEMM | 6.1 ms |
| sampler fused softmax/argmax | 1.9 ms |
| store_kvcache | 0.6 ms |

Eager module-level view:

| Module/event | CUDA total |
| --- | --- |
| `module.qwen3_attention` | 420.6 ms |
| `module.mlp` | 113.8 ms |
| `module.paged_decode_attention` | 86.1 ms |
| FlashAttention split-kv decode kernel | 53.0 ms |
| GEMM kernels | 33.2 ms + smaller GEMMs |

Supported conclusion:

- the main steady decode hotspots are attention decode and GEMM;
- small Triton kernels are numerous but not dominant in total time;
- RoPE, RMSNorm, sampler, logits, and KV-store are not the first optimization
  targets.

## 4. Nsight Systems Status

The current AutoDL image does not include `nsys`:

```text
which nsys -> empty
nsys --version -> command not found
```

Nsight Systems timeline validation is therefore blocked by environment, not by
project code. This remains a caveat:

- PyTorch Profiler gives event-level evidence;
- Nsight Systems would still be useful to inspect timeline gaps, graph replay
  ordering, synchronization, and kernel spacing;
- do not switch images only for this unless the next optimization requires
  timeline-level proof.

## 5. Phase 4 Conclusion

Phase 4 supports this bottleneck statement:

```text
In the current workload, steady decode is dominated by FlashAttention decode
and BF16 GEMM. Python scheduler/postprocess, sampler, logits, RoPE, RMSNorm,
and small Triton kernels are not first-order bottlenecks.
```

Do not implement a Triton RoPE/RMSNorm kernel based on this profile. The next
optimization decision should compare attention decode and GEMM.
