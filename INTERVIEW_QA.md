# Interview Q&A: nano-vLLM Kernel Lab

## 1. What is this project?

It is a profile-driven LLM inference systems project built on top of nano-vLLM.
I used a small engine so I could understand the full request lifecycle:
scheduler, BlockManager, KV cache, ModelRunner, CUDA Graph decode, and profiling.

The project is not a production framework. It is a controlled optimization lab
on Qwen3-0.6B and RTX 3090.

## 2. What is your main contribution?

The main original contribution is decode-aware scheduling for chunked prefill.

Upstream chunked prefill splits a long prefill into chunks, but it can still run
multiple prefill chunks in a row and delay active decode requests. I added a
policy that explicitly gives active decode requests a service opportunity after
N prefill chunks.

In the tested workload, this reduced active decode max gap from about 182 ms to
about 35 ms with N=1. N=2 gave a more balanced trade-off at about 64 ms max gap
and lower TTFT penalty.

## 3. Why is active decode max gap important?

It measures the worst service interval experienced by already-running decode
requests. Throughput alone can hide this problem. A system can produce many
tokens overall while some active requests experience large pauses.

For serving, this matters because users experience decode continuity, not just
aggregate tokens per second.

## 4. Why not just say chunked prefill improves throughput?

Because that would overclaim. The project found that the real improvement was
about scheduling continuity and service gaps. Chunking alone does not guarantee
decode interleaving; the scheduler policy decides whether decode gets a chance
between prefill chunks.

## 5. Why did you audit the benchmark?

The early smoke benchmark had suspicious behavior: in some cases a longer input
looked faster than a shorter input. I did not want to build claims on unstable
numbers.

The audit controlled order, repeats, warmup, CUDA synchronization, graph/eager
mode, and output completeness. The conclusion was that Phase 2 smoke numbers
should be treated as environment validation, not final performance evidence.

## 6. What did profiling show?

The profiling path was layered:

```text
engine timing -> ModelRunner timing -> PyTorch Profiler
```

The result was that steady decode is dominated by FlashAttention decode and
BF16 GEMM. Scheduler, postprocess, sampler, logits, RoPE, RMSNorm, and small
Triton kernels were not first-order bottlenecks in the tested workload.

## 7. Why did CUDA Graph mode beat eager mode so much?

The ModelRunner breakdown showed that the difference is mainly in decode
forward / graph replay. Eager mode also had a large first decode outlier caused
by cold-start or lazy initialization. After excluding the first-step outlier,
eager decode was still much slower than CUDA Graph replay.

So the supported statement is not simply "CUDA Graph is faster" in the abstract.
It is:

```text
For this decode workload, graph replay removes much of the repeated eager
execution overhead in ModelRunner forward.
```

## 8. Why did you not write a Triton RoPE or RMSNorm kernel?

Because profiling did not support it. RoPE and RMSNorm appeared as small
contributors compared with attention decode and BF16 GEMM. Writing those kernels
would look active, but it would not be profile-driven.

## 9. Why is attention replacement no-go?

Attention decode is a major end-to-end hotspot, but the standalone
FlashAttention decode baseline was already correct and fast across the tested
Qwen3-0.6B shapes.

The project validated the KV cache layout, block table, GQA mapping, and BF16
correctness contract. Then it benchmarked the existing FlashAttention path. The
evidence did not show a clear replacement window for a custom Triton attention
kernel.

## 10. Is GEMM still worth studying?

Possibly, but only as a future standalone study. The MLP/GEMM baseline showed
that `gate_up` dominates larger-token standalone MLP cases, with `down` as the
second important GEMM segment.

That is a go for deeper standalone GEMM measurement, not a go for immediate
W4A16, fused MLP, or engine integration.

## 11. What would you do next if you continued this project?

I would not add another scheduler policy immediately. I would either:

- run Nsight Systems in an environment that supports it to validate graph replay
  timeline behavior; or
- continue a standalone GEMM study with clear correctness and timing criteria.

But for the current project phase, the better decision is to stop main
development and package the results.

## 12. What did you learn?

The main lesson is that optimization should be evidence-driven.

A hotspot does not automatically justify a custom kernel. A benchmark anomaly
does not automatically become a performance claim. A scheduler improvement
should be explained through workload behavior and service metrics, not just
aggregate throughput.
