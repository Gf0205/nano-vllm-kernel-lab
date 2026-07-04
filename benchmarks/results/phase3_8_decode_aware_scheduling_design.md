# Phase 3.8 Decode-Aware Scheduling Design

Phase 3.8 is a design phase, not an implementation phase. It follows the
Phase 3.7 finding that upstream chunked prefill performs compute chunking but
does not actively interleave decode execution between consecutive prefill
chunks.

## 1. Motivation

The initial hypothesis was:

```text
Chunked prefill shortens a single long-prefill blocking step
    -> active decode may get more execution opportunities
    -> decode tail latency may improve
```

Phase 3.7 showed only the first part is true in the current scheduler.

## 2. Upstream Timeline Evidence

In the long-prompt interference workload:

```text
8 active decode requests
+ inject one 3072-token prompt
+ prefill budget 512
```

the upstream chunked timeline looked like:

```text
prefill 512
prefill 512
prefill 512
prefill 512
prefill 512
prefill 512
decode ...
```

It did not look like:

```text
prefill 512
decode
prefill 512
decode
...
```

## 3. Current Policy Definition

The current policy is best described as:

```text
compute chunking without decode-aware interleaving
```

It reduces the maximum single prefill step:

```text
~183.5 ms -> ~30.7 ms
```

but it does not materially reduce the post-injection completion window:

```text
~0.622 s vs ~0.617 s
```

and it does not materially improve active decode max gap in this workload.

## 4. Why Active Decode Gap Did Not Improve

The active decode requests do not receive execution opportunities between
prefill chunks. They wait until the long prompt finishes all scheduled chunks.

Therefore, the scheduler changes the shape of prefill work but not the service
order seen by active decode.

## 5. Candidate Policies

### Policy A: 1P:1D Fixed Interleave

After each prefill chunk, allow one decode step if running decode requests exist.

```text
prefill chunk
decode step
prefill chunk
decode step
```

Pros:

- simplest to implement;
- easiest to explain and benchmark;
- directly tests whether explicit interleaving helps active decode latency.

Cons:

- may slow long prompt completion;
- may waste opportunities when decode batch is tiny;
- not adaptive.

### Policy B: N Prefill Chunks Per Decode Step

Allow one decode step after every `N` prefill chunks.

```text
P P D
P P D
```

Candidate values:

```text
N = 1, 2, 4
```

Pros:

- exposes a clear latency/throughput trade-off curve;
- less aggressive than 1P:1D.

Cons:

- still static;
- requires a sweep after correctness is established.

### Policy C: Decode-Age / Starvation-Aware Interleave

Force decode when active decode requests have waited too long:

```text
if steps_since_last_decode >= threshold:
    schedule_decode()
else:
    continue_prefill()
```

Pros:

- closest to a real serving scheduler;
- can target tail latency directly.

Cons:

- more moving parts;
- harder to validate;
- should not be the first implementation.

## 6. Recommended First Implementation

Start with Policy A:

```text
Interleave one decode opportunity between prefill chunks.
```

This is intentionally simple. The first experiment should answer one question:

```text
Does explicit decode interleaving reduce active decode gap in this workload?
```

If Policy A does not improve active decode gap, more complex adaptive policies
are unlikely to be worth implementing immediately.

## 7. Evaluation Workload

Keep the workload fixed:

```text
model: Qwen3-0.6B
gpu: RTX 3090
active decode requests: 8
active input len: 128
active output len: 128
injected long prompt: 3072 tokens
long output len: 32
chunk budget: 512
CUDA Graph: enabled
```

Compare:

```text
upstream chunked policy
vs
decode-aware interleaving policy
```

## 8. Primary Metrics

| Metric | Expected trade-off |
| --- | --- |
| active decode gap avg / P95 / max | should improve |
| post-injection wall time | may worsen slightly |
| long request TTFT | may worsen |
| long request completion time | may worsen |
| post-injection output tok/s | may drop or stay close |
| max prefill step time | should stay bounded by chunk size |
| prefill/decode phase runs | should show interleaving |

The expected result is not "everything improves." A realistic scheduler policy
should trade some long-prefill progress for better active-decode continuity.

## 9. Failure Criteria

Do not continue to more complex policies if Policy A shows:

- active decode gap P95/max does not improve;
- post-injection wall time worsens substantially;
- decode batch histogram becomes too fragmented;
- long request TTFT grows without active-decode benefit;
- implementation destabilizes prefix cache or block allocation behavior.

## 10. Next Step

Do not implement Triton kernels, quantization, speculative decoding, or broad
profiling yet. The next engineering step is a minimal, feature-flagged scheduler
policy for explicit prefill/decode interleaving, validated only on the canonical
interference workload.
