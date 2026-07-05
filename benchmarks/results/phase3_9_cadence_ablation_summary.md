# Phase 3.9 Cadence Ablation Summary

Phase 3.9 evaluates static decode-interleave cadence. It does not introduce
adaptive scheduling.

## 1. Question

Given that Phase 3.8 proved explicit interleaving works, how often should active
decode be inserted between long-prefill chunks?

## 2. Compared Policies

| Case | Cadence |
| --- | --- |
| `chunked_long_prefill` | upstream chunked prefill, no explicit interleave |
| `decode_aware_chunked_n1` | decode after every 1 prefill chunk |
| `decode_aware_chunked_n2` | decode after every 2 prefill chunks |
| `decode_aware_chunked_n4` | decode after every 4 prefill chunks |

## 3. Repeat Summary

The AutoDL output was partially truncated, but the key `repeat_summary` rows
were available for the main metrics.

| Case | Max active decode gap mean | Long TTFT mean | Post-injection wall mean | Interleaves |
| --- | --- | --- | --- | --- |
| upstream chunked | 182.4 ms | 176.6 ms | 622.9 ms | 0 |
| N=1 | 35.3 ms | 195.2 ms | 623.3 ms | 5 |
| N=2 | 63.8 ms | 183.6 ms | 623.0 ms | 2 |
| N=4 | 124.2 ms | 184.6 ms | about 629.4 ms observed before truncation | partially truncated |

All visible rows reported `capacity_limited_runs=0`.

## 4. Interpretation

The cadence curve matches the expected trade-off:

- N=1 gives the strongest active-decode continuity. It reduces max gap from
  about 182 ms to about 35 ms, but has the highest long-request TTFT.
- N=2 is the best balanced candidate in this workload. It keeps max gap much
  lower than upstream, about 64 ms, while reducing TTFT pressure compared with
  N=1.
- N=4 is too sparse for this workload. The max gap rises to about 124 ms,
  showing that much of the continuity benefit is lost.
- post-injection wall time is broadly similar across upstream, N=1, and N=2,
  so the main trade-off is active decode continuity vs long-request TTFT.

## 5. Supported Conclusion

Phase 3.9 is closed with this conclusion:

```text
Static cadence controls the continuity/progress trade-off. N=1 minimizes
active decode max gap, while N=2 emerges as the best balanced candidate in the
current interference workload. N=4 trends back toward upstream behavior.
```

Do not call N=2 the default yet. It is only a balanced candidate for the current
Qwen3-0.6B / RTX 3090 / 8-active-decode / 3072-token-long-prefill workload.

## 6. Next Step

Pause scheduler-policy expansion. Do not implement adaptive Policy C now.

The next phase should move to bottleneck profiling:

- internal timing first;
- then PyTorch profiler;
- then Nsight Systems if needed;
- use profiler evidence to choose the next optimization target.
