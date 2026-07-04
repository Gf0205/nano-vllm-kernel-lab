# Prefix Cache Causality Summary

This note clarifies what the Phase 3 prefix-cache result proves, and what it
does not yet prove.

## Observed result

The RTX 3090 prefix-cache benchmark compared 32 requests with a shared
1024-token prefix against 32 unique 1056-token prompts.

| case | hit blocks | physical allocations | output tok/s | peak memory GB |
| --- | --- | --- | --- | --- |
| shared prefix after warmup | 128 | 32 | 2167.79 | 20.397 |
| unique prefix baseline | 0 | 160 | 1709.11 | 20.793 |

The shared prefix spans four full 256-token KV blocks. After one warmup request,
each of the 32 later requests can reuse those four blocks:

```text
32 requests * 4 reusable blocks = 128 prefix-cache hit blocks
```

## Supported causal chain

The current metrics strongly support this chain:

```text
shared full-block prefix
    -> prefix hash hits in BlockManager
    -> reused KV cache blocks
    -> fewer physical block allocations
    -> lower allocator/cache pressure
    -> better measured throughput in this workload
```

The allocation side of the causality is directly measured:

- `prefix_cache_hit_blocks`: 0 -> 128
- `physical_block_allocations`: 160 -> 32
- `prefix_cache_hit_rate`: 0.0 -> 1.0

## Throughput interpretation

Throughput improved from 1709.11 tok/s to 2167.79 tok/s in this benchmark. This
is consistent with prefix KV reuse, but the current benchmark does not yet break
the improvement into:

- prefill compute skipped;
- TTFT reduction;
- prefill step time reduction;
- decode-side changes;
- allocator pressure reduction.

Therefore the safe claim is:

```text
Prefix cache demonstrably reduces physical KV block allocation and correlates
with higher output throughput in a controlled shared-prefix workload.
```

The stronger claim:

```text
Throughput improved because prefill compute was skipped.
```

should be treated as a hypothesis until we add per-step prefill timing and TTFT
instrumentation to the prefix-cache benchmark.

## Next measurement needed

To close the full causal chain, add or collect:

- total prefill tokens actually computed after cache hits;
- number of prefill steps;
- prefill wall time;
- TTFT for shared-prefix vs unique-prefix cases;
- output throughput after separating prefill and decode time.

Once those are measured, the report can distinguish compute skipping from block
allocator/cache-pressure effects.
