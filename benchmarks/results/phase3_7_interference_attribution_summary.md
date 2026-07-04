# Phase 3.7 Interference Attribution Summary

This summary records the first Phase 3.7 attribution run. The goal was to
explain why the chunked-prefill interference workload previously showed roughly
`2.6x` higher output throughput.

## Input validity

Both cases used the full requested long prompt:

| case | capacity limited | prompt shrunk | effective long input |
| --- | --- | --- | --- |
| non-chunked | False | False | 3072 |
| chunked | False | False | 3072 |

So this run is not a KV-capacity artifact.

## Top-level comparison

| metric | non-chunked | chunked |
| --- | ---: | ---: |
| wall time s | 1.8321 | 0.7076 |
| output tok/s | 576.38 | 1492.36 |
| total prefill wall time s | 1.357916 | 0.232742 |
| total decode wall time s | 0.471625 | 0.472246 |
| decode steps | 127 | 127 |
| decode batch histogram | `{8: 96, 9: 31}` | `{8: 96, 9: 31}` |
| CUDA Graph decode steps | 127 | 127 |
| eager decode steps | 0 | 0 |

The wall-time delta is:

```text
1.8321 - 0.7076 = 1.1245 s
```

The prefill-wall-time delta is:

```text
1.357916 - 0.232742 = 1.125174 s
```

The decode-wall-time delta is effectively zero:

```text
0.471625 - 0.472246 = -0.000621 s
```

## Attribution

The 2.6x output-throughput difference is not explained by:

- decode step count;
- decode batch occupancy;
- CUDA Graph replay/fallback;
- decode wall time.

Those are essentially identical between the two cases.

The measured difference is almost entirely in total prefill wall time.

## Important caveat

`total_prefill_wall_time_s` currently includes both:

1. pre-injection active-request prefill;
2. post-injection long-request prefill.

Because of that, this run proves the difference is in prefill timing, but it
does not yet prove that the difference is entirely caused by the injected long
prompt. The benchmark has been updated to split pre-injection and
post-injection wall time in the next run.

## Supported conclusion

The previous explanation should be narrowed:

```text
Chunked prefill did not improve throughput through decode occupancy or CUDA
Graph replay differences in this run. Decode-side behavior was effectively the
same. The full wall-time delta matches the prefill-wall-time delta, so the next
question is why prefill wall time differs so much.
```

## Next run

Re-run the same workload after the pre/post-injection split and inspect:

- `pre_injection_prefill_wall_time_s`
- `post_injection_prefill_wall_time_s`
- `pre_injection_decode_wall_time_s`
- `post_injection_decode_wall_time_s`
- `pre_injection_wall_time_s`
- `post_injection_wall_time_s`

This will separate bootstrap active-prefill effects from true long-prompt
interference effects.

## Pre/post-injection split result

The follow-up run added the pre/post-injection split:

| metric | non-chunked | chunked | delta |
| --- | ---: | ---: | ---: |
| wall time s | 1.8352 | 0.7092 | 1.1260 |
| pre-injection wall time s | 1.210695 | 0.089681 | 1.121014 |
| post-injection wall time s | 0.622071 | 0.616843 | 0.005228 |
| pre-injection prefill wall time s | 1.182307 | 0.061415 | 1.120892 |
| post-injection prefill wall time s | 0.183507 | 0.174242 | 0.009265 |
| post-injection decode wall time s | 0.438565 | 0.442601 | -0.004036 |

This changes the attribution:

```text
The apparent 2.6x full-run throughput difference is dominated by the
pre-injection active-prefill region, not by the injected long-prompt
interference window.
```

The actual post-injection window is nearly the same:

```text
0.622071s vs 0.616843s
```

So the correct conclusion is now narrower:

- chunked prefill still reduces the maximum single post-injection prefill step
  from about `183.5ms` to about `30.7ms`;
- total post-injection prefill wall time is similar: `183.5ms` vs `174.2ms`;
- total post-injection wall time is also similar;
- the previous full-run throughput number was polluted by the initial
  active-request prefill region.

## Updated supported conclusion

For this workload, chunked prefill changes the shape of long-prompt prefill
work, but does not materially reduce total post-injection interference-window
time. It reduces the largest individual prefill step, which is still useful for
tail-step analysis, but the full-run `output_tokens_per_s` should not be used as
the primary metric for chunked-prefill benefit.

The benchmark has been updated to report post-injection output throughput so the
next run can focus on the true interference window rather than the initial
active-prefill bootstrap.

## Timeline verification

The chunked timeline showed consecutive prefill chunks before decode resumed:

```text
prefill 512
prefill 512
prefill 512
prefill 512
prefill 512
prefill 512
decode ...
```

So, in this scheduler/workload, chunked prefill performs compute chunking but
does not provide decode-aware interleaving between chunks. This explains why the
largest single prefill step gets much smaller while total post-injection window
time and active decode max gap remain nearly unchanged.

The benchmark now emits `post_injection_phase_runs` and
`prefill_decode_interleaved_after_injection` to make this behavior explicit in
future runs.

## Final Phase 3.7 Conclusion

Phase 3.7 is closed with the following conclusion:

```text
The upstream chunked-prefill policy in this workload performs compute chunking
but does not actively interleave decode execution between consecutive prefill
chunks.
```

It improves:

- max single prefill step latency;
- per-step prefill token budget;
- observability of prefill chunk behavior.

It does not yet improve:

- active decode max gap;
- post-injection completion time;
- decode continuity.

The design implication is that improving active decode latency requires a
scheduler-level decode-aware interleaving policy, not only smaller prefill
compute chunks.
