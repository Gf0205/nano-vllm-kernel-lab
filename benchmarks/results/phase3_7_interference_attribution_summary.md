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
