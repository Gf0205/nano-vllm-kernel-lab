# Phase 3.8 Policy A Closeout Summary

Policy A is the first project-owned scheduler optimization in this repo. It is
kept behind `decode_aware_prefill_interleave=False` by default and is evaluated
only in the canonical long-context interference workload.

## 1. Question

Does explicit decode-aware interleaving reduce active decode disruption when a
long prefill request arrives while decode requests are already running?

## 2. Compared Policies

| Case | Policy |
| --- | --- |
| `non_chunked_long_prefill` | one large long-prefill step |
| `chunked_long_prefill` | upstream chunked prefill without decode interleaving |
| `decode_aware_chunked_prefill` | Policy A: one decode opportunity after each prefill chunk |

## 3. Metric Semantics

`active_decode_gap_s_*` measures the wall-clock interval between consecutive
decode progress events for the already-active decode requests. It is a
request-service continuity metric, not a model-kernel latency metric.

`post_inject_prefill_step_s_max` measures the largest prefill step after the
long request is injected. It verifies whether chunking bounds one prefill
blocking step.

`long_request_ttft_s` measures time from long-request injection to its first
generated token. It is expected to trade off against active-decode continuity
when decode is interleaved before the long prompt completes prefill.

`post_injection_phase_runs` is the scheduler-order proof. Upstream chunked
prefill should look like `prefill... -> decode...`; Policy A should show
`prefill -> decode -> prefill -> decode...`.

## 4. Single-Run Observation

The first RTX 3090 run showed the expected scheduler-order change:

| Case | Interleaved | Max active decode gap | Long TTFT | Max post-inject prefill step |
| --- | --- | --- | --- | --- |
| non-chunked | no | ~191.8 ms | ~186.1 ms | ~186.1 ms |
| upstream chunked | no | ~181.3 ms | ~175.6 ms | ~30.8 ms |
| Policy A | yes | ~34.0 ms | ~189.7 ms | ~29.2 ms |

This single run supported the mechanism but was not enough for closeout:

- upstream chunked prefill bounds single prefill-step time but does not
  interleave active decode between chunks;
- Policy A changes scheduler order and inserts decode between chunks;
- the first run shows a large active-decode max-gap reduction;
- Policy A may increase long-request TTFT, which is the expected trade-off.

## 5. Repeat Stability Result

The follow-up RTX 3090 repeat run used `repeats=3` on the same workload.

| Case | Interleaved runs | Max active decode gap mean | Max active decode gap range | Long TTFT mean | Capacity limited |
| --- | --- | --- | --- | --- | --- |
| non-chunked | 0/3 | 114.7 ms | 81.4-180.8 ms | 108.7 ms | 0 |
| upstream chunked | 0/3 | 184.1 ms | 180.1-189.5 ms | 178.4 ms | 0 |
| Policy A | 3/3 | 34.9 ms | 33.2-36.7 ms | 195.0 ms | 0 |

Policy A also reported `num_decode_aware_interleaves=5` in every run.

## 6. Supported Conclusion

Phase 3.8 is closed with the following supported conclusion:

```text
Explicit decode-aware interleaving gives active decode requests service
opportunities between long-prefill chunks. In the canonical interference
workload, this reduces active decode max gap from ~184 ms to ~35 ms while
increasing long-request TTFT from ~178 ms to ~195 ms.
```

Interpretation:

- upstream chunked prefill is compute chunking only;
- Policy A is a true scheduler-order change;
- `active_decode_gap_s_max` is the key interference/tail metric in this
  workload because most decode steps remain near 5 ms and P95 stays flat;
- the TTFT increase is an expected trade-off because the long request no longer
  monopolizes all prefill chunks before active decode gets service;
- post-injection wall time did not materially worsen in this repeat run.

Not claimed:

- Policy A is globally optimal;
- N=1 is the best cadence;
- adaptive scheduling is necessary;
- this result transfers unchanged to larger models, higher concurrency, or
  different KV-cache pressure.

## 7. Repeat Command

The closeout run used:

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
  --timeline-limit 80 \
  --include-decode-aware \
  --repeats 3 \
  --no-write \
  --output-prefix chunked_prefill_interference_3090
```

Check the printed `repeat_summary` rows.

Phase 3.8 closeout criteria were:

- `decode_aware_chunked_prefill.interleaved_runs == repeats`;
- `num_decode_aware_interleaves_min > 0`;
- Policy A keeps `active_decode_gap_s_max_mean` clearly below upstream chunked;
- Policy A does not cause capacity-limited runs;
- `long_request_ttft_s_mean` trade-off is visible and explainable.

All criteria passed.

## 8. Next Decision

Do not modify Policy A further in this phase. The only reasonable next
question is whether the fixed cadence should remain N=1 or whether a minimal
N=1/2/4 cadence ablation exposes a better continuity/progress trade-off.
