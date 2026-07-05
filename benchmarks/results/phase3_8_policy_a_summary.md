# Phase 3.8 Policy A Summary

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

## 4. Current Single-Run Observation

The first RTX 3090 run showed:

| Case | Interleaved | Max active decode gap | Long TTFT | Max post-inject prefill step |
| --- | --- | --- | --- | --- |
| non-chunked | no | ~191.8 ms | ~186.1 ms | ~186.1 ms |
| upstream chunked | no | ~181.3 ms | ~175.6 ms | ~30.8 ms |
| Policy A | yes | ~34.0 ms | ~189.7 ms | ~29.2 ms |

Supported so far:

- upstream chunked prefill bounds single prefill-step time but does not
  interleave active decode between chunks;
- Policy A changes scheduler order and inserts decode between chunks;
- the first run shows a large active-decode max-gap reduction;
- Policy A may increase long-request TTFT, which is the expected trade-off.

Not yet claimed:

- final performance improvement;
- robustness across runs;
- optimal interleave cadence.

## 5. Repeat Stability Protocol

Run:

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

Policy A is considered stable enough to close Phase 3.8 if:

- `decode_aware_chunked_prefill.interleaved_runs == repeats`;
- `num_decode_aware_interleaves_min > 0`;
- Policy A keeps `active_decode_gap_s_max_mean` clearly below upstream chunked;
- Policy A does not cause capacity-limited runs;
- `long_request_ttft_s_mean` trade-off is visible and explainable.

## 6. Next Decision

If repeat stability holds, close Phase 3.8 as evidence for a decode-aware
scheduler optimization. Only then consider a cadence ablation such as N=1/2/4.

If repeat stability fails, do not expand Policy B. First inspect scheduler
timeline, decode batch histogram, CUDA Graph replay counts, and capacity state.
