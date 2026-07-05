# Phase 3.9 Cadence Ablation Design

Phase 3.9 is a minimal static-cadence ablation. It should not introduce
adaptive scheduling yet. The static cadence knob is implemented as
`prefill_interleave_every_n_chunks`; the closeout result is summarized in
`phase3_9_cadence_ablation_summary.md`.

## 1. Motivation

Phase 3.8 proved that explicit interleaving works:

```text
upstream chunked max active decode gap: ~184 ms
Policy A max active decode gap: ~35 ms
```

The remaining question is narrower:

```text
How often should decode be inserted between prefill chunks?
```

## 2. Candidate Cadences

Use one integer cadence:

```text
prefill_interleave_every_n_chunks = N
```

Compare:

| N | Schedule shape | Expected behavior |
| --- | --- | --- |
| 1 | P D P D P D | best active-decode continuity, highest long-TTFT pressure |
| 2 | P P D P P D | middle trade-off |
| 4 | P P P P D | closer to upstream, weaker continuity benefit |

Keep upstream chunked as the zero-interleave baseline.

## 3. Minimal Scheduler Design

Reuse the existing Policy A mechanism but replace the boolean trigger with a
small chunk counter:

```text
if decode-aware cadence is enabled:
    after each chunked prefill step:
        chunks_since_decode += 1
        if chunks_since_decode >= N and running decode requests exist:
            force one decode step next
            chunks_since_decode = 0
```

Constraints:

- default behavior must remain unchanged;
- N=1 must reproduce current Policy A behavior;
- do not add adaptive age thresholds;
- do not change KV allocation, prefix cache, CUDA Graph capture, or model
  execution paths;
- keep the benchmark workload fixed.

## 4. Evaluation Workload

Use the same canonical interference workload:

```text
model: Qwen3-0.6B
GPU: RTX 3090
active decode requests: 8
active input/output: 128/128
long prompt: 3072 tokens
long output: 32 tokens
chunk budget: 512
repeats: 3
CUDA Graph: enabled
```

Suggested command:

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
  --decode-aware-cadences 1,2,4 \
  --repeats 3 \
  --no-write \
  --output-prefix cadence_ablation_3090
```

## 5. Primary Metrics

| Metric | Why it matters |
| --- | --- |
| `active_decode_gap_s_max_mean` | main tail/interference metric |
| `active_decode_gap_s_max_min/max` | stability across repeats |
| `long_request_ttft_s_mean` | long-request progress trade-off |
| `post_injection_wall_time_s_mean` | overall completion window |
| `interleaved_runs` | confirms the policy actually changed schedule order |
| `prefill_interleave_every_n_chunks` | identifies N=1/2/4 cadence rows |
| `num_decode_aware_interleaves_min/max` | checks cadence execution |
| `post_injection_phase_runs` | human-readable schedule proof |
| `decode_batch_histogram` | checks whether decode batching is fragmented |
| `decode_cuda_graph_steps` | checks CUDA Graph path is preserved |
| `capacity_limited_runs` | invalidates latency interpretation if nonzero |

## 6. Expected Result

Expected shape:

```text
N=1: lowest active decode max gap, highest long TTFT
N=2: moderate gap reduction, less TTFT pressure
N=4: weaker gap reduction, closer to upstream long TTFT
```

This ablation is useful only if the curve is monotonic or at least explainable.
If measurements are noisy or inconsistent, do not build adaptive policy on top
of them.

## 7. Stop Criteria

Stop after the minimal N=1/2/4 comparison if:

- N=1 is clearly best and TTFT trade-off is acceptable;
- N=2 gives most of the gap improvement with less TTFT cost;
- N=4 collapses toward upstream behavior;
- results vary too much across repeats to support a cadence claim.

Do not proceed to adaptive Policy C until the static cadence trade-off is
understood.

Phase 3.9 is closed. N=2 is a balanced candidate in the current workload, not a
global default.
