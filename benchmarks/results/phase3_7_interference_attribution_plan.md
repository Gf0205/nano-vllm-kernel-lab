# Phase 3.7 Interference Result Attribution

Phase 3.7 exists to explain the observed wall-time gap in the chunked-prefill
interference workload before moving to profiling or kernel work.

## Question

The previous run showed the same token work:

- `total_prefill_tokens = 4096`
- `total_decode_tokens = 1047`
- full 3072-token injected prompt
- no capacity limiting

but wall time changed from roughly `1.87s` to `0.72s`.

The only safe conclusion so far is that chunking happened and the maximum
post-injection prefill step became smaller. The 2.6x output-throughput gap still
needs attribution.

## Added attribution metrics

`bench_chunked_prefill_interference.py` now reports:

- `total_prefill_wall_time_s`
- `total_decode_wall_time_s`
- `pre_injection_wall_time_s`
- `post_injection_wall_time_s`
- `pre_injection_prefill_wall_time_s`
- `post_injection_prefill_wall_time_s`
- `pre_injection_decode_wall_time_s`
- `post_injection_decode_wall_time_s`
- `decode_step_s_avg`
- `decode_step_s_p50`
- `decode_step_s_p95`
- `decode_step_s_max`
- `decode_batch_histogram`
- `decode_cuda_graph_steps`
- `decode_eager_steps`
- bounded `post_injection_timeline`

Each timeline row records:

- `step_id`
- `phase`
- `prefill_tokens`
- `decode_batch_size`
- `execution_path`
- `step_ms`
- `waiting`
- `running`
- `active_decode_unfinished`

## AutoDL command

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
  --timeline-limit 48 \
  --no-write \
  --output-prefix chunked_prefill_interference_3090
```

## Supported conclusions before this phase

- Full long prompt was used when `capacity_limited=False` and
  `long_prompt_shrunk=False`.
- Chunking happened when `num_chunked_prefill_steps > 0`.
- Single post-injection prefill step duration dropped in the chunked case.

## Not yet supported

Do not claim that the full wall-time gap is caused by smoother prefill/decode
interleaving until the new timeline explains where the saved time comes from.
