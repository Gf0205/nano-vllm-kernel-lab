# Benchmark Audit Summary

This report formally closes Phase 3.6. The goal was not to improve throughput,
but to validate whether the Phase 2 smoke benchmark numbers were safe to use as
performance claims.

## Original anomaly

The first Phase 2 smoke run reported:

| mode | batch | input len | output len | output tok/s |
| --- | --- | --- | --- | --- |
| CUDA Graph | 1 | 128 | 128 | 85.84 |
| CUDA Graph | 1 | 512 | 128 | 310.00 |
| eager | 1 | 128 | 128 | 24.00 |
| eager | 1 | 512 | 128 | 40.45 |

This looked suspicious because the model, batch size, and output length were the
same, but the longer prompt appeared much faster.

## Audit hypothesis

The anomaly was likely caused by benchmark order and warmup effects rather than
a real model/runtime property. Candidate causes included first-shape overhead,
CUDA Graph replay warmup, lazy initialization, and one-shot smoke measurement
noise.

## Experimental design

The audit benchmark kept the timed region to `llm.generate` only and recorded
`llm_init_time_s` separately. It also:

- synchronized CUDA before and after every timed generation;
- used `ignore_eos=True`;
- checked `total_output_tokens == expected_output_tokens`;
- ran the same cases in `natural`, `reverse`, and `shuffle` order;
- ran two repeats per order;
- compared CUDA Graph against eager mode.

## Output completeness

All audited rows had:

```text
output_complete = True
total_output_tokens == expected_output_tokens
```

Therefore the anomaly was not caused by early EOS, missing generated tokens, or
a denominator mismatch in output-token throughput.

## Audited results

The suspicious single-request cases became stable after order/repeat controls:

| mode | case | avg tok/s | min tok/s | max tok/s |
| --- | --- | --- | --- | --- |
| CUDA Graph | bs1_in128_out128 | ~313 | ~308 | ~316 |
| CUDA Graph | bs1_in512_out128 | ~313 | ~311 | ~314 |
| eager | bs1_in128_out128 | ~41 | ~40 | ~42 |
| eager | bs1_in512_out128 | ~41 | ~40 | ~42 |

The audited conclusion is that `input_len=512` is not materially faster than
`input_len=128` for `batch=1,output_len=128`.

## Invalidated old conclusion

The Phase 2 smoke result should not be used to claim that longer prompts improve
single-request decode throughput. That interpretation is invalidated by the
audit.

## Current supported conclusion

Phase 2 remains useful as a smoke test, but the audited baseline should be used
for performance discussion:

- CUDA Graph mode is still much faster than eager in steady-state generation.
- The gap should be described as an end-to-end generation-path improvement,
  including reduced Python/runtime overhead, not as pure kernel speedup.
- Single-request `input_len=128` and `input_len=512` have similar output-token
  throughput after warmup/order controls.

## Remaining caveats

- The audit is still based on synthetic token IDs, not real user prompts.
- The timed region excludes LLM initialization and CUDA Graph capture.
- The benchmark reports output-token throughput, not total input+output token
  throughput.
- TPOT is an aggregate decode-cadence estimate after average TTFT; it is not
  expected to equal the reciprocal of system throughput.

## Status

Phase 3.6 is closed. Future optimization work should use this audited baseline
or a workload-specific benchmark, not the original one-shot Phase 2 smoke table.
