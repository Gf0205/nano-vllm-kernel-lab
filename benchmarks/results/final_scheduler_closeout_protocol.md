# Final Scheduler Evidence Protocol

Status: **implementation ready; RTX 3090 results pending**.

This is the final validation round for the nano-vLLM scheduler study. It does
not add an engine feature or change the decode-aware scheduling policy.

## Question

Does the decode-aware interleave effect remain visible when active decode
concurrency and injected prefill length change, and is the continuity/TTFT
trade-off reproducible across independent benchmark runs?

## Frozen Environment

- Model: Qwen3-0.6B
- GPU: one AutoDL RTX 3090
- Decode mode: CUDA Graph, unless a result explicitly reports eager fallback
- Active request input/output: 128/128 tokens
- Injected request output: 32 tokens
- Injection point: after 8 active decode steps
- Chunked-prefill token budget: 512
- Synthetic token inputs with recorded seed

## Workload Matrix

| Dimension | Values |
| --- | --- |
| active decode concurrency | 4, 8, 16 |
| injected prefill length | 1024, 3072 |
| policies | upstream chunked, decode-aware N=1, decode-aware N=2 |
| repeats | 5 per configuration |

The canonical `active=8, long_input=3072` workload additionally runs N=4 to
replace the previously truncated cadence evidence.

## Run On AutoDL

```bash
python benchmarks/bench_scheduler_closeout.py \
  --model /root/huggingface/Qwen3-0.6B \
  --active-decode-seqs 4,8,16 \
  --long-input-lens 1024,3072 \
  --repeats 5 \
  --output-prefix final_scheduler_robustness_3090
```

Do not add `--no-write`. This run must preserve the raw JSONL, complete console
logs, per-workload tables, and the combined matrix summary. Each child run
starts a fresh JSONL file so reruns cannot silently append stale records.

## Primary Metrics

- post-injection active decode service-gap p50/p95;
- per-run post-injection max active decode gap;
- median and p20/p80 of per-run max gap across repeats;
- long-request TTFT median and p20/p80;
- post-injection wall time;
- interleave count, capacity-limited/prompt-shrunk runs, and eager fallbacks.

The per-run max gap is the tail-continuity observation. Requests advanced in
the same batched decode step are not treated as independent samples.

## Decision Rules

The current scheduler result may be presented as a robustness-supported case
study only if:

1. all workloads complete without prompt shrinking or capacity limitation;
2. N=1 lowers the median per-run max post-injection gap versus upstream in all
   or nearly all tested workloads;
3. N=2 shows the expected intermediate continuity behavior often enough to
   support a workload-specific trade-off, not a universal default;
4. the TTFT and wall-time costs are reported even when noisy or unfavorable;
5. the full N=4 canonical evidence is preserved.

If these conditions fail, retain the original single-workload result and state
that robustness was not established. Do not tune the policy after seeing this
matrix; that would turn the closeout set into a training set.

After the run, package the evidence for review instead of pasting the long
terminal output:

```bash
tar -czf /root/autodl-tmp/final_scheduler_robustness_3090.tar.gz \
  benchmarks/results/final_scheduler_robustness_3090_*
```

## Publication Boundary

Even after a successful run, the supported claim remains limited to a
controlled Qwen3-0.6B / RTX 3090 synthetic interference study. It is not a
production scheduler evaluation, an online-serving SLO result, or evidence of
general improvement across vLLM/SGLang workloads.
