import argparse
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

from utils import (
    LatencyRecord,
    add_repo_to_path,
    append_jsonl,
    collect_env,
    ensure_results_dir,
    make_token_ids,
    peak_memory_gb,
    reset_peak_memory,
    write_markdown_table,
)


add_repo_to_path()

from nanovllm import LLM, SamplingParams  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * q)))
    return values[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nano-vLLM latency baseline benchmark.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-prefix", default="latency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = os.path.expanduser(args.model)
    results_dir = ensure_results_dir()
    jsonl_path = results_dir / f"{args.output_prefix}.jsonl"
    md_path = results_dir / f"{args.output_prefix}.md"

    llm = LLM(
        model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    prompts = make_token_ids(args.num_seqs, args.input_len, llm.tokenizer.vocab_size, args.seed)
    sampling = SamplingParams(
        temperature=args.temperature,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    for prompt in prompts:
        llm.add_request(prompt, sampling)
    # Keep direct Sequence references so we can observe first-token arrival.
    tracked = list(llm.scheduler.waiting)

    append_jsonl(jsonl_path, {"type": "env", "env": collect_env()})
    reset_peak_memory()
    start = perf_counter()
    first_token_times: dict[int, float] = {}
    last_completion_total = 0

    while not llm.is_finished():
        llm.step()
        now = perf_counter()
        for seq in tracked:
            if seq.seq_id not in first_token_times and seq.num_completion_tokens > 0:
                first_token_times[seq.seq_id] = now - start
        last_completion_total = sum(seq.num_completion_tokens for seq in tracked)

    wall_time = perf_counter() - start
    ttfts = list(first_token_times.values())
    # TPOT excludes the first-token wait, so it reflects decode cadence after TTFT.
    avg_ttft = mean(ttfts) if ttfts else 0.0
    tpot_denominator = max(1, last_completion_total - len(tracked))
    tpot = max(0.0, wall_time - avg_ttft) / tpot_denominator
    record = LatencyRecord(
        model=Path(model).name,
        enforce_eager=args.enforce_eager,
        num_seqs=args.num_seqs,
        input_len=args.input_len,
        output_len=args.output_len,
        total_output_tokens=last_completion_total,
        wall_time_s=round(wall_time, 4),
        ttft_s_avg=round(avg_ttft, 4),
        ttft_s_p50=round(percentile(ttfts, 0.50), 4),
        ttft_s_p90=round(percentile(ttfts, 0.90), 4),
        tpot_s=round(tpot, 6),
        output_tokens_per_s=round(last_completion_total / wall_time, 2),
        peak_memory_gb=round(peak_memory_gb(), 3),
    )
    row = record.to_dict()
    append_jsonl(jsonl_path, {"type": "latency", **row})
    write_markdown_table(md_path, [row], list(row.keys()))
    print(row)
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
