import argparse
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

from utils import (
    add_repo_to_path,
    append_jsonl,
    collect_env,
    cuda_sync,
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
    parser = argparse.ArgumentParser(description="Internal timing breakdown for nano-vLLM steps.")
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
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--output-prefix", default="internal_timing")
    return parser.parse_args()


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_s_sum": round(sum(values), 6),
        f"{prefix}_s_avg": round(mean(values), 6) if values else 0.0,
        f"{prefix}_s_p50": round(percentile(values, 0.50), 6),
        f"{prefix}_s_p95": round(percentile(values, 0.95), 6),
        f"{prefix}_s_max": round(max(values), 6) if values else 0.0,
    }


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

    timings = {
        "prefill_schedule": [],
        "prefill_model": [],
        "prefill_postprocess": [],
        "prefill_step": [],
        "decode_schedule": [],
        "decode_model": [],
        "decode_postprocess": [],
        "decode_step": [],
    }
    prefill_tokens = 0
    decode_tokens = 0
    prefill_steps = 0
    decode_steps = 0

    if not args.no_write:
        append_jsonl(jsonl_path, {"type": "env", "env": collect_env()})

    reset_peak_memory()
    cuda_sync()
    start = perf_counter()
    while not llm.is_finished():
        step_start = perf_counter()

        schedule_start = perf_counter()
        seqs, is_prefill = llm.scheduler.schedule()
        schedule_s = perf_counter() - schedule_start
        scheduled_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs)

        cuda_sync()
        model_start = perf_counter()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        cuda_sync()
        model_s = perf_counter() - model_start

        post_start = perf_counter()
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)
        post_s = perf_counter() - post_start

        step_s = perf_counter() - step_start
        phase = "prefill" if is_prefill else "decode"
        timings[f"{phase}_schedule"].append(schedule_s)
        timings[f"{phase}_model"].append(model_s)
        timings[f"{phase}_postprocess"].append(post_s)
        timings[f"{phase}_step"].append(step_s)
        if is_prefill:
            prefill_steps += 1
            prefill_tokens += scheduled_tokens
        else:
            decode_steps += 1
            decode_tokens += scheduled_tokens

    cuda_sync()
    wall_time = perf_counter() - start
    row = {
        "model": Path(model).name,
        "enforce_eager": args.enforce_eager,
        "num_seqs": args.num_seqs,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "wall_time_s": round(wall_time, 4),
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "output_tokens_per_s": round(decode_tokens / wall_time, 2) if wall_time > 0 else 0.0,
        "peak_memory_gb": round(peak_memory_gb(), 3),
    }
    for key, values in timings.items():
        row.update(summarize(values, key))

    if not args.no_write:
        append_jsonl(jsonl_path, {"type": "internal_timing", **row})
        write_markdown_table(md_path, [row], list(row.keys()))

    print(row)
    if not args.no_write:
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {md_path}")

    llm.exit()


if __name__ == "__main__":
    main()
