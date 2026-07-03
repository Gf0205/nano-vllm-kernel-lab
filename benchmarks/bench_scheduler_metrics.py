import argparse
import os
from pathlib import Path

from utils import (
    add_repo_to_path,
    append_jsonl,
    collect_env,
    ensure_results_dir,
    make_token_ids,
    peak_memory_gb,
    reset_peak_memory,
    timed,
    write_markdown_table,
)


add_repo_to_path()

from nanovllm import LLM, SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Scheduler prefill/decode/chunked-prefill metrics.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--normal-budget", type=int, default=16384)
    parser.add_argument("--chunked-budget", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-prefix", default="scheduler_metrics_smoke")
    return parser.parse_args()


def run_case(args: argparse.Namespace, case: str, budget: int, prompts: list[list[int]]) -> dict:
    model = os.path.expanduser(args.model)
    llm = LLM(
        model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=budget,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    try:
        reset_peak_memory()
        outputs, wall_time = timed(lambda: llm.generate(prompts, sampling, use_tqdm=False))
        total_output_tokens = sum(len(item["token_ids"]) for item in outputs)
        scheduler_metrics = llm.scheduler.metrics()
        block_metrics = llm.scheduler.block_manager.metrics()
        return {
            "case": case,
            "model": Path(model).name,
            "enforce_eager": args.enforce_eager,
            "num_seqs": len(prompts),
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_num_batched_tokens": budget,
            "total_output_tokens": total_output_tokens,
            "wall_time_s": round(wall_time, 4),
            "output_tokens_per_s": round(total_output_tokens / wall_time, 2),
            "peak_memory_gb": round(peak_memory_gb(), 3),
            **scheduler_metrics,
            "block_peak_used_blocks": block_metrics["peak_used_blocks"],
            "block_peak_reuse_ratio": block_metrics["peak_block_reuse_ratio"],
        }
    finally:
        llm.exit()


def main() -> None:
    args = parse_args()
    model = os.path.expanduser(args.model)
    results_dir = ensure_results_dir()
    jsonl_path = results_dir / f"{args.output_prefix}.jsonl"
    md_path = results_dir / f"{args.output_prefix}.md"

    # Reuse the same synthetic prompts across both cases so only the scheduler budget changes.
    vocab_size = AutoTokenizer.from_pretrained(model, use_fast=True).vocab_size
    prompts = make_token_ids(args.num_seqs, args.input_len, vocab_size, args.seed)

    append_jsonl(jsonl_path, {"type": "env", "env": collect_env()})
    rows = [
        run_case(args, "normal_prefill_budget", args.normal_budget, prompts),
        run_case(args, "chunked_prefill_budget", args.chunked_budget, prompts),
    ]
    for row in rows:
        append_jsonl(jsonl_path, {"type": "scheduler_metrics", **row})
        print(row)

    write_markdown_table(
        md_path,
        rows,
        [
            "case",
            "num_seqs",
            "input_len",
            "output_len",
            "max_num_batched_tokens",
            "wall_time_s",
            "output_tokens_per_s",
            "num_prefill_steps",
            "num_chunked_prefill_steps",
            "num_decode_steps",
            "max_prefill_batch_tokens",
            "avg_prefill_tokens_per_step",
            "max_decode_batch_size",
            "avg_decode_batch_size",
            "num_preemptions",
            "peak_waiting",
            "peak_running",
        ],
    )
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
