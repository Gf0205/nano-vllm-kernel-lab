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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure prefix-cache reuse with shared synthetic token prefixes.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--prefix-len", type=int, default=1024)
    parser.add_argument("--suffix-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-prefix", default="prefix_cache_smoke")
    return parser.parse_args()


def metric_delta(before: dict, after: dict, keys: list[str]) -> dict:
    return {key: after[key] - before[key] for key in keys}


def make_shared_prefix_prompts(vocab_size: int, num_seqs: int, prefix_len: int, suffix_len: int, seed: int):
    prefix = make_token_ids(1, prefix_len, vocab_size, seed)[0]
    suffixes = make_token_ids(num_seqs, suffix_len, vocab_size, seed + 1)
    return [prefix + suffix for suffix in suffixes]


def run_case(llm: LLM, name: str, prompts: list[list[int]], sampling: SamplingParams) -> dict:
    bm = llm.scheduler.block_manager
    before = bm.metrics()
    reset_peak_memory()
    outputs, wall_time = timed(lambda: llm.generate(prompts, sampling, use_tqdm=False))
    after = bm.metrics()
    total_output_tokens = sum(len(item["token_ids"]) for item in outputs)
    delta = metric_delta(
        before,
        after,
        [
            "allocation_requests",
            "logical_blocks_allocated",
            "physical_block_allocations",
            "prefix_cache_eligible_blocks",
            "prefix_cache_hit_blocks",
        ],
    )
    # Prefix-cache effectiveness is a delta metric because the engine is reused.
    hit_rate = (
        delta["prefix_cache_hit_blocks"] / delta["prefix_cache_eligible_blocks"]
        if delta["prefix_cache_eligible_blocks"] else 0.0
    )
    return {
        "case": name,
        "num_seqs": len(prompts),
        "prompt_len": len(prompts[0]) if prompts else 0,
        "total_output_tokens": total_output_tokens,
        "wall_time_s": round(wall_time, 4),
        "output_tokens_per_s": round(total_output_tokens / wall_time, 2),
        "peak_memory_gb": round(peak_memory_gb(), 3),
        **delta,
        "prefix_cache_hit_rate": round(hit_rate, 4),
        "cached_block_entries_after": after["cached_block_entries"],
        "peak_block_reuse_ratio_after": after["peak_block_reuse_ratio"],
        "peak_shared_blocks_after": after["peak_shared_blocks"],
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
    vocab_size = llm.tokenizer.vocab_size
    sampling = SamplingParams(
        temperature=args.temperature,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    shared_prompts = make_shared_prefix_prompts(
        vocab_size, args.num_seqs, args.prefix_len, args.suffix_len, args.seed
    )
    unique_prompts = make_token_ids(
        args.num_seqs, args.prefix_len + args.suffix_len, vocab_size, args.seed + 100
    )

    append_jsonl(jsonl_path, {"type": "env", "env": collect_env()})

    # Warm one shared-prefix request so its full prefix blocks are hashed.
    warm_sampling = SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=1)
    llm.generate([shared_prompts[0]], warm_sampling, use_tqdm=False)

    rows = [
        run_case(llm, "shared_prefix_after_warmup", shared_prompts, sampling),
        run_case(llm, "unique_prefix_baseline", unique_prompts, sampling),
    ]
    for row in rows:
        append_jsonl(jsonl_path, {"type": "prefix_cache", "model": Path(model).name, **row})
        print(row)

    write_markdown_table(
        md_path,
        rows,
        [
            "case",
            "num_seqs",
            "prompt_len",
            "total_output_tokens",
            "wall_time_s",
            "output_tokens_per_s",
            "peak_memory_gb",
            "prefix_cache_eligible_blocks",
            "prefix_cache_hit_blocks",
            "prefix_cache_hit_rate",
            "physical_block_allocations",
            "peak_block_reuse_ratio_after",
            "peak_shared_blocks_after",
        ],
    )
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
