import argparse
import os
import random
from pathlib import Path
from time import perf_counter

from utils import (
    add_repo_to_path,
    append_jsonl,
    collect_env,
    ensure_results_dir,
    make_token_ids,
    parse_int_list,
    peak_memory_gb,
    reset_peak_memory,
    timed,
    write_markdown_table,
)


add_repo_to_path()

from nanovllm import LLM, SamplingParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit benchmark ordering, warmup, and metric semantics.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--input-lens", default="128,512")
    parser.add_argument("--output-lens", default="128")
    parser.add_argument("--orders", default="natural,reverse,shuffle")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-prefix", default="benchmark_audit_smoke")
    parser.add_argument("--no-write", action="store_true", help="Print rows only; do not write jsonl/md result files.")
    return parser.parse_args()


def build_cases(args: argparse.Namespace) -> list[dict]:
    cases = []
    for batch_size in parse_int_list(args.batch_sizes):
        for input_len in parse_int_list(args.input_lens):
            for output_len in parse_int_list(args.output_lens):
                cases.append(
                    {
                        "case_id": f"bs{batch_size}_in{input_len}_out{output_len}",
                        "batch_size": batch_size,
                        "input_len": input_len,
                        "output_len": output_len,
                    }
                )
    return cases


def order_cases(cases: list[dict], order_name: str, seed: int) -> list[dict]:
    ordered = list(cases)
    if order_name == "natural":
        return ordered
    if order_name == "reverse":
        return list(reversed(ordered))
    if order_name == "shuffle":
        rng = random.Random(seed)
        rng.shuffle(ordered)
        return ordered
    raise ValueError(f"Unsupported order: {order_name}")


def main() -> None:
    args = parse_args()
    model = os.path.expanduser(args.model)
    results_dir = ensure_results_dir()
    jsonl_path = results_dir / f"{args.output_prefix}.jsonl"
    md_path = results_dir / f"{args.output_prefix}.md"

    init_start = perf_counter()
    llm = LLM(
        model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    init_time_s = perf_counter() - init_start
    vocab_size = llm.tokenizer.vocab_size

    env_record = {
        "type": "env",
        "env": collect_env(),
        "metric_notes": {
            "timed_region": "llm.generate only; LLM initialization and CUDA Graph capture are excluded",
            "throughput": "output tokens per wall-clock second for completed generated tokens",
            "sync": "torch.cuda.synchronize is called before and after each timed generation",
        },
    }
    if not args.no_write:
        append_jsonl(jsonl_path, env_record)

    try:
        if args.warmup_tokens > 0:
            # Keep warmup outside timed rows so order effects are easier to see.
            llm.generate(["Benchmark audit warmup"], SamplingParams(max_tokens=args.warmup_tokens), use_tqdm=False)

        rows = []
        cases = build_cases(args)
        for order_name in [name.strip() for name in args.orders.split(",") if name.strip()]:
            for repeat in range(args.repeats):
                ordered_cases = order_cases(cases, order_name, args.seed + repeat)
                for sequence_index, case in enumerate(ordered_cases):
                    prompts = make_token_ids(case["batch_size"], case["input_len"], vocab_size, args.seed + repeat)
                    sampling = SamplingParams(
                        temperature=args.temperature,
                        ignore_eos=True,
                        max_tokens=case["output_len"],
                    )
                    reset_peak_memory()
                    outputs, wall_time = timed(lambda: llm.generate(prompts, sampling, use_tqdm=False))
                    total_output_tokens = sum(len(item["token_ids"]) for item in outputs)
                    expected_output_tokens = case["batch_size"] * case["output_len"]
                    row = {
                        "model": Path(model).name,
                        "enforce_eager": args.enforce_eager,
                        "order": order_name,
                        "repeat": repeat,
                        "sequence_index": sequence_index,
                        "case_id": case["case_id"],
                        "batch_size": case["batch_size"],
                        "input_len": case["input_len"],
                        "output_len": case["output_len"],
                        "total_output_tokens": total_output_tokens,
                        "expected_output_tokens": expected_output_tokens,
                        "output_complete": total_output_tokens == expected_output_tokens,
                        "wall_time_s": round(wall_time, 4),
                        "output_tokens_per_s": round(total_output_tokens / wall_time, 2),
                        "peak_memory_gb": round(peak_memory_gb(), 3),
                        "llm_init_time_s": round(init_time_s, 4),
                    }
                    rows.append(row)
                    if not args.no_write:
                        append_jsonl(jsonl_path, {"type": "benchmark_audit", **row})
                    print(row)
    finally:
        llm.exit()

    if not args.no_write:
        write_markdown_table(
            md_path,
            rows,
            [
                "order",
                "repeat",
                "sequence_index",
                "case_id",
                "enforce_eager",
                "batch_size",
                "input_len",
                "output_len",
                "total_output_tokens",
                "expected_output_tokens",
                "output_complete",
                "wall_time_s",
                "output_tokens_per_s",
                "peak_memory_gb",
            ],
        )
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
