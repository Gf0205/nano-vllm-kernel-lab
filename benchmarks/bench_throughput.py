import argparse
import os
from pathlib import Path

from utils import (
    ThroughputRecord,
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
    parser = argparse.ArgumentParser(description="Nano-vLLM throughput baseline benchmark.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--batch-sizes", default="1,8,32,128")
    parser.add_argument("--input-lens", default="128,512,1024")
    parser.add_argument("--output-lens", default="128,512")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-prefix", default="throughput")
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
    vocab_size = llm.tokenizer.vocab_size

    if args.warmup:
        llm.generate(["Benchmark warmup"], SamplingParams(max_tokens=8), use_tqdm=False)

    rows = []
    env = collect_env()
    append_jsonl(jsonl_path, {"type": "env", "env": env})

    for batch_size in parse_int_list(args.batch_sizes):
        for input_len in parse_int_list(args.input_lens):
            for output_len in parse_int_list(args.output_lens):
                prompts = make_token_ids(batch_size, input_len, vocab_size, args.seed)
                sampling = SamplingParams(
                    temperature=args.temperature,
                    ignore_eos=True,
                    max_tokens=output_len,
                )
                reset_peak_memory()
                outputs, wall_time = timed(lambda: llm.generate(prompts, sampling, use_tqdm=False))
                total_output_tokens = sum(len(item["token_ids"]) for item in outputs)
                record = ThroughputRecord(
                    model=Path(model).name,
                    enforce_eager=args.enforce_eager,
                    batch_size=batch_size,
                    input_len=input_len,
                    output_len=output_len,
                    total_output_tokens=total_output_tokens,
                    wall_time_s=round(wall_time, 4),
                    output_tokens_per_s=round(total_output_tokens / wall_time, 2),
                    peak_memory_gb=round(peak_memory_gb(), 3),
                )
                row = record.to_dict()
                rows.append(row)
                append_jsonl(jsonl_path, {"type": "throughput", **row})
                print(row)

    write_markdown_table(
        md_path,
        rows,
        [
            "model",
            "enforce_eager",
            "batch_size",
            "input_len",
            "output_len",
            "total_output_tokens",
            "wall_time_s",
            "output_tokens_per_s",
            "peak_memory_gb",
        ],
    )
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
