import argparse
import gc
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
import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class CapacityLimitError(RuntimeError):
    pass


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * q)))
    return values[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure long-prefill interference with active decode requests.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--active-decode-seqs", type=int, default=8)
    parser.add_argument("--active-input-len", type=int, default=128)
    parser.add_argument("--active-output-len", type=int, default=128)
    parser.add_argument("--long-input-len", type=int, default=3072)
    parser.add_argument("--long-output-len", type=int, default=32)
    parser.add_argument("--inject-after-decode-steps", type=int, default=8)
    parser.add_argument("--normal-budget", type=int, default=8192)
    parser.add_argument("--chunked-budget", type=int, default=512)
    parser.add_argument("--long-decode-reserve-blocks", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-write", action="store_true", help="Print rows only; do not write jsonl/md result files.")
    parser.add_argument("--output-prefix", default="chunked_prefill_interference")
    return parser.parse_args()


def validate_lengths(args: argparse.Namespace, model: str) -> int:
    hf_config = AutoConfig.from_pretrained(model)
    effective_max_model_len = min(args.max_model_len, hf_config.max_position_embeddings)
    if args.long_decode_reserve_blocks < 0:
        raise ValueError("--long-decode-reserve-blocks must be non-negative")
    active_total_len = args.active_input_len + args.active_output_len
    long_total_len = args.long_input_len + args.long_output_len
    if active_total_len > effective_max_model_len:
        raise ValueError(
            "active_input_len + active_output_len must fit the model context: "
            f"{active_total_len} > {effective_max_model_len}"
        )
    if long_total_len > effective_max_model_len:
        raise ValueError(
            "long_input_len + long_output_len must fit the model context: "
            f"{long_total_len} > {effective_max_model_len}. "
            "For Qwen3-0.6B on this project, use --long-input-len 3072 with --long-output-len 32."
        )
    return effective_max_model_len


def add_tracked_request(llm: LLM, prompt: list[int], sampling: SamplingParams):
    llm.add_request(prompt, sampling)
    return llm.scheduler.waiting[-1]


def fit_prompt_to_free_blocks(
    llm: LLM,
    prompt: list[int],
    output_len: int,
    reserve_blocks: int,
) -> tuple[list[int], bool]:
    block_manager = llm.scheduler.block_manager
    free_blocks = len(block_manager.free_block_ids)
    usable_blocks = free_blocks - reserve_blocks
    if usable_blocks <= 0:
        raise CapacityLimitError(
            f"No KV blocks left for the injected long prompt: free_blocks={free_blocks}, "
            f"reserve_blocks={reserve_blocks}."
        )
    # BlockManager currently allocates all prompt blocks up front even when the
    # prefill compute is chunked, so this benchmark trims only when capacity
    # would otherwise make the scheduler hit its empty-schedule assertion.
    max_prompt_tokens = usable_blocks * block_manager.block_size - output_len
    if len(prompt) <= max_prompt_tokens:
        return prompt, False
    if max_prompt_tokens <= 0:
        raise CapacityLimitError(
            f"Injected prompt plus output cannot fit: free_blocks={free_blocks}, "
            f"reserve_blocks={reserve_blocks}, output_len={output_len}."
        )
    return prompt[:max_prompt_tokens], True


def run_step(llm: LLM) -> tuple[float, bool]:
    cuda_sync()
    start = perf_counter()
    _, num_tokens = llm.step()
    cuda_sync()
    return perf_counter() - start, num_tokens > 0


def run_case(args: argparse.Namespace, case: str, budget: int, vocab_size: int) -> dict:
    model = os.path.expanduser(args.model)
    llm = LLM(
        model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=budget,
    )
    active_sampling = SamplingParams(
        temperature=args.temperature,
        ignore_eos=True,
        max_tokens=args.active_output_len,
    )
    long_sampling = SamplingParams(
        temperature=args.temperature,
        ignore_eos=True,
        max_tokens=args.long_output_len,
    )
    active_prompts = make_token_ids(args.active_decode_seqs, args.active_input_len, vocab_size, args.seed)
    long_prompt = make_token_ids(1, args.long_input_len, vocab_size, args.seed + 100_000)[0]
    active_seqs = [add_tracked_request(llm, prompt, active_sampling) for prompt in active_prompts]

    reset_peak_memory()
    active_decode_steps_before_inject = 0
    long_seq = None
    long_injected_at = 0.0
    long_ttft_s = 0.0
    last_active_progress_time = None
    active_decode_gaps = []
    active_decode_step_durations = []
    prefill_step_durations_after_inject = []
    injected = False
    effective_long_input_len = len(long_prompt)
    long_prompt_shrunk = False
    capacity_limited = False
    capacity_limit_reason = ""
    kv_total_blocks = 0
    kv_free_blocks_at_inject = 0

    start = perf_counter()
    try:
        while not llm.is_finished():
            before_active_tokens = sum(seq.num_completion_tokens for seq in active_seqs)
            before_long_tokens = long_seq.num_completion_tokens if long_seq is not None else 0

            step_duration, is_prefill = run_step(llm)
            now = perf_counter()

            after_active_tokens = sum(seq.num_completion_tokens for seq in active_seqs)
            active_progressed = after_active_tokens > before_active_tokens and not is_prefill
            if active_progressed:
                if last_active_progress_time is not None:
                    active_decode_gaps.append(now - last_active_progress_time)
                last_active_progress_time = now
                active_decode_step_durations.append(step_duration)
                if not injected:
                    active_decode_steps_before_inject += 1

            if injected and is_prefill:
                prefill_step_durations_after_inject.append(step_duration)

            if (
                not injected
                and active_decode_steps_before_inject >= args.inject_after_decode_steps
                and all(not seq.is_finished for seq in active_seqs)
            ):
                block_manager = llm.scheduler.block_manager
                kv_total_blocks = len(block_manager.blocks)
                kv_free_blocks_at_inject = len(block_manager.free_block_ids)
                try:
                    fitted_long_prompt, long_prompt_shrunk = fit_prompt_to_free_blocks(
                        llm,
                        long_prompt,
                        args.long_output_len,
                        args.long_decode_reserve_blocks,
                    )
                except CapacityLimitError as exc:
                    capacity_limited = True
                    capacity_limit_reason = str(exc)
                    injected = True
                    continue
                effective_long_input_len = len(fitted_long_prompt)
                long_seq = add_tracked_request(llm, fitted_long_prompt, long_sampling)
                long_injected_at = perf_counter()
                injected = True

            if long_seq is not None and long_ttft_s == 0.0 and before_long_tokens == 0 and long_seq.num_completion_tokens > 0:
                long_ttft_s = now - long_injected_at

        wall_time = perf_counter() - start
        active_output_tokens = sum(seq.num_completion_tokens for seq in active_seqs)
        long_output_tokens = long_seq.num_completion_tokens if long_seq is not None else 0
        scheduler_metrics = llm.scheduler.metrics()
        return {
            "case": case,
            "model": Path(model).name,
            "enforce_eager": args.enforce_eager,
            "max_num_batched_tokens": budget,
            "active_decode_seqs": args.active_decode_seqs,
            "active_input_len": args.active_input_len,
            "active_output_len": args.active_output_len,
            "requested_long_input_len": args.long_input_len,
            "effective_long_input_len": effective_long_input_len,
            "long_prompt_shrunk": long_prompt_shrunk,
            "long_output_len": args.long_output_len,
            "long_decode_reserve_blocks": args.long_decode_reserve_blocks,
            "inject_after_decode_steps": args.inject_after_decode_steps,
            "capacity_limited": capacity_limited,
            "capacity_limit_reason": capacity_limit_reason,
            "kv_total_blocks": kv_total_blocks or len(llm.scheduler.block_manager.blocks),
            "kv_free_blocks_at_inject": kv_free_blocks_at_inject,
            "wall_time_s": round(wall_time, 4),
            "active_output_tokens": active_output_tokens,
            "long_output_tokens": long_output_tokens,
            "output_tokens_per_s": round((active_output_tokens + long_output_tokens) / wall_time, 2),
            "active_decode_gap_s_avg": round(mean(active_decode_gaps), 6) if active_decode_gaps else 0.0,
            "active_decode_gap_s_p95": round(percentile(active_decode_gaps, 0.95), 6),
            "active_decode_gap_s_max": round(max(active_decode_gaps), 6) if active_decode_gaps else 0.0,
            "active_decode_step_s_avg": round(mean(active_decode_step_durations), 6) if active_decode_step_durations else 0.0,
            "long_request_ttft_s": round(long_ttft_s, 6),
            "post_inject_prefill_step_s_max": round(max(prefill_step_durations_after_inject), 6)
            if prefill_step_durations_after_inject
            else 0.0,
            "peak_memory_gb": round(peak_memory_gb(), 3),
            **scheduler_metrics,
        }
    finally:
        llm.exit()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    model = os.path.expanduser(args.model)
    effective_max_model_len = validate_lengths(args, model)
    results_dir = ensure_results_dir()
    jsonl_path = results_dir / f"{args.output_prefix}.jsonl"
    md_path = results_dir / f"{args.output_prefix}.md"

    vocab_size = AutoTokenizer.from_pretrained(model, use_fast=True).vocab_size
    rows = [
        run_case(args, "non_chunked_long_prefill", args.normal_budget, vocab_size),
        run_case(args, "chunked_long_prefill", args.chunked_budget, vocab_size),
    ]

    if not args.no_write:
        append_jsonl(
            jsonl_path,
            {
                "type": "env",
                "env": collect_env(),
                "effective_max_model_len": effective_max_model_len,
            },
        )

    for row in rows:
        if not args.no_write:
            append_jsonl(jsonl_path, {"type": "chunked_prefill_interference", **row})
        print(row)

    if not args.no_write:
        write_markdown_table(
            md_path,
            rows,
            [
                "case",
                "max_num_batched_tokens",
                "requested_long_input_len",
                "effective_long_input_len",
                "long_prompt_shrunk",
                "capacity_limited",
                "kv_total_blocks",
                "kv_free_blocks_at_inject",
                "wall_time_s",
                "output_tokens_per_s",
                "active_decode_gap_s_avg",
                "active_decode_gap_s_p95",
                "active_decode_gap_s_max",
                "long_request_ttft_s",
                "post_inject_prefill_step_s_max",
                "num_prefill_steps",
                "num_chunked_prefill_steps",
                "num_decode_steps",
                "peak_waiting",
                "peak_running",
            ],
        )
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
