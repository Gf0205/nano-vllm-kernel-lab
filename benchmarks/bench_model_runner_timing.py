import argparse
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

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
from nanovllm.utils.context import get_context, reset_context  # noqa: E402


SEGMENTS = [
    "prepare_input",
    "prepare_sample",
    "graph_setup",
    "forward_or_graph",
    "logits",
    "sampler",
    "run_total",
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * q)))
    return values[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Break down ModelRunner.run into prepare/model/logits/sampler timing.")
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
    parser.add_argument("--output-prefix", default="model_runner_timing")
    return parser.parse_args()


def timed_cuda(fn):
    cuda_sync()
    start = perf_counter()
    out = fn()
    cuda_sync()
    return out, perf_counter() - start


@torch.inference_mode()
def run_model_breakdown(llm: LLM, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool) -> tuple[torch.Tensor, dict]:
    runner = llm.model_runner
    timings = {"graph_setup": 0.0}
    if is_prefill or runner.enforce_eager or input_ids.size(0) > 512:
        hidden_states, timings["forward_or_graph"] = timed_cuda(lambda: runner.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = runner.graphs[next(x for x in runner.graph_bs if x >= bs)]
        graph_vars = runner.graph_vars

        def setup_graph_inputs():
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        _, timings["graph_setup"] = timed_cuda(setup_graph_inputs)
        _, timings["forward_or_graph"] = timed_cuda(graph.replay)
        hidden_states = graph_vars["outputs"][:bs]

    logits, timings["logits"] = timed_cuda(lambda: runner.model.compute_logits(hidden_states))
    return logits, timings


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_s_sum": round(sum(values), 6),
        f"{prefix}_s_avg": round(mean(values), 6) if values else 0.0,
        f"{prefix}_s_p50": round(percentile(values, 0.50), 6),
        f"{prefix}_s_p95": round(percentile(values, 0.95), 6),
        f"{prefix}_s_max": round(max(values), 6) if values else 0.0,
    }


def summarize_phase(records: list[dict], phase: str) -> dict:
    rows = [record for record in records if record["phase"] == phase]
    out = {f"{phase}_steps": len(rows)}
    for segment in SEGMENTS:
        values = [row[segment] for row in rows]
        out.update(summarize(values, f"{phase}_{segment}"))
    if rows:
        max_row = max(rows, key=lambda row: row["run_total"])
        first_row = rows[0]
        out[f"{phase}_run_total_s_first"] = round(first_row["run_total"], 6)
        out[f"{phase}_run_total_s_max_step"] = max_row["phase_step"]
        out[f"{phase}_run_total_s_max_path"] = max_row["execution_path"]
        for segment in ["prepare_input", "graph_setup", "forward_or_graph", "logits", "sampler"]:
            out[f"{phase}_{segment}_s_first"] = round(first_row[segment], 6)
    return out


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

    if not args.no_write:
        append_jsonl(jsonl_path, {"type": "env", "env": collect_env()})

    records = []
    phase_steps = {"prefill": 0, "decode": 0}
    reset_peak_memory()
    cuda_sync()
    start = perf_counter()

    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        phase = "prefill" if is_prefill else "decode"
        phase_steps[phase] += 1
        record = {"phase": phase, "phase_step": phase_steps[phase]}

        run_start = perf_counter()
        if is_prefill:
            (input_ids, positions), record["prepare_input"] = timed_cuda(lambda: llm.model_runner.prepare_prefill(seqs))
        else:
            (input_ids, positions), record["prepare_input"] = timed_cuda(lambda: llm.model_runner.prepare_decode(seqs))

        temperatures, record["prepare_sample"] = timed_cuda(lambda: llm.model_runner.prepare_sample(seqs))
        logits, model_timings = run_model_breakdown(llm, input_ids, positions, is_prefill)
        record.update(model_timings)
        token_ids, record["sampler"] = timed_cuda(lambda: llm.model_runner.sampler(logits, temperatures).tolist())
        record["run_total"] = perf_counter() - run_start
        if is_prefill:
            record["execution_path"] = "prefill_eager"
        elif llm.model_runner.enforce_eager or input_ids.size(0) > 512:
            record["execution_path"] = "decode_eager"
        else:
            record["execution_path"] = "decode_cuda_graph"
        records.append(record)

        reset_context()
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)

    cuda_sync()
    wall_time = perf_counter() - start
    row = {
        "model": Path(model).name,
        "enforce_eager": args.enforce_eager,
        "num_seqs": args.num_seqs,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "wall_time_s": round(wall_time, 4),
        "peak_memory_gb": round(peak_memory_gb(), 3),
    }
    row.update(summarize_phase(records, "prefill"))
    row.update(summarize_phase(records, "decode"))

    if not args.no_write:
        append_jsonl(jsonl_path, {"type": "model_runner_timing", **row})
        write_markdown_table(md_path, [row], list(row.keys()))

    print(row)
    if not args.no_write:
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {md_path}")
    llm.exit()


if __name__ == "__main__":
    main()
