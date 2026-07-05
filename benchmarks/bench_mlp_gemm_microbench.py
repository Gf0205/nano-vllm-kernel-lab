from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def parse_int_cases(value: str) -> list[int]:
    cases = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not cases:
        raise argparse.ArgumentTypeError("At least one token-count case is required.")
    if any(case <= 0 for case in cases):
        raise argparse.ArgumentTypeError("Token-count cases must be positive integers.")
    return cases


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * q)))
    return values[idx]


def summarize_ms(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_ms_avg": round(mean(values), 4) if values else 0.0,
        f"{prefix}_ms_p50": round(median(values), 4) if values else 0.0,
        f"{prefix}_ms_p95": round(percentile(values, 0.95), 4),
        f"{prefix}_ms_min": round(min(values), 4) if values else 0.0,
        f"{prefix}_ms_max": round(max(values), 4) if values else 0.0,
    }


def percent_of_total(value: float, total: float) -> float:
    return round(value * 100.0 / total, 4) if total else 0.0


def mlp_projection_shapes(hidden_size: int, intermediate_size: int, num_tokens: int) -> dict[str, tuple[int, ...]]:
    return {
        "input_shape": (num_tokens, hidden_size),
        "gate_up_weight_shape": (intermediate_size * 2, hidden_size),
        "gate_up_output_shape": (num_tokens, intermediate_size * 2),
        "activation_output_shape": (num_tokens, intermediate_size),
        "down_weight_shape": (hidden_size, intermediate_size),
        "down_output_shape": (num_tokens, hidden_size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Qwen3 MLP BF16 GEMM microbenchmark.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument(
        "--token-cases",
        default="1,8,16,32,64,128,256,512,1024",
        help="Comma-separated token counts for one MLP call.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--timing-iters", type=int, default=100)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--output-prefix", default="mlp_gemm_microbench")
    return parser.parse_args()


def make_weights(
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    gate_up_weight = torch.randn(intermediate_size * 2, hidden_size, device=device, dtype=dtype)
    down_weight = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    return gate_up_weight, down_weight


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def mlp_forward(x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    gate_up = F.linear(x, gate_up_weight)
    activated = silu_and_mul(gate_up)
    return F.linear(activated, down_weight)


def cuda_time_ms(fn, warmup_iters: int, timing_iters: int) -> list[float]:
    import torch

    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(timing_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return timings


def run_case(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    warmup_iters: int,
    timing_iters: int,
    seed: int,
) -> dict:
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    torch.manual_seed(seed + num_tokens)
    device = "cuda"
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    gate_up_weight, down_weight = make_weights(hidden_size, intermediate_size, dtype, device)
    gate_up = F.linear(x, gate_up_weight)
    activated = silu_and_mul(gate_up)

    gate_up_times = cuda_time_ms(lambda: F.linear(x, gate_up_weight), warmup_iters, timing_iters)
    act_times = cuda_time_ms(lambda: silu_and_mul(gate_up), warmup_iters, timing_iters)
    down_times = cuda_time_ms(lambda: F.linear(activated, down_weight), warmup_iters, timing_iters)
    full_times = cuda_time_ms(lambda: mlp_forward(x, gate_up_weight, down_weight), warmup_iters, timing_iters)

    row = {
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": str(dtype),
        "warmup_iters": warmup_iters,
        "timing_iters": timing_iters,
    }
    row.update(mlp_projection_shapes(hidden_size, intermediate_size, num_tokens))
    row.update(summarize_ms(gate_up_times, "gate_up"))
    row.update(summarize_ms(act_times, "silu_mul"))
    row.update(summarize_ms(down_times, "down"))
    row.update(summarize_ms(full_times, "full_mlp"))
    full_avg = row["full_mlp_ms_avg"]
    row["gate_up_pct_of_full"] = percent_of_total(row["gate_up_ms_avg"], full_avg)
    row["silu_mul_pct_of_full"] = percent_of_total(row["silu_mul_ms_avg"], full_avg)
    row["down_pct_of_full"] = percent_of_total(row["down_ms_avg"], full_avg)
    row["gate_up_tokens_per_s"] = round(num_tokens * 1000.0 / row["gate_up_ms_avg"], 2) if row["gate_up_ms_avg"] else 0.0
    row["down_tokens_per_s"] = round(num_tokens * 1000.0 / row["down_ms_avg"], 2) if row["down_ms_avg"] else 0.0
    row["full_mlp_tokens_per_s"] = round(num_tokens * 1000.0 / full_avg, 2) if full_avg else 0.0
    return row


def main() -> None:
    args = parse_args()
    benchmarks_dir = Path(__file__).resolve().parent
    if str(benchmarks_dir) not in sys.path:
        sys.path.insert(0, str(benchmarks_dir))
    from utils import add_repo_to_path, append_jsonl, collect_env, ensure_results_dir, write_markdown_table

    add_repo_to_path()
    import torch
    from nanovllm.config import Config

    model = os.path.expanduser(args.model)
    config = Config(model)
    hf_config = config.hf_config
    dtype = hf_config.dtype
    hidden_size = hf_config.hidden_size
    intermediate_size = hf_config.intermediate_size
    token_cases = parse_int_cases(args.token_cases)
    results_dir = ensure_results_dir()
    jsonl_path = results_dir / f"{args.output_prefix}.jsonl"
    md_path = results_dir / f"{args.output_prefix}.md"
    rows = []

    if not args.no_write:
        append_jsonl(jsonl_path, {"type": "env", "env": collect_env()})

    for num_tokens in token_cases:
        row = run_case(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            warmup_iters=args.warmup_iters,
            timing_iters=args.timing_iters,
            seed=args.seed,
        )
        row["model"] = Path(model).name
        rows.append(row)
        print(row)
        if not args.no_write:
            append_jsonl(jsonl_path, {"type": "mlp_gemm_microbench", **row})

    if not args.no_write:
        columns = [
            "model",
            "num_tokens",
            "hidden_size",
            "intermediate_size",
            "dtype",
            "gate_up_weight_shape",
            "down_weight_shape",
            "gate_up_ms_avg",
            "silu_mul_ms_avg",
            "down_ms_avg",
            "full_mlp_ms_avg",
            "gate_up_pct_of_full",
            "silu_mul_pct_of_full",
            "down_pct_of_full",
            "gate_up_tokens_per_s",
            "down_tokens_per_s",
            "full_mlp_tokens_per_s",
        ]
        write_markdown_table(md_path, rows, columns)
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
