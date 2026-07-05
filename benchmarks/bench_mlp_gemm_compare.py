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


def parse_str_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def default_variants() -> list[str]:
    return ["linear", "matmul_t", "matmul_contiguous_t"]


def projection_shape(projection: str, num_tokens: int, hidden_size: int, intermediate_size: int) -> dict:
    if projection == "gate_up":
        return {
            "projection": projection,
            "input_shape": (num_tokens, hidden_size),
            "weight_shape": (intermediate_size * 2, hidden_size),
            "output_shape": (num_tokens, intermediate_size * 2),
        }
    if projection == "down":
        return {
            "projection": projection,
            "input_shape": (num_tokens, intermediate_size),
            "weight_shape": (hidden_size, intermediate_size),
            "output_shape": (num_tokens, hidden_size),
        }
    raise ValueError(f"Unsupported projection: {projection}")


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


def speedup_vs_baseline(baseline_ms: float, candidate_ms: float) -> float:
    if baseline_ms <= 0.0 or candidate_ms <= 0.0:
        return 0.0
    return round(baseline_ms / candidate_ms, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare torch/cuBLAS BF16 GEMM call variants for Qwen3 MLP shapes.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--token-cases", default="128,256,512,1024")
    parser.add_argument("--projections", default="gate_up,down")
    parser.add_argument("--variants", default=",".join(default_variants()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--timing-iters", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--output-prefix", default="mlp_gemm_compare")
    return parser.parse_args()


def make_tensors(shape: dict, dtype: torch.dtype, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    x = torch.randn(*shape["input_shape"], device=device, dtype=dtype)
    weight = torch.randn(*shape["weight_shape"], device=device, dtype=dtype)
    return x, weight


def run_variant(variant: str, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    if variant == "linear":
        return F.linear(x, weight)
    if variant == "matmul_t":
        return torch.matmul(x, weight.t())
    if variant == "matmul_contiguous_t":
        return torch.matmul(x, weight.t().contiguous())
    if variant == "matmul_pretransposed":
        return torch.matmul(x, weight.t().contiguous())
    raise ValueError(f"Unsupported variant: {variant}")


def time_variant(
    variant: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    warmup_iters: int,
    timing_iters: int,
) -> list[float]:
    import torch

    if variant == "matmul_pretransposed":
        weight_t = weight.t().contiguous()

        def fn():
            return torch.matmul(x, weight_t)
    else:
        def fn():
            return run_variant(variant, x, weight)

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
    projection: str,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    variants: list[str],
    warmup_iters: int,
    timing_iters: int,
    seed: int,
    rtol: float,
    atol: float,
) -> list[dict]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    torch.manual_seed(seed + num_tokens + (0 if projection == "gate_up" else 1000000))
    device = "cuda"
    shape = projection_shape(projection, num_tokens, hidden_size, intermediate_size)
    x, weight = make_tensors(shape, dtype, device)
    baseline = run_variant("linear", x, weight)
    rows = []
    baseline_avg = 0.0
    for variant in variants:
        out = run_variant(variant, x, weight)
        diff = (baseline.float() - out.float()).abs()
        max_abs_err = diff.max().item()
        mean_abs_err = diff.mean().item()
        passed = torch.allclose(baseline.float(), out.float(), rtol=rtol, atol=atol)
        if not passed:
            raise AssertionError(f"{projection=} {num_tokens=} {variant=} failed: {max_abs_err=}, {mean_abs_err=}")
        timings = time_variant(variant, x, weight, warmup_iters, timing_iters)
        row = {
            "projection": projection,
            "variant": variant,
            "num_tokens": num_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "dtype": str(dtype),
            "warmup_iters": warmup_iters,
            "timing_iters": timing_iters,
            "correctness_baseline": "linear",
            "passed": passed,
            "max_abs_err": round(max_abs_err, 6),
            "mean_abs_err": round(mean_abs_err, 6),
        }
        row.update(shape)
        row.update(summarize_ms(timings, "latency"))
        if variant == "linear":
            baseline_avg = row["latency_ms_avg"]
        row["speedup_vs_linear"] = speedup_vs_baseline(baseline_avg, row["latency_ms_avg"])
        row["tokens_per_s"] = round(num_tokens * 1000.0 / row["latency_ms_avg"], 2) if row["latency_ms_avg"] else 0.0
        rows.append(row)
    for row in rows:
        row["speedup_vs_linear"] = speedup_vs_baseline(baseline_avg, row["latency_ms_avg"])
    return rows


def main() -> None:
    args = parse_args()
    benchmarks_dir = Path(__file__).resolve().parent
    if str(benchmarks_dir) not in sys.path:
        sys.path.insert(0, str(benchmarks_dir))
    from utils import add_repo_to_path, append_jsonl, collect_env, ensure_results_dir, write_markdown_table

    add_repo_to_path()
    from nanovllm.config import Config

    model = os.path.expanduser(args.model)
    config = Config(model)
    hf_config = config.hf_config
    dtype = hf_config.dtype
    hidden_size = hf_config.hidden_size
    intermediate_size = hf_config.intermediate_size
    token_cases = parse_int_cases(args.token_cases)
    projections = parse_str_list(args.projections)
    variants = parse_str_list(args.variants)
    results_dir = ensure_results_dir()
    jsonl_path = results_dir / f"{args.output_prefix}.jsonl"
    md_path = results_dir / f"{args.output_prefix}.md"
    rows = []

    if not args.no_write:
        append_jsonl(jsonl_path, {"type": "env", "env": collect_env()})

    for projection in projections:
        for num_tokens in token_cases:
            case_rows = run_case(
                projection=projection,
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                variants=variants,
                warmup_iters=args.warmup_iters,
                timing_iters=args.timing_iters,
                seed=args.seed,
                rtol=args.rtol,
                atol=args.atol,
            )
            for row in case_rows:
                row["model"] = Path(model).name
                rows.append(row)
                print(row)
                if not args.no_write:
                    append_jsonl(jsonl_path, {"type": "mlp_gemm_compare", **row})

    if not args.no_write:
        columns = [
            "model",
            "projection",
            "variant",
            "num_tokens",
            "input_shape",
            "weight_shape",
            "output_shape",
            "dtype",
            "passed",
            "max_abs_err",
            "mean_abs_err",
            "latency_ms_avg",
            "latency_ms_p50",
            "latency_ms_p95",
            "speedup_vs_linear",
            "tokens_per_s",
        ]
        write_markdown_table(md_path, rows, columns)
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
