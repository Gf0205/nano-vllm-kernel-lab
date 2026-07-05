from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from nanovllm.config import Config


def parse_cases(value: str) -> list[tuple[int, int]]:
    cases = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise argparse.ArgumentTypeError(f"Expected BxC case, got {item!r}")
        batch, context = item.split("x", 1)
        cases.append((int(batch), int(context)))
    if not cases:
        raise argparse.ArgumentTypeError("At least one BxC case is required.")
    return cases


def required_num_blocks(batch_size: int, context_len: int, block_size: int) -> int:
    return batch_size * math.ceil(context_len / block_size)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * q)))
    return values[idx]


def summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "latency_ms_avg": round(mean(values), 4) if values else 0.0,
        "latency_ms_p50": round(median(values), 4) if values else 0.0,
        "latency_ms_p95": round(percentile(values, 0.95), 4),
        "latency_ms_min": round(min(values), 4) if values else 0.0,
        "latency_ms_max": round(max(values), 4) if values else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone FlashAttention paged decode microbenchmark.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument(
        "--cases",
        default="1x128,4x513,8x512,16x512,32x512,32x1024,64x512",
        help="Comma-separated batch/context pairs, for example: 4x513,32x512.",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--timing-iters", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--no-reference", action="store_true", help="Skip the PyTorch reference correctness check.")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--output-prefix", default="attention_decode_microbench")
    return parser.parse_args()


def repeat_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    import torch

    batch, seq_len, num_kv_heads, head_dim = x.shape
    if num_heads == num_kv_heads:
        return x
    repeat = num_heads // num_kv_heads
    return x[:, :, :, None, :].expand(batch, seq_len, num_kv_heads, repeat, head_dim).reshape(
        batch, seq_len, num_heads, head_dim
    )


def reference_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    import torch

    batch, num_heads, _ = q.shape
    block_size = k_cache.size(1)
    outputs = []
    for b in range(batch):
        seq_len = int(context_lens[b].item())
        keys = []
        values = []
        for token_idx in range(seq_len):
            table_idx = token_idx // block_size
            offset = token_idx % block_size
            block_id = int(block_tables[b, table_idx].item())
            keys.append(k_cache[block_id, offset])
            values.append(v_cache[block_id, offset])
        k = repeat_kv(torch.stack(keys, dim=0).unsqueeze(0).float(), num_heads)[0]
        v = repeat_kv(torch.stack(values, dim=0).unsqueeze(0).float(), num_heads)[0]
        scores = torch.einsum("hd,shd->hs", q[b].float(), k) * scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hs,shd->hd", probs, v))
    return torch.stack(outputs, dim=0).to(q.dtype)


def make_case_tensors(
    batch_size: int,
    context_len: int,
    block_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch

    blocks_per_seq = math.ceil(context_len / block_size)
    num_blocks = required_num_blocks(batch_size, context_len, block_size)
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).view(batch_size, blocks_per_seq)
    context_lens = torch.full((batch_size,), context_len, device=device, dtype=torch.int32)
    return q, k_cache, v_cache, block_tables, context_lens


def flash_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    from flash_attn import flash_attn_with_kvcache

    return flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=context_lens,
        block_table=block_tables,
        softmax_scale=scale,
        causal=True,
    ).squeeze(1)


def time_flash_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    warmup_iters: int,
    timing_iters: int,
) -> list[float]:
    import torch

    for _ in range(warmup_iters):
        flash_decode(q, k_cache, v_cache, block_tables, context_lens, scale)
    torch.cuda.synchronize()

    timings = []
    for _ in range(timing_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        flash_decode(q, k_cache, v_cache, block_tables, context_lens, scale)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return timings


def run_case(
    batch_size: int,
    context_len: int,
    config: Config,
    block_size: int,
    warmup_iters: int,
    timing_iters: int,
    seed: int,
    check_reference: bool,
    rtol: float,
    atol: float,
) -> dict:
    import torch

    hf_config = config.hf_config
    dtype = hf_config.dtype
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads must be divisible by num_kv_heads: {num_heads=} {num_kv_heads=}")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    torch.manual_seed(seed + batch_size * 100000 + context_len)
    device = "cuda"
    q, k_cache, v_cache, block_tables, context_lens = make_case_tensors(
        batch_size,
        context_len,
        block_size,
        num_heads,
        num_kv_heads,
        head_dim,
        dtype,
        device,
    )
    scale = head_dim**-0.5
    flash_out = flash_decode(q, k_cache, v_cache, block_tables, context_lens, scale)
    passed = True
    max_abs_err = 0.0
    mean_abs_err = 0.0
    if check_reference:
        ref_out = reference_decode(q, k_cache, v_cache, block_tables, context_lens, scale)
        diff = (flash_out.float() - ref_out.float()).abs()
        max_abs_err = diff.max().item()
        mean_abs_err = diff.mean().item()
        passed = torch.allclose(flash_out.float(), ref_out.float(), rtol=rtol, atol=atol)
        if not passed:
            raise AssertionError(f"Correctness failed for {batch_size=} {context_len=}: {max_abs_err=}, {mean_abs_err=}")

    timings = time_flash_decode(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale,
        warmup_iters,
        timing_iters,
    )
    row = {
        "model": Path(config.model).name,
        "batch_size": batch_size,
        "context_len": context_len,
        "block_size": block_size,
        "blocks_per_seq": math.ceil(context_len / block_size),
        "num_blocks": k_cache.size(0),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "gqa_group_size": num_heads // num_kv_heads,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "q_shape_for_flash": tuple(q.unsqueeze(1).shape),
        "k_cache_shape": tuple(k_cache.shape),
        "k_cache_stride": tuple(k_cache.stride()),
        "block_tables_shape": tuple(block_tables.shape),
        "block_tables_stride": tuple(block_tables.stride()),
        "warmup_iters": warmup_iters,
        "timing_iters": timing_iters,
        "correctness_checked": check_reference,
        "passed": passed,
        "max_abs_err": round(max_abs_err, 6),
        "mean_abs_err": round(mean_abs_err, 6),
    }
    row.update(summarize_ms(timings))
    row["latency_us_per_token_avg"] = round(row["latency_ms_avg"] * 1000.0 / batch_size, 4)
    row["tokens_per_s"] = round(batch_size * 1000.0 / row["latency_ms_avg"], 2) if row["latency_ms_avg"] else 0.0
    return row


def main() -> None:
    args = parse_args()
    benchmarks_dir = Path(__file__).resolve().parent
    if str(benchmarks_dir) not in sys.path:
        sys.path.insert(0, str(benchmarks_dir))
    from utils import add_repo_to_path, append_jsonl, collect_env, ensure_results_dir, write_markdown_table

    add_repo_to_path()
    from nanovllm.config import Config

    model = os.path.expanduser(args.model)
    cases = parse_cases(args.cases)
    config = Config(model, kvcache_block_size=args.block_size)
    results_dir = ensure_results_dir()
    jsonl_path = results_dir / f"{args.output_prefix}.jsonl"
    md_path = results_dir / f"{args.output_prefix}.md"
    rows = []

    if not args.no_write:
        append_jsonl(jsonl_path, {"type": "env", "env": collect_env()})

    for batch_size, context_len in cases:
        row = run_case(
            batch_size=batch_size,
            context_len=context_len,
            config=config,
            block_size=args.block_size,
            warmup_iters=args.warmup_iters,
            timing_iters=args.timing_iters,
            seed=args.seed,
            check_reference=not args.no_reference,
            rtol=args.rtol,
            atol=args.atol,
        )
        rows.append(row)
        print(row)
        if not args.no_write:
            append_jsonl(jsonl_path, {"type": "attention_decode_microbench", **row})

    if not args.no_write:
        columns = [
            "model",
            "batch_size",
            "context_len",
            "blocks_per_seq",
            "num_heads",
            "num_kv_heads",
            "gqa_group_size",
            "head_dim",
            "dtype",
            "passed",
            "max_abs_err",
            "mean_abs_err",
            "latency_ms_avg",
            "latency_ms_p50",
            "latency_ms_p95",
            "latency_ms_min",
            "latency_ms_max",
            "latency_us_per_token_avg",
            "tokens_per_s",
        ]
        write_markdown_table(md_path, rows, columns)
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
