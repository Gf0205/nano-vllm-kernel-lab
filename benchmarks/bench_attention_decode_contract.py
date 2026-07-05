import argparse
import math
import os
from pathlib import Path

import torch
from flash_attn import flash_attn_with_kvcache

from utils import add_repo_to_path, append_jsonl, collect_env, ensure_results_dir


add_repo_to_path()

from nanovllm.config import Config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate attention decode KV-cache layout and FlashAttention contract.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=513)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--output-prefix", default="attention_decode_contract")
    return parser.parse_args()


def repeat_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
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
    batch, num_heads, head_dim = q.shape
    outputs = []
    for b in range(batch):
        seq_len = int(context_lens[b].item())
        keys = []
        values = []
        for token_idx in range(seq_len):
            table_idx = token_idx // k_cache.size(1)
            offset = token_idx % k_cache.size(1)
            block_id = int(block_tables[b, table_idx].item())
            keys.append(k_cache[block_id, offset])
            values.append(v_cache[block_id, offset])
        k = repeat_kv(torch.stack(keys, dim=0).unsqueeze(0).float(), num_heads)[0]
        v = repeat_kv(torch.stack(values, dim=0).unsqueeze(0).float(), num_heads)[0]
        scores = torch.einsum("hd,shd->hs", q[b].float(), k) * scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hs,shd->hd", probs, v)
        outputs.append(out)
    return torch.stack(outputs, dim=0).to(q.dtype)


def main() -> None:
    args = parse_args()
    model = os.path.expanduser(args.model)
    config = Config(model, kvcache_block_size=args.block_size)
    hf_config = config.hf_config
    dtype = hf_config.dtype
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads must be divisible by num_kv_heads: {num_heads=} {num_kv_heads=}")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(args.batch_size, num_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(args.num_blocks, args.block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(args.num_blocks, args.block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    blocks_per_seq = math.ceil(args.context_len / args.block_size)
    if args.batch_size * blocks_per_seq > args.num_blocks:
        raise ValueError("Increase --num-blocks so every synthetic sequence has unique blocks.")
    block_tables = torch.arange(args.batch_size * blocks_per_seq, device=device, dtype=torch.int32).view(
        args.batch_size, blocks_per_seq
    )
    context_lens = torch.full((args.batch_size,), args.context_len, device=device, dtype=torch.int32)
    scale = head_dim**-0.5

    flash_out = flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=context_lens,
        block_table=block_tables,
        softmax_scale=scale,
        causal=True,
    ).squeeze(1)
    ref_out = reference_decode(q, k_cache, v_cache, block_tables, context_lens, scale)
    max_abs_err = (flash_out.float() - ref_out.float()).abs().max().item()
    mean_abs_err = (flash_out.float() - ref_out.float()).abs().mean().item()
    passed = torch.allclose(flash_out.float(), ref_out.float(), rtol=args.rtol, atol=args.atol)

    row = {
        "model": Path(model).name,
        "batch_size": args.batch_size,
        "context_len": args.context_len,
        "block_size": args.block_size,
        "blocks_per_seq": blocks_per_seq,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "gqa_group_size": num_heads // num_kv_heads,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "k_cache_shape": tuple(k_cache.shape),
        "k_cache_stride": tuple(k_cache.stride()),
        "block_tables_shape": tuple(block_tables.shape),
        "block_tables_stride": tuple(block_tables.stride()),
        "q_shape_for_flash": tuple(q.unsqueeze(1).shape),
        "max_abs_err": round(max_abs_err, 6),
        "mean_abs_err": round(mean_abs_err, 6),
        "passed": passed,
    }
    print(row)

    if not args.no_write:
        results_dir = ensure_results_dir()
        jsonl_path = results_dir / f"{args.output_prefix}.jsonl"
        append_jsonl(jsonl_path, {"type": "env", "env": collect_env()})
        append_jsonl(jsonl_path, {"type": "attention_decode_contract", **row})
        print(f"Wrote {jsonl_path}")

    if not passed:
        raise AssertionError(f"FlashAttention decode contract check failed: {max_abs_err=}, {mean_abs_err=}")


if __name__ == "__main__":
    main()
