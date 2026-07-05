import argparse
import json
import os
from pathlib import Path
from types import MethodType

import torch

from utils import (
    add_repo_to_path,
    append_jsonl,
    collect_env,
    cuda_sync,
    ensure_results_dir,
    make_token_ids,
)


add_repo_to_path()

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP  # noqa: E402
from nanovllm.layers.attention import Attention  # noqa: E402
from nanovllm.layers.activation import SiluAndMul  # noqa: E402
from nanovllm.layers.layernorm import RMSNorm  # noqa: E402
from nanovllm.layers.rotary_embedding import RotaryEmbedding  # noqa: E402


LABELS = {
    Qwen3Attention: "module.qwen3_attention",
    Attention: "module.paged_decode_attention",
    Qwen3MLP: "module.mlp",
    RMSNorm: "module.rmsnorm",
    RotaryEmbedding: "module.rope",
    SiluAndMul: "module.silu_mul",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch profiler for steady-state decode forward.")
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
    parser.add_argument("--warmup-decode-steps", type=int, default=4)
    parser.add_argument("--profile-decode-steps", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--export-trace", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--output-prefix", default="decode_profiler")
    return parser.parse_args()


def cuda_us(event, attr: str) -> float:
    value = getattr(event, attr, None)
    if value is None and attr.startswith("cuda_"):
        value = getattr(event, attr.replace("cuda_", "device_"), 0.0)
    return float(value or 0.0)


def install_eager_module_labels(llm: LLM) -> None:
    # These labels only affect eager runs. CUDA Graph replay has already been
    # captured during LLM construction, so graph-mode internals are analyzed via
    # CUDA kernel events instead.
    for module in llm.model_runner.model.modules():
        label = next((name for cls, name in LABELS.items() if isinstance(module, cls)), None)
        if label is None or getattr(module, "_nanovllm_profile_wrapped", False):
            continue
        original_forward = module.forward

        def wrapped_forward(self, *args, __forward=original_forward, __label=label, **kwargs):
            with torch.profiler.record_function(__label):
                return __forward(*args, **kwargs)

        module.forward = MethodType(wrapped_forward, module)
        module._nanovllm_profile_wrapped = True


def top_events(prof: torch.profiler.profile, top_k: int) -> list[dict]:
    rows = []
    for event in prof.key_averages():
        cuda_total_us = cuda_us(event, "cuda_time_total")
        self_cuda_total_us = cuda_us(event, "self_cuda_time_total")
        if cuda_total_us <= 0 and self_cuda_total_us <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "count": int(event.count),
                "cuda_total_ms": round(cuda_total_us / 1000.0, 4),
                "self_cuda_total_ms": round(self_cuda_total_us / 1000.0, 4),
                "cuda_avg_us": round(cuda_total_us / max(1, event.count), 3),
                "cpu_total_ms": round(float(getattr(event, "cpu_time_total", 0.0)) / 1000.0, 4),
            }
        )
    rows.sort(key=lambda row: row["cuda_total_ms"], reverse=True)
    return rows[:top_k]


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No CUDA profiler events captured.")
        return
    columns = ["name", "count", "cuda_total_ms", "self_cuda_total_ms", "cuda_avg_us", "cpu_total_ms"]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(str(row[col]) for col in columns) + " |")


def main() -> None:
    args = parse_args()
    model = os.path.expanduser(args.model)
    results_dir = ensure_results_dir()
    jsonl_path = results_dir / f"{args.output_prefix}.jsonl"
    trace_path = results_dir / f"{args.output_prefix}.json"

    llm = LLM(
        model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    if args.enforce_eager:
        install_eager_module_labels(llm)

    prompts = make_token_ids(args.num_seqs, args.input_len, llm.tokenizer.vocab_size, args.seed)
    sampling = SamplingParams(
        temperature=args.temperature,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    for prompt in prompts:
        llm.add_request(prompt, sampling)

    # Run prefill and skip early decode steps so the profile is steady-state.
    llm.step()
    for _ in range(args.warmup_decode_steps):
        if not llm.is_finished():
            llm.step()
    cuda_sync()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.record_shapes,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(args.profile_decode_steps):
            if llm.is_finished():
                break
            with torch.profiler.record_function("steady_decode_step"):
                llm.step()

    cuda_sync()
    rows = top_events(prof, args.top_k)
    record = {
        "type": "decode_profiler",
        "model": Path(model).name,
        "enforce_eager": args.enforce_eager,
        "num_seqs": args.num_seqs,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "warmup_decode_steps": args.warmup_decode_steps,
        "profile_decode_steps": args.profile_decode_steps,
        "top_events": rows,
    }

    print(json.dumps({k: v for k, v in record.items() if k != "top_events"}, ensure_ascii=False))
    print_table(rows)

    if not args.no_write:
        append_jsonl(jsonl_path, {"type": "env", "env": collect_env()})
        append_jsonl(jsonl_path, record)
        if args.export_trace:
            prof.export_chrome_trace(str(trace_path))
            print(f"Wrote {trace_path}")
        print(f"Wrote {jsonl_path}")

    llm.exit()


if __name__ == "__main__":
    main()
