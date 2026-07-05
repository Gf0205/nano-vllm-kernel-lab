import argparse
import os
from time import perf_counter

import torch

from utils import add_repo_to_path, cuda_sync, make_token_ids


add_repo_to_path()

from nanovllm import LLM, SamplingParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nsight Systems steady-state decode capture helper.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-decode-steps", type=int, default=8)
    parser.add_argument("--profile-decode-steps", type=int, default=32)
    return parser.parse_args()


def nvtx_range(name: str):
    class Range:
        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(name)

        def __exit__(self, exc_type, exc, tb):
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()

    return Range()


def main() -> None:
    args = parse_args()
    model = os.path.expanduser(args.model)
    llm = LLM(
        model,
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

    with nvtx_range("prefill"):
        llm.step()
    for i in range(args.warmup_decode_steps):
        if llm.is_finished():
            break
        with nvtx_range(f"warmup_decode_step_{i}"):
            llm.step()

    cuda_sync()
    if torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStart()
    start = perf_counter()
    steps = 0
    for i in range(args.profile_decode_steps):
        if llm.is_finished():
            break
        with nvtx_range(f"nsys_profile_decode_step_{i}"):
            llm.step()
        steps += 1
    cuda_sync()
    elapsed = perf_counter() - start
    if torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStop()

    print(
        {
            "profiled_decode_steps": steps,
            "elapsed_s": round(elapsed, 6),
            "avg_step_s": round(elapsed / steps, 6) if steps else 0.0,
        }
    )
    llm.exit()


if __name__ == "__main__":
    main()
