import os
import time
import argparse
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Original Nano-vLLM throughput smoke benchmark.")
    parser.add_argument(
        "--model",
        default=os.environ.get("NANOVLLM_MODEL", "~/huggingface/Qwen3-0.6B/"),
        help="Local model directory. Can also be set with NANOVLLM_MODEL.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser(args.model)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Model directory not found: {path}\n"
            "Download it first, for example:\n"
            "huggingface-cli download --resume-download Qwen/Qwen3-0.6B "
            "--local-dir ~/huggingface/Qwen3-0.6B/ --local-dir-use-symlinks False"
        )
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
