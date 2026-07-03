import os
import argparse
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small Nano-vLLM generation example.")
    parser.add_argument(
        "--model",
        default=os.environ.get("NANOVLLM_MODEL", "~/huggingface/Qwen3-0.6B/"),
        help="Local model directory. Can also be set with NANOVLLM_MODEL.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = os.path.expanduser(args.model)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Model directory not found: {path}\n"
            "Download it first, for example:\n"
            "huggingface-cli download --resume-download Qwen/Qwen3-0.6B "
            "--local-dir ~/huggingface/Qwen3-0.6B/ --local-dir-use-symlinks False"
        )
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
