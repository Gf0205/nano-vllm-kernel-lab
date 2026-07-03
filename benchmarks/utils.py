import importlib.metadata
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks" / "results"


def add_repo_to_path() -> None:
    # Running a script under benchmarks/ makes Python prefer that directory.
    # Insert the repo root so `import nanovllm` works without installing first.
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def collect_env() -> dict[str, str | int | None]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu_name,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "triton": package_version("triton"),
        "transformers": package_version("transformers"),
        "flash_attn": package_version("flash-attn"),
    }


def make_token_ids(num_seqs: int, input_len: int, vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    # Avoid very small special-token ids in synthetic prompts.
    lo = 100
    hi = max(lo + 1, vocab_size - 1)
    return [[rng.randint(lo, hi) for _ in range(input_len)] for _ in range(num_seqs)]


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def timed(fn):
    cuda_sync()
    start = perf_counter()
    out = fn()
    cuda_sync()
    return out, perf_counter() - start


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown_table(path: Path, rows: Iterable[dict], columns: list[str]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |\n")


@dataclass(slots=True)
class ThroughputRecord:
    model: str
    enforce_eager: bool
    batch_size: int
    input_len: int
    output_len: int
    total_output_tokens: int
    wall_time_s: float
    output_tokens_per_s: float
    peak_memory_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class LatencyRecord:
    model: str
    enforce_eager: bool
    num_seqs: int
    input_len: int
    output_len: int
    total_output_tokens: int
    wall_time_s: float
    ttft_s_avg: float
    ttft_s_p50: float
    ttft_s_p90: float
    tpot_s: float
    output_tokens_per_s: float
    peak_memory_gb: float

    def to_dict(self) -> dict:
        return asdict(self)
