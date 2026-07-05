import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "bench_attention_decode_microbench.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_attention_decode_microbench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_cases_accepts_batch_context_pairs():
    bench = load_module()

    cases = bench.parse_cases("1x128, 8x512,32x1024")

    assert cases == [(1, 128), (8, 512), (32, 1024)]


def test_required_num_blocks_covers_all_case_blocks():
    bench = load_module()

    num_blocks = bench.required_num_blocks(batch_size=32, context_len=513, block_size=256)

    assert num_blocks == 96


def test_summarize_ms_reports_stable_percentiles():
    bench = load_module()

    summary = bench.summarize_ms([5.0, 1.0, 3.0, 2.0])

    assert summary == {
        "latency_ms_avg": 2.75,
        "latency_ms_p50": 2.5,
        "latency_ms_p95": 5.0,
        "latency_ms_min": 1.0,
        "latency_ms_max": 5.0,
    }
