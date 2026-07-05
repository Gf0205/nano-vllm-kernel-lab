import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "bench_mlp_gemm_compare.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_mlp_gemm_compare", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_projection_shapes_for_gate_up_and_down():
    bench = load_module()

    gate_up = bench.projection_shape("gate_up", num_tokens=512, hidden_size=1024, intermediate_size=3072)
    down = bench.projection_shape("down", num_tokens=512, hidden_size=1024, intermediate_size=3072)

    assert gate_up == {
        "projection": "gate_up",
        "input_shape": (512, 1024),
        "weight_shape": (6144, 1024),
        "output_shape": (512, 6144),
    }
    assert down == {
        "projection": "down",
        "input_shape": (512, 3072),
        "weight_shape": (1024, 3072),
        "output_shape": (512, 1024),
    }


def test_default_variants_are_explicit_and_stable():
    bench = load_module()

    assert bench.default_variants() == ["linear", "matmul_t", "matmul_contiguous_t"]


def test_speedup_vs_baseline_handles_zero_and_rounding():
    bench = load_module()

    assert bench.speedup_vs_baseline(baseline_ms=2.0, candidate_ms=1.0) == 2.0
    assert bench.speedup_vs_baseline(baseline_ms=3.0, candidate_ms=2.0) == 1.5
    assert bench.speedup_vs_baseline(baseline_ms=0.0, candidate_ms=2.0) == 0.0
    assert bench.speedup_vs_baseline(baseline_ms=2.0, candidate_ms=0.0) == 0.0


def test_summarize_repeats_groups_rows_by_shape_and_variant():
    bench = load_module()
    rows = [
        {"projection": "gate_up", "variant": "linear", "num_tokens": 128, "latency_ms_avg": 1.0, "speedup_vs_linear": 1.0},
        {"projection": "gate_up", "variant": "linear", "num_tokens": 128, "latency_ms_avg": 2.0, "speedup_vs_linear": 1.0},
        {
            "projection": "gate_up",
            "variant": "matmul_pretransposed",
            "num_tokens": 128,
            "latency_ms_avg": 0.8,
            "speedup_vs_linear": 1.25,
        },
        {
            "projection": "gate_up",
            "variant": "matmul_pretransposed",
            "num_tokens": 128,
            "latency_ms_avg": 1.0,
            "speedup_vs_linear": 1.1,
        },
    ]

    summary = bench.summarize_repeats(rows)

    assert summary == [
        {
            "projection": "gate_up",
            "variant": "linear",
            "num_tokens": 128,
            "repeats": 2,
            "latency_ms_avg_mean": 1.5,
            "latency_ms_avg_min": 1.0,
            "latency_ms_avg_max": 2.0,
            "speedup_vs_linear_mean": 1.0,
            "speedup_vs_linear_min": 1.0,
            "speedup_vs_linear_max": 1.0,
            "faster_than_linear_runs": 0,
        },
        {
            "projection": "gate_up",
            "variant": "matmul_pretransposed",
            "num_tokens": 128,
            "repeats": 2,
            "latency_ms_avg_mean": 0.9,
            "latency_ms_avg_min": 0.8,
            "latency_ms_avg_max": 1.0,
            "speedup_vs_linear_mean": 1.175,
            "speedup_vs_linear_min": 1.1,
            "speedup_vs_linear_max": 1.25,
            "faster_than_linear_runs": 2,
        },
    ]
