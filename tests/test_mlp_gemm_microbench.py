import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "bench_mlp_gemm_microbench.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_mlp_gemm_microbench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_int_cases_accepts_comma_separated_tokens():
    bench = load_module()

    cases = bench.parse_int_cases("1, 8,32, 512")

    assert cases == [1, 8, 32, 512]


def test_projection_shapes_match_qwen3_mlp_contract():
    bench = load_module()

    shapes = bench.mlp_projection_shapes(hidden_size=1024, intermediate_size=2816, num_tokens=32)

    assert shapes == {
        "input_shape": (32, 1024),
        "gate_up_weight_shape": (5632, 1024),
        "gate_up_output_shape": (32, 5632),
        "activation_output_shape": (32, 2816),
        "down_weight_shape": (1024, 2816),
        "down_output_shape": (32, 1024),
    }


def test_percent_of_total_handles_zero_and_rounding():
    bench = load_module()

    assert bench.percent_of_total(2.5, 10.0) == 25.0
    assert bench.percent_of_total(1.0, 3.0) == 33.3333
    assert bench.percent_of_total(1.0, 0.0) == 0.0


def test_add_boundary_percentages_uses_consistent_full_timing():
    bench = load_module()
    row = {
        "gate_up_boundary_ms_avg": 6.0,
        "silu_mul_boundary_ms_avg": 1.5,
        "down_boundary_ms_avg": 2.5,
        "full_mlp_boundary_ms_avg": 10.0,
    }

    bench.add_boundary_percentages(row)

    assert row["gate_up_boundary_pct_of_full"] == 60.0
    assert row["silu_mul_boundary_pct_of_full"] == 15.0
    assert row["down_boundary_pct_of_full"] == 25.0
    assert row["boundary_pct_sum"] == 100.0
