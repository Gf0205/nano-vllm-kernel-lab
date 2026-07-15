import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SCRIPT = Path(__file__).with_name("bench_chunked_prefill_interference.py")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_markdown_table(path: Path, rows: Iterable[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write("| " + " | ".join(columns) + " |\n")
        output_file.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            output_file.write("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final scheduler robustness matrix on one GPU.")
    parser.add_argument("--model", default="/root/huggingface/Qwen3-0.6B")
    parser.add_argument("--active-decode-seqs", default="4,8,16")
    parser.add_argument("--long-input-lens", default="1024,3072")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--active-input-len", type=int, default=128)
    parser.add_argument("--active-output-len", type=int, default=128)
    parser.add_argument("--long-output-len", type=int, default=32)
    parser.add_argument("--inject-after-decode-steps", type=int, default=8)
    parser.add_argument("--chunked-budget", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeline-limit", type=int, default=256)
    parser.add_argument("--output-prefix", default="final_scheduler_robustness_3090")
    parser.add_argument("--dry-run", action="store_true", help="Print the workload commands without executing them.")
    return parser.parse_args()


def run_and_tee(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def load_summaries(path: Path, active_seqs: int, long_input_len: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as result_file:
        for line in result_file:
            record = json.loads(line)
            if record.get("type") != "chunked_prefill_interference_repeat_summary":
                continue
            rows.append(
                {
                    "active_decode_seqs": active_seqs,
                    "long_input_len": long_input_len,
                    **{key: value for key, value in record.items() if key != "type"},
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    active_seq_cases = parse_int_list(args.active_decode_seqs)
    long_input_cases = parse_int_list(args.long_input_lens)
    if args.repeats < 5:
        raise ValueError("The closeout protocol requires --repeats >= 5")
    if not active_seq_cases or not long_input_cases:
        raise ValueError("The workload matrix cannot be empty")

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    for active_seqs in active_seq_cases:
        for long_input_len in long_input_cases:
            case_prefix = f"{args.output_prefix}_a{active_seqs}_l{long_input_len}"
            # The canonical workload also closes the previously truncated N=4
            # cadence evidence; the robustness matrix itself remains N=1/N=2.
            cadences = "1,2,4" if active_seqs == 8 and long_input_len == 3072 else "1,2"
            jsonl_path = results_dir / f"{case_prefix}.jsonl"
            log_path = results_dir / f"{case_prefix}.log"
            command = [
                sys.executable,
                str(SCRIPT),
                "--model",
                args.model,
                "--active-decode-seqs",
                str(active_seqs),
                "--active-input-len",
                str(args.active_input_len),
                "--active-output-len",
                str(args.active_output_len),
                "--long-input-len",
                str(long_input_len),
                "--long-output-len",
                str(args.long_output_len),
                "--inject-after-decode-steps",
                str(args.inject_after_decode_steps),
                "--chunked-budget",
                str(args.chunked_budget),
                "--max-model-len",
                str(args.max_model_len),
                "--max-num-seqs",
                str(args.max_num_seqs),
                "--seed",
                str(args.seed),
                "--repeats",
                str(args.repeats),
                "--timeline-limit",
                str(args.timeline_limit),
                "--skip-non-chunked",
                "--include-decode-aware",
                "--decode-aware-cadences",
                cadences,
                "--output-prefix",
                case_prefix,
            ]
            print(f"Running active={active_seqs}, long_input={long_input_len}, repeats={args.repeats}")
            if args.dry_run:
                print(" ".join(command))
                continue
            run_and_tee(command, log_path)
            combined_rows.extend(load_summaries(jsonl_path, active_seqs, long_input_len))

    if args.dry_run:
        return

    summary_path = results_dir / f"{args.output_prefix}_matrix_summary.md"
    write_markdown_table(
        summary_path,
        combined_rows,
        [
            "active_decode_seqs",
            "long_input_len",
            "case",
            "repeats",
            "post_injection_active_decode_gap_s_p50_median",
            "post_injection_active_decode_gap_s_p95_median",
            "post_injection_active_decode_gap_s_max_median",
            "post_injection_active_decode_gap_s_max_p20",
            "post_injection_active_decode_gap_s_max_p80",
            "long_request_ttft_s_median",
            "long_request_ttft_s_p20",
            "long_request_ttft_s_p80",
            "post_injection_wall_time_s_mean",
            "interleaved_runs",
            "num_decode_aware_interleaves_min",
            "num_decode_aware_interleaves_max",
            "capacity_limited_runs",
            "long_prompt_shrunk_runs",
            "decode_eager_steps_total",
        ],
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
