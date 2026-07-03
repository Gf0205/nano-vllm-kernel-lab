# RTX 3090 Phase 2 Smoke Results

Environment:

- GPU: NVIDIA GeForce RTX 3090
- Python: 3.12.3
- PyTorch: 2.5.1+cu124
- CUDA runtime: 12.4
- Triton: 3.1.0
- Transformers: 4.51.0
- FlashAttention: 2.7.4.post1
- Model: Qwen3-0.6B

## Key Observations

- The CUDA Graph path (`enforce_eager=False`) is much faster than eager decode on this decode-heavy smoke matrix.
- The largest observed throughput in the smoke run is 5784.56 output tokens/s at batch size 32, input length 128, output length 128.
- For batch size 32 and input length 512, throughput drops from 3615.98 output tokens/s with CUDA Graph to 1186.01 output tokens/s with eager execution.
- Peak memory stays around 20 GB because Nano-vLLM pre-allocates KV cache according to `gpu_memory_utilization`, so this number should be interpreted as reserved/allocated benchmark memory rather than per-request incremental memory.

## Stop Condition

Phase 2 smoke validation is complete:

- `example.py` runs on AutoDL RTX 3090.
- Throughput smoke benchmark runs with CUDA Graph enabled.
- Throughput smoke benchmark runs with `enforce_eager=True`.
- Latency smoke benchmark records TTFT, TPOT, and output tokens/s.

Next phase: BlockManager metrics and prefix-cache benchmark.
