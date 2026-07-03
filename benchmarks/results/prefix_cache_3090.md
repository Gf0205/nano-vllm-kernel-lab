| case | num_seqs | prompt_len | total_output_tokens | wall_time_s | output_tokens_per_s | peak_memory_gb | prefix_cache_eligible_blocks | prefix_cache_hit_blocks | prefix_cache_hit_rate | physical_block_allocations | peak_block_reuse_ratio_after | peak_shared_blocks_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared_prefix_after_warmup | 32 | 1056 | 2048 | 0.9447 | 2167.79 | 20.397 | 128 | 128 | 1.0 | 32 | 4.4444 | 4 |
| unique_prefix_baseline | 32 | 1056 | 2048 | 1.1983 | 1709.11 | 20.793 | 128 | 0 | 0.0 | 160 | 1.0 | 4 |
