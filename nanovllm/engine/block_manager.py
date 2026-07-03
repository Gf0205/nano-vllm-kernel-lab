from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.active_block_refs = 0
        self.current_shared_blocks = 0
        self.peak_used_blocks = 0
        self.peak_active_block_refs = 0
        self.peak_shared_blocks = 0
        self.allocation_requests = 0
        self.logical_blocks_allocated = 0
        self.physical_block_allocations = 0
        self.prefix_cache_eligible_blocks = 0
        self.prefix_cache_hit_blocks = 0

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        self.active_block_refs += 1
        self.physical_block_allocations += 1
        self._update_peak_stats()
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _update_peak_stats(self):
        self.peak_used_blocks = max(self.peak_used_blocks, len(self.used_block_ids))
        self.peak_active_block_refs = max(self.peak_active_block_refs, self.active_block_refs)
        self.peak_shared_blocks = max(self.peak_shared_blocks, self.current_shared_blocks)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        self.allocation_requests += 1
        self.logical_blocks_allocated += seq.num_blocks
        self.prefix_cache_eligible_blocks += max(0, seq.num_blocks - 1)
        self.prefix_cache_hit_blocks += num_cached_blocks
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                if block.ref_count == 1:
                    self.current_shared_blocks += 1
                block.ref_count += 1
                self.active_block_refs += 1
                self._update_peak_stats()
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
                self.active_block_refs += 1
                self._update_peak_stats()
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            if block.ref_count == 2:
                self.current_shared_blocks -= 1
            block.ref_count -= 1
            self.active_block_refs -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def metrics(self) -> dict[str, int | float]:
        # Ratios are computed on demand so allocation/decode paths stay cheap.
        total_blocks = len(self.blocks)
        used_blocks = len(self.used_block_ids)
        free_blocks = len(self.free_block_ids)
        block_reuse_ratio = self.active_block_refs / used_blocks if used_blocks else 0.0
        peak_block_reuse_ratio = self.peak_active_block_refs / self.peak_used_blocks if self.peak_used_blocks else 0.0
        prefix_cache_hit_rate = (
            self.prefix_cache_hit_blocks / self.prefix_cache_eligible_blocks
            if self.prefix_cache_eligible_blocks else 0.0
        )
        return {
            "total_blocks": total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": free_blocks,
            "cached_block_entries": len(self.hash_to_block_id),
            "active_block_refs": self.active_block_refs,
            "current_shared_blocks": self.current_shared_blocks,
            "peak_used_blocks": self.peak_used_blocks,
            "peak_active_block_refs": self.peak_active_block_refs,
            "peak_shared_blocks": self.peak_shared_blocks,
            "block_reuse_ratio": round(block_reuse_ratio, 4),
            "peak_block_reuse_ratio": round(peak_block_reuse_ratio, 4),
            "free_block_ratio": round(free_blocks / total_blocks, 4) if total_blocks else 0.0,
            "allocation_requests": self.allocation_requests,
            "logical_blocks_allocated": self.logical_blocks_allocated,
            "physical_block_allocations": self.physical_block_allocations,
            "prefix_cache_eligible_blocks": self.prefix_cache_eligible_blocks,
            "prefix_cache_hit_blocks": self.prefix_cache_hit_blocks,
            "prefix_cache_hit_rate": round(prefix_cache_hit_rate, 4),
        }
