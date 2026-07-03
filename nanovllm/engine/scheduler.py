from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_prefill_steps = 0
        self.num_decode_steps = 0
        self.num_preemptions = 0
        self.num_chunked_prefill_steps = 0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.max_prefill_batch_tokens = 0
        self.max_decode_batch_size = 0
        self.peak_waiting = 0
        self.peak_running = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
        self._update_queue_peaks()

    def _update_queue_peaks(self):
        self.peak_waiting = max(self.peak_waiting, len(self.waiting))
        self.peak_running = max(self.peak_running, len(self.running))

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            chunked = any(seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens for seq in scheduled_seqs)
            self.num_prefill_steps += 1
            self.num_chunked_prefill_steps += int(chunked)
            self.total_prefill_tokens += num_batched_tokens
            self.max_prefill_batch_tokens = max(self.max_prefill_batch_tokens, num_batched_tokens)
            self._update_queue_peaks()
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        self.num_decode_steps += 1
        self.total_decode_tokens += len(scheduled_seqs)
        self.max_decode_batch_size = max(self.max_decode_batch_size, len(scheduled_seqs))
        self._update_queue_peaks()
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self._update_queue_peaks()

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        self._update_queue_peaks()

    def metrics(self) -> dict[str, int | float]:
        # Derived averages are computed on demand to keep scheduling lightweight.
        avg_prefill_tokens = self.total_prefill_tokens / self.num_prefill_steps if self.num_prefill_steps else 0.0
        avg_decode_batch = self.total_decode_tokens / self.num_decode_steps if self.num_decode_steps else 0.0
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "peak_waiting": self.peak_waiting,
            "peak_running": self.peak_running,
            "num_prefill_steps": self.num_prefill_steps,
            "num_decode_steps": self.num_decode_steps,
            "num_chunked_prefill_steps": self.num_chunked_prefill_steps,
            "num_preemptions": self.num_preemptions,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "max_prefill_batch_tokens": self.max_prefill_batch_tokens,
            "max_decode_batch_size": self.max_decode_batch_size,
            "avg_prefill_tokens_per_step": round(avg_prefill_tokens, 2),
            "avg_decode_batch_size": round(avg_decode_batch, 2),
        }
