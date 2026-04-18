from collections import deque
from math import ceil
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
    def __init__(self, num_blocks: int, block_size: int, block_table_idx: int = 0):
        self.block_size = block_size
        self.block_table_idx = block_table_idx
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _block_table(self, seq: Sequence) -> list[int]:
        return seq.block_tables[self.block_table_idx]

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence, bonus_blocks: int = 0) -> bool:
        return len(self.free_block_ids) >= (seq.num_blocks + bonus_blocks)

    def allocate(self, seq: Sequence):
        block_table = self._block_table(seq)
        assert not block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                if self.block_table_idx == 0:
                    seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        block_table = self._block_table(seq)
        for block_id in reversed(block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        if self.block_table_idx == 0:
            seq.num_cached_tokens = 0
        block_table.clear()

    def can_append(self, seq: Sequence, num_bonus_tokens: int = 0) -> bool:
        num_tokens_to_reserve = len(seq) + num_bonus_tokens
        block_table = self._block_table(seq)
        need_to_allocate = ceil(num_tokens_to_reserve / self.block_size)

        # len block table is how many blocks are already allocated to this
        # sequence by this block manager
        num_blocks_to_reserve = need_to_allocate - len(block_table)
        return len(self.free_block_ids) >= num_blocks_to_reserve

    def may_append(self, seq: Sequence, num_bonus_tokens: int = 0):
        block_table = self._block_table(seq)
        committed_len = len(seq)
        cur_len = committed_len + num_bonus_tokens

        # Grow the block table to cover committed + bonus (reserved) slots.
        # With spec decode, up to num_bonus_tokens new tokens may be committed
        # per step, so the previous reservation can fall short by one or more
        # blocks — may_append is responsible for allocating them now.
        need_to_allocate = ceil(cur_len / self.block_size) if cur_len else 0
        num_blocks_to_reserve = need_to_allocate - len(block_table)
        for _ in range(num_blocks_to_reserve):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        # Finalize all fully filled blocks that are still unhashed.
        # This is robust to multi-token extension per step (spec decode).
        # Only blocks whose contents are fully committed may be hashed —
        # speculative slots haven't been accepted yet, and a hashed block
        # is treated as immutable by the prefix cache.
        num_full_blocks = committed_len // self.block_size
        for block_idx in range(num_full_blocks):
            block = self.blocks[block_table[block_idx]]
            if block.hash != -1:
                continue
            token_ids = seq.block(block_idx)
            prefix = (
                self.blocks[block_table[block_idx - 1]].hash if block_idx > 0 else -1
            )
            if block_idx > 0:
                assert prefix != -1
            h = self.compute_hash(token_ids, prefix)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

        # If current length ends inside a block, it must remain writable/unhashed.
        if cur_len % self.block_size != 0:
            cur_block_idx = need_to_allocate - 1
            cur_block = self.blocks[block_table[cur_block_idx]]
            assert cur_block.hash == -1
