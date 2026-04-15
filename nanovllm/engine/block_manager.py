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

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
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
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_allocate_spec(self, seq: Sequence, n_tokens: int) -> bool:
        N = len(seq)
        blocks_needed = 0
        for i in range(n_tokens):
            pos = N + i
            block_index = pos // self.block_size
            if len(seq.block_table) <= block_index:
                blocks_needed += 1
        return len(self.free_block_ids) >= blocks_needed

    def allocate_slots_for_spec(self, seq: Sequence, n_tokens: int) -> None:
        N = len(seq)
        for i in range(n_tokens):
            pos = N + i
            block_index = pos // self.block_size
            if len(seq.block_table) <= block_index:
                if not self.free_block_ids:
                    raise RuntimeError(
                        "KV cache exhausted during specdec allocation"
                    )
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                seq.block_table.append(block_id)

    def trim_speculative_blocks(self, seq: Sequence) -> None:
        """Free blocks allocated beyond the accepted length, then finalize any
        blocks that crossed a boundary during the speculative step.

        After accepting K' tokens, seq.num_tokens = N + K'.  Blocks that were
        pre-allocated for positions N+K'..N+K are freed.  Blocks that are now
        completely full but were never processed by may_append (because several
        tokens were accepted at once) are finalized here so that the next call
        to may_append sees the invariant it expects.
        """
        num_needed = (len(seq) + self.block_size - 1) // self.block_size
        while len(seq.block_table) > num_needed:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        # Finalize every block that is now completely full.  _finalize_completed_blocks
        # uses the sequence length to determine which blocks are full, so it naturally
        # skips the (potentially partial) last block.
        self._finalize_completed_blocks(seq)

    def _finalize_completed_blocks(self, seq: Sequence) -> None:
        """Finalize (hash) every block in block_table that is completely filled
        by the current sequence but has not yet been hashed.

        Unlike the old version that stopped before the last block, this version
        processes ALL blocks including the last one — but only if the sequence
        actually covers that block fully (i.e. len(seq) >= end of block).
        Partial blocks (the active write target) are naturally skipped because
        the sequence doesn't reach their end yet.

        Processing left-to-right guarantees the prefix-hash chain is correct.
        """
        for i in range(len(seq.block_table)):
            block_id = seq.block_table[i]
            block = self.blocks[block_id]
            if block.hash != -1:
                continue  # already finalized
            end_pos = (i + 1) * self.block_size
            if end_pos > len(seq):
                break  # block is only partially covered — stop here
            token_ids = seq.block(i)
            if len(token_ids) < self.block_size:
                break  # safeguard: shouldn't happen, but don't hash partial data
            prefix_hash = self.blocks[seq.block_table[i - 1]].hash if i > 0 else -1
            h = self.compute_hash(token_ids, prefix_hash)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # Position len(seq)-1 is the first slot of a new block.
            # Standard path: allocate that new block, first ensuring the
            # previous (now-full) block is finalized.
            # Speculative path: allocate_slots_for_spec may have pre-allocated
            # this block already; detect that and skip the allocation.
            already_allocated = len(block_table) * self.block_size >= len(seq)
            if not already_allocated:
                # Finalize the previous full block if somehow it wasn't yet
                # (e.g. tokens were accepted in bulk crossing a boundary).
                if last_block.hash == -1:
                    self._finalize_completed_blocks(seq)
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                block_table.append(block_id)
            # else: block already pre-allocated by speculative step; nothing to do.
        elif len(seq) % self.block_size == 0:
            # The last token filled the block exactly.  _finalize_completed_blocks
            # (called from trim_speculative_blocks) may have already hashed it when
            # multiple tokens were accepted in a single speculative step.  Skip if so.
            if last_block.hash == -1:
                token_ids = seq.block(seq.num_blocks - 1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
        else:
            # Mid-block: should be unhashed, but _finalize_completed_blocks never
            # touches partial blocks so this remains an invariant in practice.
            assert last_block.hash == -1
