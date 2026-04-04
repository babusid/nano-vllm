from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs  # batch size in sequences
        self.max_num_batched_tokens = (
            config.max_num_batched_tokens
        )  # batch size in tokens
        self.eos = config.eos  # end of sequence token id
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )  # setup the memory pool
        self.waiting: deque[Sequence] = deque()  # manage waiting and running sequences
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        # if we have no sequences waiting or running, this scheduler
        # is finished. This doesn't really mean anything, we could still
        # continue to use it, and it would transition out of this state
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # add a new sequence to the scheduler
        # this may / may not get scheduled the next time we call schedule
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            # while we have waiting sequences and there's room in the batch
            seq = self.waiting[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                # check if allowing this sequence in satisfies the memory pool and the max batch token size
                break
            # allow the sequence into the batch
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING  # mark sequence as inflight
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            # if we scheduled waiting sequences, return them, and note we're going to do a
            # prfill pass
            return scheduled_seqs, True

        # decode
        # we didn't have any sequences that were waiting to be scheduled AND could be scheduled
        # so now we're going to try to schedule sequences that are currently running (greedy prefill)
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()  # pop head of queue from running list
            while not self.block_manager.can_append(seq):
                # while we can't append the head of the queue to the batch
                if self.running:
                    # while we have other sequences that are currently marked as running (were scheduled before)
                    # preempt the ones at the tail of the queue until we can fit
                    # this sends them back to the waiting queue
                    self.preempt(self.running.pop())
                else:
                    # if we have no other running sequences and we still can't fit
                    # this sequence, we're just going to have to preempt it / send it back to the waiting queue
                    # and break out of the while-else
                    self.preempt(seq)
                    break
            else:
                # while loop passed meaning that we can fit the current sequence now
                # if the else triggered, we don't get here
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs  # make sure something got scheduled
        self.running.extendleft(
            reversed(scheduled_seqs)
        )  # put scheduled sequences back into the running list
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        Kicks a sequence out of the memory pool and puts it back into
        the waiting queue. Sequence will have to do a full prefill again.
        BUT the actual tokens in the sequence are retained, so prefix cache might still speed it up
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        Postprocesses the sequences by adding the generated tokens to the sequence.
        If the generated token is an EOS, the sequence is marked as finished,
        it's remvoed from the waiting list, and its KV cache is deallocated.
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
