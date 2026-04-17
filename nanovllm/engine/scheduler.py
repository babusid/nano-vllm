from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.speculation import SpeculationMode


class Scheduler:
    def __init__(
        self,
        config: Config,
        block_managers: list[BlockManager],
        speculation_mode: SpeculationMode = SpeculationMode.NONE,
        speculator_config: list[Config] | None = None,
        speculation_length: int | None = None,
    ):
        self.speculation_mode = speculation_mode
        self.speculator_config = speculator_config
        self.speculation_length = speculation_length
        _spec_conf = [
            self.speculation_mode is not SpeculationMode.NONE,
            self.speculator_config,
            self.speculation_length,
        ]
        if any(_spec_conf) and not all(_spec_conf):
            raise ValueError(
                "Speculation mode, speculator config and speculation length must be specified together"
            )

        self.max_num_seqs = config.max_num_seqs  # batch size in sequences
        self.max_num_batched_tokens = (
            config.max_num_batched_tokens
        )  # batch size in tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos  # end of sequence token id
        self.block_managers = block_managers  # setup the memory pools
        self.waiting: deque[Sequence] = deque()  # manage waiting and running sequences
        self.running: deque[Sequence] = deque()

    def _can_allocate(self, seq: Sequence) -> bool:
        return all(
            block_manager.can_allocate(seq) for block_manager in self.block_managers
        )

    def _allocate(self, seq: Sequence):
        for block_manager in self.block_managers:
            block_manager.allocate(seq)

    def _can_append(self, seq: Sequence, num_bonus_tokens: int = 0) -> bool:
        return all(
            block_manager.can_append(seq, num_bonus_tokens)
            for block_manager in self.block_managers
        )

    def _may_append(self, seq: Sequence, num_bonus_tokens: int = 0):
        for block_manager in self.block_managers:
            block_manager.may_append(seq, num_bonus_tokens)

    def _deallocate(self, seq: Sequence):
        for block_manager in self.block_managers:
            block_manager.deallocate(seq)

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
            ) > self.max_num_batched_tokens or not self._can_allocate(seq):
                # check if allowing this sequence in satisfies the memory pool and the max batch token size
                break
            # allow the sequence into the batch
            num_seqs += 1
            self._allocate(seq)
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
        speculation_tokens = 0
        if self.speculation_mode is SpeculationMode.NAIVE_SPECULATION:
            # add 1 to the speculation length to account for the bonus token
            # from the verifier
            # speculation_tokens = self.speculation_length + 1
            speculation_tokens = self.speculation_length

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()  # pop head of queue from running list
            bonus_tokens = min(  # make sure the bonus tokens aren't more than the remaining context
                speculation_tokens,
                max(0, self.max_model_len - len(seq)),
            )
            while not self._can_append(seq, bonus_tokens):
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
                self._may_append(seq, bonus_tokens)
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
        self._deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self, seqs: list[Sequence], seqs_token_ids: list[list[int]]
    ) -> list[bool]:
        """
        Postprocesses the sequences by adding the generated tokens to the sequence.
        If the generated token is an EOS, the sequence is marked as finished,
        it's remvoed from the waiting list, and its KV cache is deallocated.
        """
        for seq, token_ids in zip(seqs, seqs_token_ids):
            seq.extend(token_ids)
            if (
                (
                    not seq.ignore_eos
                    and any(token_id == self.eos for token_id in token_ids)
                )
                or seq.num_completion_tokens >= seq.max_tokens
                or len(seq) >= self.max_model_len
            ):
                seq.status = SequenceStatus.FINISHED
                self._deallocate(seq)
                self.running.remove(seq)
