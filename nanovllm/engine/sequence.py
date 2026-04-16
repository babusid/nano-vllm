from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    Primary data structure for a sequence of tokens.
    This is the primitive that gets scheduled by the scheduler.
    """

    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
        num_block_tables: int = 1,
    ):
        # uniquely identify this sequence
        self.seq_id = next(Sequence.counter)

        # start off as waiting until scheduled
        self.status = SequenceStatus.WAITING

        # if we're given token_ids, copy them in
        self.token_ids = copy(token_ids)

        # staging area for draft tokens
        # one of the lists is for the verifier
        # and will be empty the whole time. This specific initialization
        # is relied in the model runner.
        self.draft_token_ids = [[] for _ in range(num_block_tables)]

        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0

        # no block memory allocated yet for this sequence
        self._block_tables = [[] for _ in range(num_block_tables)]

        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    @property
    def block_tables(self) -> list[list[int]]:
        return self._block_tables

    @block_tables.setter
    def block_tables(self, value: list[list[int]]):
        self._block_tables = value

    @property
    def block_table(self):
        return self._block_tables[0]

    @block_table.setter
    def block_table(self, value):
        self._block_tables[0] = value

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """how many tokens have been generated to complete the given prompt"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """get the tokenized prompt"""
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """get the tokenized completion"""
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        """how many blocks are cached"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """total number of blocks used by this sequence"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return ((self.num_tokens - 1) % self.block_size) + 1 if self.num_tokens else 0

    def block(self, i):
        """get ith block of this sequence"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self._block_tables,
            self.draft_token_ids,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        draft_token_ids = None
        if len(state) == 6:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                block_tables,
                draft_token_ids,
            ) = state[:-1]
        else:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                block_tables,
            ) = state[:-1]
        if block_tables and isinstance(block_tables[0], int):
            self._block_tables = [block_tables]
        else:
            self._block_tables = block_tables
        if draft_token_ids is None:
            self.draft_token_ids = [[] for _ in range(len(self._block_tables))]
        elif draft_token_ids and isinstance(draft_token_ids[0], int):
            self.draft_token_ids = [draft_token_ids]
        else:
            self.draft_token_ids = draft_token_ids
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
            self.last_token = self.token_ids[-1]
        else:
            self.last_token = state[-1]
