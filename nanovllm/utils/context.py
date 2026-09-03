from dataclasses import dataclass, field
import torch


# TODO: single global context variable might become a footgun if we ever overlap
# different models running simultaneously
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # MEDUSA tree-decode fields — only set when is_medusa_tree_decode=True
    is_medusa_tree_decode: bool = False
    # Additive bias [1, 1, medusa_len, medusa_len]: 0 for allowed, -inf for masked.
    # Pre-computed once from medusa_choices and reused every step.
    medusa_tree_mask: torch.Tensor | None = None
    # Committed-only sequence lengths [bs] used by the prefix FA call so that
    # the attention ignores the reserved tree slots already written to the cache.
    prefix_lens: torch.Tensor | None = None


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    is_medusa_tree_decode=False,
    medusa_tree_mask=None,
    prefix_lens=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        is_medusa_tree_decode,
        medusa_tree_mask,
        prefix_lens,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
