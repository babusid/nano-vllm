"""EAGLE-3 inference utilities.

EAGLE-3 (Li et al., "EAGLE-3: Scaling up Inference Acceleration of LLMs via
Training-Time Test") differs from EAGLE-1/2 in three places that affect
inference plumbing:

  1. Multi-layer hidden fusion. The draft head consumes a *concatenation* of
     three hidden states from the target model (typically low / mid / late
     layers). The fc layer in the head projects [3H] -> [H].
  2. Reduced draft vocabulary. The head's lm_head outputs over a smaller
     `draft_vocab_size` (e.g. 32k for Qwen3) rather than the full target
     vocabulary. A `d2t` buffer maps draft-vocab indices to target-vocab token
     ids; a `t2d` buffer marks which target tokens are reachable from the
     draft.
  3. Multi-step training-time test. Inference is unchanged from a chain-draft
     perspective, but EAGLE-3 weights tend to give longer accept runs so
     longer draft horizons (k=5..7) are productive.

The tree-decode plumbing lives on the medusa path (tree mask, prefix-split
attention, captured CUDA graphs). EAGLE will share that machinery once the
chain-draft path is validated, so this file does NOT duplicate the tree
buffer code from medusa_utils — it only owns EAGLE-3-specific bits:
vocabulary mapping, default chain/tree topologies, and acceptance helpers.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Default draft topologies
# ---------------------------------------------------------------------------

# Chain-draft length used for first-pass validation (max_num_seqs=1 path).
# At each draft step the EAGLE head emits exactly one greedy token.
DEFAULT_CHAIN_DRAFT_LEN = 5

# EAGLE-3 published static tree (mirrors the SafeAILab repo's default 26-node
# tree for chat-tuned models). Each entry is a path from root; the topology is
# fed to the same generate_*_buffers helper used by medusa, so adopting EAGLE's
# tree later is a one-line swap. Kept here for reference / future use.
EAGLE3_DEFAULT_TREE: list[list[int]] = [
    [0],
    [1],
    [2],
    [3],
    [0, 0],
    [0, 1],
    [1, 0],
    [2, 0],
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0, 0],
]


# ---------------------------------------------------------------------------
# Draft -> Target vocabulary mapping
# ---------------------------------------------------------------------------


def map_draft_to_target_token(
    draft_token: int | torch.Tensor, d2t: torch.Tensor
) -> int | torch.Tensor:
    """Translate a draft-vocab token id to the corresponding target-vocab id.

    EAGLE-3 stores `d2t` as an additive offset table: target_id = draft_id +
    d2t[draft_id]. (This compresses the mapping; the offsets are typically
    small and many are zero.) Both scalar ints and tensors are supported so
    the same helper can be used in the per-step loop and in batched tensor
    paths.
    """
    if isinstance(draft_token, int):
        return int(draft_token + d2t[draft_token].item())
    return draft_token + d2t[draft_token]


def mask_draft_logits_to_target(
    draft_logits: torch.Tensor, t2d: torch.Tensor, target_vocab: int
) -> torch.Tensor:
    """Scatter draft-vocab logits into target-vocab shape, leaving non-draft
    target tokens at -inf so sampling never selects them.

    Args:
        draft_logits: [..., draft_vocab]
        t2d:          [target_vocab] bool — True where the target token has a
                      draft-vocab counterpart.
        target_vocab: int (sanity check)

    Returns:
        target_logits: [..., target_vocab]
    """
    assert t2d.numel() == target_vocab
    out_shape = draft_logits.shape[:-1] + (target_vocab,)
    target_logits = draft_logits.new_full(out_shape, float("-inf"))
    # scatter along the last dim into positions where t2d is True
    idx = t2d.nonzero(as_tuple=False).squeeze(-1)  # [draft_vocab]
    assert idx.numel() == draft_logits.size(-1), (
        f"t2d True-count ({idx.numel()}) must match draft vocab "
        f"({draft_logits.size(-1)})"
    )
    target_logits[..., idx] = draft_logits
    return target_logits


# ---------------------------------------------------------------------------
# Acceptance (chain draft)
# ---------------------------------------------------------------------------


def accept_chain_greedy(
    draft_tokens: list[int],
    verifier_tokens: list[int],
) -> tuple[list[int], int]:
    """Greedy acceptance over a chain-shaped draft.

    The verifier produced one token per (committed_last + draft_prefix) input,
    i.e. verifier_tokens has length len(draft_tokens) + 1. Acceptance walks
    the chain and stops at the first mismatch; on mismatch we still commit the
    verifier's token at that position (since it's a real argmax sample from
    the target). On full match we additionally commit the verifier's "bonus"
    token after the last draft.

    Returns:
        accepted: list of committed token ids (length 1 .. len(draft)+1)
        n_accepted_drafts: number of draft tokens that matched
    """
    assert (
        len(verifier_tokens) == len(draft_tokens) + 1
    ), f"verifier emitted {len(verifier_tokens)} tokens for {len(draft_tokens)} drafts"
    accepted: list[int] = []
    for i, d in enumerate(draft_tokens):
        if d == verifier_tokens[i]:
            accepted.append(d)
            continue
        # mismatch: commit the verifier's argmax at this position and stop
        accepted.append(verifier_tokens[i])
        return accepted, i
    # full chain matched — commit the bonus token too
    accepted.append(verifier_tokens[-1])
    return accepted, len(draft_tokens)
