"""
Abstract interface for speculative decoding strategies.

Both MEDUSA and EAGLE implement this interface so they can be swapped in the
ModelRunner without any changes to the surrounding inference machinery.

Terminology used throughout:
  N       — number of tokens already in the sequence (including prompt)
  K       — number of MEDUSA draft heads (== num_speculative_tokens in Config)
  d_0     — the base model's own next-token prediction (always accepted)
  c_1..cK — draft head predictions for positions N+1 .. N+K
  V_j     — verification logit at position N+j (produced by the base model
             after seeing d_0..c_{j-1} in the verification forward pass)

Linear-chain Phase-1 flow (no tree branching):
  draft()  → DraftOutput with draft_tokens[i] = [c_1 .. c_K] (no d_0 yet)
  verify() → AcceptOutput with accepted_tokens[i] = [d_0, (c_1, ..,) bonus]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class DraftOutput:
    """Tokens proposed by the speculative strategy for each sequence in the batch.

    draft_tokens[i] is the list [c_1, c_2, ..., c_K] for sequence i.
    d_0 (the base model's own prediction) is NOT included here; it is prepended
    by ModelRunner.run_speculative() before the verification pass.

    tree_attn_mask:
      None  → linear/causal chain (Phase 1). The verification pass uses standard
              causal attention and no extra masking is needed.
      Tensor → tree-structured mask (Phase 2, not yet implemented).
    """
    draft_tokens: list[list[int]]
    tree_attn_mask: torch.Tensor | None = None


@dataclass
class AcceptOutput:
    """Result of the accept/reject step for each sequence.

    accepted_tokens[i] is the list of tokens to append to sequence i.
    Length is at least 2 (d_0 + at least one more token from the verification
    logit) and at most K+2 (d_0 + K accepted heads + bonus).

    metrics holds per-batch statistics for ablation logging.
    """
    accepted_tokens: list[list[int]]
    metrics: dict = field(default_factory=dict)


class SpeculativeDecoder(ABC):
    """Strategy interface for speculative decoding.

    Concrete implementations: MedusaDecoder (this file), EagleDecoder (TBD).

    The ModelRunner calls these two methods once per speculative decode step:

      1. draft(hidden_states, seqs) → DraftOutput
         Called after the base model's single-token decode forward pass.
         hidden_states: [batch, hidden_size] — the last-token hidden states from
         the base model (before the LM head). Used by MEDUSA heads directly;
         EAGLE would ignore this and use a separate draft model instead.

      2. verify(seqs, draft, verify_logits, temperatures) → AcceptOutput
         Called after the verification forward pass.
         verify_logits: [batch * (K+1), vocab_size] — logits at every draft
         token position (all_logits=True context flag must be set).
    """

    @abstractmethod
    def draft(
        self,
        hidden_states: torch.Tensor,
        seqs: list,
    ) -> DraftOutput:
        """Produce draft tokens from the base model's last-token hidden states."""
        ...

    @abstractmethod
    def verify(
        self,
        seqs: list,
        draft: DraftOutput,
        verify_logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> AcceptOutput:
        """Accept or reject draft tokens based on the verification logits."""
        ...
