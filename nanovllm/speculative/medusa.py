"""
MEDUSA-1 speculative decoding implementation.

Architecture
------------
Each MedusaHead is a lightweight residual MLP:
    x → Linear(H→H) + SiLU + residual → LM-head (weight-tied to base model)

K heads are stacked as MedusaHeads, one per speculative lookahead position.
Head i predicts the token at position N+i+1 (0-indexed from 0).

Training (MEDUSA-1)
-------------------
The base model is frozen; only the heads are trained.
See train_medusa_heads.py for the training loop.

Checkpoint format
-----------------
Heads are saved as a single safetensors file with keys:
    medusa_heads.{head_idx}.blocks.{block_idx}.linear.{weight|bias}
    medusa_heads.{head_idx}.lm_head.weight  (optional; loaded from base model)
The file can live inside the model directory or at medusa_model_path.

Acceptance (Phase 1 — greedy / argmax)
---------------------------------------
Given verification logits V[0..K] for positions N..N+K:
  - V[j] predicts the token at position N+j+1
  - accepted = [d_0]                     (d_0 always accepted — base model)
  - for j in 0..K-1:
      if argmax(V[j]) == c_{j+1}:  append c_{j+1}
      else:                         append argmax(V[j]), break
  - if loop completes without break:   append argmax(V[K])  (bonus token)
  => accepted length in [2, K+2]

Phase 2 (not yet implemented): stochastic typical-acceptance scheme for
temperature > 0 and full tree attention.
"""

import os

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.speculative.base import AcceptOutput, DraftOutput, SpeculativeDecoder


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MedusaResBlock(nn.Module):
    """Single residual block used inside each MEDUSA draft head."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class MedusaHead(nn.Module):
    """One MEDUSA draft head: num_layers residual blocks + an LM head."""

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MedusaResBlock(hidden_size) for _ in range(num_layers)]
        )
        # lm_head weight is weight-tied to the base model's lm_head after init
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


class MedusaHeads(nn.Module):
    """K MEDUSA draft heads bundled into a single module."""

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [MedusaHead(hidden_size, vocab_size, num_layers) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_size]  — last-token hidden states per sequence
        Returns:
            [batch, num_heads, vocab_size]  — one logit vector per head per seq
        """
        return torch.stack([head(x) for head in self.heads], dim=1)


# ---------------------------------------------------------------------------
# Decoder (implements the SpeculativeDecoder interface)
# ---------------------------------------------------------------------------

class MedusaDecoder(SpeculativeDecoder):
    """
    Implements draft() and verify() for MEDUSA-1.

    Parameters
    ----------
    heads : MedusaHeads
        The K lightweight draft head modules.
    base_lm_head_weight : torch.Tensor
        The base model's lm_head weight matrix.  Each MedusaHead's lm_head
        will share this weight (MEDUSA-1 tying).
    """

    def __init__(self, heads: MedusaHeads, base_lm_head_weight: torch.Tensor):
        self.heads = heads
        # Weight-tie every draft head's LM head to the base model's LM head.
        for head in self.heads.heads:
            head.lm_head.weight = base_lm_head_weight

    # ------------------------------------------------------------------
    # SpeculativeDecoder interface
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def draft(
        self,
        hidden_states: torch.Tensor,
        seqs: list,
    ) -> DraftOutput:
        """Run K MEDUSA heads on the base model's hidden states.

        Args:
            hidden_states: [batch, hidden_size]
            seqs: list of Sequence objects (used for future tree extensions)

        Returns:
            DraftOutput where draft_tokens[i] = [c_1, ..., c_K] for seq i.
            tree_attn_mask is None (linear chain — Phase 1).
        """
        # heads_logits: [batch, K, vocab_size]
        heads_logits = self.heads(hidden_states)

        # Greedy argmax per head: [batch, K]
        head_preds = heads_logits.argmax(dim=-1)

        # Convert to list-of-lists: [[c_1..c_K], ...] per sequence
        draft_tokens = head_preds.tolist()

        return DraftOutput(draft_tokens=draft_tokens, tree_attn_mask=None)

    def verify(
        self,
        seqs: list,
        draft: DraftOutput,
        verify_logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> AcceptOutput:
        """Accept or reject draft tokens using greedy argmax matching.

        The full_drafts passed to prepare_verify() are [d_0, c_1, ..., c_K],
        so the verification forward produced K+1 logits per sequence:
            V[0] at position N   → predicts token at N+1 → verifies c_1
            V[1] at position N+1 → predicts token at N+2 → verifies c_2
            ...
            V[K-1] at position N+K-1 → predicts token at N+K → verifies c_K
            V[K]   at position N+K   → "bonus" token (always taken)

        Greedy acceptance rule for each seq:
            accepted = [d_0]
            for j in 0..K-1:
                base_pred = argmax(V[j])
                if base_pred == c_{j+1}: accept c_{j+1}
                else: accept base_pred, stop
            if all K verified and accepted: also take argmax(V[K]) as bonus

        Args:
            seqs:          list of Sequence (for length; not mutated here)
            draft:         DraftOutput from draft() — draft_tokens[i] = [c_1..c_K]
                           NOTE: by the time verify() is called, full_drafts
                           (with d_0 prepended) are stored on draft.draft_tokens.
            verify_logits: [sum(K+1), vocab_size] — ALL logits from verification
            temperatures:  [batch] — not used in Phase-1 greedy; reserved for
                           Phase-2 stochastic acceptance

        Returns:
            AcceptOutput with accepted_tokens[i] (length in [2, K+2]) and metrics.
        """
        K = len(draft.draft_tokens[0]) - 1  # full_drafts has K+1 entries per seq

        accepted_tokens: list[list[int]] = []
        total_steps = 0
        total_accepted = 0
        # Per-head acceptance counters: how often each head's prediction was used
        per_head_hits = [0] * K
        per_head_tries = [0] * K

        offset = 0
        for seq_idx, full_draft in enumerate(draft.draft_tokens):
            d0 = full_draft[0]          # base model's token (always accepted)
            head_preds = full_draft[1:]  # [c_1, ..., c_K] from MEDUSA heads

            # Logits for this sequence: K+1 positions
            seq_logits = verify_logits[offset: offset + K + 1]  # [K+1, vocab_size]
            offset += K + 1

            # Greedy predictions from the base model at each position
            base_preds = seq_logits.argmax(dim=-1).tolist()  # [K+1]

            accepted = [d0]
            for j in range(K):
                per_head_tries[j] += 1
                base_pred = base_preds[j]   # what base model says comes at N+j+1
                draft_pred = head_preds[j]  # what MEDUSA head j predicted

                if base_pred == draft_pred:
                    accepted.append(draft_pred)
                    per_head_hits[j] += 1
                else:
                    # Take the corrected base-model token and stop
                    accepted.append(base_pred)
                    break
            else:
                # All K drafts accepted — also take the bonus from V[K]
                accepted.append(base_preds[K])

            accepted_tokens.append(accepted)
            total_steps += 1
            total_accepted += len(accepted)

        metrics = {
            "total_steps": total_steps,
            "total_accepted": total_accepted,
            "mean_accepted_per_step": total_accepted / total_steps if total_steps else 0.0,
            "per_head_hits": per_head_hits,
            "per_head_tries": per_head_tries,
            "per_head_acceptance_rate": [
                h / t if t > 0 else 0.0
                for h, t in zip(per_head_hits, per_head_tries)
            ],
        }

        return AcceptOutput(accepted_tokens=accepted_tokens, metrics=metrics)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_medusa_heads(
    heads: MedusaHeads,
    model_path: str,
    medusa_model_path: str = "",
) -> None:
    """Load MEDUSA head weights from a safetensors checkpoint.

    Search order:
      1. medusa_model_path  (if non-empty and is a file)
      2. <model_path>/medusa_heads.safetensors
      3. <model_path>/medusa_heads.pt

    If no checkpoint is found, head weights remain randomly initialised
    (useful for measuring overhead without a trained checkpoint).

    Weight-tying of lm_head is NOT done here — the caller (ModelRunner) must
    do it after loading the base model weights.
    """
    from safetensors.torch import load_file

    candidates = []
    if medusa_model_path and os.path.isfile(medusa_model_path):
        candidates.append(medusa_model_path)
    candidates.append(os.path.join(model_path, "medusa_heads.safetensors"))
    candidates.append(os.path.join(model_path, "medusa_heads.pt"))

    ckpt_path = None
    for path in candidates:
        if os.path.isfile(path):
            ckpt_path = path
            break

    if ckpt_path is None:
        print(
            "[MEDUSA] No checkpoint found — heads are randomly initialised. "
            "Run train_medusa_heads.py to train them or provide a checkpoint."
        )
        return

    print(f"[MEDUSA] Loading heads from {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Strip an optional "medusa_heads." prefix so the dict matches MedusaHeads
    cleaned = {}
    for k, v in state_dict.items():
        key = k.removeprefix("medusa_heads.")
        cleaned[key] = v

    missing, unexpected = heads.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[MEDUSA] Missing keys (will use random init): {missing}")
    if unexpected:
        print(f"[MEDUSA] Unexpected keys (ignored): {unexpected}")
