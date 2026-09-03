"""Pure-PyTorch MEDUSA utilities ported from medusa_repo.

Only the functions needed for inference (candidate generation and
posterior evaluation) are included here. HuggingFace-specific routines
(initialize_medusa, tree_decoding, update_inference_inputs, etc.) are
intentionally omitted because nano-vllm handles the forward pass and KV
management through its own ModelRunner / paged-KV infrastructure.

Sources:
  medusa_repo/Medusa/medusa/model/utils.py
  medusa_repo/Medusa/medusa/model/medusa_model.py   (ResBlock)
  medusa_repo/Medusa/medusa/model/medusa_choices.py (default topologies)
"""

import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOPK = 10  # number of top-k tokens sampled per Medusa head per depth level

# Repo root (parent of ``nanovllm/``) so relative paths work when CWD is not the repo.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _medusa_choices_file_candidates(spec: str) -> list[str]:
    """Paths to try for a tree JSON file (Modal / bench CWD may vary)."""
    out = [spec, os.path.join(os.getcwd(), spec), os.path.join(_REPO_ROOT, spec)]
    # Dedupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def resolve_medusa_choices_file(spec: str) -> str | None:
    """Return the absolute/normalized path opened for ``spec``, or ``None`` if no file matches."""
    spec = (spec or "").strip()
    if not spec:
        return None
    for path in _medusa_choices_file_candidates(spec):
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def load_medusa_choices(spec: str) -> list | None:
    """Parse ``MEDUSA_CHOICES`` into a list of paths.

    * If ``spec`` is empty → ``None`` (caller uses built-in default).
    * If ``spec`` resolves to an existing file (tries CWD, then repo root) →
      load JSON (must be a JSON array of integer paths).
    * Otherwise → parse ``spec`` as JSON (legacy inline JSON string).
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    resolved = resolve_medusa_choices_file(spec)
    if resolved is not None:
        with open(resolved, encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(spec)
    except json.JSONDecodeError as e:
        tried = ", ".join(repr(p) for p in _medusa_choices_file_candidates(spec))
        raise ValueError(
            "MEDUSA_CHOICES is not a readable JSON file and is not valid JSON. "
            f"Tried paths: {tried}. JSON error: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Default tree topologies (ported from medusa_choices.py)
# ---------------------------------------------------------------------------

# Small two-head tree for high-batch-throughput MEDUSA runs.  The 63-choice
# defaults still leave 33-38 choices after clipping to two heads, which makes
# tree decode verify far more candidates than a weak/low-acceptance head pair
# can justify.  This keeps the most likely one-token branches and a few depth-2
# continuations while cutting medusa_len from 39 to 7 for the default topology.
vicuna_33b_heads2_fast = [
    (0,),
    (1,),
    (2,),
    (0, 0),
    (0, 1),
    (1, 0),
]

# ---------------------------------------------------------------------------
# ResBlock (Medusa head building block)
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Single residual block used inside each Medusa head.

    A Linear layer with SiLU activation whose output is added back to the
    input (residual connection). Weights are initialised to zero so that at
    the start of training the block acts as an identity map.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class MedusaBlock(nn.Module):
    """One Medusa head: ``num_layers`` :class:`ResBlock` then a vocab ``Linear``.

    Implemented as ``nn.Sequential`` under ``self.net`` so ``forward`` can be
    ``torch.compile``d as a single region. Checkpoints from the older layout
    (``ModuleList`` of bare ``Sequential``, keys ``<h>.0.linear…``) are
    adapted by :func:`remap_medusa_lm_head_state_dict` before ``load_state_dict``.
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int):
        super().__init__()
        layers: list[nn.Module] = [
            *(ResBlock(hidden_size) for _ in range(num_layers)),
            nn.Linear(hidden_size, vocab_size, bias=False),
        ]
        self.net = nn.Sequential(*layers)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_MEDUSA_HEAD_KEY = re.compile(r"^(\d+)\.(\d+)\.")


def remap_medusa_lm_head_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map flat ``medusa_lm_head.pt`` keys into :class:`MedusaBlock` layout.

    HuggingFace / Medusa checkpoints use ``{head_idx}.{seq_idx}.…`` (one
    ``nn.Sequential`` per head, no wrapper). We store ``Sequential`` as
    ``head.net``, so keys become ``{head_idx}.net.{seq_idx}.…``.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if ".net." in k:
            out[k] = v
            continue
        m = _MEDUSA_HEAD_KEY.match(k)
        if m:
            h, s = m.group(1), m.group(2)
            rest = k[m.end() :]
            out[f"{h}.net.{s}.{rest}"] = v
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Buffer generation (called once at engine init)
# ---------------------------------------------------------------------------


def _pad_path(path: list, length: int, pad_value: int = -2) -> list:
    return path + [pad_value] * (length - len(path))


def generate_medusa_buffers(
    medusa_choices: list,
    device: str = "cuda",
) -> dict:
    """Pre-compute all static tensors required for tree-structured decoding.

    Returns a dict with keys:
      - medusa_attn_mask   : [1, 1, medusa_len, medusa_len]  additive bias
                             (0 where attention is allowed, -inf elsewhere)
      - tree_indices       : [medusa_len]  maps flat candidate vector → tree node
      - medusa_position_ids: [medusa_len]  depth of each tree node (for RoPE)
      - retrieve_indices   : [num_paths, max_depth+1]  tree node → cartesian path
      - medusa_len         : int  total number of tree nodes (root + all paths)
    """
    sorted_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_choices) + 1  # +1 for the root node

    # Count how many choices exist at each depth level
    depth_counts: list[int] = []
    prev_depth = 0
    for path in sorted_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    # Build attention mask: 1 where node can attend, 0 elsewhere.
    # The root (index 0) is attended by everyone; each node also attends
    # to itself and all its ancestors.
    attn_mask = torch.eye(medusa_len, medusa_len)
    attn_mask[:, 0] = 1  # all nodes attend to root
    start = 0
    for i, count in enumerate(depth_counts):
        for j in range(count):
            cur = sorted_choices[start + j]
            if len(cur) == 1:
                continue
            ancestor_idx = [
                sorted_choices.index(cur[: c + 1]) + 1 for c in range(len(cur) - 1)
            ]
            attn_mask[start + j + 1, ancestor_idx] = 1
        start += count

    # Convert to additive bias: 0 → attend, -inf → mask out
    additive_mask = torch.zeros_like(attn_mask)
    additive_mask[attn_mask == 0] = float("-inf")

    # tree_indices: maps position in the flat candidate vector to tree node
    tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i, count in enumerate(depth_counts):
        for j in range(count):
            cur = sorted_choices[start + j]
            tree_indices[start + j + 1] = cur[-1] + TOPK * i + 1
        start += count

    # medusa_position_ids: depth of each tree node (used to offset RoPE)
    position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i, count in enumerate(depth_counts):
        position_ids[start + 1 : start + count + 1] = i + 1
        start += count

    # retrieve_indices: for each leaf-to-root path, the ordered tree node indices
    retrieve_indices_nest: list[list[int]] = []
    seen_paths: list = []
    for i in range(len(sorted_choices)):
        cur = sorted_choices[-i - 1]
        if cur in seen_paths:
            continue
        path_indices = []
        for c in range(len(cur)):
            path_indices.append(sorted_choices.index(cur[: c + 1]))
            seen_paths.append(cur[: c + 1])
        retrieve_indices_nest.append(path_indices)

    max_len = max(len(p) for p in retrieve_indices_nest)
    retrieve_indices = torch.tensor(
        [_pad_path(p, max_len) for p in retrieve_indices_nest], dtype=torch.long
    )
    # Shift by 1 (root is index 0 in the tree) and prepend a column of zeros
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat(
        [
            torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long),
            retrieve_indices,
        ],
        dim=1,
    )

    buffers = {
        "medusa_attn_mask": additive_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "medusa_position_ids": position_ids,
        "retrieve_indices": retrieve_indices,
        "medusa_len": medusa_len,
    }
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in buffers.items()
    }


# ---------------------------------------------------------------------------
# Candidate generation (called every decode step on CPU/CUDA)
# ---------------------------------------------------------------------------


def _get_typical_one_token_batched(
    logit: torch.Tensor,  # [B, vocab]
    temperature: float,
    posterior_threshold: float,
    posterior_alpha: float,
) -> torch.Tensor:  # [B, 1]
    # Upcast to fp32: at very low temperature (e.g. 1e-4) `logit / temperature`
    # trivially overflows fp16 (max 65504) and produces +inf, which causes
    # softmax to return NaN and multinomial to raise a device-side assert.
    logit = logit.float() / temperature
    probs = torch.softmax(logit, dim=-1)  # [B, V]
    entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1)  # [B]
    threshold = torch.minimum(
        torch.ones_like(entropy) * posterior_threshold,
        torch.exp(-entropy) * posterior_alpha,
    )  # [B]
    logit = logit.masked_fill(probs < threshold.unsqueeze(-1), float("-inf"))
    return torch.multinomial(F.softmax(logit, dim=-1), 1)  # [B, 1]


def _get_nucleus_one_token_batched(
    logit: torch.Tensor,  # [B, vocab]
    temperature: float,
    top_p: float,
) -> torch.Tensor:  # [B, 1]
    # fp32 upcast — same overflow concern as _get_typical_one_token_batched.
    logit = logit.float()
    if top_p >= 1:
        return torch.multinomial(F.softmax(logit / temperature, dim=-1), 1)
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_logits, dim=-1)
    remove = cum_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = 0
    # scatter `remove` back to original-logit ordering, then mask
    remove_in_orig = torch.zeros_like(remove).scatter(
        dim=1, index=sorted_indices, src=remove
    )
    logit = logit.masked_fill(remove_in_orig, float("-inf"))
    return torch.multinomial(F.softmax(logit, dim=-1), 1)


def generate_candidates(
    medusa_logits: torch.Tensor,
    logits: torch.Tensor,
    tree_indices: torch.Tensor,
    retrieve_indices: torch.Tensor,
    temperature: float = 0.0,
    posterior_threshold: float = 0.3,
    posterior_alpha: float = 0.09,
    top_p: float = 0.8,
    sampling: str = "typical",
    fast: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build tree candidates from the last-accepted-position logits (batched).

    Args:
        medusa_logits: [num_heads, B, 1, vocab_size]   head logits at last pos
        logits:        [B, 1, vocab_size]              LM-head logits at last pos
        tree_indices:      [medusa_len]   from generate_medusa_buffers
        retrieve_indices:  [num_paths, depth+1]  from generate_medusa_buffers
        temperature: scalar, applied uniformly across the batch (all seqs in a
            single MEDUSA step must share the same temperature — see
            _medusa_step for enforcement).
        posterior_threshold / posterior_alpha / top_p / sampling / fast:
            sampling hyper-parameters (mirror the original MEDUSA API).

    Returns:
        cart_candidates:  [B, num_paths, depth+1]  cartesian candidate paths
        tree_candidates:  [B, medusa_len]           token IDs in tree layout
    """
    B = logits.size(0)
    # Top-1 from LM head, shape [B, 1]
    if temperature == 0 or fast:
        candidates_logit = torch.argmax(logits[:, -1], dim=-1, keepdim=True)  # [B, 1]
    else:
        if sampling == "typical":
            candidates_logit = _get_typical_one_token_batched(
                logits[:, -1], temperature, posterior_threshold, posterior_alpha
            )  # [B, 1]
        elif sampling == "nucleus":
            candidates_logit = _get_nucleus_one_token_batched(
                logits[:, -1], temperature, top_p
            )
        else:
            raise NotImplementedError(f"Unknown sampling strategy: {sampling!r}")

    # Top-k from each head, shape [num_heads, B, TOPK] → [B, num_heads * TOPK]
    candidates_medusa = torch.topk(medusa_logits[:, :, -1], TOPK, dim=-1).indices
    candidates_medusa = candidates_medusa.permute(1, 0, 2).reshape(B, -1)

    # Flat candidate vector: [B, 1 + num_heads * TOPK]
    candidates = torch.cat([candidates_logit, candidates_medusa], dim=-1)

    # Map to tree layout: [B, medusa_len]
    tree_candidates = candidates[:, tree_indices]

    # Extend with a pad column of zeros so retrieve_indices == -? safely indexes 0.
    pad = torch.zeros(B, 1, dtype=tree_candidates.dtype, device=tree_candidates.device)
    tree_candidates_ext = torch.cat(
        [tree_candidates, pad], dim=-1
    )  # [B, medusa_len + 1]
    cart_candidates = tree_candidates_ext[
        :, retrieve_indices
    ]  # [B, num_paths, depth+1]

    return cart_candidates, tree_candidates


# ---------------------------------------------------------------------------
# Posterior evaluation (called every decode step on CPU/CUDA)
# ---------------------------------------------------------------------------


def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float,
    posterior_threshold: float = 0.3,
    posterior_alpha: float = 0.09,
    top_p: float = 0.8,
    sampling: str = "typical",
    fast: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the best candidate and its acceptance length (batched).

    Args:
        logits:     [B, num_paths, depth+1, vocab_size]  per-path per-position logits
        candidates: [B, num_paths, depth+1]              candidate token IDs
        temperature: scalar, uniform across the batch.
        posterior_threshold / posterior_alpha / top_p / sampling / fast:
            sampling hyper-parameters.

    Returns:
        best_candidate: [B] LongTensor — index of chosen path per seq
        accept_length:  [B] LongTensor — number of accepted speculative tokens per seq
    """
    # Greedy path (temperature == 0): deterministic argmax match.
    if temperature == 0:
        posterior_mask = (
            candidates[:, :, 1:] == torch.argmax(logits[:, :, :-1], dim=-1)
        ).int()  # [B, P, D]
        accept_len_per_path = torch.cumprod(posterior_mask, dim=-1).sum(
            dim=-1
        )  # [B, P]
        accept_length = accept_len_per_path.max(dim=-1).values  # [B]
        best_candidate = accept_len_per_path.argmax(dim=-1).to(torch.long)  # [B]
        return best_candidate, accept_length

    # Fast typical sampling — the default path used by the engine.
    if sampling == "typical" and fast:
        # fp32 upcast: avoids overflow at low temperature (logit / 1e-4 easily
        # exceeds fp16 range and yields NaN probabilities).
        posterior_prob = torch.softmax(
            logits[:, :, :-1].float() / temperature, dim=-1
        )  # [B, P, D, V]
        candidates_prob = torch.gather(
            posterior_prob, dim=-1, index=candidates[:, :, 1:].unsqueeze(-1)
        ).squeeze(
            -1
        )  # [B, P, D]
        posterior_entropy = -torch.sum(
            posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
        )  # [B, P, D]
        threshold = torch.minimum(
            torch.ones_like(posterior_entropy) * posterior_threshold,
            torch.exp(-posterior_entropy) * posterior_alpha,
        )
        posterior_mask = (candidates_prob > threshold).int()  # [B, P, D]
        accept_len_per_path = torch.cumprod(posterior_mask, dim=-1).sum(
            dim=-1
        )  # [B, P]
        accept_length = accept_len_per_path.max(dim=-1).values  # [B]

        # For seqs with nonzero accept_length, pick the path with the highest
        # joint log-prob among those achieving the max length. For seqs with
        # zero, best_candidate is unused downstream (accept_length = 0 short-
        # circuits to "commit only the bonus token"), so argmax of all-zero
        # lengths returning 0 is fine.
        is_max = accept_len_per_path == accept_length.unsqueeze(-1)  # [B, P]
        # log-prob sum up to the accept length; pad past accept_length with 0.
        log_prob = torch.log(candidates_prob.clamp_min(1e-9))  # [B, P, D]
        depth = log_prob.size(-1)
        pos = torch.arange(depth, device=log_prob.device)  # [D]
        valid = pos.unsqueeze(0) < accept_length.unsqueeze(-1)  # [B, D]
        masked_log_prob = log_prob * valid.unsqueeze(1).float()  # [B, P, D]
        likelihood = masked_log_prob.sum(dim=-1)  # [B, P]
        # Penalize non-max paths by -inf so argmax selects among is_max paths.
        likelihood = likelihood.masked_fill(~is_max, float("-inf"))
        best_candidate = likelihood.argmax(dim=-1).to(torch.long)  # [B]
        return best_candidate, accept_length

    raise NotImplementedError(
        f"Batched evaluate_posterior currently supports temperature==0 or "
        f"sampling='typical' with fast=True. Got sampling={sampling!r}, fast={fast!r}."
    )
