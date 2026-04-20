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

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOPK = 10  # number of top-k tokens sampled per Medusa head per depth level

# ---------------------------------------------------------------------------
# Default tree topologies (ported from medusa_choices.py)
# ---------------------------------------------------------------------------

# General-purpose 63-node tree (model-agnostic, good default)
mc_sim_7b_63 = [
    [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3],
    [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7],
    [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9],
    [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4],
    [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0],
    [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4],
    [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0],
    [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6],
    [0, 7, 0],
]

# Vicuna-7B stage-2 optimised topology (63 paths)
vicuna_7b_stage2 = [
    (0,), (0, 0), (1,), (0, 1), (0, 0, 0), (1, 0), (2,), (0, 2),
    (0, 0, 1), (0, 3), (3,), (0, 1, 0), (2, 0), (4,), (0, 0, 2),
    (0, 4), (1, 1), (1, 0, 0), (0, 0, 0, 0), (5,), (0, 0, 3), (0, 5),
    (0, 2, 0), (3, 0), (0, 1, 1), (0, 6), (6,), (0, 7), (0, 0, 4),
    (4, 0), (1, 2), (0, 8), (7,), (0, 3, 0), (0, 0, 0, 1), (0, 0, 5),
    (2, 1), (0, 0, 6), (1, 0, 1), (0, 0, 1, 0), (2, 0, 0), (5, 0),
    (0, 9), (0, 1, 2), (8,), (0, 4, 0), (0, 2, 1), (1, 3), (0, 0, 7),
    (0, 0, 0, 2), (0, 0, 8), (1, 1, 0), (0, 1, 0, 0), (6, 0), (9,),
    (0, 1, 3), (0, 0, 0, 3), (1, 0, 2), (0, 5, 0), (3, 1), (0, 0, 2, 0),
    (7, 0), (1, 4),
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
                sorted_choices.index(cur[: c + 1]) + 1
                for c in range(len(cur) - 1)
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
        [torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
        dim=1,
    )

    buffers = {
        "medusa_attn_mask": additive_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "medusa_position_ids": position_ids,
        "retrieve_indices": retrieve_indices,
        "medusa_len": medusa_len,
    }
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in buffers.items()}


# ---------------------------------------------------------------------------
# Candidate generation (called every decode step on CPU/CUDA)
# ---------------------------------------------------------------------------

def _get_typical_one_token(
    logit: torch.Tensor,
    temperature: float,
    posterior_threshold: float,
    posterior_alpha: float,
) -> torch.Tensor:
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1)
    threshold = torch.minimum(
        torch.ones_like(entropy) * posterior_threshold,
        torch.exp(-entropy) * posterior_alpha,
    )
    logit[probs < threshold.unsqueeze(-1)] = float("-inf")
    return torch.multinomial(F.softmax(logit, dim=-1), 1)


def _get_nucleus_one_token(
    logit: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if top_p >= 1:
        return torch.multinomial(F.softmax(logit / temperature, dim=-1), 1)
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_logits, dim=-1)
    remove = cum_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = 0
    logit[remove.scatter(dim=1, index=sorted_indices, src=remove)] = float("-inf")
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
    """Build tree candidates from the last-accepted-position logits.

    Args:
        medusa_logits: [num_heads, 1, 1, vocab_size]  head logits at last pos
        logits:        [1, 1, vocab_size]              LM-head logits at last pos
        tree_indices:  [medusa_len]   from generate_medusa_buffers
        retrieve_indices: [num_paths, depth+1]  from generate_medusa_buffers
        temperature / posterior_threshold / posterior_alpha / top_p / sampling:
            sampling hyper-parameters (mirror the orignal MEDUSA API).
        fast: if True use the faster typical-sampling approximation.

    Returns:
        cart_candidates:  [num_paths, max_depth+1]  cartesian candidate paths
        tree_candidates:  [1, medusa_len]            token IDs in tree layout
    """
    if temperature == 0 or fast:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        if sampling == "typical":
            candidates_logit = _get_typical_one_token(
                logits[:, -1], temperature, posterior_threshold, posterior_alpha
            ).squeeze(0)
        elif sampling == "nucleus":
            candidates_logit = _get_nucleus_one_token(
                logits[:, -1], temperature, top_p
            ).squeeze(0)
        else:
            raise NotImplementedError(f"Unknown sampling strategy: {sampling!r}")

    # Top-k from each head, shape [num_heads, TOPK]
    candidates_medusa = torch.topk(medusa_logits[:, 0, -1], TOPK, dim=-1).indices

    # Flat candidate vector: [1 + num_heads * TOPK]
    candidates = torch.cat([candidates_logit, candidates_medusa.view(-1)], dim=-1)

    # Map to tree layout
    tree_candidates = candidates[tree_indices]
    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros(1, dtype=torch.long, device=tree_candidates.device)],
        dim=0,
    )
    cart_candidates = tree_candidates_ext[retrieve_indices]

    return cart_candidates, tree_candidates.unsqueeze(0)


# ---------------------------------------------------------------------------
# Posterior evaluation (called every decode step on CPU/CUDA)
# ---------------------------------------------------------------------------

def _get_typical_posterior_mask(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float,
    posterior_threshold: float,
    posterior_alpha: float,
) -> torch.Tensor:
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[:2]
    logits = logits.view(n_samples * n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1)
    threshold = torch.minimum(
        torch.ones_like(entropy) * posterior_threshold,
        torch.exp(-entropy) * posterior_alpha,
    )
    logits[probs < threshold.unsqueeze(-1)] = float("-inf")
    sampled = torch.multinomial(F.softmax(logits, dim=-1), 1).view(n_samples, n_tokens)
    return (candidates[:, 1:] == sampled).int()


def _get_nucleus_posterior_mask(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[:2]
    logits = logits.view(n_samples * n_tokens, -1)
    if top_p >= 1:
        sampled = torch.multinomial(F.softmax(logits, dim=-1), 1).view(n_samples, n_tokens)
        return (candidates[:, 1:] == sampled).int()
    probs = F.softmax(logits, dim=-1)
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_logits, dim=-1)
    remove = cum_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = 0
    logits[remove.scatter(dim=1, index=sorted_indices, src=remove)] = float("-inf")
    sampled = torch.multinomial(F.softmax(logits, dim=-1), 1).view(n_samples, n_tokens)
    return (candidates[:, 1:] == sampled).int()


def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float,
    posterior_threshold: float = 0.3,
    posterior_alpha: float = 0.09,
    top_p: float = 0.8,
    sampling: str = "typical",
    fast: bool = True,
) -> tuple[torch.Tensor, int]:
    """Select the best candidate and its acceptance length.

    Args:
        logits:     [num_paths, depth+1, vocab_size]  per-path per-position logits
        candidates: [num_paths, depth+1]              candidate token IDs
        temperature / posterior_threshold / posterior_alpha / top_p / sampling / fast:
            sampling hyper-parameters.

    Returns:
        best_candidate: scalar LongTensor — index of chosen path
        accept_length:  int — number of accepted speculative tokens (≥ 0)
    """
    if temperature == 0:
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
        accept_length = candidates_accept_length.max()
        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, int(accept_length)

    if sampling == "typical":
        if fast:
            posterior_prob = torch.softmax(logits[:, :-1] / temperature, dim=-1)
            candidates_prob = torch.gather(
                posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            posterior_entropy = -torch.sum(
                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
            )
            threshold = torch.minimum(
                torch.ones_like(posterior_entropy) * posterior_threshold,
                torch.exp(-posterior_entropy) * posterior_alpha,
            )
            posterior_mask = candidates_prob > threshold
            candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
            accept_length = candidates_accept_length.max()
            if accept_length == 0:
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidates = torch.where(candidates_accept_length == accept_length)[0]
                likelihood = torch.sum(
                    torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1
                )
                best_candidate = best_candidates[torch.argmax(likelihood)]
            return best_candidate, int(accept_length)

        posterior_mask = _get_typical_posterior_mask(
            logits, candidates, temperature, posterior_threshold, posterior_alpha
        )
        candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
        accept_length = candidates_accept_length.max()
        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, int(accept_length)

    if sampling == "nucleus":
        assert top_p < 1.0 + 1e-6, "top_p must be in (0, 1]"
        posterior_mask = _get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
        accept_length = candidates_accept_length.max()
        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, int(accept_length)

    raise NotImplementedError(f"Unknown sampling strategy: {sampling!r}")
