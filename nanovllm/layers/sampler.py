import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        sample_tokens = torch.empty(
            logits.size(0), dtype=torch.int64, device=logits.device
        )
        greedy_mask = temperatures <= 0
        if greedy_mask.any():
            sample_tokens[greedy_mask] = logits[greedy_mask].argmax(dim=-1)
        sample_mask = ~greedy_mask
        if sample_mask.any():
            sampled_logits = torch.div(
                logits[sample_mask], temperatures[sample_mask].unsqueeze(dim=1)
            )
            probs = torch.softmax(sampled_logits, dim=-1)
            noise = torch.clamp_min(torch.empty_like(probs).exponential_(1), 1e-10)
            sample_tokens[sample_mask] = torch.div(probs, noise).argmax(dim=-1)
        return sample_tokens
