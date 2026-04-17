import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        stochastic = temperatures >= 1e-10  # False → greedy for that position

        if not stochastic.any():
            return logits.argmax(dim=-1)

        safe_temps = temperatures.clamp(min=1e-10).unsqueeze(1)
        probs = torch.softmax(logits / safe_temps, dim=-1)
        sampled = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        if stochastic.all():
            return sampled

        # Mixed batch: some sequences greedy, some stochastic
        greedy = logits.argmax(dim=-1)
        return torch.where(stochastic, sampled, greedy)
