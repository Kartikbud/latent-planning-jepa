"""Goal-conditioned autoregressive policy over standardized primitive actions.

Fixed five-action adaptation, not a complete FlexiWorld reproduction. The actor
sees the current fused latent, goal intent, their product, the previous action
block, and the shifted within-block prefix. It never sees the future state.
"""
import math

import torch
from torch import nn


class GoalActor(nn.Module):
    def __init__(self, d_model=192, action_dim=7, block_size=5,
                 nlayers=3, nheads=4, ff_dim=768, dropout=0.1):
        super().__init__()
        self.d_model, self.action_dim, self.block_size = d_model, action_dim, block_size
        self.config = dict(d_model=d_model, action_dim=action_dim, block_size=block_size,
                           nlayers=nlayers, nheads=nheads, ff_dim=ff_dim, dropout=dropout)
        self.context = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.previous = nn.Sequential(nn.Linear(block_size * action_dim, 64), nn.GELU(),
                                      nn.Linear(64, d_model))
        self.action = nn.Linear(action_dim, d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.randn(1, 4 + block_size, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nheads, ff_dim, dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, nlayers, norm=nn.LayerNorm(d_model),
                                                 enable_nested_tensor=False)
        self.mean = nn.Linear(d_model, action_dim)
        self.log_std = nn.Linear(d_model, action_dim)

    def forward(self, current, goal, previous, prefix):
        """prefix [B,L,7] predicts L+1 actions (max five), using only past tokens."""
        if prefix.shape[1] >= self.block_size:
            raise ValueError('Prefix must be shorter than the action block')
        intent = goal - current
        context = torch.stack([p(x) for p, x in zip(
            self.context, (current, intent, current * intent))] +
            [self.previous(previous.flatten(1))], dim=1)
        tokens = torch.cat((context, self.bos.expand(current.shape[0], -1, -1),
                            self.action(prefix)), dim=1)
        tokens = tokens + self.position[:, :tokens.shape[1]]
        causal = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool,
                            device=tokens.device).triu(1)
        hidden = self.transformer(tokens, mask=causal)[:, 4:]
        return self.mean(hidden), self.log_std(hidden).clamp(-5, 2)

    @torch.no_grad()
    def generate(self, current, goal, previous):
        prefix = current.new_empty(current.shape[0], 0, self.action_dim)
        for _ in range(self.block_size):
            mean, _ = self(current, goal, previous, prefix)
            prefix = torch.cat((prefix, mean[:, -1:]), dim=1)
        return prefix


def actor_objective(actor, current, goal, previous, targets, valid, student_probability=0.0):
    """Masked per-sample Gaussian NLL; Student Forcing only changes prefixes.

    Goal branch detached; current branch remains attached. No privileged state.
    Validation should pass student_probability=0 for deterministic teacher forcing.
    """
    prefix = targets[:, :-1]
    if student_probability > 0:
        was_training = actor.training
        actor.eval()
        try:
            generated = actor.generate(current.detach(), goal.detach(), previous.detach())
        finally:
            actor.train(was_training)
        select = torch.rand(targets.shape[0], 1, 1, device=targets.device) < student_probability
        prefix = torch.where(select, generated[:, :-1], prefix)
    mean, log_std = actor(current, goal.detach(), previous, prefix)
    mean, log_std, targets = mean.float(), log_std.float(), targets.float()
    nll = 0.5 * ((targets - mean) * torch.exp(-log_std)).square() + log_std + 0.5 * math.log(2 * math.pi)
    mask = valid.float()
    denominator = mask.sum(1).clamp_min(1)
    loss = ((nll.mean(-1) * mask).sum(1) / denominator).mean()
    mse = (((targets - mean).square().mean(-1) * mask).sum(1) / denominator).mean()
    return loss, mse
