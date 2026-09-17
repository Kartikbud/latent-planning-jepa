import sys
from pathlib import Path

import torch
from torch import nn


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from architectures.modules import ActionEncoder, ProjectionMLP, TransformerBlock


class PredictorNetwork(nn.Module):
    """Action-conditioned latent predictor matching the released LeWM design."""

    def __init__(
        self,
        d_model: int = 192,
        nheads: int = 16,
        d_action: int = 2,
        dropout: float = 0.1,
        nlayers: int = 6,
        window: int = 3,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        action_smoothing_dim: int = 10,
        projector_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_action = d_action
        self.window = window
        self.action_encoder = ActionEncoder(
            input_dim=d_action,
            smoothed_dim=action_smoothing_dim,
            embedding_dim=d_model,
        )
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nheads=nheads,
                    d_action=d_model,
                    dropout=dropout,
                    causal_masking=True,
                    adaptive=True,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                )
                for _ in range(nlayers)
            ]
        )
        self.position_emb = nn.Parameter(torch.empty(1, window, d_model))
        nn.init.trunc_normal_(self.position_emb, std=0.02)
        self.final_norm = nn.LayerNorm(d_model)
        self.projector = ProjectionMLP(d_model, projector_hidden_dim, d_model)

    def forward(self, z, a):
        batch_size, sequence_length, embedding_dim = z.shape
        if embedding_dim != self.d_model:
            raise ValueError(f"expected embedding size {self.d_model}, got {embedding_dim}")
        if sequence_length > self.window:
            raise ValueError(f"sequence length {sequence_length} exceeds window {self.window}")
        if a.shape[:2] != z.shape[:2] or a.shape[-1] != self.d_action:
            raise ValueError(
                f"expected actions [B, {sequence_length}, {self.d_action}], got {tuple(a.shape)}"
            )

        z = z + self.position_emb[:, :sequence_length]
        action_embedding = self.action_encoder(a)
        for layer in self.transformer_layers:
            z = layer(z, action_embedding)
        z = self.final_norm(z)
        z = self.projector(z.reshape(batch_size * sequence_length, self.d_model))
        return z.reshape(batch_size, sequence_length, self.d_model)


if __name__ == "__main__":
    dummy = PredictorNetwork(d_action=35)
    print(sum(parameter.numel() for parameter in dummy.parameters()))
