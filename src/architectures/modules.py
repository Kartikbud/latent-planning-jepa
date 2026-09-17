import torch
import torch.nn.functional as F
from torch import nn


class ProjectionMLP(nn.Module):
    """LeWM projection head used after encoder and predictor features."""

    def __init__(self, input_dim: int = 192, hidden_dim: int = 2048, output_dim: int = 192):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class ActionEncoder(nn.Module):
    """Encode one ordered primitive-action block for AdaLN conditioning."""

    def __init__(self, input_dim: int, smoothed_dim: int = 10, embedding_dim: int = 192):
        super().__init__()
        self.smoother = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(smoothed_dim, 4 * embedding_dim),
            nn.SiLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, actions):
        batch_size, sequence_length, _ = actions.shape
        actions = self.smoother(actions.transpose(1, 2)).transpose(1, 2)
        return self.mlp(actions.reshape(batch_size * sequence_length, -1)).reshape(
            batch_size, sequence_length, -1
        )


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional AdaLN-Zero conditioning."""

    def __init__(
        self,
        d_model: int = 192,
        nheads: int = 16,
        d_action: int | None = None,
        dropout: float = 0.1,
        causal_masking: bool = True,
        adaptive: bool = True,
        dim_head: int | None = None,
        mlp_dim: int | None = None,
    ):
        super().__init__()
        self.adaptive = adaptive
        hidden_dim = mlp_dim or 4 * d_model

        self.norm_one = nn.LayerNorm(d_model, elementwise_affine=not adaptive)
        self.norm_two = nn.LayerNorm(d_model, elementwise_affine=not adaptive)
        if adaptive:
            if d_action is None:
                raise ValueError("d_action is required for an adaptive transformer block")
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_action, 6 * d_model),
            )
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)

        self.attn_block = MultiHeadAttention(
            d_model=d_model,
            nheads=nheads,
            dim_head=dim_head,
            dropout=dropout,
            causal_masking=causal_masking,
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, z, a=None):
        if self.adaptive:
            if a is None:
                raise ValueError("action conditioning is required for an adaptive block")
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(a).chunk(6, dim=-1)
            )
            z = z + gate_attn * self.attn_block(
                self._modulate(self.norm_one(z), shift_attn, scale_attn)
            )
            z = z + gate_mlp * self.mlp(
                self._modulate(self.norm_two(z), shift_mlp, scale_mlp)
            )
        else:
            z = z + self.attn_block(self.norm_one(z))
            z = z + self.mlp(self.norm_two(z))
        return z


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 192,
        nheads: int = 16,
        dim_head: int | None = None,
        dropout: float = 0.1,
        causal_masking: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.d_k = dim_head or (d_model // nheads)
        self.nheads = nheads
        self.inner_dim = self.nheads * self.d_k
        self.causal_masking = causal_masking

        self.W_q = nn.Linear(d_model, self.inner_dim)
        self.W_k = nn.Linear(d_model, self.inner_dim)
        self.W_v = nn.Linear(d_model, self.inner_dim)
        self.W_o = nn.Linear(self.inner_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        queries = self.W_q(x).reshape(
            batch_size, sequence_length, self.nheads, self.d_k
        ).transpose(1, 2)
        keys = self.W_k(x).reshape(
            batch_size, sequence_length, self.nheads, self.d_k
        ).transpose(1, 2)
        values = self.W_v(x).reshape(
            batch_size, sequence_length, self.nheads, self.d_k
        ).transpose(1, 2)

        attention = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal_masking,
        )
        attention = attention.transpose(1, 2).reshape(
            batch_size, sequence_length, self.inner_dim
        )
        return self.drop(self.W_o(attention))
