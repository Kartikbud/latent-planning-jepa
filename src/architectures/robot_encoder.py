import sys
from pathlib import Path

from torch import nn
import torch


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from architectures.encoder import EncoderNetwork
from architectures.modules import ProjectionMLP

"""
This is the modified encoder network that extends it to take in
2 camera observations plus robot state information and then output
a final embedding in latent space. Also modifying it to be 84x84 to match
the mimicgen dataset instead of 224x224 which was for the original definition.
This encoder will now seperately pass through each input image through
the default encoder and then concatenate them together with the robot state
embedding and then this concatenated vector will be passed through a fusion MLP
to compress it back to the original embedding space. The output of this encoder
will still be the standard dimension so it can be passed through the default
predictor
"""


class RobotEncoder(nn.Module):
    def __init__(self,
                 patch : int = 14,
                 nheads : int = 3,
                 nlayers : int = 12,
                 d_model : int = 192,
                 d_state : int = 9,
                 d_state_emb : int = 64,
                 npatches : int = 36,
                 projector_hidden_dim: int = 2048):  # 84x84, patch 14 → 6x6 patches
        super().__init__()
        self.encoder = EncoderNetwork(patch=patch,
                                      nheads=nheads,
                                      nlayers=nlayers,
                                      d_model=d_model,
                                      npatches=npatches,
                                      projector_hidden_dim=projector_hidden_dim,
                                      project_output=False)

        # small projection network for the 
        self.state_encoder = nn.Sequential(
            nn.Linear(in_features=d_state, out_features=2*d_state),
            nn.GELU(),
            nn.Linear(in_features=2*d_state, out_features=d_state_emb)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_features=(d_model * 2 + d_state_emb), out_features=(d_model * 2)),
            nn.GELU(),
            nn.Linear(in_features=(d_model * 2), out_features=d_model),
        )
        self.projector = ProjectionMLP(d_model, projector_hidden_dim, d_model)

    def forward(self, above, wrist, state):
        # above : [Batch, Window, 3, 84, 84] | wrist : [Batch, Window, 3, 84, 84] | state : [Batch, Window, 9]
        # Run both views together so the shared encoder's BatchNorm sees the
        # same mixed camera distribution during training and inference.
        batch_size = above.shape[0]
        view_emb = self.encoder(torch.cat((above, wrist), dim=0))
        above_emb, wrist_emb = view_emb.split(batch_size, dim=0)
        state_emb = self.state_encoder(state) # [Batch, Window, 64]
        fusion = torch.cat((above_emb, wrist_emb, state_emb), dim=-1) # [Batch, Window, 448]
        fusion = self.fusion_mlp(fusion)
        flat = fusion.reshape(-1, fusion.shape[-1])
        return self.projector(flat).reshape_as(fusion) # [Batch, Window, 192]


if __name__ == "__main__":
    dummy = RobotEncoder()
    above = torch.ones(3, 3, 3, 84, 84)
    wrist = torch.ones(3, 3, 3, 84, 84)
    state = torch.ones(3, 3, 9)
    print(dummy(above, wrist, state).shape)
