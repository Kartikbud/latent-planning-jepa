from torch import nn
import torch
from architectures.vit_tiny import CustomVitTiny
from architectures.modules import ProjectionMLP

"""
This is the network definition for the encoder network which in the paper
was just ViT-Tiny which is a visual transformer with the following parameters:
- patch size of 14, 12 layers, 3 attention heads, and a hidden dimension of 192
- there is a "projection" layer added to the end of this because visual transformers
  typically have layernorms at the end but this ruins the regularization used to
  prevent representation collapse
    - this projection layer is a single layer with batch normalization to undo the
      affects of the layernorm at the end
"""
# right now the image size is defaulted to 224x224

class EncoderNetwork(nn.Module):
    def __init__(self,
                 patch : int = 14,
                 nheads : int = 3,
                 nlayers : int = 12,
                 d_model : int = 192,
                 npatches : int = 256,
                 projector_hidden_dim: int = 2048,
                 project_output: bool = True): # npatches is (H/patch_size)*(W/patch_size)
        super().__init__()
        self.d_model = d_model
        
        self.vit = CustomVitTiny(patch, nheads, nlayers, d_model, npatches)
        self.projector = (
            ProjectionMLP(d_model, projector_hidden_dim, d_model)
            if project_output else nn.Identity()
        )
    
    def forward(self, x):
        # shape of input (x): (B, N, C, H, W)
        B, N, C, H, W = x.shape
        z = self.vit(x)
        #projection step
        z = z.reshape(B*N, self.d_model)
        z = self.projector(z)
        z = z.reshape(B, N, self.d_model)

        return z
    

# TEST Code
# dummy_image = torch.ones(24, 3, 3, 224, 224)
# enc = EncoderNetwork()
# print(enc(dummy_image).shape)

