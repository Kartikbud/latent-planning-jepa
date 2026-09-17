from torch import nn
import torch
from architectures.modules import TransformerBlock

"""
Thought it would be fun to define the architecture of the ViT-Tiny
from scratch so here it is using the transformers I defined as well
"""

class CustomVitTiny(nn.Module):
    def __init__(self,
                 patch : int = 14,
                 nheads : int = 3,
                 nlayers : int = 12,
                 d_model : int = 192,
                 npatches : int = 256):
        super().__init__()

        self.npatches = npatches
        self.d_model = d_model

        self.conv_flattener = nn.Conv2d(in_channels=3,
                                        out_channels=d_model,
                                        kernel_size=patch,
                                        stride=patch,
                                        padding=0)
        
        self.transformer_layers = nn.ModuleList([])
        for _ in range(nlayers):
            self.transformer_layers.append(TransformerBlock(d_model=d_model, 
                                           nheads=nheads,                                            
                                           dropout=0.0,
                                           causal_masking=False,
                                           adaptive=False))
        # learned positional embeddings
        self.position_emb = nn.Parameter(torch.empty(npatches + 1, d_model))
        # learned [cls] embedding
        self.cls_emb = nn.Parameter(torch.empty(1, 1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.position_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x):
        # shape of x: (B, N, C, H, W)
        B, N, C, H, W = x.shape
        x = x.reshape(B*N, C, H, W)
        emb = self.conv_flattener(x)
        emb = emb.reshape(B, N, self.d_model, self.npatches).permute(0, 1, 3, 2)
        # shape of emb now: (512, 3, 256, 192)
        cls_emb = self.cls_emb.expand(B, N, 1, self.d_model)
        emb = torch.cat([cls_emb, emb], dim=2)
        emb = emb + self.position_emb
        # shape of emb now: (512, 3, 257, 192)
        emb = emb.reshape(B*N, self.npatches + 1, self.d_model) # now ready to go into the transformer

        for layer in (self.transformer_layers):
           emb = layer(emb)
        emb = self.norm(emb)

        emb = emb.reshape(B, N, self.npatches + 1, self.d_model)
        emb = emb[:, :, 0, :] # shape : (512, 3, 192), this is what the predictor expects as the input

        return emb





# TEST CODE for Shape
# dummy_conv = nn.Conv2d(in_channels=3,
#           out_channels=192,
#           kernel_size=14,
#           stride=14,
#           padding=0)

# dummy_frame = torch.ones((512, 3, 3, 224, 224))
# dummy_frame = dummy_frame.reshape(512*3, 3, 224, 224)
# emb = dummy_conv(dummy_frame)

# print(emb.shape)
# print("after reshaping")
# emb = emb.reshape((512, 3, 192, 256))
# emb = emb.permute(0, 1, 3, 2)
# print(emb.shape)
# cls = torch.ones(1, 1, 1, 192)
# cls = cls.expand(512, 3, 1, cls.shape[-1])
# emb = torch.cat([cls, emb], dim=2)
# print(emb.shape)
# emb = emb[:, :, 0, :]
# print(emb.shape)

# TEST CODE for the Architecture
# test = CustomVitTiny()
# dummy = torch.ones(2, 3, 3, 224, 224)
# dummy = test(dummy)
# print(dummy.shape)
