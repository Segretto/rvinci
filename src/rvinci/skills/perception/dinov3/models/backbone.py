import torch
import torch.nn as nn

class DinoBackbone(nn.Module):
    """
    Wraps a DINO vision transformer to return spatial feature maps (B, C, H, W).
    """
    def __init__(self, dino_model, n_layers=12):
        super().__init__()
        self.dino = dino_model
        self.n_layers = n_layers

    def forward(self, x):
        feats = self.dino.get_intermediate_layers(
            x, n=range(self.n_layers), reshape=True, norm=False
        )
        return feats
