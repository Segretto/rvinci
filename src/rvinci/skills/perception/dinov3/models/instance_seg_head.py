import torch
from torch import nn
from typing import List
from transformers import AutoModelForUniversalSegmentation, AutoConfig

class Adapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: List[int]):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, out_ch, kernel_size=1) for out_ch in out_channels
        ])

        self.up_x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.up_x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.down_x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        projected = [self.projections[i](feat) for i, feat in enumerate(features)]
        out0 = self.up_x4(projected[0])
        out1 = self.up_x2(projected[1])
        out2 = projected[2]
        out3 = self.down_x2(projected[3])
        return [out0, out1, out2, out3]

class InstanceSegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Load mask2former with 1 class for initialization
        config = AutoConfig.from_pretrained("facebook/mask2former-swin-small-coco-instance")
        config.num_labels = 1
        mask2former_model = AutoModelForUniversalSegmentation.from_config(config)

        self.adapter = Adapter(in_channels=384, out_channels=[96, 192, 384, 768])
        self.mask2former = mask2former_model
    
    def forward(self, features: List[torch.Tensor]) -> dict:
        layers_to_extract = [2, 5, 8, 11]   
        selected_features = [features[i] for i in layers_to_extract]

        adapted = self.adapter(selected_features)

        decoder_output = self.mask2former.model.pixel_level_module.decoder(adapted)
        output = self.mask2former.model.transformer_module(
            multi_scale_features=decoder_output.multi_scale_features,
            mask_features=decoder_output.mask_features,
        )

        class_queries_logits = self.mask2former.class_predictor(output.last_hidden_state)
        masks_queries_logits = output.masks_queries_logits[-1]
        
        seg_out = {
            "class_queries_logits": class_queries_logits,
            "masks_queries_logits": masks_queries_logits
        }
        return seg_out
