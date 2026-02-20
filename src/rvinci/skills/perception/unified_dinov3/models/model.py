import torch
import torch.nn as nn
from rvinci.skills.perception.unified_dinov3.models.backbone import DinoBackbone
from rvinci.skills.perception.unified_dinov3.models.depth_head import DepthHeadLite
from rvinci.skills.perception.unified_dinov3.models.instance_seg_head import InstanceSegmentationHead

class UnifiedDINOv3(nn.Module):
    def __init__(self, dino_model, num_classes: int, n_layers: int = 12, embed_dim: int = 384, depth_out_size: tuple = (640, 640), seg_target_size: tuple = (320, 320)):
        """
        Unified Depth and Instance Segmentation on top of DINOv3.
        dino_model: The pre-loaded DINOv3 model (e.g. via torch.hub)
        num_classes: Number of segmentation classes
        n_layers: Number of layers in DINO backbone to extract feature from (usually 12 for vits)
        embed_dim: Embedding dimension of DINO backbone (384 for vits)
        depth_out_size: Output resolution for depth map
        seg_target_size: Output resolution for segmentation map
        """
        super().__init__()
        self.backbone = DinoBackbone(dino_model, n_layers=n_layers)
        self.depth_head = DepthHeadLite(in_ch=embed_dim, out_size=depth_out_size)
        self.seg_head = InstanceSegmentationHead()

        # Remove the pixel value encoder as we rely entirely on DINOv3 features
        del self.seg_head.mask2former.model.pixel_level_module.encoder

        # Note: In the original implementation, the classifier weight was updated using from pretrained,
        # but in inference scripts it was overridden. The unified config can define how to handle num_classes if a custom head is needed, 
        # but for now we follow the script's loading patterns.

    def forward(self, x):
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        feat = self.backbone(x)
        
        # In inference, stream logic works best when synchronous operations don't block.
        # However, for pure nn.Module simplicity we can conditionally enable streams if available,
        # but pure sequential works perfectly well on small sizes. 
        # Leaving the stream logic since original authors used it for performance gains.
        if x.is_cuda:
            with torch.cuda.stream(s1):    
                depth_out = self.depth_head(feat[-1])
            with torch.cuda.stream(s2):
                seg_out = self.seg_head(feat)
            torch.cuda.synchronize()
        else:
            depth_out = self.depth_head(feat[-1])
            seg_out = self.seg_head(feat)

        return depth_out, seg_out
