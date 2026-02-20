from .schemas.config import UnifiedDINOv3Config
from .models.model import UnifiedDINOv3
import torch

def build_unified_dinov3(dino_backbone: torch.nn.Module, config: UnifiedDINOv3Config) -> UnifiedDINOv3:
    """
    Builds the UnifiedDINOv3 model from a loaded DINOv3 backbone and configuration.
    
    Args:
        dino_backbone: The pre-loaded DINOv3 model (e.g., via torch.hub)
        config: Expected Pydantic config containing hyper-parameters.
    
    Returns:
        Instantiated and initialized UnifiedDINOv3 model.
    """
    return UnifiedDINOv3(
        dino_model=dino_backbone,
        num_classes=config.num_classes,
        n_layers=config.n_layers,
        embed_dim=config.embed_dim,
        depth_out_size=config.depth_out_size,
        seg_target_size=config.seg_target_size
    )

__all__ = ["UnifiedDINOv3Config", "UnifiedDINOv3", "build_unified_dinov3"]
