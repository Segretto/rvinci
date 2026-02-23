from pydantic import BaseModel, ConfigDict
from typing import Tuple

class UnifiedDINOv3Config(BaseModel):
    """
    Configuration for the Unified DINOv3 model architecture.
    """
    model_config = ConfigDict(extra="forbid")
    
    num_classes: int = 150
    n_layers: int = 12
    embed_dim: int = 384
    depth_out_size: Tuple[int, int] = (640, 640)
    seg_target_size: Tuple[int, int] = (160, 160)
