from pydantic import BaseModel, Field
from typing import Optional

class VGGTConfig(BaseModel):
    """Configuration for VGGT Depth skill."""
    model_id: str = Field(default="facebook/VGGT-1B", description="Hugging Face model ID or local path")
    device: str = Field(default="cuda", description="Device for inference (cuda, cpu)")
    batch_size: int = Field(default=16, ge=1, description="Batch size for inference")
    scale_m: float = Field(default=1.0, gt=0, description="Metric scale factor for depth results")
    amp_enabled: bool = Field(default=True, description="Enable automatic mixed precision")

    class Config:
        extra = "forbid"
