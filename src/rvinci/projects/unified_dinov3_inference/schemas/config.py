from pydantic import BaseModel, ConfigDict, Field
from rvinci.skills.perception.unified_dinov3.schemas.config import UnifiedDINOv3Config

class ProjectConfig(BaseModel):
    """Configuration for running Unified DINOv3 Inference."""
    model_config = ConfigDict(extra="forbid")

    input_source: str = Field(description="Camera ID (int) or Video Path/Image Path (str)")
    is_video: bool = Field(default=False, description="Whether the input source is a video/camera stream")
    fp16: bool = Field(default=False, description="Use Half Precision (FP16)")
    
    # Paths
    unified_weights: str = Field(description="Path to merged checkpoint")
    backbone_weights: str = Field(description="Path to DINOv3 backbone weights")
    dino_dir: str = Field(description="Path to local dinov3 code")
    dino_model_type: str = Field(default="dinov3_vits16plus", description="Type of DINOv3 backbone")
    class_names: str = Field(description="Path to class names")

    # Image sizes
    image_size: int = Field(default=640, description="Input inference size")

    # Unified DINOv3 options
    dinov3_config: UnifiedDINOv3Config = Field(default_factory=UnifiedDINOv3Config)
