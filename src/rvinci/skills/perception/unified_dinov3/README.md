# Unified DINOv3 Skill

Provides the deep-learning capability for unified depth estimation and instance segmentation using a DINOv3 backbone. 
This skill encapsulates model heads originally implemented for customized multi-task prediction.

## Public API Usage

```python
from rvinci.skills.perception.unified_dinov3.api import UnifiedDINOv3Config, build_unified_dinov3
import torch

# Load a DINOv3 backbone (this should be handled outside the skill, usually in the project layer)
dino_backbone = torch.hub.load(
    repo_or_dir="facebookresearch/dinov2", # or local path
    model="dinov2_vts14",
)

config = UnifiedDINOv3Config(num_classes=150, embed_dim=384)
model = build_unified_dinov3(dino_backbone, config)

# Inputs: (B, 3, H, W)
depth_out, seg_out = model(image_tensor)
```

## Expected Inputs/Outputs
- **Input**: A normalized image tensor exactly matching what the chosen DINOv3 backbone expects.
- **Output**: Tuple containing depth map estimation and a dictionary of segmentation query logits.

> **Note:** This skill is Hydra-agnostic; orchestration belongs in projects.
