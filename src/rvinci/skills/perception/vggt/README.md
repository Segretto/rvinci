# VGGT Perception Skill

This skill provides metric depth estimation using the VGGT (Vision-Guided Geometry Transformer) model.

## Purpose
Enables 3D perception by predicting per-pixel metric depth from standard RGB images.

## Features
- Scalable depth estimation (facebook/VGGT-1B).
- Support for batch processing and Mixed Precision (AMP).
- Metric depth output (meters).

## Usage

```python
from rvinci.skills.perception.vggt.api import VGGTAPI
from rvinci.skills.perception.vggt.schemas.config import VGGTConfig
from PIL import Image

# Initialize with custom config
config = VGGTConfig(device="cuda", batch_size=4)
api = VGGTAPI(config)

# Load image
img = Image.open("scene.jpg")

# Predict metric depth (meters)
depth = api.predict_single(img)
print(f"Mean depth: {depth.mean():.2f}m")
```

## Note
This skill is Hydra-agnostic. Orchestration and configuration management belong in project-level code.
