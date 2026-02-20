# Unified DINOv3 Inference Project

This project provides an orchestration layer to run inference using the `unified_dinov3` skill. It supports both single image processing and real-time video/camera stream inference.

## Features
- **Multi-task Inference**: Simultaneously predicts depth maps and instance segmentation.
- **Dynamic Input**: Automatically switches between image and video modes based on the `is_video` flag.
- **TensorRT Support**: Uses `torch.compile` with the TensorRT backend for optimized performance (requires compatible environment).

## Setup

Ensure you have the following weights available (paths can be configured in `configs/config.yaml` or via CLI):
1. **DINOv3 Backbone Weights**
2. **Unified Model Weights** (containing `depth_head` and `seg_head` state dicts)
3. **Class Names File** (text file with one class per line)

## Usage

Run the project via the canonical `uv` command:

### 1. Image Inference
```bash
uv run python -m rvinci.projects.dinov3_segdepth_inference.cli \
    input_source="path/to/your/image.jpg" \
    is_video=false
```

### 2. Video/Camera Inference
```bash
uv run python -m rvinci.projects.dinov3_segdepth_inference.cli \
    input_source="0" \
    is_video=true
```

## Configuration

The project uses Hydra for configuration. You can find the default values in `src/rvinci/projects/dinov3_segdepth_inference/configs/config.yaml`.

Key parameters:
- `unified_weights`: Path to the merged checkpoint.
- `backbone_weights`: Path to the pre-trained DINOv3 backbone.
- `fp16`: Set to `true` for half-precision inference.
- `image_size`: Resolution for inference (default: 640).

## Output
- For images: A combined visualization is saved to `outputs/inference_result.png`.
- For video: A real-time window displays the side-by-side results.
