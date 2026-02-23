import os
import cv2
import time
import torch
import numpy as np
from transformers import AutoImageProcessor
from transformers.utils import ModelOutput

from rvinci.core.logging import get_logger
from rvinci.projects.dinov3_inf.schemas.config import ProjectConfig
from rvinci.libs.vision_data.processing import image_to_tensor
from rvinci.libs.visualization.drawing import (
    depth_to_colormap,
    generate_instance_overlay,
)
from rvinci.libs.vision_data.constants import IMG_MEAN, IMG_STD
from rvinci.libs.inference.optimization import optimize_and_warmup
from rvinci.libs.utils.video import get_media_stream
from rvinci.skills.perception.dinov3.api import build_dinov3

log = get_logger(__name__)


def run_pipeline(config: ProjectConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # Load class names
    if not os.path.exists(config.class_names):
        log.error(f"Class names file not found at {config.class_names}")
        raise FileNotFoundError(config.class_names)

    with open(config.class_names, "r") as f:
        class_names = {i: line.strip() for i, line in enumerate(f)}
    log.info(f"Loaded {len(class_names)} classes.")

    # Override config num_classes with loaded classes if needed, but it's already in dinov3_config
    if config.dinov3_config.num_classes != len(class_names):
        log.warning(
            f"Config num_classes ({config.dinov3_config.num_classes}) != file lines ({len(class_names)})"
        )

    # Load DINOv3 Backbone
    log.info(
        f"Loading DINOv3 backbone from {config.dino_dir} weights {config.backbone_weights}"
    )
    dino_model = torch.hub.load(
        repo_or_dir=config.dino_dir,
        model=config.dino_model_type,
        source="local",
        weights=config.backbone_weights,
    )

    # Build Unified DINOv3 Skill Model
    model = build_dinov3(dino_model, config.dinov3_config)
    model.to(device)

    # Load Custom Weights
    log.info("Loading custom weights from ...")

    try:
        base_dir = os.path.dirname(config.unified_weights)
        model.depth_head.load_state_dict(
            torch.load(os.path.join(base_dir, "depth_head.pth"), map_location=device)
        )
        model.seg_head.load_state_dict(
            torch.load(
                os.path.join(base_dir, "instance_seg_head.pt"), map_location=device
            )
        )
    except Exception as e:
        log.error(f"Failed to load head weights: {e}")
        raise

    model.eval()
    if config.fp16 and device == "cuda":
        model = model.half()

    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-small-coco-instance"
    )

    # Optimize model (TensorRT + Warmup) if on Aarch64
    trt_model = optimize_and_warmup(model, config, device)

    # Prepare Normalization Parameters
    mean_np = np.array(IMG_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std_np = np.array(IMG_STD, dtype=np.float32).reshape(3, 1, 1)
    dtype = torch.float16 if config.fp16 else torch.float32

    # Get unified media stream
    stream = get_media_stream(config.input_source, is_video=config.is_video)

    for frame_idx, original_bgr_frame, original_rgb_frame in stream:
        log.info(f"Processing frame {frame_idx}...")

        # Resize to network resolution
        input_rgb = cv2.resize(
            original_rgb_frame,
            (config.image_size, config.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        # Some draw logic expects original BGR (or resized BGR)
        input_bgr = cv2.resize(
            original_bgr_frame,
            (config.image_size, config.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        img_tensor = image_to_tensor(input_rgb, mean_np, std_np)
        img_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=dtype)

        with torch.inference_mode():
            t0 = time.time()
            depth_pred, seg_logits = trt_model(img_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            inf_time = (t1 - t0) * 1000
            fps = 1.0 / (t1 - t0 + 1e-9)
            if not config.is_video:
                log.info(f"Inference time: {inf_time:.2f} ms")

            depth_map = depth_pred.squeeze().float().cpu().numpy()
            results = processor.post_process_instance_segmentation(
                ModelOutput(seg_logits),
                target_sizes=[(config.image_size, config.image_size)],
            )

        depth_vis = depth_to_colormap(depth_map, bgr=True)
        seg_vis = generate_instance_overlay(
            input_bgr if config.is_video else original_bgr_frame,
            results[0]["segmentation"],
            results[0]["segments_info"],
            class_names,
            target_size=(config.image_size, config.image_size),
        )
        seg_vis = cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR)

        combined = np.hstack((seg_vis, depth_vis))

        # Add FPS overlay if video
        if config.is_video:
            cv2.putText(
                combined,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        if not config.is_video:
            output_path = "outputs/inference_result.png"
            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite(output_path, combined)
            log.info(f"Saved visualization to {output_path}")

        window_name = "Unified DINOv3 (Left: Seg, Right: Depth)"
        try:
            cv2.imshow(window_name, combined)
            if not config.is_video:
                log.info(
                    "Visualization open. Press any key or close the window to exit."
                )

            wait_time = 1 if config.is_video else 50
            if not config.is_video:
                while True:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if cv2.waitKey(wait_time) != -1:
                        break
            else:
                if (
                    cv2.waitKey(wait_time) & 0xFF == ord("q")
                    or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
        except cv2.error:
            log.warning("Display visualization failed (likely no DISPLAY available).")

    cv2.destroyAllWindows()
