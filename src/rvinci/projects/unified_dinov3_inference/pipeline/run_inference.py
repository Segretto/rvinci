import os
import cv2
import time
import torch
import numpy as np
from transformers import AutoImageProcessor
from transformers.utils import ModelOutput
import torch_tensorrt

from rvinci.core.logging import get_logger
from rvinci.projects.dinov3_segdepth_inference.schemas.config import ProjectConfig
from rvinci.projects.dinov3_segdepth_inference.pipeline.utils import (
    image_to_tensor, depth_to_colormap, generate_instance_overlay
)
from rvinci.skills.perception.unified_dinov3.api import build_unified_dinov3

log = get_logger(__name__)

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

def run_pipeline(config: ProjectConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # Load class names
    if not os.path.exists(config.class_names):
        log.error(f"Class names file not found at {config.class_names}")
        raise FileNotFoundError(config.class_names)
    
    with open(config.class_names, 'r') as f:
        class_names = {i: line.strip() for i, line in enumerate(f)}
    log.info(f"Loaded {len(class_names)} classes.")

    # Override config num_classes with loaded classes if needed, but it's already in dinov3_config
    if config.dinov3_config.num_classes != len(class_names):
        log.warning(f"Config num_classes ({config.dinov3_config.num_classes}) != file lines ({len(class_names)})")

    # Load DINOv3 Backbone
    log.info(f"Loading DINOv3 backbone from {config.dino_dir} weights {config.backbone_weights}")
    dino_model = torch.hub.load(
        repo_or_dir=config.dino_dir,
        model=config.dino_model_type,
        source="local",
        weights=config.backbone_weights
    )

    # Build Unified DINOv3 Skill Model
    model = build_unified_dinov3(dino_model, config.dinov3_config)
    model.to(device)

    # Load Custom Weights
    log.info(f"Loading custom weights from {config.unified_weights}")
    checkpoint = torch.load(config.unified_weights, map_location=device)
    if 'depth_head' in checkpoint and 'seg_head' in checkpoint:
        model.depth_head.load_state_dict(checkpoint['depth_head'])
        model.seg_head.load_state_dict(checkpoint['seg_head'])
    else:
        log.warning("Unified weights didn't contain 'depth_head' and 'seg_head' keys directly. Trying to load individual files if present in the same directory.")
        try:
            base_dir = os.path.dirname(config.unified_weights)
            model.depth_head.load_state_dict(torch.load(os.path.join(base_dir, 'depth_head.pth'), map_location=device))
            model.seg_head.load_state_dict(torch.load(os.path.join(base_dir, 'instance_seg_head.pt'), map_location=device))
        except Exception as e:
            log.error(f"Failed to load head weights: {e}")
            raise

    model.eval()
    if config.fp16 and device == "cuda":
        model = model.half()

    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")

    opts = {
        "torch_executed_ops": {"aten.native_group_norm.default"},
        "min_block_size": 10,
        "debug": False,
    }
    if config.fp16:
        opts["enabled_precisions"] = {torch.float16}

    trt_model = torch.compile(model, backend="tensorrt", options=opts)
    
    # Warmup
    log.info("Warming up model...")
    dtype = torch.float16 if config.fp16 else torch.float32
    example_input = torch.randn(1, 3, config.image_size, config.image_size, dtype=dtype, device=device)
    with torch.no_grad(), torch.inference_mode():
        trt_model(example_input)

    # Prepare Image Transformations
    mean_tensor = torch.tensor(IMG_MEAN, dtype=dtype, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(IMG_STD, dtype=dtype, device=device).view(1, 3, 1, 1)

    if config.is_video:
        source = int(config.input_source) if config.input_source.isdigit() else config.input_source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        log.info("Starting stream. Press 'q' to exit.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                input_frame = cv2.resize(frame, (config.image_size, config.image_size), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                
                img_tensor = torch.from_numpy(img_rgb).to(device)
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(dtype)
                img_tensor = img_tensor / 255.0
                img_tensor = (img_tensor - mean_tensor) / std_tensor

                with torch.inference_mode():
                    t0 = time.time()
                    depth_pred, seg_logits = trt_model(img_tensor)
                    t1 = time.time()
                    
                    depth_map = depth_pred.squeeze().float().cpu().numpy()
                    results = processor.post_process_instance_segmentation(
                        ModelOutput(seg_logits),
                        target_sizes=[(config.image_size, config.image_size)]
                    )

                depth_vis = depth_to_colormap(depth_map, bgr=True)
                seg_vis = generate_instance_overlay(input_frame, results[0]["segmentation"], results[0]["segments_info"], class_names)
                seg_vis = cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR)

                combined = np.hstack((seg_vis, depth_vis))
                cv2.putText(combined, f"FPS: {1.0/(t1-t0):.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Unified DINOv3 (Left: Seg, Right: Depth)", combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        log.info(f"Processing image {config.input_source}...")
        image = cv2.imread(config.input_source)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {config.input_source}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (config.image_size, config.image_size), interpolation=cv2.INTER_LINEAR)

        img_tensor = torch.from_numpy(image_resized).to(device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(dtype)
        img_tensor = img_tensor / 255.0
        img_tensor = (img_tensor - mean_tensor) / std_tensor

        with torch.inference_mode():
            t0 = time.time()
            depth_pred, seg_logits = trt_model(img_tensor)
            torch.cuda.synchronize()
            t1 = time.time()
            log.info(f"Inference time: {(t1-t0)*1000:.2f} ms")

            depth_map = depth_pred.squeeze().float().cpu().numpy()
            results = processor.post_process_instance_segmentation(
                ModelOutput(seg_logits),
                target_sizes=[(config.image_size, config.image_size)]
            )

        depth_vis = depth_to_colormap(depth_map, bgr=True)
        seg_vis = generate_instance_overlay(image_resized, results[0]["segmentation"], results[0]["segments_info"], class_names)
        seg_vis = cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR)

        combined = np.hstack((seg_vis, depth_vis))
        output_path = "outputs/inference_result.png"
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(output_path, combined)
        log.info(f"Saved visualization to {output_path}")

        # Try to show it if display is available
        try:
            cv2.imshow("Unified DINOv3 (Left: Seg, Right: Depth)", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass
