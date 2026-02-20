import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import random

from rvinci.core.logging import get_logger
from rvinci.projects.sam3_pipeline.schemas.config import ProjectConfig
from rvinci.projects.sam3_pipeline.pipeline.utils import resolve_run_dir, load_coco_data, save_coco_format
from rvinci.projects.sam3_pipeline.pipeline.interactive import InteractiveLabeler, format_inputs_for_sam3

from rvinci.skills.perception.sam3.api import Sam3VideoTracker, Sam3ImagePredictor
from rvinci.skills.perception.sam3.schemas.config import Sam3TrackerConfig, Sam3PredictorConfig

from rvinci.libs.vision_data.filtering import filter_coco_by_confidence
from rvinci.libs.visualization.drawing import draw_bounding_boxes, draw_segmentation_masks

log = get_logger(__name__)


def get_image_paths(images_dir: str):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
    paths.sort()
    return paths

def random_color(seed: int):
    random.seed(seed)
    return (int(random.random() * 255), int(random.random() * 255), int(random.random() * 255))


def run_tracking(config: ProjectConfig, run_dir: str):
    cfg = config.tracking
    log.info(f"Running SAM3 Video Tracking on {cfg.images_dir}")

    coco_data = load_coco_data(str(cfg.input_json))
    all_image_paths = get_image_paths(str(cfg.images_dir))

    if not all_image_paths:
        log.error("No images found.")
        return

    start_idx = 0
    end_idx = len(all_image_paths) - 1

    if cfg.range:
        start_idx = max(0, cfg.range[0] - 1)
        end_idx = min(len(all_image_paths) - 1, cfg.range[1])

    target_paths = all_image_paths[start_idx:end_idx + 1]

    # Manual Labeling
    inference_prompts = {}

    for frame_idx in cfg.manual_ids:
        if frame_idx < start_idx or frame_idx > end_idx:
            continue

        log.info(f"Labeling frame {frame_idx}")
        labeler = InteractiveLabeler(all_image_paths[frame_idx])
        user_data = labeler.run()

        if user_data:
            obj_ids, points, labels = format_inputs_for_sam3(user_data)
            inference_prompts[frame_idx - start_idx] = {
                "obj_ids": obj_ids,
                "points": [points], # Add batch dimension
                "labels": [labels], # Add batch dimension
            }

    if not inference_prompts:
        log.warning("No manual inputs provided. Exiting.")
        return

    tracker_cfg = Sam3TrackerConfig(model_id=cfg.model_id)
    tracker = Sam3VideoTracker(tracker_cfg)

    # Load frames as video
    video_frames = []
    for path in target_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video_frames.append(img)

    log.info("Propagating video segments...")
    video_segments = tracker.track(video_frames, inference_prompts)

    # Convert to COCO
    new_annotations = []
    next_ann_id = max([ann["id"] for ann in coco_data.get("annotations", [])], default=0) + 1

    for session_idx, objects in video_segments.items():
        global_idx = start_idx + session_idx
        filename = os.path.basename(all_image_paths[global_idx])

        image_id = None
        for img in coco_data["images"]:
            if img["file_name"] == filename:
                image_id = img["id"]
                break
        
        if image_id is None:
            continue

        for obj_id, mask in objects.items():
            mask = np.array(mask)
            while mask.ndim > 2:
                mask = mask[0]

            if mask.ndim != 2:
                continue

            mask_binary = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            segmentation = []
            for contour in contours:
                if len(contour) >= 3:
                    segmentation.append(contour.flatten().tolist())

            if not segmentation:
                continue

            x, y, w, h = cv2.boundingRect(mask_binary)
            area = float(w * h)

            ann = {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": segmentation,
                "area": area,
                "bbox": [x, y, w, h],
                "iscrowd": 0,
                "attributes": {"track_id": obj_id},
            }

            new_annotations.append(ann)
            next_ann_id += 1

    if "annotations" not in coco_data:
        coco_data["annotations"] = []
    coco_data["annotations"].extend(new_annotations)

    out_path = os.path.join(run_dir, "tracking_annotations.json")
    if str(cfg.output_json) != "outputs/tracking_annotations.json":
        out_path = str(cfg.output_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    save_coco_format(out_path, coco_data)
    log.info(f"Done tracking. Saved to {out_path}")


def run_text_prompt(config: ProjectConfig, run_dir: str):
    cfg = config.text_prompt
    log.info(f"Running SAM3 Text Prompt inference on {cfg.images_dir}")

    pred_cfg = Sam3PredictorConfig(model_id=cfg.model_id)
    predictor = Sam3ImagePredictor(pred_cfg)

    image_paths = get_image_paths(str(cfg.images_dir))
    
    if not image_paths:
        log.error("No images found.")
        return

    coco_output = {
        "info": {"description": "SAM3 Auto-Annotations"},
        "images": [],
        "annotations": [],
        "categories": [],
    }

    prompt_to_cat_id = {}
    for idx, prompt in enumerate(cfg.prompts):
        cat_id = idx + 1
        coco_output["categories"].append({
            "id": cat_id,
            "name": prompt,
            "supercategory": "object"
        })
        prompt_to_cat_id[prompt] = cat_id

    ann_id = 1
    
    for img_idx, img_path in enumerate(tqdm(image_paths, desc="Text inference")):
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        image_id = img_idx + 1

        coco_output["images"].append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "height": height,
            "width": width,
        })

        predictions = predictor.predict(image, cfg.prompts)
        
        for prompt, instances in predictions.items():
            cat_id = prompt_to_cat_id[prompt]
            
            for inst in instances:
                mask_np = inst["mask"]
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                for contour in contours:
                    if len(contour) < 3:
                        continue
                        
                    x, y, w, h = cv2.boundingRect(contour)
                    area = float(cv2.contourArea(contour))
                    polygon = contour.flatten().tolist()
                    
                    coco_output["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [x, y, w, h],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [polygon],
                        "score": inst["score"]
                    })
                    ann_id += 1

    out_path = os.path.join(run_dir, "text_prompt_annotations.json")
    if str(cfg.output_json) != "outputs/text_prompt_annotations.json":
        out_path = str(cfg.output_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
    save_coco_format(out_path, coco_output)
    log.info(f"Done text prompt sequence. Saved to {out_path}")


def run_filter(config: ProjectConfig, run_dir: str):
    cfg = config.filter
    log.info(f"Filtering dataset COCO at {cfg.input_json}")
    
    out_path = os.path.join(run_dir, "filtered_annotations.json")
    if str(cfg.output_json) != "outputs/filtered_annotations.json":
        out_path = str(cfg.output_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
    filter_coco_by_confidence(
        input_json_path=cfg.input_json,
        output_json_path=out_path,
        min_score=cfg.threshold,
        remove_empty_images=cfg.remove_empty_images
    )
    log.info(f"Filtered annotations saved at {out_path}")


def run_visualize(config: ProjectConfig, run_dir: str):
    cfg = config.visualize
    log.info(f"Visualizing datasets from {cfg.annotations}")

    out_dir = os.path.join(run_dir, "visualizations")
    if str(cfg.output_dir) != "outputs/visualizations":
        out_dir = str(cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(cfg.annotations, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    
    # Custom visualization specific category logic to bridge to `drawing.py`
    class_map = {}
    for cat in coco.get("categories", []):
        class_map[cat["id"]] = {
            "name": cat["name"],
            "color": random_color(cat["id"])
        }

    anns_per_image = {}
    for ann in coco.get("annotations", []):
        anns_per_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, image_info in tqdm(images.items(), desc="Visualizing"):
        file_name = image_info["file_name"]
        image_path = os.path.join(cfg.images_dir, file_name)

        if not os.path.exists(image_path):
            continue

        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size
        annotations = anns_per_image.get(image_id, [])

        boxes_to_render = []
        masks_to_render = []

        for ann in annotations:
            cat_id = ann["category_id"]
            if cat_id not in class_map:
                continue

            color = class_map[cat_id]["color"]
            cat_name = class_map[cat_id]["name"]
            
            label_text = f"{cat_name} (id:{cat_id})"
            if not cfg.hide_score and "score" in ann:
                label_text += f" {ann['score']:.2f}"

            if "bbox" in ann:
                # coco bbox is [x, y, w, h]
                x, y, w, h = ann["bbox"]
                bbox = [x, y, x+w, y+h]
                boxes_to_render.append({
                    "box": bbox,
                    "color": color,
                    "text": label_text,
                })
                
            if "segmentation" in ann and isinstance(ann["segmentation"], list):
                for poly in ann["segmentation"]:
                    poly_norm = []
                    # drawing.py expects absolute format for flat arrays if passed properly or normalized.
                    # It accepts absolute: abs_polygon = [(int(polygon[i]), int(polygon[i+1]))] but wait,
                    # drawing.py assumes normalized polygon [0, 1] usually because it does `int(pt[0] * img_width)`.
                    # Let's normalize it here!
                    for i in range(0, len(poly), 2):
                        poly_norm.append([poly[i] / img_w, poly[i+1] / img_h])
                        
                    masks_to_render.append({
                        "polygon": poly_norm,
                        "color": color,
                        "text": None # We drew text via box already
                    })

        if masks_to_render:
            img = draw_segmentation_masks(img, masks_to_render, class_map, alpha=0.4)
            
        if boxes_to_render:
            img = draw_bounding_boxes(img, boxes_to_render, class_map)
            
        img.save(os.path.join(out_dir, file_name))

    log.info(f"Visualizations saved to {out_dir}")


def run_pipeline(config: ProjectConfig) -> None:
    run_dir = resolve_run_dir(config.project.runs_root, config.project.name)
    log.info(f"Starting sam3_pipeline in mode '{config.mode}' ... [Run Dir: {run_dir}]")

    if config.mode == "tracking":
        run_tracking(config, run_dir)
    elif config.mode == "text_prompt":
        run_text_prompt(config, run_dir)
    elif config.mode == "filter":
        run_filter(config, run_dir)
    elif config.mode == "visualize":
        run_visualize(config, run_dir)
    else:
        log.error(f"Unknown mode: {config.mode}")

