#!/usr/bin/env python3
import argparse
import sys
import os
import logging
from collections import defaultdict
import numpy as np
from pathlib import Path
import cv2
from PIL import Image, ImageDraw



from rvinci.libs.utils.coco import load_coco, ann_to_rle
from rvinci.libs.visualization.drawing import draw_bounding_boxes, draw_segmentation_masks, get_font
from rvinci.libs.visualization.palette import get_error_color, PaletteManager
from rvinci.libs.vision_data.analysis import analyze_predictions
import pycocotools.mask as mask_utils

# Set up logging early
logger = logging.getLogger(__name__)

def get_class_map(coco_data):
    class_map = {}
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255)
    ]
    for i, cat in enumerate(coco_data.get("categories", [])):
        color = palette[i % len(palette)]
        class_map[cat["id"]] = {
            "name": cat["name"],
            "color": color
        }
    return class_map

def get_items_to_render(tp, fp, fn, mis, class_map, segmentation, img_shape, pred_id_to_gt_id=None):
    boxes = []
    masks = []
    h_img, w_img, _ = img_shape

    # TP: Correct
    for p, g in tp:
        cat_name = class_map.get(g["category_id"], {"name": "Unknown"})["name"]
        color = get_error_color(cat_name, "Correct")
        add_item(p, color, None, boxes, masks, segmentation, h_img, w_img)

    # FP: False Positive
    for p in fp:
        p_cat_id = p["category_id"]
        if pred_id_to_gt_id:
            p_cat_id = pred_id_to_gt_id.get(p_cat_id, p_cat_id)
        cat_name = class_map.get(p_cat_id, {"name": "Unknown"})["name"]
        color = get_error_color(cat_name, "False Positive")
        add_item(p, color, None, boxes, masks, segmentation, h_img, w_img)

    # FN: False Negative
    for g in fn:
        cat_name = class_map.get(g["category_id"], {"name": "Unknown"})["name"]
        color = get_error_color(cat_name, "False Negative")
        add_item(g, color, None, boxes, masks, segmentation, h_img, w_img)

    # Misclassification
    for p, g in mis:
        p_cat_id = p["category_id"]
        if pred_id_to_gt_id:
            p_cat_id = pred_id_to_gt_id.get(p_cat_id, p_cat_id)
        cat_name_pred = class_map.get(p_cat_id, {"name": "?"})["name"]
        color = get_error_color(cat_name_pred, "Misclassification")
        add_item(p, color, None, boxes, masks, segmentation, h_img, w_img)

    return boxes, masks
    
def add_item(ann, color, text, boxes, masks, segmentation, h_img, w_img):
    if not segmentation and "bbox" in ann:
        x, y, w, h = ann["bbox"]
        boxes.append({"box": (x, y, x + w, y + h), "color": color, "text": text})
    elif segmentation:
        try:
            rle = ann_to_rle(ann, h_img, w_img)
            mask = mask_utils.decode(rle)
            if len(mask.shape) == 3: mask = mask[:, :, 0]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for contour in contours:
                if len(contour) >= 3: polygons.append(contour.flatten().tolist())
            
            for poly in polygons:
                poly_np = np.array(poly).reshape(-1, 2)
                poly_norm = poly_np / [w_img, h_img]
                masks.append({"polygon": poly_norm.tolist(), "color": color, "text": text})
        except Exception as e:
            logger.debug(f"Failed to extract polygons from annotation: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize images with most errors.")
    parser.add_argument("gt_json", help="Path to GT COCO JSON.")
    parser.add_argument("pred_json", help="Path to Predictions COCO JSON.")
    parser.add_argument("--image_dir", help="Directory containing images.")
    parser.add_argument("--output_dir", default="error_vis", help="Output directory.")
    parser.add_argument("--top_k", type=int, default=1, help="Top K images per error type.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold.")
    parser.add_argument("--segmentation", action="store_true", help="Use segmentation masks.")
    parser.add_argument("--match_n", type=int, default=1, help="Allow multiple matches per GT.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--gt_debug", action="store_true", help="Also save GT-only versions of images.")
    parser.add_argument("--all", action="store_true", help="Visualize all images with errors.")
    parser.add_argument("--mask_alpha", type=float, default=0.4, help="Alpha transparency for segmentation masks.")
    parser.add_argument("--hide_legend", action="store_true", help="Do not render the legend on the images.")
    parser.add_argument("--font_size", type=int, default=40, help="Font size for bounding boxes.")
    parser.add_argument("--mask_font_size", type=int, default=32, help="Font size for segmentation masks.")
    parser.add_argument("--palette_config", help="Optional path to a palette config file.")
    
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)
    
    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)
    
    logger.info("Loading Data...")
    gt_data = load_coco(args.gt_json)
    pred_data = load_coco(args.pred_json)
    class_map = get_class_map(gt_data)
    palette_manager = PaletteManager(args.palette_config)

    # Use centralized analysis
    results = analyze_predictions(
        gt_data,
        pred_data,
        iou_threshold=args.iou_threshold,
        match_n=args.match_n,
        segmentation=args.segmentation,
        ignore_missing_classes=True
    )
    
    image_stats = results["image_stats"]
    pred_id_to_gt_id = results.get("pred_id_to_gt_id")
    gt_img_map = results.get("gt_img_map")
    
    # Enrich image stats for detailed selection
    # Need to group by category for Top-K selection
    
    # ... (Selection Logic remains mostly same, need to loop over image_stats) ...
    # Wait, the selection logic was complex inside visualize_errors.py
    # I should preserve that.
    
    # Map image stats to cat_counts
    for stat in image_stats:
        cat_counts = defaultdict(lambda: {"fp": 0, "fn": 0, "mis": 0})
        fp = stat["fp"]
        fn = stat["fn"]
        mis = stat["mis"]
        
        for p in fp:
            p_cat_id = p["category_id"]
            if pred_id_to_gt_id: p_cat_id = pred_id_to_gt_id.get(p_cat_id, p_cat_id)
            cname = class_map.get(p_cat_id, {"name": "unknown"})["name"].lower()
            cat_counts[cname]["fp"] += 1
        for g in fn:
             cname = class_map.get(g["category_id"], {"name": "unknown"})["name"].lower()
             cat_counts[cname]["fn"] += 1
        for p, g in mis:
             cname = class_map.get(g["category_id"], {"name": "unknown"})["name"].lower()
             cat_counts[cname]["mis"] += 1
        
        stat["cat_counts"] = dict(cat_counts)
        stat["num_fp"] = len(fp)
        stat["num_fn"] = len(fn)
        stat["num_mis"] = len(mis)

    # Build index if needed
    image_path_index = {}
    if args.image_dir:
        logger.info(f"Indexing images in {args.image_dir}...")
        for img_path in Path(args.image_dir).rglob("*"):
             if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                image_path_index[img_path.stem] = str(img_path)

    def get_image_path(img_id):
        img_info = gt_img_map.get(img_id)
        if not img_info: return None
        file_name = img_info["file_name"]
        stem = Path(file_name).stem
        if stem in image_path_index: return image_path_index[stem]
        fallback_path = os.path.join(os.path.dirname(args.gt_json), os.path.basename(file_name))
        if os.path.exists(fallback_path): return fallback_path
        elif os.path.exists(file_name): return file_name
        return None

    # Selection Logic
    images_to_vis = defaultdict(list)
    
    if args.all:
        for stat in image_stats:
            img_id = stat["img_id"]
            if stat["num_fp"] + stat["num_fn"] + stat["num_mis"] > 0:
                if get_image_path(img_id):
                     images_to_vis[img_id].append(None)
    else:
        # Top-K
        categories = sorted([c["name"] for c in gt_data["categories"]])
        types = ["fp", "fn", "mis"]
        
        for cat in categories:
            for t in types:
                def get_c(s, c, et):
                    counts = s.get("cat_counts", {})
                    c_low = c.lower()
                    for k,v in counts.items():
                        if k.lower() == c_low: return v.get(et, 0)
                    return 0
                
                sorted_stats = sorted(image_stats, key=lambda x: get_c(x, cat, t), reverse=True)
                count = 0
                for stat in sorted_stats:
                    if count >= args.top_k: break
                    if get_c(stat, cat, t) == 0: break
                    
                    if get_image_path(stat["img_id"]):
                        suffix = f"_{count+1}" if count > 0 else ""
                        custom_name = f"{cat}_{t}{suffix}"
                        images_to_vis[stat["img_id"]].append(custom_name)
                        count += 1

    if not images_to_vis:
        logger.warning("No images selected.")
        return

    # Visualization Loop
    worst_case_mapping = []
    
    # Legend setup
    vis_legend_map = {}
    idx = 0
    error_types = ["Correct", "False Positive", "False Negative", "Misclassification"]
    sorted_cat_ids = sorted(class_map.keys())
    for cat_id in sorted_cat_ids:
        cat_name = class_map[cat_id]["name"]
        for et in error_types:
            color = palette_manager.get_color(cat_name, et)
            vis_legend_map[idx] = {"name": f"{cat_name} - {et}", "color": color}
            idx += 1
            
    # Process
    for img_id, custom_names in images_to_vis.items():
        img_path = get_image_path(img_id)
        if not img_path: continue
        
        img_cv = cv2.imread(img_path)
        if img_cv is None: continue
        
        stat = next(s for s in image_stats if s["img_id"] == img_id)
        
        # 1. Error Vis
        boxes, masks = get_items_to_render(stat["tp"], stat["fp"], stat["fn"], stat["mis"], class_map, args.segmentation, img_cv.shape, pred_id_to_gt_id)
        img_err_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        legend_to_pass = vis_legend_map if not args.hide_legend else {}
        
        if boxes: 
            img_err_pil = draw_bounding_boxes(img_err_pil, boxes, legend_to_pass, font_size=args.font_size)
        if masks: 
            img_err_pil = draw_segmentation_masks(img_err_pil, masks, legend_to_pass, alpha=args.mask_alpha, font_size=args.mask_font_size)
            
        file_name = gt_img_map[img_id]["file_name"]
        stem = Path(file_name).stem
        
        for custom_name in custom_names:
            out_name = f"{custom_name}.png" if custom_name else f"{stem}_error.png"
            out_path = os.path.join(args.output_dir, out_name)
            img_err_pil.convert("RGB").save(out_path)
            if custom_name: worst_case_mapping.append((custom_name, file_name))
            
        # GT Debug
        if args.gt_debug:
            gt_anns_for_img = [ann for ann in gt_data["annotations"] if ann["image_id"] == img_id]
            gt_boxes, gt_masks = [], []
            h_img, w_img, _ = img_cv.shape
            for g in gt_anns_for_img:
                cat_name = class_map.get(g["category_id"], {"name": "Unknown"})["name"]
                color = palette_manager.get_color(cat_name, "Correct")
                add_item(g, color, cat_name, gt_boxes, gt_masks, args.segmentation, h_img, w_img)
            
            img_gt_raw = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_gt_vis = img_gt_raw
            
            # Use same legend for consistency
            gt_legend = legend_to_pass
            
            # Apply boxes/masks to GT-only version
            if gt_boxes or not args.hide_legend:
                img_gt_vis = draw_bounding_boxes(img_gt_vis, gt_boxes, gt_legend, font_size=args.font_size)
            if gt_masks:
                img_gt_vis = draw_segmentation_masks(img_gt_vis, gt_masks, gt_legend, alpha=args.mask_alpha, font_size=args.mask_font_size)
                
            # Combined side-by-side
            w_err, h_err = img_err_pil.size
            w_gt, h_gt = img_gt_vis.size
            
            combined = Image.new('RGB', (w_err + w_gt, max(h_err, h_gt)), (255, 255, 255))
            combined.paste(img_err_pil, (0, 0))
            combined.paste(img_gt_vis, (w_err, 0))
            
            sidebyside_path = os.path.join(args.output_dir, f"{stem}_debug_sidebyside.png")
            combined.save(sidebyside_path)

    if worst_case_mapping:
        with open(os.path.join(args.output_dir, "worst_cases_mapping.txt"), "w") as f:
            for c, o in worst_case_mapping: f.write(f"{c},{o}\n")

if __name__ == "__main__":
    main()
