import logging
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from pathlib import Path

from rvinci.libs.utils.coco import match_annotations, compute_iou_masks, compute_iou_boxes

logger = logging.getLogger(__name__)

def analyze_predictions(
    gt_data: Dict[str, Any],
    pred_data: Dict[str, Any],
    iou_threshold: float = 0.5,
    match_n: int = 1,
    segmentation: bool = False,
    ignore_missing_classes: bool = False
) -> Dict[str, Any]:
    """
    Analyze predictions against ground truth to identify TPs, FPs, FNs, and Misclassifications.
    Returns a dictionary containing:
    - per_image_stats: [ {img_id, tp, fp, fn, mis, ...}, ... ]
    - aggregate_stats: { "TP": count, "FP": count, "FN": count }
    - confusion_matrix: np.ndarray (if classes align)
    - classes: list of class names
    """
    
    # 1. Setup Categories
    categories = gt_data.get("categories", [])
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    cat_name_to_id = {c["name"]: c["id"] for c in categories}
    
    # Map Pred Categories
    pred_id_to_gt_id = {}
    present_pred_category_ids = set()
    
    if "categories" in pred_data:
        for p_cat in pred_data["categories"]:
            present_pred_category_ids.add(p_cat["id"])
            if p_cat["name"] in cat_name_to_id:
                pred_id_to_gt_id[p_cat["id"]] = cat_name_to_id[p_cat["name"]]
    else:
         pred_anns_list = pred_data if isinstance(pred_data, list) else pred_data.get("annotations", [])
         for ann in pred_anns_list:
             present_pred_category_ids.add(ann["category_id"])

    # Determine Valid Classes
    valid_gt_cat_ids = set()
    if not ignore_missing_classes:
        for p_id in present_pred_category_ids:
            if p_id in pred_id_to_gt_id:
                valid_gt_cat_ids.add(pred_id_to_gt_id[p_id])
            elif p_id in cat_id_to_name:
                valid_gt_cat_ids.add(p_id)
        if not valid_gt_cat_ids:
             logger.warning("No matching classes found. Using all GT classes.")
             valid_gt_cat_ids = set(cat_id_to_name.keys())
    else:
        valid_gt_cat_ids = set(cat_id_to_name.keys())

    sorted_cat_names = sorted([cat_id_to_name[i] for i in valid_gt_cat_ids])
    
    # 2. Organize Data
    gt_anns = [ann for ann in gt_data.get("annotations", []) if ann["category_id"] in valid_gt_cat_ids]
    pred_anns = pred_data.get("annotations", []) if isinstance(pred_data, dict) else pred_data
    
    gt_img_map = {img["id"]: img for img in gt_data.get("images", [])}
    gt_by_img = defaultdict(list)
    for ann in gt_anns: gt_by_img[ann["image_id"]].append(ann)
    
    pred_by_img = defaultdict(list)
    # Robust Image Mapping Logic
    # (Simplified for library: assumes IDs match or caller handled mapping? 
    #  No, the script had robust filename-based mapping. We should probably preserve that or require aligned inputs.)
    # For a reusable lib, it's cleaner to assume inputs are aligned or provide a separate alignment utility.
    # However, to maintain parity with the script, let's implement the robust mapping here.
    
    pred_img_map = {img["id"]: img for img in pred_data.get("images", [])} if isinstance(pred_data, dict) else {}
    filename_to_gt_id = {img["file_name"]: img["id"] for img in gt_data.get("images", [])}
    stem_to_gt_id = {Path(img["file_name"]).stem: img["id"] for img in gt_data.get("images", [])}
    
    # Process Pred Anns to align Image IDs
    for ann in pred_anns:
        p_img_id = ann["image_id"]
        target_gt_id = None
        
        # 1. ID Match
        if p_img_id in gt_img_map:
             # Check consistency if possible
             target_gt_id = p_img_id
        
        # 2. Filename/Stem Match
        if target_gt_id is None and p_img_id in pred_img_map:
            p_fname = pred_img_map[p_img_id]["file_name"]
            if p_fname in filename_to_gt_id:
                target_gt_id = filename_to_gt_id[p_fname]
            elif Path(p_fname).stem in stem_to_gt_id:
                target_gt_id = stem_to_gt_id[Path(p_fname).stem]
                
        if target_gt_id:
            pred_by_img[target_gt_id].append(ann)
            
    # 3. Analyze Images
    # We want to analyze all images that were "processed".
    # If pred_data provides a list of images, we use that as the boundary.
    # Otherwise, we use all images mentioned in predictions or all images in GT.
    
    if isinstance(pred_data, dict) and "images" in pred_data:
        # The user provided a specific set of images in the prediction file.
        # We should only analyze these.
        processed_ids = set()
        for p_img in pred_data["images"]:
            p_img_id = p_img["id"]
            # Try to map to GT
            if p_img_id in gt_img_map:
                processed_ids.add(p_img_id)
            else:
                p_fname = p_img["file_name"]
                if p_fname in filename_to_gt_id:
                    processed_ids.add(filename_to_gt_id[p_fname])
                elif Path(p_fname).stem in stem_to_gt_id:
                    processed_ids.add(stem_to_gt_id[Path(p_fname).stem])
        # Also include any images that had detections but weren't in 'images' list? 
        # (Though that would be an invalid COCO file)
        processed_ids |= set(pred_by_img.keys())
        processed_ids = sorted(list(processed_ids))
    else:
        # No image constraint in pred_data (e.g. it's just a list of annotations).
        # We assume the user wants to see all errors against all GT images.
        processed_ids = sorted(list(set(pred_by_img.keys()) | set(gt_img_map.keys())))
    
    image_stats = []
    
    all_tp = []
    all_fp = []
    all_fn = []
    all_mis = [] # List of tuples (pred, gt)
    
    for img_id in processed_ids:
        gts = gt_by_img.get(img_id, [])
        preds = pred_by_img.get(img_id, [])
        img_info = gt_img_map.get(img_id)
        h, w = (img_info["height"], img_info["width"]) if img_info else (None, None)
        
        # Match
        tp, fp, fn = match_annotations(gts, preds, pred_id_to_gt_id, iou_threshold, segmentation, match_n, h, w)
        
        # Identify Misclassifications (FP vs FN overlap)
        mis = []
        already_matched_fp = set()
        already_matched_fn = set()
        
        for g in fn:
            best_iou = 0
            best_p = None
            for p in fp:
                if id(p) in already_matched_fp: continue
                
                iou = 0
                if segmentation and h and w:
                    try: iou = compute_iou_masks(p, g, h, w, g.get("iscrowd", 0))
                    except: 
                        if "bbox" in p and "bbox" in g: iou = compute_iou_boxes(p["bbox"], g["bbox"])
                elif "bbox" in p and "bbox" in g:
                    iou = compute_iou_boxes(p["bbox"], g["bbox"])
                
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_p = p
            
            if best_p:
                mis.append((best_p, g))
                already_matched_fp.add(id(best_p))
                already_matched_fn.add(id(g))
                
        # Filter pure FP/FN
        pure_fp = [p for p in fp if id(p) not in already_matched_fp]
        pure_fn = [g for g in fn if id(g) not in already_matched_fn]
        
        image_stats.append({
            "img_id": img_id,
            "tp": tp,
            "fp": pure_fp,
            "fn": pure_fn,
            "mis": mis
        })
        
        all_tp.extend(tp)
        all_fp.extend(pure_fp)
        all_fn.extend(pure_fn)
        all_mis.extend(mis)

    # 4. Build Confusion Matrix
    labels = sorted_cat_names + ["Background"]
    label_to_idx = {name: i for i, name in enumerate(labels)}
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill CM
    # TP (Diagonal)
    for p, g in all_tp:
        cname = cat_id_to_name.get(g["category_id"])
        if cname and cname in label_to_idx:
            idx = label_to_idx[cname]
            cm[idx, idx] += 1
            
    # Mis (Row=GT, Col=Pred)
    for p, g in all_mis:
        gt_cname = cat_id_to_name.get(g["category_id"])
        
        p_cat = p["category_id"]
        if pred_id_to_gt_id: p_cat = pred_id_to_gt_id.get(p_cat, p_cat)
        p_cname = cat_id_to_name.get(p_cat)
        
        if gt_cname and p_cname and gt_cname in label_to_idx and p_cname in label_to_idx:
            row = label_to_idx[gt_cname]
            col = label_to_idx[p_cname]
            cm[row, col] += 1
            
    # FN (Row=GT, Col=Background)
    bg_idx = label_to_idx["Background"]
    for g in all_fn:
        cname = cat_id_to_name.get(g["category_id"])
        if cname and cname in label_to_idx:
            row = label_to_idx[cname]
            cm[row, bg_idx] += 1
            
    # FP (Row=Background, Col=Pred)
    for p in all_fp:
        p_cat = p["category_id"]
        if pred_id_to_gt_id: p_cat = pred_id_to_gt_id.get(p_cat, p_cat)
        cname = cat_id_to_name.get(p_cat)
        if cname and cname in label_to_idx:
            col = label_to_idx[cname]
            cm[bg_idx, col] += 1
            
    return {
        "image_stats": image_stats,
        "confusion_matrix": cm,
        "labels": labels,
        "aggregate": {
            "tp": len(all_tp),
            "fp": len(all_fp),
            "fn": len(all_fn),
            "mis": len(all_mis)
        },
        "class_map": cat_id_to_name,
        "pred_id_to_gt_id": pred_id_to_gt_id,
        "gt_img_map": gt_img_map,
        "all_processed_ids": processed_ids
    }
