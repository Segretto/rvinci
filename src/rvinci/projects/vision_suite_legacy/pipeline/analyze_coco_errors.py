#!/usr/bin/env python3
import argparse
import sys
import os
import logging
import numpy as np
import csv


from rvinci.libs.utils.coco import load_coco
from rvinci.libs.vision_data.analysis import analyze_predictions
from rvinci.libs.visualization.plots import plot_confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Analyze COCO detection/segmentation errors.")
    parser.add_argument("gt_json", help="Path to Ground Truth COCO JSON.")
    parser.add_argument("pred_json", help="Path to Predictions COCO JSON.")
    parser.add_argument("--ignore_missing_classes", action="store_true", help="Do not filter GT classes based on predictions.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for matching.")
    parser.add_argument("--match_n", type=int, default=1, help="Number of matches allowed per GT.")
    parser.add_argument("--output_dir", default="analysis_results", help="Directory to save results.")
    parser.add_argument("--normalize", action="store_true", help="Normalize confusion matrix.")
    parser.add_argument("--segmentation", action="store_true", help="Use segmentation masks for IoU.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    logger.info("Loading Data...")
    gt_data = load_coco(args.gt_json)
    pred_data = load_coco(args.pred_json)
    
    logger.info("Analyzing Predictions...")
    results = analyze_predictions(
        gt_data,
        pred_data,
        iou_threshold=args.iou_threshold,
        match_n=args.match_n,
        segmentation=args.segmentation,
        ignore_missing_classes=args.ignore_missing_classes
    )
    
    agg = results["aggregate"]
    logger.info(f"Stats: TP={agg['tp']}, FP={agg['fp']}, FN={agg['fn']}, Misclassifications={agg['mis']}")
    
    # Save Confusion Matrix CSV
    cm = results["confusion_matrix"]
    labels = results["labels"]
    csv_path = os.path.join(args.output_dir, "confusion_matrix.csv")
    
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["GT \\ Pred"] + labels)
        for i, row_label in enumerate(labels):
            writer.writerow([row_label] + list(cm[i]))
            
    # Plot
    plot_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels, plot_path, normalize=args.normalize)
        
if __name__ == "__main__":
    main()
