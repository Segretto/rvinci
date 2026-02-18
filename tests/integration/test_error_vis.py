#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import shutil
from PIL import Image
import pytest

def create_dummy_data(output_dir):
    # Ground Truth: 3 images. 
    # img1: has 1 MIS. 
    # img2: has 1 FP, 1 FN.
    # img5: has 2 FN. (img4 missing on disk)
    gt = {
        "images": [
            {"id": 1, "file_name": "img1.png", "width": 100, "height": 100},
            {"id": 2, "file_name": "img2.jpg", "width": 100, "height": 100},
            {"id": 4, "file_name": "img4.jpg", "width": 100, "height": 100},
            {"id": 5, "file_name": "img5.jpg", "width": 100, "height": 100}
        ],
        "categories": [
            {"id": 1, "name": "dog"},
            {"id": 2, "name": "cat"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 50, 30, 30], "area": 900, "iscrowd": 0},
            # Image 4: many errors (3 FNs) - BUT MISSING ON DISK
            {"id": 4, "image_id": 4, "category_id": 1, "bbox": [1, 1, 1, 1], "area": 1, "iscrowd": 0},
            {"id": 5, "image_id": 4, "category_id": 1, "bbox": [2, 2, 2, 2], "area": 1, "iscrowd": 0},
            {"id": 6, "image_id": 4, "category_id": 1, "bbox": [3, 3, 3, 3], "area": 1, "iscrowd": 0},
            # Image 5: some errors (2 FNs)
            {"id": 7, "image_id": 5, "category_id": 1, "bbox": [4, 4, 1, 1], "area": 1, "iscrowd": 0},
            {"id": 8, "image_id": 5, "category_id": 1, "bbox": [5, 5, 1, 1], "area": 1, "iscrowd": 0}
        ]
    }
    
    # Predictions: Subset (Images 1, 2)
    pred = [
        {"image_id": 1, "category_id": 2, "bbox": [12, 12, 18, 18], "score": 0.9}, # Misclassification
        {"image_id": 2, "category_id": 2, "bbox": [10, 10, 10, 10], "score": 0.8}  # False Positive (no GT match)
    ]
    
    with open(output_dir / "test_err_gt.json", "w") as f: json.dump(gt, f)
    with open(output_dir / "test_err_pred.json", "w") as f: json.dump(pred, f)
    
    # Create blank images on disk
    import numpy as np
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(output_dir / "img1.png"), img)
    cv2.imwrite(str(output_dir / "img2.jpg"), img)
    cv2.imwrite(str(output_dir / "img5.jpg"), img)

@pytest.mark.integration
def test_error_vis_main(tmp_path):
    print("Testing Error Visualization Enhancements...")
    
    # Use tmp_path for data
    data_dir = tmp_path
    create_dummy_data(data_dir)
    
    out_dir = tmp_path / "test_error_vis_out"
    # pytest cleans up tmp_path, no need to manual rm

    # Run script with --all and --gt_debug
    # Point to JSONs in data_dir
    # Point --image_dir to data_dir
    cmd = [
        sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.visualize_errors",
        str(data_dir / "test_err_gt.json"), 
        str(data_dir / "test_err_pred.json"),
        "--image_dir", str(data_dir),
        "--output_dir", str(out_dir),
        "--gt_debug",
        "--all"
    ]
    
    cmd_str = ' '.join(cmd)
    print(f"Running: {cmd_str}")
    subprocess.run(cmd, check=True)

    # Expected files in new naming convention
    # All images with errors (1, 2, 5) should be visualized
    expected = [
        "img1_error.png", "img1_debug_sidebyside.png",
        "img2_error.png", "img2_debug_sidebyside.png",
        "img5_error.png", "img5_debug_sidebyside.png"
    ]
    
    if not out_dir.exists():
        print(f"FAIL: Output directory {out_dir} not created.")
        sys.exit(1)
        
    files = os.listdir(out_dir)
    print(f"Generated files: {files}")
    
    for f in expected:
        if f not in files:
            print(f"FAIL: Missing expected file {f}")
            sys.exit(1)
            
    # Check if images have headers (height 100 + 60 = 160)
    comp = Image.open(out_dir / "img1_debug_sidebyside.png")
    if comp.size[1] != 160:
        print(f"FAIL: Composite height is {comp.size[1]}, expected 160")
        sys.exit(1)

    print("PASS: Error visualization enhancements verified.")

    # Cleanup handled by pytest tmp_path fixture

if __name__ == "__main__":
    # Simulate tmp_path for manual run
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        test_error_vis_main(Path(tmp))
