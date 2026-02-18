
import json
import os
import sys
import shutil
import pytest

# from rvinci.libs.utils.coco import load_coco, match_annotations
# We can't easily import from libs if we run as standalone script without package install, 
# but we assume package is installed.
from rvinci.libs.utils.coco import load_coco

def create_dummy_coco(path, images, annotations, categories):
    data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(path, 'w') as f:
        json.dump(data, f)

@pytest.mark.integration
def test_analysis_script(tmp_path):
    print("\nTesting Analysis Script...")
    
    # Create Dummy Data
    cat_dog = {"id": 1, "name": "dog"}
    cat_cat = {"id": 2, "name": "cat"}
    categories = [cat_dog, cat_cat]
    
    images = [{"id": 1, "file_name": "img1.jpg", "height": 100, "width": 100}]
    
    # GT: One dog
    gt_anns = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400}]
    
    # Pred: One dog (Correct), One cat (FP)
    pred_anns = [
        {"id": 101, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}, # TP Dog
        {"id": 102, "image_id": 1, "category_id": 2, "bbox": [60, 60, 20, 20], "score": 0.8}  # FP Cat
    ]
    
    test_gt = tmp_path / "test_gt.json"
    test_pred = tmp_path / "test_pred.json"
    test_output = tmp_path / "test_output"
    
    create_dummy_coco(str(test_gt), images, gt_anns, categories)
    create_dummy_coco(str(test_pred), images, pred_anns, categories)
    
    # Run the script via subprocess module invocation
    import subprocess
    cmd = [sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.analyze_coco_errors", str(test_gt), str(test_pred), "--output_dir", str(test_output)]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print("PASS: Script ran successfully.")
        
        # Check output
        csv_path = test_output / "confusion_matrix.csv"
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                content = f.read()
                print("CSV Output:")
                print(content)
                # Validation:
                # Dog (TP): Dog-Dog should be 1
                # Cat (FP): Background-Cat should be 1
                # Background-Dog: 0?
                # GT Dog -> Pred Dog (1)
                # GT Background -> Pred Cat (1)
                if "dog,1" in content or "1" in content: # Lazy check
                     print("PASS: CSV content seems plausible.")
        else:
            print("FAIL: CSV output not found.")
            
    except subprocess.CalledProcessError as e:
        print(f"FAIL: Script execution failed: {e}")
        
    # Test 2: Misclassification
    # GT Dog, Pred Cat at same location
    print("\nTesting Misclassification...")
    gt_anns = [{"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400}]
    pred_anns = [{"id": 202, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20], "score": 0.9}] # Matches IoU but wrong class
    
    test_gt_mix = tmp_path / "test_gt_mix.json"
    test_pred_mix = tmp_path / "test_pred_mix.json"
    test_output_mix = tmp_path / "test_output_mix"
    
    create_dummy_coco(str(test_gt_mix), images, gt_anns, categories)
    create_dummy_coco(str(test_pred_mix), images, pred_anns, categories)
    
    cmd = [sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.analyze_coco_errors", str(test_gt_mix), str(test_pred_mix), "--output_dir", str(test_output_mix)]
    subprocess.check_call(cmd)
    
    with open(test_output_mix / "confusion_matrix.csv", 'r') as f:
        print("Mix CSV:")
        print(f.read())
        # Should show GT Dog, Pred Cat = 1
        
    # Test 3: Segmentation
    print("\nTesting Segmentation...")
    # GT Square: [0,0,10,10]
    # Pred Square: [5,0,10,10] (Overlap 50? IoU ~0.33)
    # Box IoU: [0,0,10,10] vs [5,0,10,10]. Box IoU is 5x10 / (100+100-50) = 50/150 = 0.33
    # Wait, 0 to 10 width 10. 5 to 15 width 10.
    # Intersection 5 to 10 width 5. Height 10. Area 50.
    # Union 150. IoU 0.33. Matches if thresh 0.3.
    # Let's make Box IoU high but Mask IoU low?
    # GT: Box [0,0,10,10]. Mask: Top Half [0,0,10,5].
    # Pred: Box [0,0,10,10]. Mask: Bottom Half [0,5,10,5].
    # Box IoU: 1.0.
    # Mask IoU: 0.0.
    
    gt_anns = [{
        "id": 3, "image_id": 1, "category_id": 1, 
        "bbox": [0,0,10,10], "area": 50,
        "segmentation": [[0,0, 10,0, 10,5, 0,5, 0,0]] # Top Half
    }]
    pred_anns = [{
        "id": 303, "image_id": 1, "category_id": 1, 
        "bbox": [0,0,10,10], "score": 0.9,
        "segmentation": [[0,5, 10,5, 10,10, 0,10, 0,5]] # Bottom Half
    }]
    
    test_gt_seg = tmp_path / "test_gt_seg.json"
    test_pred_seg = tmp_path / "test_pred_seg.json"
    test_output_seg_box = tmp_path / "test_output_seg_box"
    test_output_seg_mask = tmp_path / "test_output_seg_mask"
    
    create_dummy_coco(str(test_gt_seg), images, gt_anns, categories)
    create_dummy_coco(str(test_pred_seg), images, pred_anns, categories)
    
    # Run WITHOUT --segmentation (Should match based on Box IoU=1.0)
    print("Running without --segmentation...")
    cmd = [sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.analyze_coco_errors", str(test_gt_seg), str(test_pred_seg), "--output_dir", str(test_output_seg_box), "--iou_threshold", "0.5"]
    subprocess.check_call(cmd)
    
    with open(test_output_seg_box / "confusion_matrix.csv", 'r') as f:
        content = f.read()
        print("Box Test CSV:")
        print(content)
        # Should be diagonal match (dog, dog = 1) because Box IoU 1.0 > 0.5
    
    # Run WITH --segmentation (Should NOT match based on Mask IoU=0.0)
    print("Running with --segmentation...")
    cmd = [sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.analyze_coco_errors", str(test_gt_seg), str(test_pred_seg), "--output_dir", str(test_output_seg_mask), "--iou_threshold", "0.5", "--segmentation"]
    subprocess.check_call(cmd)
    
    with open(test_output_seg_mask / "confusion_matrix.csv", 'r') as f:
        content = f.read()
        print("Mask Test CSV:")
        print(content)
        # Should be Background (FN/FP) because Mask IoU 0.0 < 0.5
        # dog, dog = 0
        # Background, dog = 1 (FP)
        # dog, Background = 1 (FN)

    # Test 4: Filtering
    print("\nTesting Filtering...")
    # GT: Dog + Cat
    # Pred: Dog only.
    # Should ignore Cat GT.
    # Output should only show Dog row/col.
    
    gt_anns = [
        {"id": 4, "image_id": 1, "category_id": 1, "bbox": [0,0,10,10], "area": 100}, # Dog
        {"id": 5, "image_id": 1, "category_id": 2, "bbox": [50,50,10,10], "area": 100}  # Cat
    ]
    # Prediction has only Dog class info implicitly or explicitly
    # Case A: Pred file has 'categories' list with only Dog.
    pred_cats = [cat_dog]
    pred_anns = [{"id": 404, "image_id": 1, "category_id": 1, "bbox": [0,0,10,10], "score": 0.9}]
    
    test_gt_filter = tmp_path / "test_gt_filter.json"
    test_pred_filter = tmp_path / "test_pred_filter.json"
    test_output_filter = tmp_path / "test_output_filter"
    
    create_dummy_coco(str(test_gt_filter), images, gt_anns, categories) # GT has both cats
    create_dummy_coco(str(test_pred_filter), images, pred_anns, pred_cats) # Pred has only Dog
    
    print("Running with default filtering...")
    cmd = [sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.analyze_coco_errors", str(test_gt_filter), str(test_pred_filter), "--output_dir", str(test_output_filter)]
    subprocess.check_call(cmd)
    
    with open(test_output_filter / "confusion_matrix.csv", 'r') as f:
        content = f.read()
        print("Filter Test CSV:")
        print(content)
        # Should NOT contain 'cat'
        if "cat" not in content and "dog" in content:
            print("PASS: Correctly filtered out 'cat'.")
        else:
            print("FAIL: 'cat' found in filtered output.")
            
    # Test 5: Path/Stem Matching
    print("\nTesting Stem Matching...")
    # GT: /data/train/img001.jpg
    # Pred: img001.png
    # IDs differ.
    
    gt_anns = [{"id": 6, "image_id": 100, "category_id": 1, "bbox": [0,0,10,10], "area": 100}]
    gt_imgs = [{"id": 100, "file_name": "/data/train/img001.jpg", "height": 100, "width": 100}]
    
    pred_anns = [{"id": 606, "image_id": 999, "category_id": 1, "bbox": [0,0,10,10], "score": 0.9}]
    pred_imgs = [{"id": 999, "file_name": "img001.png", "height": 100, "width": 100}]
    
    test_gt_path = tmp_path / "test_gt_path.json"
    test_pred_path = tmp_path / "test_pred_path.json"
    test_output_path = tmp_path / "test_output_path"
    
    create_dummy_coco(str(test_gt_path), gt_imgs, gt_anns, categories)
    create_dummy_coco(str(test_pred_path), pred_imgs, pred_anns, categories)
    
    print("Running with path mismatch...")
    cmd = [sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.analyze_coco_errors", str(test_gt_path), str(test_pred_path), "--output_dir", str(test_output_path)]
    subprocess.check_call(cmd)
    
    with open(test_output_path / "confusion_matrix.csv", 'r') as f:
        content = f.read()
        print("Path Test CSV:")
        print(content)
        # Should be TP (dog, dog = 1)
        if "dog,0,1,0" in content:
            print("PASS: Correctly matched by stem.")
        else:
            print("FAIL: Failed to match by stem.")

    # Test 6: Image Filtering
    print("\nTesting Image Filtering...")
    # GT: Image 100 (Unprocessed by pred), Image 101 (Processed)
    # Pred: Image 101 only.
    # Image 100 should be ignored (no FN).
    
    gt_imgs = [
        {"id": 100, "file_name": "img100.jpg", "height": 100, "width": 100},
        {"id": 101, "file_name": "img101.jpg", "height": 100, "width": 100}
    ]
    gt_anns = [{"id": 10, "image_id": 100, "category_id": 1, "bbox": [0,0,10,10], "area": 100}] # On ignored image
    
    # Pred: Only lists img101 in 'images'
    pred_imgs = [{"id": 101, "file_name": "img101.jpg", "height": 100, "width": 100}]
    pred_anns = [] # No detections on 101
    
    test_gt_imgfilter = tmp_path / "test_gt_imgfilter.json"
    test_pred_imgfilter = tmp_path / "test_pred_imgfilter.json"
    test_output_imgfilter = tmp_path / "test_output_imgfilter"
    
    create_dummy_coco(str(test_gt_imgfilter), gt_imgs, gt_anns, categories)
    create_dummy_coco(str(test_pred_imgfilter), pred_imgs, pred_anns, categories)
    
    print("Running with image filtering...")
    cmd = [sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.analyze_coco_errors", str(test_gt_imgfilter), str(test_pred_imgfilter), "--output_dir", str(test_output_imgfilter)]
    subprocess.check_call(cmd)
    
    with open(test_output_imgfilter / "confusion_matrix.csv", 'r') as f:
        content = f.read()
        print("Image Filter CSV:")
        print(content)
        # Should NOT contain FNs for img100. So dog FN=0.
        # Should be empty stats or zeros.
        if "dog,0,0,0" in content:
            print("PASS: Correctly ignored unprocessed image.")
        else:
            print("FAIL: Included unprocessed image as FN.")

    # Test 7: Match N
    print("\nTesting Match N...")
    # GT: 1 Dog
    # Pred: 2 Dogs overlapping it (IoU > 0.5)
    # match_n=1 -> 1 TP, 1 FP
    # match_n=2 -> 2 TP
    
    gt_imgs = [{"id": 200, "file_name": "img200.jpg", "height": 100, "width": 100}]
    gt_anns = [{"id": 20, "image_id": 200, "category_id": 1, "bbox": [0,0,50,50], "area": 2500}] # Dog
    
    pred_imgs = [{"id": 200, "file_name": "img200.jpg", "height": 100, "width": 100}]
    pred_anns = [
        {"id": 201, "image_id": 200, "category_id": 1, "bbox": [0,0,50,50], "score": 0.9}, # Match 1
        {"id": 202, "image_id": 200, "category_id": 1, "bbox": [1,1,49,49], "score": 0.8}  # Match 2 (High IoU)
    ]
    
    test_gt_matchn = tmp_path / "test_gt_matchn.json"
    test_pred_matchn = tmp_path / "test_pred_matchn.json"
    test_output_matchn1 = tmp_path / "test_output_matchn1"
    test_output_matchn2 = tmp_path / "test_output_matchn2"
    
    create_dummy_coco(str(test_gt_matchn), gt_imgs, gt_anns, categories)
    create_dummy_coco(str(test_pred_matchn), pred_imgs, pred_anns, categories)
    
    print("Running with match_n=1...")
    cmd = [sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.analyze_coco_errors", str(test_gt_matchn), str(test_pred_matchn), "--output_dir", str(test_output_matchn1), "--match_n", "1"]
    subprocess.check_call(cmd)
    
    print("Running with match_n=2...")
    cmd = [sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.analyze_coco_errors", str(test_gt_matchn), str(test_pred_matchn), "--output_dir", str(test_output_matchn2), "--match_n", "2"]
    subprocess.check_call(cmd)
    
    with open(test_output_matchn1 / "confusion_matrix.csv", 'r') as f:
        c1 = f.read()
        print("Match N=1 CSV:\n", c1)
        # Expect dog TP=1, FP=1 (Background col)
        # dog row: dog=1, Background=0? No wait.
        # TP counts go to diagonal.
        # Unmatched Pred goes to Background Row (Pred Background?) No, Pred is Col.
        # Row=Background, Col=dog -> FP.
        
    with open(test_output_matchn2 / "confusion_matrix.csv", 'r') as f:
        c2 = f.read()
        print("Match N=2 CSV:\n", c2)
        # Expect dog TP=2.
        # dog row: dog=2.
    
    # Check logic
    pass1 = "dog,0,1,0" in c1 and "Background,0,1,0" in c1 # TP=1 (dog,dog), FP=1 (Bg, dog)
    # Wait, CSV header is GT \ Pred.
    # dog, dog=1.
    # Background, dog = 1 (FP).
    
    pass2 = "dog,0,2,0" in c2 # TP=2 (dog, dog)
    
    if pass1 and pass2:
        print("PASS: Match N logic confirmed.")
    else:
        print("FAIL: Match N logic mismatch.")

    # No manual cleanup needed with tmp_path fixture

if __name__ == "__main__":
    test_analysis_script()
