#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import pytest

def create_dummy(path):
    data = {
        "categories": [
            {"id": 1, "name": "dog"},
            {"id": 2, "name": "cat"},
            {"id": 3, "name": "bird"}
        ],
        "images": [
            {"id": 1, "file_name": "img1.jpg"}
        ],
        "annotations": [
            {"id": 1, "category_id": 1, "image_id": 1, "bbox": [0,0,10,10]},
            {"id": 2, "category_id": 2, "image_id": 1, "bbox": [10,10,10,10]},
            {"id": 3, "category_id": 3, "image_id": 1, "bbox": [20,20,10,10]}
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f)

@pytest.mark.integration
def test_filter_coco_main(tmp_path):
    src_path = tmp_path / "test_filter_src.json"
    exclude_path = tmp_path / "test_filtered_exclude.json"
    include_path = tmp_path / "test_filtered_include.json"
    
    create_dummy(src_path)
    
    # Test Exclude
    print("Testing Exclude...")
    subprocess.run([sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.filter_coco_classes", str(src_path), str(exclude_path), "--exclude", "bird"], check=True)
    with open(exclude_path, "r") as f:
        data = json.load(f)
    cats = [c["name"] for c in data["categories"]]
    anns = [a["category_id"] for a in data["annotations"]]
    assert "bird" not in cats
    assert 3 not in anns
    assert len(cats) == 2
    assert len(anns) == 2
    print("Exclude PASS")

    # Test Include
    print("Testing Include...")
    subprocess.run([sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.filter_coco_classes", str(src_path), str(include_path), "--include", "dog", "cat"], check=True)
    with open(include_path, "r") as f:
        data = json.load(f)
    cats = [c["name"] for c in data["categories"]]
    anns = [a["category_id"] for a in data["annotations"]]
    assert "bird" not in cats
    assert 3 not in anns
    assert "dog" in cats
    assert "cat" in cats
    assert len(cats) == 2
    assert len(anns) == 2
    print("Include PASS")
    
    # Cleanup handled by tmp_path

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        test_filter_coco_main(Path(tmp))
