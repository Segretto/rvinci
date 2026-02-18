#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil
import numpy as np
import cv2
import pytest

@pytest.mark.integration
def create_rotated_data(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Landscape
    img1 = np.zeros((900, 1200, 3), dtype=np.uint8)
    img1[:] = [100, 100, 255] # Reddish
    cv2.imwrite(str(output_dir / "soy_fp.png"), img1)
    
    # Portrait
    img2 = np.zeros((1200, 900, 3), dtype=np.uint8)
    img2[:] = [100, 255, 100] # Greenish
    cv2.imwrite(str(output_dir / "soy_fn.png"), img2)

@pytest.mark.integration
def test_rotation_main(tmp_path):
    print("Testing Image Rotation and Legend Position...")
    
    in_dir = tmp_path / "test_rotation"
    out_file = tmp_path / "test_rotation_grid.png"
    
    create_rotated_data(in_dir)
    
    cmd = [
        sys.executable, "-m", "rvinci.projects.vision_suite_legacy.pipeline.generate_error_grid",
        str(in_dir),
        "--output", str(out_file),
        "--padding", "20",
        "--radius", "40",
        "--whitespace", "15"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    if out_file.exists():
        print("PASS: Grid generated.")
    else:
        print("FAIL: Grid not generated.")
        sys.exit(1)

    print("Verification complete.")
    # tmp_path cleanup handled by pytest

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        test_rotation_main(Path(tmp))
