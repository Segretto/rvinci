# SAM3 Pipeline Project

Unified orchestration for SAM3 interactive tracking, zero-shot segmentation via text prompts, and COCO dataset utilities (filtering and visualization).

## Canonical Execution

Run the project via `uv` using the project's CLI entrypoint:

```bash
uv run python -m rvinci.projects.sam3_pipeline.cli mode=<MODE> [mode_specific_options]
```

---

## Operational Modes

### 1. `visualize`
Generates visualization images with bounding boxes and segmentation masks overlaid, using the repository's canonical visualization library.

**Options:**
- `visualize.images_dir`: Path to the directory containing images.
- `visualize.annotations`: Path to the COCO JSON annotation file.
- `visualize.output_dir`: (Optional) Path to save visualizations. Defaults to `outputs/visualizations` within the run directory.
- `visualize.hide_score`: (Boolean) If true, confidence scores are hidden in labels.

**Example:**
```bash
uv run python -m rvinci.projects.sam3_pipeline.cli mode=visualize \
    visualize.images_dir=data/images \
    visualize.annotations=data/annotations.json
```

---

### 2. `tracking`
Performs interactive video object tracking. It opens a GUI for manual point selection on specific frames and propagates those labels through the video sequence using SAM3's temporal capabilities.

**How to use:**
1. Execute the command.
2. A window opens for each frame index provided in `manual_ids`.
3. **Left Click** to add a positive point (object).
4. **Right Click** to add a negative point (background).
5. Press **'n'** to increment to the next object ID.
6. Press **'c'** to clear points for the current frame.
7. Press **'q'** or **ESC** to finish labeling and start propagation.

**Options:**
- `tracking.input_json`: Seed COCO JSON (used for image metadata and existing IDs).
- `tracking.images_dir`: Path to the video frames directory.
- `tracking.manual_ids`: List of frame indices (integer) to manually prompt (e.g., `[0, 50]`).
- `tracking.output_json`: (Optional) Where to save resulting tracking annotations.
- `tracking.range`: (Optional) List of `[start, end]` frame indices to process.
- `tracking.model_id`: (Optional) SAM3 model identifier.

**Example:**
```bash
uv run python -m rvinci.projects.sam3_pipeline.cli mode=tracking \
    tracking.images_dir=data/frames \
    tracking.input_json=data/metadata.json \
    tracking.manual_ids=[0]
```

---

### 3. `text_prompt`
Runs zero-shot object detection/segmentation on a directory of images based on descriptive text prompts (Natural Language).

**Options:**
- `text_prompt.images_dir`: Path to images to auto-annotate.
- `text_prompt.prompts`: List of text strings (e.g., `["red car", "person wearing a hat"]`).
- `text_prompt.output_json`: (Optional) Path to save the new COCO dataset.
- `text_prompt.model_id`: (Optional) SAM3 model identifier.

**Example:**
```bash
uv run python -m rvinci.projects.sam3_pipeline.cli mode=text_prompt \
    text_prompt.images_dir=data/images \
    text_prompt.prompts='["apple", "banana"]'
```

---

### 4. `filter`
Utility to clean up COCO datasets by removing low-confidence predictions.

**Options:**
- `filter.input_json`: Input COCO JSON file.
- `filter.output_json`: Output filtered COCO JSON file.
- `filter.threshold`: (Float) Minimum confidence score to keep.
- `filter.remove_empty_images`: (Boolean) If true, images with no remaining annotations are stripped from the dataset.

**Example:**
```bash
uv run python -m rvinci.projects.sam3_pipeline.cli mode=filter \
    filter.input_json=data/results.json \
    filter.output_json=data/filtered.json \
    filter.threshold=0.75
```

---

## Technical Details

- **Model Integration**: Leverages the `rvinci.skills.perception.sam3` skill wrapper around Transformers `Sam3Model` and `Sam3TrackerVideoModel`.
- **Inference Optimization**: Uses `accelerate` for optimized device placement (`bfloat16` by default for tracking).
- **Configuration**: Managed via Hydra in `cli.py` and validated through Pydantic schemas in `schemas/config.py`.
- **Outputs**: All runs generate a unique run directory under `runs/sam3_pipeline/` containing logs and resolved configs.
