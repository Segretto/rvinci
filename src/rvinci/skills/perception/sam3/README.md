# sam3

SAM3 skill wrapping Facebook's SAM3 models for zero-shot image segmentation and video tracking.

## Usage

```python
from rvinci.skills.perception.sam3.api import Sam3ImagePredictor, Sam3VideoTracker
from rvinci.skills.perception.sam3.schemas.config import Sam3PredictorConfig, Sam3TrackerConfig

# Image segmentation via text prompts
predictor = Sam3ImagePredictor(Sam3PredictorConfig())
masks = predictor.predict(image, prompts=["car", "person"])

# Video object tracking via points/boxes
tracker = Sam3VideoTracker(Sam3TrackerConfig())
masks_across_frames = tracker.track(video_frames, object_stubs)
```

## Note
This skill is Hydra-agnostic; orchestration belongs in projects.
