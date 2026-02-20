import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union
from accelerate import Accelerator

from rvinci.skills.perception.sam3.schemas.config import Sam3PredictorConfig, Sam3TrackerConfig


class Sam3ImagePredictor:
    """Zero-shot object segmentation from text prompts using SAM3."""
    
    def __init__(self, config: Sam3PredictorConfig):
        from transformers import Sam3Processor, Sam3Model
        
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = Sam3Processor.from_pretrained(self.config.model_id)
        self.model = Sam3Model.from_pretrained(self.config.model_id).to(self.device)
        self.model.eval()

    def predict(self, image: Union[Image.Image, np.ndarray], prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Runs SAM3 inference on a single image with a list of text prompts.
        
        Args:
            image: PIL Image or numpy array (RGB).
            prompts: List of string prompts for zero-shot object detection.
            
        Returns:
            A list of predictions. Each entry belongs to one prompt and contains
            a list of dictionaries describing detected instances.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image or numpy array.")

        # Prepare outputs
        predictions_by_prompt = {prompt: [] for prompt in prompts}
        
        for prompt in prompts:
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            result = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.config.confidence_threshold,
                mask_threshold=self.config.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            masks = result["masks"]
            boxes = result["boxes"]
            scores = result["scores"]

            instances = []
            for i in range(len(masks)):
                mask = masks[i].cpu().numpy().astype(np.uint8)
                score = float(scores[i].cpu().item())
                box = boxes[i].cpu().tolist() # [x_min, y_min, x_max, y_max]
                
                instances.append({
                    "mask": mask,
                    "score": score,
                    "bbox": box
                })
            
            predictions_by_prompt[prompt] = instances

        return predictions_by_prompt


class Sam3VideoTracker:
    """Interactive video object tracking using SAM3."""
    
    def __init__(self, config: Sam3TrackerConfig):
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        
        self.config = config
        
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        self.model = Sam3TrackerVideoModel.from_pretrained(
            self.config.model_id
        ).to(self.device, dtype=torch.bfloat16)
        
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(self.config.model_id)

    def track(self, video_frames: List[np.ndarray], prompts: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Tracks multiple objects through a video sequence given initial prompts.
        
        Args:
            video_frames: List of RGB numpy arrays representing frames of the video.
            prompts: A dict where keys are frame indices, and values are dicts containing:
                     {
                         "obj_ids": list of ints,
                         "points": list of list of [x, y] coordinates,
                         "labels": list of list of ints (1 positive, 0 negative)
                     }
        Returns:
            A dictionary mapping frame indices to a dictionary mapping object IDs 
            to their corresponding boolean/uint8 2D mask (np.ndarray).
        """
        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            dtype=torch.bfloat16,
        )

        for frame_idx, data in prompts.items():
            self.processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_ids=data["obj_ids"],
                input_points=data["points"],
                input_labels=data["labels"],
            )
            # Run first segmentation locally for the frame
            self.model(inference_session=inference_session, frame_idx=frame_idx)

        video_segments = {}
        for output in self.model.propagate_in_video_iterator(inference_session):
            frame_idx = output.frame_idx

            masks = self.processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[inference_session.video_height,
                                 inference_session.video_width]],
                binarize=True,
            )[0]

            video_segments[frame_idx] = {
                obj_id: masks[i].cpu().numpy().astype(np.uint8)
                for i, obj_id in enumerate(inference_session.obj_ids)
            }

        return video_segments
