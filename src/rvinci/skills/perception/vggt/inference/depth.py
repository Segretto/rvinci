import logging
import time
from typing import List, Union
import numpy as np
import torch
from PIL import Image

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
except ImportError:
    VGGT = None
    load_and_preprocess_images = None

from rvinci.skills.perception.vggt.schemas.config import VGGTConfig

logger = logging.getLogger(__name__)

class VGGTInference:
    """Internal implementation of VGGT depth inference."""
    
    def __init__(self, config: VGGTConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self._init_model()

    def _init_model(self):
        if VGGT is None:
            raise ImportError("The 'vggt' package is required but not installed.")
            
        logger.info(f"Loading VGGT model from {self.config.model_id}...")
        self.model = VGGT.from_pretrained(self.config.model_id).to(self.device).eval()
        
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        logger.info(f"VGGT initialized on {self.device} with AMP dtype {self.amp_dtype}")

    def run(self, images: List[Union[Image.Image, np.ndarray, str]]) -> List[np.ndarray]:
        """Runs batch inference."""
        if not images:
            return []

        image_inputs = []
        for img in images:
            if isinstance(img, str):
                image_inputs.append(img)
            elif hasattr(img, "filename") and img.filename:
                image_inputs.append(img.filename)
            else:
                raise ValueError("VGGT skill currently requires image paths or PIL images with a .filename attribute.")

        processed_images = load_and_preprocess_images(image_inputs).to(self.device)
        n_total = processed_images.shape[0]
        bs = self.config.batch_size
        
        all_depths = []
        
        for start in range(0, n_total, bs):
            end = min(start + bs, n_total)
            chunk = processed_images[start:end]
            n_chunk = chunk.shape[0]
            
            with torch.no_grad():
                images_batched = chunk.unsqueeze(0)
                
                with torch.cuda.amp.autocast(enabled=self.config.amp_enabled, dtype=self.amp_dtype):
                    aggregated_tokens_list, ps_idx = self.model.aggregator(images_batched)
                    depth_map, _ = self.model.depth_head(aggregated_tokens_list, images_batched, ps_idx)
                
                if depth_map.dim() == 5 and depth_map.size(-1) == 1:
                    depth_map = depth_map.squeeze(-1)
                
                for i in range(n_chunk):
                    d_raw = depth_map[0, i].float().cpu().numpy()
                    all_depths.append(d_raw * self.config.scale_m)
                    
        return all_depths
