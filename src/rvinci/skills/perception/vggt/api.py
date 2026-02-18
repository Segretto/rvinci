from typing import List, Union
import numpy as np
from PIL import Image
from rvinci.skills.perception.vggt.schemas.config import VGGTConfig
from rvinci.skills.perception.vggt.inference.depth import VGGTInference

class VGGTAPI:
    """Public interface for VGGT Depth Estimator."""
    
    def __init__(self, config: VGGTConfig = None):
        self.config = config or VGGTConfig()
        self._inference = None

    def _get_inference(self):
        if self._inference is None:
            self._inference = VGGTInference(self.config)
        return self._inference

    def predict(self, images: List[Union[Image.Image, np.ndarray, str]]) -> List[np.ndarray]:
        """
        Predict metric depth for a list of images.
        
        Args:
            images: List of PIL Images, numpy arrays (RGB), or image paths.
            
        Returns:
            List of depth maps as numpy arrays (meters).
        """
        return self._get_inference().run(images)

    def predict_single(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        """Convenience method for a single image."""
        return self.predict([image])[0]
