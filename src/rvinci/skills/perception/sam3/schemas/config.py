from pydantic import BaseModel, ConfigDict

class Sam3PredictorConfig(BaseModel):
    """Configuration for SAM3 text-prompt zero-shot image prediction."""
    model_id: str = "facebook/sam3"
    confidence_threshold: float = 0.5
    mask_threshold: float = 0.5
    
    model_config = ConfigDict(extra="forbid")


class Sam3TrackerConfig(BaseModel):
    """Configuration for SAM3 interactive video tracking."""
    model_id: str = "facebook/sam3"
    
    model_config = ConfigDict(extra="forbid")
