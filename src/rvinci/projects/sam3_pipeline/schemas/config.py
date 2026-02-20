from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional, List
from pathlib import Path


class ProjectInfo(BaseModel):
    name: str = "sam3_pipeline"
    runs_root: str = "runs"
    notes: str = ""


class TrackingConfig(BaseModel):
    input_json: Path
    images_dir: Path
    output_json: Path
    range: Optional[List[int]] = None
    manual_ids: List[int]
    model_id: str = "facebook/sam3"


class TextPromptConfig(BaseModel):
    images_dir: Path
    output_json: Path
    prompts: List[str]
    model_id: str = "facebook/sam3"


class FilterConfig(BaseModel):
    input_json: Path
    output_json: Path
    threshold: float
    remove_empty_images: bool = False


class VisualizeConfig(BaseModel):
    images_dir: Path
    annotations: Path
    output_dir: Path
    hide_score: bool = False


class ProjectConfig(BaseModel):
    project: ProjectInfo = Field(default_factory=ProjectInfo)
    mode: Literal["tracking", "text_prompt", "filter", "visualize"]
    
    tracking: Optional[TrackingConfig] = None
    text_prompt: Optional[TextPromptConfig] = None
    filter: Optional[FilterConfig] = None
    visualize: Optional[VisualizeConfig] = None
    
    model_config = ConfigDict(extra="ignore")
