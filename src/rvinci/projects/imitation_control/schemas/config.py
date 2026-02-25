from pydantic import BaseModel, Field
from rvinci.skills.perception.dinov3.api import UnifiedDINOv3Config
from rvinci.skills.manipulation.imitation.api import ImitationConfig


class SkillsConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: ["perception.dinov3", "manipulation.imitation"]
    )
    dinov3: UnifiedDINOv3Config = Field(default_factory=UnifiedDINOv3Config)
    imitation: ImitationConfig = Field(default_factory=ImitationConfig)


class ProjectMetaConfig(BaseModel):
    name: str = "imitation_control"
    runs_root: str = "runs"
    notes: str = ""


class ProjectConfig(BaseModel):
    project: ProjectMetaConfig = Field(default_factory=ProjectMetaConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)

    # Spot / Imitation Specific Parameters
    hostname: str = Field(..., description="Hostname or IP of the Spot Robot")
    head_weights_dir: str = Field(
        "models", description="Directory where custom DINOv3 head weights are stored"
    )
    model_path: str = Field(
        ..., description="Path to the trained PyTorch Imitation network weights"
    )
    bottleneck_path: str = Field(
        ..., description="Path to the bottleneck pose json configuration"
    )
    sleep_freq: float = Field(0.1, description="Inference loop sleep frequency")

    model_config = {"extra": "forbid"}
