from pydantic import BaseModel, ConfigDict, Field

class ProjectConfigSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    runs_root: str
    notes: str = ""

class SkillsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: list[str] = Field(default_factory=list)

class SpotClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hostname: str
    username: str
    password: str
    frequency_hz: float = 20.0
    output_dir: str = "positions"

class SpotTrajectoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_positions: int = 200
    x_distance_range: float = 0.55
    y_range: float = 0.5
    z_range: float = 0.4
    y_range_plane: float = 0.3
    z_range_plane: float = 0.3
    roll_range_deg: float = 95.0
    move_duration: float = 4.0
    frame_rate: int = 10

class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectConfigSettings
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    spot: SpotClientConfig
    trajectory: SpotTrajectoryConfig = Field(default_factory=SpotTrajectoryConfig)
