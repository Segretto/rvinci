from typing import Any, Dict
from pydantic import BaseModel, Field
import hydra
from omegaconf import DictConfig

from rvinci.core.logging import get_logger
from rvinci.projects.imitation_control.schemas.config import (
    ProjectMetaConfig,
    SkillsConfig,
)

log = get_logger(__name__)


class DatasetGenConfig(BaseModel):
    input_dir: str = Field(description="Directory containing the JSON/image dataset.")
    output_file: str = Field(
        description="Path to output the generated dataset (.pt or .pkl)"
    )
    feature_mode: str = Field(
        "query", description="Which DINOv3 features to extract: 'query' or 'mask'"
    )


class GenProjectConfig(BaseModel):
    project: ProjectMetaConfig = Field(default_factory=ProjectMetaConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    dataset: DatasetGenConfig


@hydra.main(version_base=None, config_path="configs", config_name="generate_dataset")
def main(cfg: DictConfig) -> None:
    # 1. Init Config
    dict_cfg: Dict[str, Any] = dict(cfg)
    project_config = GenProjectConfig(**dict_cfg)

    log.info(
        f"Starting Offline Dataset Generation for Project: {project_config.project.name}"
    )

    # 2. Run Pipeline
    from rvinci.projects.imitation_control.pipeline.generate_dataset import (
        generate_pipeline,
    )

    generate_pipeline(project_config)


if __name__ == "__main__":
    main()
