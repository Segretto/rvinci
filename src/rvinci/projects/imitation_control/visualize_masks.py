from typing import Any, Dict
from pydantic import BaseModel, Field
import hydra
from omegaconf import DictConfig

from rvinci.core.logging import get_logger
from rvinci.projects.imitation_control.schemas.config import ProjectMetaConfig

log = get_logger(__name__)


class VisualizeConfig(BaseModel):
    data_path: str = Field(description="Path to the offline dataset.pkl to visualize.")
    sleep_freq: float = Field(0.1, description="Simulation playback sleep in seconds.")


class VisProjectConfig(BaseModel):
    project: ProjectMetaConfig
    visualize: VisualizeConfig


@hydra.main(version_base=None, config_path="configs", config_name="visualize_masks")
def main(cfg: DictConfig) -> None:
    dict_cfg: Dict[str, Any] = dict(cfg)
    project_config = VisProjectConfig(**dict_cfg)

    log.info(f"Starting Visualization: {project_config.project.name}")

    from rvinci.projects.imitation_control.pipeline.visualize_masks import (
        visualize_pipeline,
    )

    visualize_pipeline(project_config)


if __name__ == "__main__":
    main()
