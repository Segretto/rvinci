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


class TrainingConfig(BaseModel):
    train_data_path: str = Field(description="Path to the training .pkl dataset.")
    val_data_path: str = Field(description="Path to the validation .pkl dataset.")
    batch_size: int = Field(1, description="Training batch size.")
    learning_rate: float = Field(1e-4, description="Adam optimizer learning rate.")
    num_epochs: int = Field(100, description="Number of training epochs.")


class TrainProjectConfig(BaseModel):
    project: ProjectMetaConfig = Field(default_factory=ProjectMetaConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    training: TrainingConfig


@hydra.main(version_base=None, config_path="configs", config_name="train_imitation")
def main(cfg: DictConfig) -> None:
    # 1. Init Config
    dict_cfg: Dict[str, Any] = dict(cfg)
    project_config = TrainProjectConfig(**dict_cfg)

    log.info(f"Starting Imitation Training for Project: {project_config.project.name}")

    # 2. Run Pipeline
    from rvinci.projects.imitation_control.pipeline.train_imitation import (
        train_pipeline,
    )

    train_pipeline(project_config)


if __name__ == "__main__":
    main()
