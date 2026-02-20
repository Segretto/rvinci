import hydra
from omegaconf import DictConfig
from rvinci.core.config import validate_config
from rvinci.core.logging import get_logger
from rvinci.projects.sam3_pipeline.schemas.config import ProjectConfig
from rvinci.projects.sam3_pipeline.pipeline.run import run_pipeline

log = get_logger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        config: ProjectConfig = validate_config(cfg, ProjectConfig)
        run_pipeline(config)
    except Exception:
        log.exception("Run failed")
        raise

if __name__ == "__main__":
    main()
