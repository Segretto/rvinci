from rvinci.core.logging import get_logger
from rvinci.projects.vision_suite_legacy.schemas.config import ProjectConfig

log = get_logger(__name__)

def run_pipeline(config: ProjectConfig) -> None:
    log.info("Running vision_suite_legacy pipeline")
    log.info(f"Config: {config}")
    # Here we would dispatch to different logic based on config
    # For now, it's just a placeholder.
