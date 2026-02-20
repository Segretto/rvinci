from pathlib import Path

from rvinci.core.logging import get_logger
from rvinci.projects.spot_data_collection.schemas.config import ProjectConfig
from rvinci.projects.spot_data_collection.pipeline.recorder import JointAndFrameRecorder

log = get_logger(__name__)

def run_pipeline(config: ProjectConfig) -> None:
    log.info(f"Starting {config.project.name} run pipeline")

    # Resolve run directory
    run_dir = Path(config.project.runs_root) / config.project.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Run directory: {run_dir}")

    # Launch the Spot connection and recording loop
    recorder = JointAndFrameRecorder(run_dir)
    recorder.run(config.spot, config.trajectory)

    log.info("Finished Spot data collection.")
