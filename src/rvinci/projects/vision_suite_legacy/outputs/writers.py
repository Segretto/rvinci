from typing import Any, Dict
import json
import yaml
from pathlib import Path

def write_run_manifest(run_dir: Path, manifest: Dict[str, Any]) -> None:
    """
    Write the run manifest to json.
    """
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

def write_resolved_config(run_dir: Path, config: Dict[str, Any]) -> None:
    """
    Write the fully resolved configuration to yaml.
    """
    with open(run_dir / "config_resolved.yaml", "w") as f:
        yaml.dump(config, f)
