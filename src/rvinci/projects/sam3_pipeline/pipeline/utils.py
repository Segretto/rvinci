import os
import json
import uuid
from typing import Dict, Any

def resolve_run_dir(configs_root: str, project_name: str) -> str:
    """Creates and resolves a unique run directory."""
    run_id = str(uuid.uuid4())[:8]
    run_dir = os.path.join(configs_root, project_name, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def load_coco_data(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def save_coco_format(path: str, data: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
