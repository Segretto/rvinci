# Spot Data Collection Project

This project connects to a Boston Dynamics Spot robot to record joint states, gripper state, and Cartesian poses at a fixed frequency. It was migrated and translated from legacy scripts.

## Requirements

You must install the Spot optional dependencies:
```bash
uv sync --extra spot
# or
uv add bosdyn-client bosdyn-core --optional spot
```

## Running the Project

Run via the canonical CLI pattern:
```bash
uv run python -m rvinci.projects.spot_data_collection.cli \
    spot.hostname="192.168.80.3" \
    spot.username="admin" \
    spot.password="spotadmin2017"
```

## Interactive Commands

Once connected, the CLI will prompt for commands:
- `s`: Start recording the unified trajectory
- `e`: Stop recording and save to the output directory
- `f`: Record the target fixed frame (bottleneck pose)
- `q`: Quit

## Output

Data is saved to the run directory configured via Hydra (default: `runs/spot_data_collection/positions/`).
