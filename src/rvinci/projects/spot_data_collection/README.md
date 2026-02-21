# Spot Data Collection Project

This project provides a unified implementation for collecting robotics data from a Boston Dynamics Spot robot. It supports both manual recording of joint/Cartesian trajectories and automated generation of randomized arm trajectories relative to a reference frame.

## Overview

The project integrates functionality from multiple legacy scripts into a single, clean rvinci project. It allows you to:
1. **Manually record** high-frequency joint states and hand poses.
2. **Define a reference frame** (Bottleneck) for subsequent automated tasks.
3. **Generate randomized trajectories** relative to the bottleneck frame for MLP training or data augmentation.

## Requirements

The Spot SDK is kept separate from the core deep learning dependencies. To use this project, you must install the `spot` optional dependency group:

```bash
uv sync --extra spot
```

## Running the Project

Run the project using the canonical CLI pattern. Hydra configurations allow you to override robot credentials or trajectory parameters easily.

```bash
uv run python -m rvinci.projects.spot_data_collection.cli \
    spot.hostname="YOUR_SPOT_IP" \
    spot.username="admin" \
    spot.password="YOUR_PASSWORD"
```

### Configuration Options

You can override any parameter defined in `configs/config.yaml`. For example, to change the number of randomized positions:

```bash
uv run python -m rvinci.projects.spot_data_collection.cli trajectory.num_positions=50
```

Refer to `src/rvinci/projects/spot_data_collection/schemas/config.py` for all available configuration fields.

## Interactive Workflow

Once the script connects and takes the robot lease, it will enter an interactive loop:

| Command | Action                                                                                                                              |
| :-----: | :---------------------------------------------------------------------------------------------------------------------------------- |
|   `s`   | **Start** recording a manual unified trajectory (joints + hand pose).                                                               |
|   `e`   | **End** recording and save the manual trajectory to the output folder.                                                              |
|   `f`   | **Fix** the bottleneck frame. Saves the current hand pose as the reference for randomization.                                       |
|   `g`   | **Generate** randomized trajectories. Interleaves random 3D moves and plane-constrained moves relative to the frame saved with `f`. |
|   `q`   | **Quit** the application and release the lease.                                                                                     |

## Data Structure & Outputs

All data is saved within the run directory (default: `runs/spot_data_collection/`).

### Manual Trajectories
Saved in the directory specified by `spot.output_dir` (default: `positions/`):
- `joints_and_gripper.json`: High-frequency snapshot of joint states and hand poses.
- `gripper_bottleneck_frame.json`: The reference frame recorded with command `f`.

### Randomized Datasets
Saved in `MLP_Datasets/Dataset_YYYY-MM-DD_HH-MM-SS/`:
- **position_N/**: Contains data for a random 3D move.
- **position_N_to_plane/**: Contains data for a move returning to the reference plane.
- Each position folder contains:
  - `image_X.jpg`: RGB hand camera frame.
  - `depth_X.npy`: Raw depth data.
  - `image_X.json`: full transform snapshot, wrist frame, and gripper state.

## Safety & Leases

This project uses the `LeaseClient` to **forcefully take** the body lease of the Spot robot. It maintains a `LeaseKeepAlive` for the duration of the script. Ensure no other users are currently operating the robot before running.
