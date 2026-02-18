---
name: rvinci-run
description: Executes a specific rvinci project or skill using the canonical CLI. Use this to train models, run evaluations, or execute robot tasks.
---

# Goal
Execute a pipeline or task within an `rvinci` project.

# Instructions
1.  Identify the project name from the user's request (e.g., "spot_pick_place", "dinov3_benchmark").
2.  Construct the command using `uv run`.
3.  Execute the command.

# Examples
User: "Run the training loop for the dinov3 benchmark."
Command: `uv run python -m rvinci.projects.dinov3_benchmark.cli train`

User: "Start the spot pick and place demo."
Command: `uv run python -m rvinci.projects.spot_pick_place.cli run`