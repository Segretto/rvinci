# Imitation Control Pipeline

This project deploys a trained imitation network on a Spot robot. It orchestrates the `perception.dinov3` skill to extract visual features from the gripper camera, and runs them through the `manipulation.imitation` fusion head alongside proprioceptive state to compute 7D relative displacement vectors towards a bottleneck target.

## Running the Project

```bash
uv run python -m rvinci.projects.imitation_control.cli \
    hostname=192.168.80.3 \
    model_path=/path/to/fusion.pth \
    bottleneck_path=/path/to/target.json \
    sleep_freq=0.1
```
