# Imitation Control Pipeline

This project encompasses the full loco-manipulation pipeline for deploying a trained imitation network on a Boston Dynamics Spot robot. 

It orchestrates the `perception.dinov3` skill to extract visual features from the gripper camera, and runs them through the `manipulation.imitation` fusion head alongside proprioceptive state to compute 7D relative displacement vectors towards a bottleneck target. The project includes everything from data extraction and training to the real-time inference wrapper.

---

## 1. Offline Dataset Generation

Before training a regression model, you need to extract the DINOv3 visual representations alongside the Spot robot's proprioceptive logs. The `generate_dataset.py` pipeline runs the `UnifiedDINOv3` model over your annotated images to construct a feature dataset.

By default, the script extracts the winning Mask2Former latent query representing the target object. It can also be configured to extract the full spatial masks.

**Configuration file:** `configs/generate_dataset.yaml`

```bash
uv run python -m rvinci.projects.imitation_control.generate_dataset \
    dataset.input_dir="/path/to/raw/data_sample" \
    dataset.output_file="runs/dataset.pkl" \
    dataset.feature_mode="query" 
```

### Parameters
- **`dataset.input_dir`**: Path to the directory containing raw `.json` files and their accompanying images logged from Spot.
- **`dataset.output_file`**: Destination to save the feature extraction result (`.pkl` or `.pt`).
- **`dataset.feature_mode`**: Can be `"query"` to tap into the Mask2Former decoder's semantic queries, or `"mask"` for standard DINOv3 segmentation logits.

---

## 2. Imitation Training

The `train_imitation.py` pipeline loads your structured feature datasets into an `OfflineFeatureDataset` memory object and trains the `PoseFusionHead` from the imitation skill. 

It learns to map the extracted visual queries and the robot's existing 7D state (`wrist_pose`) to a 7D relative transformation offset targeting the task task bottleneck. The models optimize MSE loss against the known ground-truth `SE3Pose` relative differences (`wrist_pose^-1 * target_pose`).

PyTorch checkpoints are automatically saved inside your Hydra `runs/` output directories.

**Configuration file:** `configs/train_imitation.yaml`

```bash
uv run python -m rvinci.projects.imitation_control.train_imitation \
    training.train_data_path="runs/dataset_train.pkl" \
    training.val_data_path="runs/dataset_val.pkl" \
    training.batch_size=4 \
    training.learning_rate=1e-4 \
    training.num_epochs=100
```

---

## 3. Robot Control Deployment

After a model completes training, it can be tested live inside the `run.py` inference loop. The project script retrieves RGB images arriving directly from the robot, repeats the vision query extraction strategy dynamically, computes the relative target transformation offsets, and issues execution commands via the Spot SDK.

**Configuration file:** `configs/config.yaml` or `configs/hydra/default.yaml`

```bash
uv run python -m rvinci.projects.imitation_control.cli \
    hostname=192.168.80.3 \
    model_path=runs/imitation_head_epoch_100.pth \
    bottleneck_path=/path/to/target.json \
    sleep_freq=0.1
```

### Parameters
- **`hostname`**: Network IP of the Spot Robot.
- **`model_path`**: Path to the `.pth` weights you finished training in step 2.
- **`bottleneck_path`**: Ground-truth baseline target definition JSON configuration if running a strict evaluation scenario.
- **`sleep_freq`**: Frequency loop rate limit in seconds.
