import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig

from rvinci.core.logging import get_logger
from rvinci.skills.manipulation.imitation.datasets.dataset import OfflineFeatureDataset
from rvinci.skills.manipulation.imitation.models.fusion_head import PoseFusionHead
from rvinci.projects.imitation_control.pipeline.run import (
    _pose_3dof_to_se3,
    SE3Pose,
    Quat,
)

log = get_logger(__name__)


def compute_relative_target(
    wrist_pose: torch.Tensor, target_pose: torch.Tensor
) -> torch.Tensor:
    """
    Computes the relative delta needed to move from wrist_pose to target_pose.
    Both inputs are [XYZ, WXYZ] shape (7,).
    Returns a (7,) tensor for [dx, dy, dz, qw, qx, qy, qz].
    """
    wp = SE3Pose(
        x=wrist_pose[0].item(),
        y=wrist_pose[1].item(),
        z=wrist_pose[2].item(),
        rot=Quat(
            w=wrist_pose[3].item(),
            x=wrist_pose[4].item(),
            y=wrist_pose[5].item(),
            z=wrist_pose[6].item(),
        ),
    )

    tp = SE3Pose(
        x=target_pose[0].item(),
        y=target_pose[1].item(),
        z=target_pose[2].item(),
        rot=Quat(
            w=target_pose[3].item(),
            x=target_pose[4].item(),
            y=target_pose[5].item(),
            z=target_pose[6].item(),
        ),
    )

    # We want delta such that wp * delta = tp  =>  delta = wp^-1 * tp
    delta = wp.inverse() * tp
    return torch.tensor(
        [delta.x, delta.y, delta.z, delta.rot.w, delta.rot.x, delta.rot.y, delta.rot.z],
        dtype=torch.float32,
    )


def train_pipeline(config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    try:
        hydra_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        hydra_dir = "runs/imitation_training_debug"
        os.makedirs(hydra_dir, exist_ok=True)

    log.info(f"Saving checkpoints to: {hydra_dir}")

    # 1. Load Data
    log.info("Loading Datasets...")
    train_dataset = OfflineFeatureDataset(config.training.train_data_path)
    val_dataset = OfflineFeatureDataset(config.training.val_data_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4
    )

    # 2. Initialize Model
    log.info("Initializing PoseFusionHead...")
    model = PoseFusionHead(
        query_dim=config.skills.imitation.query_dim,
        proprio_dim=config.skills.imitation.proprio_dim,
        hidden_dim=config.skills.imitation.hidden_dim,
        output_quaternion=config.skills.imitation.output_quaternion,
    ).to(device)

    # 3. Optimization
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion_translation = nn.MSELoss()
    criterion_rotation = nn.MSELoss()

    # 4. Training Loop
    log.info("Starting Training...")
    for epoch in range(config.training.num_epochs):
        model.train()
        running_loss = 0.0

        for visual_features, wrist_poses, target_poses in train_loader:
            visual_features = visual_features.to(device)
            wrist_poses = wrist_poses.to(device)
            target_poses = target_poses.to(device)

            # Compute Ground Truth Deltas
            # Bx7 dimension matching
            batch_size = visual_features.size(0)
            gt_deltas = torch.stack(
                [
                    compute_relative_target(wrist_poses[i], target_poses[i])
                    for i in range(batch_size)
                ]
            ).to(device)

            gt_translation = gt_deltas[:, :3]
            gt_rotation = gt_deltas[:, 3:]

            # Forward pass
            optimizer.zero_grad()
            pred_translation, pred_rotation = model(visual_features, wrist_poses)

            loss_t = criterion_translation(pred_translation, gt_translation)
            loss_r = criterion_rotation(pred_rotation, gt_rotation)

            # Simple uniform weighting for now
            loss = loss_t + loss_r

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        log.info(
            f"Epoch [{epoch + 1}/{config.training.num_epochs}], Train Loss: {avg_loss:.6f}"
        )

        # Basic Validation
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for visual_features, wrist_poses, target_poses in val_loader:
                visual_features = visual_features.to(device)
                wrist_poses = wrist_poses.to(device)
                target_poses = target_poses.to(device)

                batch_size = visual_features.size(0)
                gt_deltas = torch.stack(
                    [
                        compute_relative_target(wrist_poses[i], target_poses[i])
                        for i in range(batch_size)
                    ]
                ).to(device)

                gt_translation = gt_deltas[:, :3]
                gt_rotation = gt_deltas[:, 3:]

                pred_translation, pred_rotation = model(visual_features, wrist_poses)
                loss_t = criterion_translation(pred_translation, gt_translation)
                loss_r = criterion_rotation(pred_rotation, gt_rotation)
                val_loss += (loss_t + loss_r).item()

        avg_val_loss = val_loss / len(val_loader)
        log.info(
            f"Epoch [{epoch + 1}/{config.training.num_epochs}], Val Loss: {avg_val_loss:.6f}"
        )

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0 or epoch == config.training.num_epochs - 1:
            checkpoint_path = os.path.join(
                hydra_dir, f"imitation_head_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            log.info(f"Saved checkpoint: {checkpoint_path}")

    log.info("Training Complete!")
