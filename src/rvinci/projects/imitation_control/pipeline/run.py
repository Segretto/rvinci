import time
import json
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict
from transformers import AutoImageProcessor

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.client.estop

from bosdyn.api import estop_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient

from rvinci.core.logging import get_logger
from rvinci.projects.imitation_control.schemas.config import ProjectConfig

from rvinci.skills.perception.dinov3.api import build_dinov3
from rvinci.skills.manipulation.imitation.api import PoseFusionHead, parse_pose_dict

log = get_logger(__name__)

MOVE_DURATION = 5.0


def _pose_3dof_to_se3(
    x: float, y: float, z: float, phi: float
) -> Dict[str, Dict[str, float]]:
    """
    Converts 3DOF [y, z, phi] into full 6DOF pose with fixed x (e.g., gripper TCP x-aligned).
    Returns dict with 'position': {x,y,z} and 'rotation': quaternion {w,x,y,z}.
    """
    log.debug(
        f"Converting 3DOF pose to SE(3): x={x:.3f}, y={y:.3f}, z={z:.3f}, phi={phi:.3f}"
    )
    position = {"x": x, "y": y, "z": z}
    # Handling scipy rotation (scipy 1.4+ supports scalar_first, but manual reorder is safer for generic envs)
    r = R.from_euler("xyz", [phi, 0, 0])
    q_xyzw = r.as_quat()
    quat = {"w": q_xyzw[3], "x": q_xyzw[0], "y": q_xyzw[1], "z": q_xyzw[2]}
    return {"position": position, "rotation": quat}


# Copied utility functions from original imitation_control.py (simplified for space)
def pose_to_SE3(pose: dict) -> SE3Pose:
    pos = pose["position"]
    rot = pose["rotation"]
    return SE3Pose(
        x=pos["x"],
        y=pos["y"],
        z=pos["z"],
        rot=Quat(w=rot["w"], x=rot["x"], y=rot["y"], z=rot["z"]),
    )


def SE3_to_pose(se3_pose: SE3Pose) -> dict:
    return {
        "position": {"x": se3_pose.x, "y": se3_pose.y, "z": se3_pose.z},
        "rotation": {
            "w": se3_pose.rot.w,
            "x": se3_pose.rot.x,
            "y": se3_pose.rot.y,
            "z": se3_pose.rot.z,
        },
    }


def get_bottleneck_pose(bottleneck_path: str):
    with open(bottleneck_path, "r") as f:
        bottleneck_pose = json.load(f)
    return bottleneck_pose["gripper_frame"]


def verify_estop(robot):
    client = robot.ensure_client(bosdyn.client.estop.EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        raise Exception(
            "Robot is estopped. Use an external E-Stop client to clear the stop condition."
        )


def power_on(robot):
    robot.logger.info("Powering on robot...")
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), "Robot power on failed."
    robot.logger.info("Robot powered on.")


def deploy_arm(command_client, robot):
    robot.logger.info("Unstowing arm...")
    unstow = RobotCommandBuilder.arm_ready_command()
    cmd_id = command_client.robot_command(unstow)
    block_until_arm_arrives(command_client, cmd_id, timeout_sec=10)
    robot.logger.info("Arm is ready.")


def safe_stow(command_client, robot, timeout_sec=10):
    try:
        robot.logger.info("Stowing arm safely...")
        robot_cmd = RobotCommandBuilder.arm_stow_command()
        cmd_id = command_client.robot_command(robot_cmd)
        block_until_arm_arrives(command_client, cmd_id, timeout_sec)
        robot.logger.info("Arm is now in stow position.")
    except Exception as e:
        robot.logger.error(f"Failed to stow arm: {e}")


def get_gripper_rgb_image(image_client, robot) -> np.ndarray | None:
    image_response = image_client.get_image_from_sources(["hand_color_image"])
    if not image_response:
        robot.logger.error("Failed to capture image from gripper camera.")
        return None
    # We decode the image data stream using cv2
    image_data = np.frombuffer(image_response[0].shot.image.data, dtype=np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return img


def run_pipeline(config: ProjectConfig) -> None:
    """
    Orchestrates the robot behavior, DINOv3 pipeline, and PoseFusion head.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Models
    log.info("Loading DINOv3 Backbone...")
    # NOTE: Hardcoding repo for simplicity in this porting step
    dino_model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model="dinov2_vits14",
    )

    log.info("Building Unified DINOv3 Feature Extractor...")
    dinov3_model = build_dinov3(dino_model, config.skills.dinov3)
    dinov3_model.to(device)
    dinov3_model.eval()

    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-small-coco-instance"
    )

    log.info("Loading PoseFusionHead...")
    fusion_head = PoseFusionHead(
        query_dim=config.skills.imitation.query_dim,
        proprio_dim=config.skills.imitation.proprio_dim,
        hidden_dim=config.skills.imitation.hidden_dim,
        output_quaternion=config.skills.imitation.output_quaternion,
    )
    # Load Weights
    # fusion_head.load_state_dict(torch.load(config.model_path, map_location=device))
    fusion_head.to(device)
    fusion_head.eval()

    # 2. Setup Robot Client
    sdk = bosdyn.client.create_standard_sdk("NeuralNetworkPoseController")
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    robot.logger = log

    # Unused placeholder for Bottleneck absolute target if needed in the future
    _ = get_bottleneck_pose(config.bottleneck_path)

    lease_client = robot.ensure_client(
        bosdyn.client.lease.LeaseClient.default_service_name
    )
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    assert robot.has_arm(), "This example requires a robot with an arm."
    verify_estop(robot)

    try:
        with bosdyn.client.lease.LeaseKeepAlive(
            lease_client, must_acquire=True, return_at_exit=True
        ):
            power_on(robot)
            log.info("Commanding robot to stand...")
            blocking_stand(command_client, timeout_sec=10)
            deploy_arm(command_client, robot)

            gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
                open_fraction=1.0
            )
            command_client.robot_command(gripper_cmd)

            keep_running = True

            while keep_running:
                # Get Observation
                original_bgr_frame = get_gripper_rgb_image(image_client, robot)
                if original_bgr_frame is None:
                    continue

                # Process Vision
                rgb_frame = cv2.cvtColor(original_bgr_frame, cv2.COLOR_BGR2RGB)

                # Ensure image logic matches the DINOv3 pipeline which uses 518x518 standard
                # Force don't resize in processor so DINOv2 patch embedding (14x14) works
                image_size = config.skills.dinov3.image_size
                rgb_frame = cv2.resize(rgb_frame, (image_size, image_size))
                inputs = processor(
                    images=rgb_frame,
                    return_tensors="pt",
                    do_resize=False,
                ).to(device)

                with torch.inference_mode():
                    # 1. Get raw model outputs (depth and the full seg_out dict)
                    _, seg_out = dinov3_model(inputs.pixel_values)

                    # 2. Extract the latent queries from the decoder
                    all_queries = seg_out["transformer_decoder_last_hidden_state"]

                    # 3. Find the 'winning' query for target_class_id = 1
                    class_logits = seg_out["class_queries_logits"]
                    probs = torch.softmax(class_logits, dim=-1)
                    target_class_id = 1
                    query_indices = probs[..., target_class_id].argmax(dim=-1)

                    # 4. Extract only that specific query vector
                    batch_idx = torch.arange(all_queries.size(0))
                    global_query = all_queries[batch_idx, query_indices]

                # Get Proprioception
                robot_state = robot_state_client.get_robot_state()
                current_se3 = get_a_tform_b(
                    robot_state.kinematic_state.transforms_snapshot,
                    BODY_FRAME_NAME,
                    HAND_FRAME_NAME,
                )
                current_pose = SE3_to_pose(current_se3)

                # Transform to network expected input (7D Tensor)
                tcp_tensor = parse_pose_dict(current_pose).unsqueeze(0).to(device)

                # Predict Offset
                with torch.inference_mode():
                    delta_translation, delta_rotation = fusion_head(
                        global_query, tcp_tensor
                    )

                # Extract Predictions
                dx, dy, dz = delta_translation.cpu().squeeze().numpy().tolist()
                qw, qx, qy, qz = delta_rotation.cpu().squeeze().numpy().tolist()

                # Predicted translation and rotation are relative (hand_tform_bottleneck)
                predicted_delta_se3 = SE3Pose(
                    x=dx, y=dy, z=dz, rot=Quat(w=qw, x=qx, y=qy, z=qz)
                )

                # Calculate absolute target in body frame: body_tform_bottleneck = body_tform_hand * hand_tform_bottleneck
                target_pose = current_se3 * predicted_delta_se3

                log.info(
                    f"Predicted Deltas - trans: {dx:.3f}, {dy:.3f}, {dz:.3f} | rot: {qw:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f}"
                )
                log.info(f"Commanding Arm to Target: \n{target_pose}")

                # Command the robot arm
                robot_cmd = RobotCommandBuilder.arm_pose_command(
                    x=target_pose.x,
                    y=target_pose.y,
                    z=target_pose.z,
                    qw=target_pose.rot.w,
                    qx=target_pose.rot.x,
                    qy=target_pose.rot.y,
                    qz=target_pose.rot.z,
                    frame_name=BODY_FRAME_NAME,
                    seconds=2.0,  # Smoothing duration
                )
                command_client.robot_command(robot_cmd)

                time.sleep(config.sleep_freq)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt detected. Stopping robot.")
    except Exception as e:
        log.error(f"Exception occurred: {e}")
    finally:
        safe_stow(command_client, robot)
        robot.power_off(cut_immediately=False, timeout_sec=20)
        log.info("Robot safely powered off.")
