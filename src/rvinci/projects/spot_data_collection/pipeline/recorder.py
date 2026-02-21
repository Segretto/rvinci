import time
import json
import os
import math
import logging
from pathlib import Path
from threading import Thread, Event
from typing import Optional

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
)
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    HAND_FRAME_NAME,
    VISION_FRAME_NAME,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    get_a_tform_b,
)

from rvinci.core.logging import get_logger
from rvinci.projects.spot_data_collection.schemas.config import (
    SpotClientConfig,
    SpotTrajectoryConfig,
)

log = get_logger(__name__)


ARM_JOINTS = ["arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1"]

# ---------------------------------------------------------------------------
# Helper Functions Refactored
# ---------------------------------------------------------------------------


def load_bottleneck_frame(run_dir: Path, output_dir_name: str):
    """Loads the bottleneck frame from the JSON file relative to the run dir."""
    filename = run_dir / output_dir_name / "gripper_bottleneck_frame.json"
    if not filename.exists():
        raise FileNotFoundError(
            f"File {filename} not found. Please record fixed frame first with 'f'"
        )
    with open(filename, "r") as f:
        data = json.load(f)
    gripper_frame = data["gripper_frame"]
    pos = gripper_frame["position"]
    rot = gripper_frame["rotation"]
    recorded_body_height = data.get("recorded_body_height", 0.0)

    pose = SE3Pose(
        x=pos["x"],
        y=pos["y"],
        z=pos["z"],
        rot=Quat(w=rot["w"], x=rot["x"], y=rot["y"], z=rot["z"]),
    )
    return pose, recorded_body_height


def generate_random_position(
    bottleneck_frame: SE3Pose, traj_config: SpotTrajectoryConfig
):
    """Generates a randomized 3D position relative to the bottleneck, with rotation on the gripper's x-axis."""
    x_distance = np.random.triangular(0, 0, traj_config.x_distance_range)
    y_offset = np.random.triangular(-traj_config.y_range, 0, traj_config.y_range)
    z_offset = np.random.triangular(-traj_config.z_range, 0, traj_config.z_range)

    # Calculate translation in BODY frame (additive)
    x = bottleneck_frame.x - x_distance - 0.2
    y = bottleneck_frame.y + y_offset
    z = bottleneck_frame.z + z_offset

    # Calculate rotation (roll relative to bottleneck)
    cos_half_roll = np.cos(0 / 2)
    sin_half_roll = np.sin(0 / 2)
    roll_quat = Quat(w=cos_half_roll, x=sin_half_roll, y=0, z=0)
    # final_quat = bottleneck_frame.rot * roll_quat

    return SE3Pose(x=x, y=y, z=z, rot=roll_quat)


def generate_plane_position(
    bottleneck_frame: SE3Pose, traj_config: SpotTrajectoryConfig
):
    """Generates a position on the bottleneck plane (fixed x, reduced y/z variations, random rotation)."""
    y_offset = np.random.triangular(
        -traj_config.y_range_plane, 0, traj_config.y_range_plane
    )
    z_offset = np.random.triangular(
        -traj_config.z_range_plane, 0, traj_config.z_range_plane
    )

    roll = np.random.uniform(
        -math.radians(traj_config.roll_range_deg),
        math.radians(traj_config.roll_range_deg),
    )

    # Keep x fixed from bottleneck, vary y and z in BODY frame
    x = bottleneck_frame.x - 0.2
    y = bottleneck_frame.y + y_offset
    z = bottleneck_frame.z + z_offset

    cos_half_roll = np.cos(roll / 2)
    sin_half_roll = np.sin(roll / 2)
    roll_quat = Quat(w=cos_half_roll, x=sin_half_roll, y=0, z=0)
    # final_quat = bottleneck_frame.rot * roll_quat

    return SE3Pose(x=x, y=y, z=z, rot=roll_quat)


def process_rgb_image(image_response_rgb):
    """Processes the RGB image."""
    cv_visual = cv2.imdecode(
        np.frombuffer(image_response_rgb.shot.image.data, dtype=np.uint8), -1
    )
    visual_rgb = (
        cv_visual
        if len(cv_visual.shape) == 3
        else cv2.cvtColor(cv_visual, cv2.COLOR_GRAY2RGB)
    )
    return visual_rgb


def process_depth_image(image_response_depth):
    """Processes the depth image, returning a raw NumPy array."""
    depth_data = np.frombuffer(image_response_depth.shot.image.data, dtype=np.uint16)
    depth_image = depth_data.reshape(
        image_response_depth.shot.image.rows, image_response_depth.shot.image.cols
    )
    return depth_image


def save_data(all_rgb, all_depth, all_transforms, position_dir: Path):
    """Saves all RGB images, depth images, and transforms into the specified directory for the position."""
    position_dir.mkdir(parents=True, exist_ok=True)

    for idx, (rgb, depth, transform) in enumerate(
        zip(all_rgb, all_depth, all_transforms)
    ):
        filename_rgb = position_dir / f"image_{idx}.jpg"
        filename_depth = position_dir / f"depth_{idx}.npy"
        filename_json = position_dir / f"image_{idx}.json"
        cv2.imwrite(str(filename_rgb), rgb)
        np.save(str(filename_depth), depth)
        with open(filename_json, "w") as f:
            json.dump(transform, f)
    return position_dir


def transform_snapshot_to_dict(
    transforms_snapshot,
    wrist_frame: SE3Pose,
    gripper_state: float,
    image_filename: str,
    hand_tform_bottleneck: SE3Pose,
):
    """Converts the transforms snapshot into a dictionary, including hand_tform_bottleneck."""
    output_dict = {
        "image_filename": str(image_filename),
        "gripper_open_angle_degrees": gripper_state * 90,
        "transforms": {},
        "wrist_frame": {
            "position": {"x": wrist_frame.x, "y": wrist_frame.y, "z": wrist_frame.z},
            "rotation": {
                "w": wrist_frame.rot.w,
                "x": wrist_frame.rot.x,
                "y": wrist_frame.rot.y,
                "z": wrist_frame.rot.z,
            },
        },
        "gripper_state": gripper_state,
        "hand_tform_bottleneck": {
            "position": {
                "x": hand_tform_bottleneck.x,
                "y": hand_tform_bottleneck.y,
                "z": hand_tform_bottleneck.z,
            },
            "rotation": {
                "w": hand_tform_bottleneck.rot.w,
                "x": hand_tform_bottleneck.rot.x,
                "y": hand_tform_bottleneck.rot.y,
                "z": hand_tform_bottleneck.rot.z,
            },
        },
    }
    for key, value in transforms_snapshot.child_to_parent_edge_map.items():
        transform_dict = {
            "parent_frame_name": value.parent_frame_name
            if hasattr(value, "parent_frame_name")
            else None,
            "parent_tform_child": {
                "position": {
                    "x": value.parent_tform_child.position.x
                    if hasattr(value.parent_tform_child.position, "x")
                    else 0.0,
                    "y": value.parent_tform_child.position.y
                    if hasattr(value.parent_tform_child.position, "y")
                    else 0.0,
                    "z": value.parent_tform_child.position.z
                    if hasattr(value.parent_tform_child.position, "z")
                    else 0.0,
                },
                "rotation": {
                    "x": value.parent_tform_child.rotation.x
                    if hasattr(value.parent_tform_child.rotation, "x")
                    else 0.0,
                    "y": value.parent_tform_child.rotation.y
                    if hasattr(value.parent_tform_child.rotation, "y")
                    else 0.0,
                    "z": value.parent_tform_child.rotation.z
                    if hasattr(value.parent_tform_child.rotation, "z")
                    else 0.0,
                    "w": value.parent_tform_child.rotation.w
                    if hasattr(value.parent_tform_child.rotation, "w")
                    else 1.0,
                },
            },
        }
        output_dict["transforms"][key] = transform_dict
    return output_dict


def calculate_gripper_to_bottleneck_transform(robot_state_client, bottleneck_frame):
    """Calculates the current gripper transform relative to the bottleneck."""
    robot_state = robot_state_client.get_robot_state()
    body_tform_hand = get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        BODY_FRAME_NAME,
        HAND_FRAME_NAME,
    )
    body_tform_bottleneck = bottleneck_frame
    hand_tform_bottleneck = body_tform_bottleneck.inverse() * body_tform_hand
    return hand_tform_bottleneck


# ---------------------------------------------------------------------------
# Main Recorder Class
# ---------------------------------------------------------------------------


class JointAndFrameRecorder:
    def __init__(self, run_dir: Path):
        self.recording = False
        self.data_collection: list[dict] = []
        self.robot = None
        self.robot_state_client: Optional[RobotStateClient] = None
        self.command_client: Optional[RobotCommandClient] = None
        self.image_client: Optional[ImageClient] = None
        self.lease_keep_alive: Optional[LeaseKeepAlive] = None
        self.stop_event = Event()
        self.run_dir = run_dir

    def record_loop(self, frequency_hz: float):
        freq = 1.0 / frequency_hz
        while not self.stop_event.is_set():
            if self.recording:
                start_time = time.time()
                try:
                    if self.robot_state_client is None:
                        continue

                    robot_state = self.robot_state_client.get_robot_state()
                    joint_values = {
                        link.name: {
                            "position": link.position.value,
                            "velocity": link.velocity.value,
                        }
                        for link in robot_state.kinematic_state.joint_states
                        if link.name in ARM_JOINTS
                    }
                    gripper_pose_tform = get_a_tform_b(
                        robot_state.kinematic_state.transforms_snapshot,
                        BODY_FRAME_NAME,
                        HAND_FRAME_NAME,
                    )
                    gripper_state = {
                        "open_percentage": robot_state.manipulator_state.gripper_open_percentage,
                    }
                    timestamp = (
                        robot_state.kinematic_state.acquisition_timestamp.seconds
                        + robot_state.kinematic_state.acquisition_timestamp.nanos / 1e9
                    )
                    snapshot = {
                        "timestamp": timestamp,
                        "joints": joint_values,
                        "gripper_pose": {
                            "position": {
                                "x": gripper_pose_tform.x,
                                "y": gripper_pose_tform.y,
                                "z": gripper_pose_tform.z,
                            },
                            "rotation": {
                                "w": gripper_pose_tform.rot.w,
                                "x": gripper_pose_tform.rot.x,
                                "y": gripper_pose_tform.rot.y,
                                "z": gripper_pose_tform.rot.z,
                            },
                        },
                        "gripper_state": gripper_state,
                    }
                    self.data_collection.append(snapshot)

                    elapsed = time.time() - start_time
                    if elapsed < freq:
                        time.sleep(freq - elapsed)
                except Exception as e:
                    log.error(f"Error during recording: {e}")
                    self.recording = False
            else:
                time.sleep(0.1)

    def _gather_randomized_trajectories(
        self, spot_config: SpotClientConfig, traj_config: SpotTrajectoryConfig
    ):
        log.info("Loading bottleneck frame...")
        try:
            bottleneck_frame, recorded_body_height = load_bottleneck_frame(
                self.run_dir, spot_config.output_dir
            )
            log.info(
                f"Bottleneck frame loaded (recorded_body_h={recorded_body_height:.3f})"
            )
        except Exception as e:
            log.error(f"Error loading bottleneck frame: {e}")
            return

        log.info("Preparing robot...")
        if not self.robot.is_powered_on():
            self.robot.power_on(timeout_sec=20)
            if not self.robot.is_powered_on():
                log.error("Failed to power on the robot.")
                return
        log.info("Robot powered on.")

        log.info(
            f"Standing up robot to recorded height: {recorded_body_height:.3f} m..."
        )
        stand_cmd = RobotCommandBuilder.synchro_stand_command(
            body_height=recorded_body_height
        )
        self.command_client.robot_command(stand_cmd)
        time.sleep(3)

        log.info(f"Bottleneck z (in BODY frame): {bottleneck_frame.z:.3f}")
        # log.info("Preparing arm...")
        # arm_cmd = RobotCommandBuilder.arm_ready_command()
        # self.command_client.robot_command(arm_cmd)
        # time.sleep(2)

        log.info("Opening gripper...")
        gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
            open_fraction=1.0
        )
        self.command_client.robot_command(gripper_cmd)
        time.sleep(2)

        # Build trajectory datasets output dir inside the RUN directory
        mlp_datasets_dir = self.run_dir / "MLP_Datasets"
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        base_dir = mlp_datasets_dir / f"Dataset_{timestamp}"
        base_dir.mkdir(parents=True, exist_ok=True)

        frames_hz = traj_config.frame_rate
        frame_interval = 1.0 / frames_hz
        sources = ["hand_color_image", "hand_depth"]

        log.info(
            f"Starting dataset collection with {traj_config.num_positions} positions at {frames_hz} Hz..."
        )

        # We interleave a regular target pose with a plane_pose
        # For sequence matching, each random target move is followed by a plane target pass.
        for idx in range(traj_config.num_positions):
            # 1. Random Pose Move
            target_pose = generate_random_position(bottleneck_frame, traj_config)
            log.info(
                f"Going to position {idx + 1}/{traj_config.num_positions}: {target_pose}"
            )

            robot_cmd = RobotCommandBuilder.arm_pose_command(
                x=target_pose.x,
                y=target_pose.y,
                z=target_pose.z,
                qw=target_pose.rot.w,
                qx=target_pose.rot.x,
                qy=target_pose.rot.y,
                qz=target_pose.rot.z,
                frame_name=BODY_FRAME_NAME,
                seconds=traj_config.move_duration,
            )

            log.info(f"Executing move command to target pose: {target_pose}")
            cmd_id = self.command_client.robot_command(
                robot_cmd, end_time_secs=time.time() + traj_config.move_duration
            )

            all_rgb, all_depth, all_transforms = [], [], []
            position_dir = base_dir / f"position_{idx + 1}"
            start_time = time.time()

            while time.time() - start_time < traj_config.move_duration:
                loop_start = time.time()
                image_responses = self.image_client.get_image_from_sources(sources)
                if len(image_responses) < 2:
                    continue

                rgb = process_rgb_image(image_responses[0])
                depth = process_depth_image(image_responses[1])
                robot_state = self.robot_state_client.get_robot_state()
                wrist_frame = get_a_tform_b(
                    robot_state.kinematic_state.transforms_snapshot,
                    BODY_FRAME_NAME,
                    HAND_FRAME_NAME,
                )
                hand_tform_bottleneck = calculate_gripper_to_bottleneck_transform(
                    self.robot_state_client, bottleneck_frame
                )

                rel_img_filename = f"position_{idx + 1}/image_{len(all_rgb)}.jpg"
                transform = transform_snapshot_to_dict(
                    image_responses[0].shot.transforms_snapshot,
                    wrist_frame,
                    1.0,
                    rel_img_filename,
                    hand_tform_bottleneck,
                )

                all_rgb.append(rgb)
                all_depth.append(depth)
                all_transforms.append(transform)

                if int((time.time() - start_time) * frames_hz) % frames_hz == 0:
                    distance = math.sqrt(
                        hand_tform_bottleneck.x**2
                        + hand_tform_bottleneck.y**2
                        + hand_tform_bottleneck.z**2
                    )
                    log.info(
                        f"Pos {idx + 1}/{traj_config.num_positions} - Frame {len(all_rgb)}. Dist: {distance:.3f} m"
                    )

                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

            block_until_arm_arrives(self.command_client, cmd_id, timeout_sec=1.0)
            save_data(all_rgb, all_depth, all_transforms, position_dir)
            log.info(f"Saved frames to {position_dir}")

            # 2. Plane Pose Move
            plane_pose = generate_plane_position(bottleneck_frame, traj_config)
            robot_cmd = RobotCommandBuilder.arm_pose_command(
                x=plane_pose.x,
                y=plane_pose.y,
                z=plane_pose.z,
                qw=plane_pose.rot.w,
                qx=plane_pose.rot.x,
                qy=plane_pose.rot.y,
                qz=plane_pose.rot.z,
                frame_name=BODY_FRAME_NAME,
                seconds=traj_config.move_duration,
            )
            cmd_id = self.command_client.robot_command(
                robot_cmd, end_time_secs=time.time() + traj_config.move_duration
            )

            all_rgb, all_depth, all_transforms = [], [], []
            position_dir = base_dir / f"position_{idx + 1}_to_plane"
            start_time = time.time()

            while time.time() - start_time < traj_config.move_duration:
                loop_start = time.time()
                image_responses = self.image_client.get_image_from_sources(sources)
                if len(image_responses) < 2:
                    continue

                rgb = process_rgb_image(image_responses[0])
                depth = process_depth_image(image_responses[1])
                robot_state = self.robot_state_client.get_robot_state()
                wrist_frame = get_a_tform_b(
                    robot_state.kinematic_state.transforms_snapshot,
                    BODY_FRAME_NAME,
                    HAND_FRAME_NAME,
                )
                hand_tform_bottleneck = calculate_gripper_to_bottleneck_transform(
                    self.robot_state_client, bottleneck_frame
                )

                rel_img_filename = (
                    f"position_{idx + 1}_to_plane/image_{len(all_rgb)}.jpg"
                )
                transform = transform_snapshot_to_dict(
                    image_responses[0].shot.transforms_snapshot,
                    wrist_frame,
                    1.0,
                    rel_img_filename,
                    hand_tform_bottleneck,
                )

                all_rgb.append(rgb)
                all_depth.append(depth)
                all_transforms.append(transform)

                if int((time.time() - start_time) * frames_hz) % frames_hz == 0:
                    distance = math.sqrt(
                        hand_tform_bottleneck.x**2
                        + hand_tform_bottleneck.y**2
                        + hand_tform_bottleneck.z**2
                    )
                    log.info(
                        f"Plane pos {idx + 1}/{traj_config.num_positions} - Frame {len(all_rgb)}. Dist: {distance:.3f} m"
                    )

                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

            block_until_arm_arrives(self.command_client, cmd_id, timeout_sec=1.0)
            save_data(all_rgb, all_depth, all_transforms, position_dir)

        log.info("Finished gathering randomized trajectories. Stowing arm...")
        arm_cmd = RobotCommandBuilder.arm_stow_command()
        self.command_client.robot_command(arm_cmd)
        time.sleep(2)
        log.info("Done.")

    def run(self, spot_config: SpotClientConfig, traj_config: SpotTrajectoryConfig):
        bosdyn.client.util.setup_logging(False)
        sdk = bosdyn.client.create_standard_sdk("JointAndFrameRecorder")

        os.environ["BOSDYN_CLIENT_USERNAME"] = spot_config.username
        os.environ["BOSDYN_CLIENT_PASSWORD"] = spot_config.password

        log.info(f"Connecting to Spot at {spot_config.hostname}...")
        self.robot = sdk.create_robot(spot_config.hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()

        self.robot_state_client = self.robot.ensure_client(
            RobotStateClient.default_service_name
        )

        # Verify it has an arm
        assert self.robot.has_arm(), (
            "Robot requires an arm to run trajectory generation"
        )

        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.command_client = self.robot.ensure_client(
            RobotCommandClient.default_service_name
        )

        lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        try:
            lease_client.take()
            log.info("Successfully took lease for the entire spot")
            self.lease_keep_alive = LeaseKeepAlive(
                lease_client, must_acquire=True, return_at_exit=True, warnings=False
            )
        except Exception as e:
            log.error(f"Failed to take lease: {e}")
            return

        log.info(
            "Connected. Type 's' to start recording, 'e' to stop, "
            "'f' to record fixed frame, 'g' to start randomized trajectories, 'q' to quit."
        )

        record_thread = Thread(
            target=self.record_loop, args=(spot_config.frequency_hz,)
        )
        record_thread.start()

        output_dir = self.run_dir / spot_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            while True:
                command = input("Command: ").strip().lower()
                if command == "q":
                    self.stop_event.set()
                    break
                elif command == "s" and not self.recording:
                    self.recording = True
                    self.data_collection = []
                    log.info("RECORDING unified trajectory...")
                elif command == "e" and self.recording:
                    self.recording = False
                    if self.data_collection:
                        output_file = output_dir / "joints_and_gripper.json"
                        with open(output_file, "w") as f:
                            json.dump(self.data_collection, f, indent=4)
                        log.info(f"Trajectory saved to {output_file}")
                elif command == "f":
                    state = self.robot_state_client.get_robot_state()
                    pose = get_a_tform_b(
                        state.kinematic_state.transforms_snapshot,
                        GRAV_ALIGNED_BODY_FRAME_NAME,
                        HAND_FRAME_NAME,
                    )
                    vision_tform_body = get_a_tform_b(
                        state.kinematic_state.transforms_snapshot,
                        VISION_FRAME_NAME,
                        GRAV_ALIGNED_BODY_FRAME_NAME,
                    )

                    fixed_file = output_dir / "gripper_bottleneck_frame.json"
                    with open(fixed_file, "w") as f:
                        json.dump(
                            {
                                "gripper_frame": {
                                    "position": {"x": pose.x, "y": pose.y, "z": pose.z},
                                    "rotation": {
                                        "w": pose.rot.w,
                                        "x": pose.rot.x,
                                        "y": pose.rot.y,
                                        "z": pose.rot.z,
                                    },
                                },
                                "recorded_body_height": vision_tform_body.z,
                            },
                            f,
                            indent=4,
                        )
                    log.info(
                        f"Fixed bottleneck frame recorded (z={pose.z:.3f}, body_h={vision_tform_body.z:.3f})."
                    )
                elif command == "g":
                    self._gather_randomized_trajectories(spot_config, traj_config)

        except KeyboardInterrupt:
            self.stop_event.set()

        record_thread.join()
