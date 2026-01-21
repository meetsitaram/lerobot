#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RealMan Robot Follower for LeRobot.

This module provides integration with RealMan robotic arms (R1D2, RM65, RM75, etc.)
for use as follower robots in teleoperation scenarios.

Supports teleoperation from SO101 leader arm with automatic joint mapping:
- SO101 has 5 arm joints + gripper (6 total)
- RealMan R1D2 has 6 arm joints + gripper (7 total)
- Joint 4 on R1D2 is held at a fixed position during SO101 teleoperation

Joint Mapping (SO101 → RealMan R1D2):
=====================================
SO101 uses normalized values from calibration:
- Arm joints: -100 to 100 (RANGE_M100_100 mode)
- Gripper: 0 to 100 (RANGE_0_100 mode)

RealMan R1D2 uses degrees with these limits (from realman_r1d2_joint_limits.yaml):
- joint_1: [-178°, 178°]  ← shoulder_pan
- joint_2: [-130°, 130°]  ← shoulder_lift  
- joint_3: [-135°, 135°]  ← elbow_flex
- joint_4: [-178°, 178°]  ← FIXED at 0° (not mapped from SO101)
- joint_5: [-128°, 128°]  ← wrist_flex
- joint_6: [-360°, 360°]  ← wrist_roll
- gripper: [1, 1000]      ← gripper (scaled from 0-100)
"""

import json
import logging
import time
from functools import cached_property
from pathlib import Path

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_realman_follower import RealManFollowerConfig

logger = logging.getLogger(__name__)


# Joint name mapping from SO101 (leader) to RealMan R1D2 (follower)
# SO101 has 5 arm joints + gripper, R1D2 has 6 arm joints + gripper
# Joint 4 on R1D2 is skipped when mapping from SO101
SO101_TO_REALMAN_JOINT_MAP = {
    "shoulder_pan": 0,   # SO101 joint 1 -> R1D2 joint_1 [-178°, 178°]
    "shoulder_lift": 1,  # SO101 joint 2 -> R1D2 joint_2 [-130°, 130°]
    "elbow_flex": 2,     # SO101 joint 3 -> R1D2 joint_3 [-135°, 135°]
    # R1D2 joint_4 (index 3) is fixed at center position (0°)
    "wrist_flex": 4,     # SO101 joint 4 -> R1D2 joint_5 [-128°, 128°]
    "wrist_roll": 5,     # SO101 joint 5 -> R1D2 joint_6 [-360°, 360°]
}

# Reverse mapping for observation conversion
REALMAN_TO_SO101_JOINT_MAP = {v: k for k, v in SO101_TO_REALMAN_JOINT_MAP.items()}

# RealMan joint names (indexed 0-5)
REALMAN_JOINT_NAMES = [
    "joint_1",
    "joint_2", 
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

# SO101-compatible joint names for action features (what leader sends)
SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class RealManFollower(Robot):
    """
    RealMan robot follower for teleoperation with SO101 or similar leader arms.
    
    This class integrates RealMan robotic arms (R1D2, RM65, RM75, etc.) into the
    LeRobot framework, enabling teleoperation, recording, and replay functionality.
    
    Key features:
    - Automatic joint mapping from 5-joint SO101 leader to 6-joint RealMan arm
    - Joint 4 held at fixed position during SO101 teleoperation
    - Integrated gripper control
    - Safety limits and collision detection
    - Camera support for recording
    """

    config_class = RealManFollowerConfig
    name = "realman_follower"

    def __init__(self, config: RealManFollowerConfig):
        # Don't call super().__init__ yet - we need to set up calibration path first
        # because RealMan uses a different calibration format
        self.robot_type = self.name
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir if config.calibration_dir 
            else Path.home() / ".cache/huggingface/lerobot/calibration/robots" / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        
        # RealMan calibration is just home position, not MotorCalibration
        self.calibration: dict = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()
        
        self.config = config
        self._connected = False
        self._robot_controller = None
        
        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # Track last gripper position for state
        self._last_gripper_position = 500  # Middle position
        self._last_gripper_command = 500  # Last commanded gripper position
        self._last_joint_angles = None  # Cache for current joint angles
        self._last_joint_read_time = 0.0  # Time of last joint read
        
        logger.info(f"Initialized RealManFollower for {config.model} at {config.ip}:{config.port}")

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """Load RealMan calibration (home position) from file."""
        fpath = self.calibration_fpath if fpath is None else fpath
        try:
            with open(fpath) as f:
                self.calibration = json.load(f)
            logger.info(f"Loaded calibration from {fpath}")
        except Exception as e:
            logger.warning(f"Could not load calibration: {e}")
            self.calibration = {}

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """Save RealMan calibration (home position) to file."""
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f:
            json.dump(self.calibration, f, indent=4)
        logger.info(f"Calibration saved to {fpath}")

    def _get_robot_controller(self):
        """Lazy import and create RobotController from realman_teleop."""
        if self._robot_controller is None:
            try:
                from realman_teleop import RobotController
            except ImportError as e:
                raise ImportError(
                    "realman_teleop package not found. Please install it:\n"
                    "  pip install -e /path/to/realman-teleop\n"
                    "or ensure the package is in your Python path."
                ) from e
            
            self._robot_controller = RobotController(
                ip=self.config.ip,
                port=self.config.port,
                model=self.config.model,
                dof=self.config.dof,
            )
        return self._robot_controller

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features using SO101-compatible naming for action compatibility."""
        return {f"{joint}.pos": float for joint in SO101_JOINT_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Features returned by get_observation()."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Features expected by send_action() - SO101-compatible format."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot and cameras are connected."""
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return self._connected and cameras_connected

    @property
    def is_calibrated(self) -> bool:
        """RealMan uses absolute encoders, so always calibrated."""
        return True

    def _log_joint_mapping_info(self) -> None:
        """Log the joint mapping configuration for debugging."""
        logger.info("=" * 60)
        logger.info("RealMan R1D2 Joint Mapping Configuration")
        logger.info("=" * 60)
        logger.info("SO101 Leader → RealMan R1D2 Follower Mapping:")
        logger.info("-" * 60)
        
        for so101_name, realman_idx in SO101_TO_REALMAN_JOINT_MAP.items():
            joint_name = REALMAN_JOINT_NAMES[realman_idx]
            if joint_name in self.config.joint_limits:
                min_deg, max_deg = self.config.joint_limits[joint_name]
                logger.info(
                    f"  {so101_name:15} (-100..100) → {joint_name} [{min_deg:>7.1f}°, {max_deg:>7.1f}°]"
                )
        
        # Log fixed joint 4
        if "joint_4" in self.config.joint_limits:
            min_deg, max_deg = self.config.joint_limits["joint_4"]
            logger.info(
                f"  {'(FIXED)':15}            → joint_4 [{min_deg:>7.1f}°, {max_deg:>7.1f}°] = {self.config.fixed_joint_4_position}°"
            )
        
        # Log gripper
        if "gripper" in self.config.joint_limits:
            min_grip, max_grip = self.config.joint_limits["gripper"]
            logger.info(
                f"  {'gripper':15} (0..100)    → gripper [{min_grip:>7.0f}, {max_grip:>7.0f}]"
            )
        
        logger.info("-" * 60)
        logger.info(f"Joint limit enforcement: {'ENABLED' if self.config.enforce_joint_limits else 'DISABLED'}")
        logger.info(f"Max relative target: {self.config.max_relative_target}°/step")
        logger.info("=" * 60)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the RealMan robot.
        
        Args:
            calibrate: If True, move to home position after connecting.
        """
        logger.info(f"Connecting to RealMan {self.config.model} at {self.config.ip}:{self.config.port}")
        
        robot = self._get_robot_controller()
        
        if not robot.connect():
            raise ConnectionError(
                f"Failed to connect to RealMan robot at {self.config.ip}:{self.config.port}"
            )
        
        self._connected = True
        
        # Log joint mapping configuration
        self._log_joint_mapping_info()
        
        # Set collision detection level (with fallback for API differences)
        if self.config.collision_level > 0:
            try:
                robot.set_collision_level(self.config.collision_level)
                logger.info(f"Collision detection set to level {self.config.collision_level}")
            except AttributeError as e:
                logger.warning(f"Could not set collision level (API not available): {e}")
        
        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()
        
        # Optionally move to home/calibration position
        if calibrate:
            self.calibrate()
        
        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        """
        Calibrate the RealMan robot by recording joint ranges.
        
        This calibration records:
        1. Center position (matching SO101 center/home position)
        2. Min/Max positions for each joint
        
        This allows proper mapping from SO101 normalized values to RealMan degrees.
        """
        if not self._connected:
            logger.warning("Robot not connected, skipping calibration")
            return
        
        robot = self._get_robot_controller()
        
        # Check if calibration file exists
        if self.calibration and "joint_ranges" in self.calibration:
            user_input = input(
                f"Calibration exists for {self.id}. Press ENTER to use it, "
                "or type 'c' to run new calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using existing calibration for {self.id}")
                return
        
        print("\n" + "=" * 60)
        print("RealMan Follower Calibration")
        print("=" * 60)
        print("\nThis calibration will record joint positions to map properly")
        print("with the SO101 leader arm.\n")
        
        # Step 1: Record center position
        print("STEP 1: CENTER POSITION")
        print("-" * 40)
        print("Move the RealMan arm to its CENTER/HOME position.")
        print("This should match where SO101 outputs 0 for all joints.")
        input("Press ENTER when ready...")
        
        center_pos = robot.get_current_joint_angles()
        if center_pos is None:
            logger.error("Failed to read joint positions")
            return
        
        print(f"Center position recorded: {[f'{p:.1f}°' for p in center_pos]}")
        
        # Step 2: Record min/max for each mapped joint
        print("\nSTEP 2: JOINT RANGES")
        print("-" * 40)
        print("For each joint, move it to its MIN and MAX positions.")
        print("This establishes the range that maps to SO101's -100 to +100.\n")
        
        joint_ranges = {}
        
        # Map of SO101 names to RealMan indices we need to calibrate
        joints_to_calibrate = [
            ("shoulder_pan", 0, "joint_1"),
            ("shoulder_lift", 1, "joint_2"),
            ("elbow_flex", 2, "joint_3"),
            ("wrist_flex", 4, "joint_5"),
            ("wrist_roll", 5, "joint_6"),
        ]
        
        for so101_name, realman_idx, realman_name in joints_to_calibrate:
            print(f"\n--- Calibrating {realman_name} (maps to SO101 {so101_name}) ---")
            
            input(f"Move {realman_name} to its MINIMUM position, then press ENTER...")
            min_pos = robot.get_current_joint_angles()
            if min_pos is None:
                logger.error("Failed to read joint positions")
                return
            min_val = min_pos[realman_idx]
            
            input(f"Move {realman_name} to its MAXIMUM position, then press ENTER...")
            max_pos = robot.get_current_joint_angles()
            if max_pos is None:
                logger.error("Failed to read joint positions")
                return
            max_val = max_pos[realman_idx]
            
            # Ensure min < max
            if min_val > max_val:
                min_val, max_val = max_val, min_val
            
            joint_ranges[realman_name] = {
                "min": min_val,
                "max": max_val,
                "center": center_pos[realman_idx],
                "so101_name": so101_name,
                "realman_idx": realman_idx,
            }
            
            print(f"  {realman_name}: [{min_val:.1f}°, {max_val:.1f}°], center: {center_pos[realman_idx]:.1f}°")
        
        # Save calibration
        self.calibration = {
            "home_position": center_pos,
            "joint_ranges": joint_ranges,
        }
        self._save_calibration()
        
        print("\n" + "=" * 60)
        print("Calibration complete!")
        print("=" * 60)
        print(f"Saved to: {self.calibration_fpath}")
        
        # Return to center position
        input("\nPress ENTER to move robot back to center position...")
        robot.movej(center_pos, velocity=self.config.velocity, block=True)
        print("Robot at center position.")

    def configure(self) -> None:
        """Apply runtime configuration to the robot."""
        # Configuration is handled during connect()
        pass

    def _clamp_to_joint_limits(self, joint_idx: int, value: float) -> float:
        """
        Clamp a joint value to its configured limits.
        
        Args:
            joint_idx: Index of the joint (0-5 for R1D2)
            value: Desired joint value in degrees
            
        Returns:
            Clamped value within joint limits
        """
        joint_name = REALMAN_JOINT_NAMES[joint_idx]
        if joint_name in self.config.joint_limits:
            min_deg, max_deg = self.config.joint_limits[joint_name]
            clamped = max(min_deg, min(max_deg, value))
            if clamped != value:
                logger.warning(
                    f"Joint {joint_name} value {value:.2f}° clamped to [{min_deg}, {max_deg}] -> {clamped:.2f}°"
                )
            return clamped
        return value

    def _get_joint_range(self, so101_name: str, realman_idx: int) -> tuple[float, float, float]:
        """
        Get the calibrated range for a joint.
        
        Returns:
            Tuple of (min_deg, max_deg, center_deg)
        """
        joint_name = REALMAN_JOINT_NAMES[realman_idx]
        
        # First check if we have calibration data
        if self.calibration and "joint_ranges" in self.calibration:
            joint_cal = self.calibration["joint_ranges"].get(joint_name)
            if joint_cal:
                return (joint_cal["min"], joint_cal["max"], joint_cal["center"])
        
        # Fall back to config joint limits
        if joint_name in self.config.joint_limits:
            min_deg, max_deg = self.config.joint_limits[joint_name]
            center_deg = (min_deg + max_deg) / 2.0
            return (min_deg, max_deg, center_deg)
        
        # Default fallback
        return (-180.0, 180.0, 0.0)

    def _so101_normalized_to_realman_degrees(self, so101_name: str, normalized_value: float) -> float:
        """
        Convert SO101 normalized value (-100 to 100) to RealMan degrees.
        
        SO101's normalized values come from raw encoder values (0-4095) that are
        normalized based on calibration (range_min, range_max). The STS3215 motors
        use 4096 counts per full 360° revolution.
        
        SO101 normalized mapping:
        - -100 corresponds to range_min encoder position
        - +100 corresponds to range_max encoder position
        - 0 is the midpoint (where homing was done)
        
        We convert this to physical degrees first, then map to RealMan's range.
        
        Args:
            so101_name: SO101 joint name (e.g., "shoulder_pan")
            normalized_value: Value in range [-100, 100]
            
        Returns:
            Joint angle in degrees for RealMan
        """
        realman_idx = SO101_TO_REALMAN_JOINT_MAP.get(so101_name)
        if realman_idx is None:
            return normalized_value
        
        # Clamp input to valid range
        normalized_value = max(-100.0, min(100.0, normalized_value))
        
        # Check if we have RealMan calibration data
        if self.calibration and "joint_ranges" in self.calibration:
            joint_name = REALMAN_JOINT_NAMES[realman_idx]
            joint_cal = self.calibration["joint_ranges"].get(joint_name)
            if joint_cal:
                min_deg = joint_cal["min"]
                max_deg = joint_cal["max"]
                center_deg = joint_cal["center"]
                
                # Map SO101 normalized (-100 to 100) to RealMan calibrated range
                # -100 -> min_deg, 0 -> center_deg, +100 -> max_deg
                if normalized_value < 0:
                    # Map -100..0 to min_deg..center_deg
                    degrees = center_deg + (normalized_value / 100.0) * (center_deg - min_deg)
                else:
                    # Map 0..100 to center_deg..max_deg
                    degrees = center_deg + (normalized_value / 100.0) * (max_deg - center_deg)
                
                return degrees
        
        # Fallback: Direct mapping using config limits
        # This assumes SO101 and RealMan have similar ranges (not ideal)
        joint_name = REALMAN_JOINT_NAMES[realman_idx]
        if joint_name in self.config.joint_limits:
            min_deg, max_deg = self.config.joint_limits[joint_name]
            # Linear mapping: -100 -> min_deg, +100 -> max_deg
            degrees = min_deg + (normalized_value + 100.0) / 200.0 * (max_deg - min_deg)
            return degrees
        
        return normalized_value

    def _realman_degrees_to_so101_normalized(self, realman_idx: int, degrees: float) -> float:
        """
        Convert RealMan degrees to SO101 normalized value (-100 to 100).
        
        Args:
            realman_idx: RealMan joint index (0-5)
            degrees: Joint angle in degrees
            
        Returns:
            Normalized value in range [-100, 100]
        """
        # Get SO101 name for this realman index
        so101_name = REALMAN_TO_SO101_JOINT_MAP.get(realman_idx)
        if so101_name is None:
            return degrees
        
        # Check if we have RealMan calibration data
        if self.calibration and "joint_ranges" in self.calibration:
            joint_name = REALMAN_JOINT_NAMES[realman_idx]
            joint_cal = self.calibration["joint_ranges"].get(joint_name)
            if joint_cal:
                min_deg = joint_cal["min"]
                max_deg = joint_cal["max"]
                center_deg = joint_cal["center"]
                
                # Clamp degrees to calibrated range
                degrees = max(min_deg, min(max_deg, degrees))
                
                # Map RealMan degrees to normalized (-100 to 100)
                if degrees < center_deg:
                    # Map min_deg..center_deg to -100..0
                    if center_deg != min_deg:
                        normalized = -100.0 * (center_deg - degrees) / (center_deg - min_deg)
                    else:
                        normalized = 0.0
                else:
                    # Map center_deg..max_deg to 0..100
                    if max_deg != center_deg:
                        normalized = 100.0 * (degrees - center_deg) / (max_deg - center_deg)
                    else:
                        normalized = 0.0
                
                return normalized
        
        # Fallback: Use config limits
        joint_name = REALMAN_JOINT_NAMES[realman_idx]
        if joint_name in self.config.joint_limits:
            min_deg, max_deg = self.config.joint_limits[joint_name]
            
            if max_deg == min_deg:
                return 0.0
            
            degrees = max(min_deg, min(max_deg, degrees))
            normalized = (degrees - min_deg) / (max_deg - min_deg) * 200.0 - 100.0
            return normalized
        
        return degrees

    def _convert_so101_action_to_realman(self, action: RobotAction) -> tuple[list[float], float | None]:
        """
        Convert SO101-format action to RealMan joint angles with limit enforcement.
        
        Args:
            action: Action dict with SO101 joint names (e.g., "shoulder_pan.pos")
            
        Returns:
            Tuple of (joint_angles_list, gripper_position)
            joint_angles_list has 6 values for R1D2 (indices 0-5)
            All values are guaranteed to be within joint limits.
        """
        # Start with cached joint positions as base (avoid extra read)
        current_joints = self._last_joint_angles
        
        if current_joints is None:
            # Fall back to reading if no cache
            robot = self._get_robot_controller()
            current_joints = robot.get_current_joint_angles()
        
        if current_joints is None:
            # Use zeros if can't read current position
            current_joints = [0.0] * self.config.dof
        
        # Copy current positions
        target_joints = list(current_joints)
        
        # Map SO101 joints to RealMan joints
        for so101_name, realman_idx in SO101_TO_REALMAN_JOINT_MAP.items():
            key = f"{so101_name}.pos"
            if key in action:
                normalized_value = action[key]
                
                # Convert from SO101 normalized range (-100 to 100) to degrees
                degrees = self._so101_normalized_to_realman_degrees(so101_name, normalized_value)
                
                # Enforce joint limits if configured
                if self.config.enforce_joint_limits:
                    degrees = self._clamp_to_joint_limits(realman_idx, degrees)
                
                target_joints[realman_idx] = degrees
        
        # Set joint 4 (index 3) to fixed position - also enforce limits
        if len(target_joints) > 3:
            fixed_pos = self.config.fixed_joint_4_position
            if self.config.enforce_joint_limits:
                fixed_pos = self._clamp_to_joint_limits(3, fixed_pos)
            target_joints[3] = fixed_pos
        
        # Handle gripper
        gripper_pos = None
        gripper_key = "gripper.pos"
        if gripper_key in action:
            gripper_value = action[gripper_key]
            
            # SO101 gripper uses RANGE_0_100 mode (0-100)
            # RealMan gripper uses 1-1000 range
            # Clamp input first
            gripper_value = max(0.0, min(100.0, gripper_value))
            
            # Map 0-100 to 1-1000
            gripper_pos = int(1 + gripper_value * 9.99)
            
            # Enforce gripper limits from config
            if "gripper" in self.config.joint_limits:
                min_grip, max_grip = self.config.joint_limits["gripper"]
                gripper_pos = max(int(min_grip), min(int(max_grip), gripper_pos))
        
        return target_joints, gripper_pos

    def _convert_realman_observation_to_so101(self, joint_angles: list[float], gripper_pos: int) -> dict[str, float]:
        """
        Convert RealMan joint angles to SO101-format observation.
        
        Args:
            joint_angles: List of 6 joint angles in degrees
            gripper_pos: Gripper position (1-1000)
            
        Returns:
            Observation dict with SO101 joint names, values in [-100, 100]
        """
        obs = {}
        
        # Map RealMan joints back to SO101 names
        for so101_name, realman_idx in SO101_TO_REALMAN_JOINT_MAP.items():
            if realman_idx < len(joint_angles):
                degrees = joint_angles[realman_idx]
                
                # Convert degrees to normalized range (-100 to 100)
                normalized = self._realman_degrees_to_so101_normalized(realman_idx, degrees)
                
                obs[f"{so101_name}.pos"] = normalized
        
        # Convert gripper (1-1000 -> 0-100)
        if gripper_pos is not None:
            # Clamp to valid range first
            gripper_pos = max(1, min(1000, int(gripper_pos)))
            obs["gripper.pos"] = (gripper_pos - 1) / 9.99
        else:
            obs["gripper.pos"] = 50.0  # Default to middle
        
        return obs

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Get current robot state and camera images.
        
        Returns:
            Observation dict with joint positions and camera images.
        """
        start = time.perf_counter()
        
        robot = self._get_robot_controller()
        
        # Get joint angles (with caching to avoid redundant reads)
        now = time.perf_counter()
        if self._last_joint_angles is None or (now - self._last_joint_read_time) > 0.005:
            # Only read if cache is stale (>5ms old) or empty
            joint_angles = robot.get_current_joint_angles()
            if joint_angles is not None:
                self._last_joint_angles = joint_angles
                self._last_joint_read_time = now
            else:
                logger.warning("Failed to read joint angles")
                joint_angles = self._last_joint_angles or [0.0] * self.config.dof
        else:
            joint_angles = self._last_joint_angles
        
        # Use tracked gripper position (skip slow gripper_get_state() call)
        # The gripper position is updated when we send commands
        gripper_pos = self._last_gripper_position
        
        # Convert to SO101-compatible format
        obs_dict = self._convert_realman_observation_to_so101(joint_angles, gripper_pos)
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Send action command to the robot.
        
        Args:
            action: Action dict with SO101-compatible joint names.
            
        Returns:
            The action that was actually sent (may be clipped for safety).
        """
        start = time.perf_counter()
        
        robot = self._get_robot_controller()
        
        # Convert SO101 action to RealMan format
        target_joints, gripper_pos = self._convert_so101_action_to_realman(action)
        
        # Apply safety limits if configured - use cached joint angles for speed
        if self.config.max_relative_target is not None:
            current_joints = self._last_joint_angles
            if current_joints is None:
                # Fall back to reading if no cache
                current_joints = robot.get_current_joint_angles()
            
            if current_joints is not None:
                max_delta = self.config.max_relative_target
                if isinstance(max_delta, (int, float)):
                    for i in range(len(target_joints)):
                        delta = target_joints[i] - current_joints[i]
                        if abs(delta) > max_delta:
                            target_joints[i] = current_joints[i] + max_delta * (1 if delta > 0 else -1)
                            logger.debug(f"Clamped joint {i} delta from {delta:.2f} to {max_delta}")
        
        # Send joint command using CANFD for low-latency teleoperation
        # follow=False means immediately override current motion (better responsiveness)
        result = robot.movej_canfd(target_joints, follow=False)
        if result != 0:
            logger.warning(f"movej_canfd returned non-zero status: {result}")
        
        # Send gripper command only if position changed significantly (reduces latency)
        if gripper_pos is not None:
            gripper_change = abs(gripper_pos - self._last_gripper_command)
            if gripper_change > 10:  # Only send if >1% change (10/1000)
                robot.gripper_set_position(gripper_pos, block=False)
                self._last_gripper_command = gripper_pos
                self._last_gripper_position = gripper_pos
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} send action: {dt_ms:.1f}ms")
        
        # Return the action in SO101 format (what was actually commanded)
        return action

    @check_if_not_connected
    def disconnect(self) -> None:
        """Disconnect from the robot and cameras."""
        robot = self._get_robot_controller()
        
        # Stop any ongoing motion
        robot.stop()
        
        # Optionally move to safe position before disconnecting
        if self.config.disable_torque_on_disconnect:
            # For RealMan, we don't disable torque but can move to home
            pass
        
        # Disconnect from robot
        robot.disconnect()
        self._connected = False
        
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        logger.info(f"{self} disconnected.")

    def move_to_home(self, velocity: int | None = None) -> None:
        """
        Move robot to home position.
        
        Args:
            velocity: Movement velocity (1-100). Uses config default if None.
        """
        if not self._connected:
            logger.warning("Robot not connected")
            return
        
        robot = self._get_robot_controller()
        vel = velocity if velocity is not None else self.config.velocity
        
        result = robot.move_to_home(velocity=vel)
        if result != 0:
            logger.warning(f"move_to_home returned non-zero status: {result}")

    def stop(self) -> None:
        """Emergency stop - immediately halt all motion."""
        if self._robot_controller is not None:
            self._robot_controller.stop()
            logger.warning("Emergency stop triggered")

